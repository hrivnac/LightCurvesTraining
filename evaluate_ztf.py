"""
evaluate_ztf.py

Evaluate a trained Keras lightcurve model on a NEW dataset (same semicolon-list CSV format).

What it does:
- Loads:
    - saved model (.keras)
    - label2id.json (from training run)
- Parses the new CSV into object sequences (t, mag, band) and targets y
    - uses y = canonicalize( union(collect_list(class)) ∪ maxclass )
    - labels not present in label2id are ignored (cannot be evaluated)
- Builds a tf.data.Dataset (NO training augmentations, uses full prefix by default)
- Runs evaluation:
    - loss
    - PR-AUC, ROC-AUC (multi-label)
    - micro-F1 at configurable threshold
- Optionally writes per-object top-K predictions to a CSV

Usage example:
python evaluate_ztf.py \
  --model runs/ztf_run1/best_model.keras \
  --label2id runs/ztf_run1/label2id.json \
  --data new_dataset.csv \
  --out runs/ztf_run1/eval_preds.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

def threshold_sweep(y_true, y_prob, thresholds=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3)):
    y_true_i = (y_true >= 0.5).astype(np.int32)
    rows = []
    for thr in thresholds:
        y_hat = (y_prob >= thr).astype(np.int32)
        tp = (y_hat & y_true_i).sum()
        fp = (y_hat & (1 - y_true_i)).sum()
        fn = ((1 - y_hat) & y_true_i).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = (2*prec*rec) / (prec + rec + 1e-9)
        rows.append((thr, int(tp), int(fp), int(fn), float(prec), float(rec), float(f1)))
    return rows

def per_class_report(y_true: np.ndarray,
                     y_prob: np.ndarray,
                     id2label: dict[int, str],
                     threshold: float = 0.5,
                     min_support: int = 20,
                     top_n: int | None = 50) -> pd.DataFrame:
    y_hat = (y_prob >= threshold).astype(np.int32)
    y_true_i = (y_true >= 0.5).astype(np.int32)

    tp = (y_hat & y_true_i).sum(axis=0)
    fp = (y_hat & (1 - y_true_i)).sum(axis=0)
    fn = ((1 - y_hat) & y_true_i).sum(axis=0)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = (2 * precision * recall) / (precision + recall + 1e-9)

    support   = y_true_i.sum(axis=0)
    pred_pos  = y_hat.sum(axis=0)
    mean_p    = y_prob.mean(axis=0)

    df = pd.DataFrame({
        "label":     [id2label[i] for i in range(y_true.shape[1])],
        "support":   support.astype(int),
        "pred_pos":  pred_pos.astype(int),
        "tp":        tp.astype(int),
        "fp":        fp.astype(int),
        "fn":        fn.astype(int),
        "precision": precision.astype(float),
        "recall":    recall.astype(float),
        "f1":        f1.astype(float),
        "mean_p":    mean_p.astype(float),
    })

    df = df[df["support"] >= min_support].copy()
    df = df.sort_values(["f1", "support"], ascending=[False, False])

    if top_n is not None:
        df = df.head(top_n)

    return df

class MaskedAttentionPooling(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.query = layers.Dense(self.d_model, use_bias=False)
        self.score = layers.Dense(1, use_bias=False)

    def call(self, x, mask=None, training=None):
        h = tf.nn.tanh(self.query(x))            # [B,T,D]
        logits = tf.squeeze(self.score(h), -1)   # [B,T]
        if mask is not None:
            neg_inf = tf.constant(-1e9, dtype=logits.dtype)
            logits = tf.where(mask, logits, neg_inf)
        w = tf.nn.softmax(logits, axis=-1)       # [B,T]
        z = tf.einsum("bt,btd->bd", w, x)        # [B,D]
        return z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


class LabelVectorHead(layers.Layer):
    def __init__(self, num_labels, d_model, use_cosine=False, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = int(num_labels)
        self.d_model = int(d_model)
        self.use_cosine = bool(use_cosine)

    def build(self, input_shape):
        self.L = self.add_weight(
            shape=(self.num_labels, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="label_vectors",
        )
        self.b = self.add_weight(
            shape=(self.num_labels,),
            initializer="zeros",
            trainable=True,
            name="label_bias",
        )

    def call(self, z):
        if self.use_cosine:
            z_norm = tf.nn.l2_normalize(z, axis=-1)
            L_norm = tf.nn.l2_normalize(self.L, axis=-1)
            logits = tf.matmul(z_norm, L_norm, transpose_b=True) + self.b
        else:
            logits = tf.matmul(z, self.L, transpose_b=True) + self.b
        return logits

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_labels": self.num_labels,
            "d_model": self.d_model,
            "use_cosine": self.use_cosine,
        })
        return cfg

# ---------------------------
# Parsing helpers
# ---------------------------
def split_semicol(s: str) -> List[str]:
    if not isinstance(s, str) or s.strip() == "":
        return []
    return [x.strip() for x in s.split(";") if x.strip() != ""]

def split_commas(s: str) -> List[str]:
    if not isinstance(s, str) or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]

def normalize_label(lbl: str) -> str:
    lbl = lbl.strip()
    lbl = re.sub(r"\s+", "_", lbl)
    return lbl

def fid_to_band_id_ztf(fid: int) -> int:
    # ZTF fid: 1=g, 2=r -> map to 0..1
    if fid == 1:
        return 0
    if fid == 2:
        return 1
    return int(fid)


# ---------------------------
# Ontology mapping (keep IDENTICAL to training)
# ---------------------------
SYNONYM_TO_CANON = {
    "Candidate_EB*": "EB*_Candidate",
    "RRLyr": "RRLyrae",
    # add your training-time synonyms here
}

PARENT_OF = {
    "EB*_Candidate": "EB*",
    # add your training-time parent links here
}

def canonicalize_labels(raw_labels: List[str],
                        synonym_to_canon: Dict[str, str] = SYNONYM_TO_CANON,
                        parent_of: Dict[str, str] = PARENT_OF,
                        propagate_parents: bool = True) -> List[str]:
    canon = []
    for x in raw_labels:
        x = normalize_label(x)
        x = synonym_to_canon.get(x, x)
        canon.append(x)

    if propagate_parents:
        out = set(canon)
        # 1-level parent; extend to multi-level if needed
        for x in list(out):
            p = parent_of.get(x)
            if p:
                out.add(p)
        canon = sorted(out)

    # keep consistent with training
    canon = [x for x in canon if x.lower() != "unknown"]
    return canon


# ---------------------------
# CSV -> objects
# ---------------------------
def load_objects_from_csv(csv_path: str,
                          label2id: Dict[str, int],
                          object_id_col: str = "objectId",
                          fid_col: str = "collect_list(fid)",
                          mag_col: str = "collect_list(magpsf)",
                          jd_col: str = "collect_list(jd)",
                          class_col: str = "collect_list(class)",
                          maxclass_col: str = "maxclass") -> List[dict]:
    df = pd.read_csv(csv_path)
    L = len(label2id)

    objects = []
    for _, r in df.iterrows():
        obj_id = r[object_id_col]

        fids = list(map(int, split_semicol(r[fid_col])))
        mags = list(map(float, split_semicol(r[mag_col])))
        jds  = list(map(float, split_semicol(r[jd_col])))

        if not (len(fids) == len(mags) == len(jds)):
            raise ValueError(f"Length mismatch for {obj_id}: "
                             f"fid={len(fids)}, mag={len(mags)}, jd={len(jds)}")

        band = np.array([fid_to_band_id_ztf(x) for x in fids], dtype=np.int32)
        t = np.array(jds, dtype=np.float32)
        mag = np.array(mags, dtype=np.float32)

        # Targets: union(point classes) ∪ maxclass, then canonicalize
        point_labels = split_semicol(r.get(class_col, ""))
        max_labels = split_commas(r.get(maxclass_col, ""))
        canon_labels = canonicalize_labels(point_labels + max_labels)

        y = np.zeros((L,), dtype=np.float32)
        for lbl in canon_labels:
            j = label2id.get(lbl)
            if j is not None:
                y[j] = 1.0

        objects.append({"objectId": obj_id, "t": t, "mag": mag, "band": band, "y": y})

    return objects


# ---------------------------
# tf.data: build features (NO augmentation in eval)
# ---------------------------
def _make_dt_trel(t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    t = tf.cast(t, tf.float32)
    t0 = t[0]
    trel = t - t0
    dt = tf.concat([tf.zeros([1], tf.float32), t[1:] - t[:-1]], axis=0)
    return dt, trel

def _pad_to_tmax(x: tf.Tensor, mask: tf.Tensor, tmax: int) -> Tuple[tf.Tensor, tf.Tensor]:
    n = tf.shape(x)[0]
    pad_len = tf.maximum(0, tmax - n)
    x_pad = tf.pad(x, [[0, pad_len], [0, 0]])
    mask_pad = tf.pad(mask, [[0, pad_len]], constant_values=False)
    return x_pad[:tmax], mask_pad[:tmax]

def _to_model_inputs(t, mag, band, y, tmax: int):
    # Sort by time
    t = tf.cast(t, tf.float32)
    mag = tf.cast(mag, tf.float32)
    band = tf.cast(band, tf.int32)
    y = tf.cast(y, tf.float32)

    idx = tf.argsort(t)
    t = tf.gather(t, idx)
    mag = tf.gather(mag, idx)
    band = tf.gather(band, idx)

    dt, trel = _make_dt_trel(t)
    x_num = tf.stack([dt, trel, mag], axis=-1)  # [N,3]
    mask = tf.ones([tf.shape(x_num)[0]], dtype=tf.bool)

    x_num_pad, mask_pad = _pad_to_tmax(x_num, mask, tmax)
    band_pad, _ = _pad_to_tmax(tf.expand_dims(tf.cast(band, tf.float32), -1), mask, tmax)
    band_pad = tf.cast(tf.squeeze(band_pad, axis=-1), tf.int32)

    return {"x_num": x_num_pad, "band": band_pad, "mask": mask_pad}, y

def make_eval_dataset(objects: List[dict], num_labels: int, batch_size: int, tmax: int) -> tf.data.Dataset:
    def gen():
        for o in objects:
            yield (o["t"], o["mag"], o["band"], o["y"])

    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),       # t
        tf.TensorSpec(shape=(None,), dtype=tf.float32),       # mag
        tf.TensorSpec(shape=(None,), dtype=tf.int32),         # band
        tf.TensorSpec(shape=(num_labels,), dtype=tf.float32), # y
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.map(lambda t, mag, band, y: _to_model_inputs(t, mag, band, y, tmax),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------
# Metrics: micro-F1
# ---------------------------
def micro_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_hat = (y_prob >= threshold).astype(np.float32)
    tp = (y_hat * y_true).sum()
    fp = (y_hat * (1 - y_true)).sum()
    fn = ((1 - y_hat) * y_true).sum()
    return float((2 * tp) / (2 * tp + fp + fn + 1e-9))


# ---------------------------
# Predictions export
# ---------------------------
def export_topk_predictions(objects: List[dict],
                            y_prob: np.ndarray,
                            id2label: Dict[int, str],
                            out_csv: str,
                            topk: int = 10) -> None:
    rows = []
    for i, o in enumerate(objects):
        p = y_prob[i]
        top = np.argsort(p)[-topk:][::-1]
        labels = [id2label[j] for j in top]
        scores = [float(p[j]) for j in top]
        rows.append({
            "objectId": o["objectId"],
            "top_labels": ";".join(labels),
            "top_scores": ";".join([f"{s:.6f}" for s in scores]),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to saved .keras model")
    ap.add_argument("--label2id", required=True, help="Path to label2id.json saved during training")
    ap.add_argument("--data", required=True, help="New dataset CSV (same format)")
    ap.add_argument("--batch", type=int, default=256, help="Batch size for evaluation")
    ap.add_argument("--tmax", type=int, default=64, help="Max padded length (must match training)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for micro-F1")
    ap.add_argument("--out", default="", help="Optional: write per-object top-K predictions CSV")
    ap.add_argument("--topk", type=int, default=10, help="Top-K labels to export per object")
    args = ap.parse_args()

    # Load label mapping
    with open(args.label2id, "r") as f:
        label2id = json.load(f)
    # json keys are strings already; ensure int values
    label2id = {k: int(v) for k, v in label2id.items()}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)

    print("Loading model:", args.model)
    #model = tf.keras.models.load_model(args.model, compile=False)

    model = tf.keras.models.load_model(
        args.model,
        compile=False,
        custom_objects={
            "MaskedAttentionPooling": MaskedAttentionPooling,
            "LabelVectorHead": LabelVectorHead,
        },
    )

    # Compile for loss + AUC metrics (optional but useful)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR",  multi_label=True, name="auc_pr"),
        ],
    )

    print("Parsing data:", args.data)
    objects = load_objects_from_csv(args.data, label2id)
    print("Objects:", len(objects), "Labels:", num_labels)

    ds = make_eval_dataset(objects, num_labels=num_labels, batch_size=args.batch, tmax=args.tmax)

    # Evaluate via Keras (loss + AUCs)
    print("\nKeras evaluate:")
    results = model.evaluate(ds, verbose=1, return_dict=True)
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    # Predict to compute micro-F1 and optionally export predictions
    print("\nPredicting probabilities...")
    y_prob = model.predict(ds, verbose=1)

    print("\nPred prob stats:")
    print("  shape:", y_prob.shape)
    print("  min:", float(np.min(y_prob)))
    print("  mean:", float(np.mean(y_prob)))
    print("  median:", float(np.median(y_prob)))
    print("  max:", float(np.max(y_prob)))
    print("  p99:", float(np.quantile(y_prob, 0.99)))

    # how many predicted positives at different thresholds?
    for thr in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
        pred_pos = int((y_prob >= thr).sum())
        print(f"  predicted positives @ {thr:>4}: {pred_pos}")
        y_true = np.stack([o["y"] for o in objects], axis=0)

    print("\nThreshold sweep (micro):")
    for thr, tp, fp, fn, prec, rec, f1 in threshold_sweep(y_true, y_prob):
        print(f"thr={thr:>5}: tp={tp} fp={fp} fn={fn}  prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
    
        print("\nGround truth stats:")
        per_obj = y_true.sum(axis=1)
        print("  mean labels/object:", float(per_obj.mean()))
        print("  median labels/object:", float(np.median(per_obj)))
        print("  fraction objects with 0 labels:", float((per_obj == 0).mean()))
        print("  total positives:", int(y_true.sum()))    
    
    report = per_class_report(
        y_true=y_true,
        y_prob=y_prob,
        id2label=id2label,
        threshold=args.threshold,
        min_support=20,     # change as you like
        top_n=50            # show top 50 classes
    )
    
    print("\nPer-class summary (top):")
    print(report.to_string(index=False, justify="left",
                           formatters={
                               "precision": "{:.3f}".format,
                               "recall": "{:.3f}".format,
                               "f1": "{:.3f}".format,
                               "mean_p": "{:.3f}".format,
                           }))
    
    out_table = os.path.join(os.path.dirname(args.out) if args.out else ".", "eval_per_class.csv")
    report.to_csv(out_table, index=False)
    print("\nWrote per-class table to:", out_table)    
    
    f1 = micro_f1(y_true, y_prob, threshold=args.threshold)
    print(f"\nMicro-F1 @ {args.threshold:.2f}: {f1:.6f}")

    if args.out:
        print("Writing predictions to:", args.out)
        export_topk_predictions(objects, y_prob, id2label, args.out, topk=args.topk)

    print("\nDone.")


if __name__ == "__main__":
    main()
