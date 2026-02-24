"""
End-to-end Keras-ready loader for your *semicolon-list CSV* format.

What it does:
- Reads your example.csv (one row = one object)
- Parses semicolon-separated fid/mag/jd
- Uses maxclass (comma-separated) as the object-level multi-label target
- Builds a label vocabulary (canonical labels) with optional synonym + parent propagation
- Produces a tf.data.Dataset compatible with the model skeleton I gave earlier

Notes:
- ZTF fid: 1->g, 2->r (we map to band_id 0/1)
- You can swap in LSST by providing band letters or mapping to 0..5 similarly.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from conf import *

# ---------------------------
# Parsing helpers
# ---------------------------
def split_semicol(s: str) -> list[str]:
    if not isinstance(s, str) or s.strip() == "":
        return []
    # keep order; strip whitespace
    return [x.strip() for x in s.split(";") if x.strip() != ""]


def split_commas(s: str) -> list[str]:
    if not isinstance(s, str) or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def normalize_label(lbl: str) -> str:
    # minimal normalization; customize as you like
    lbl = lbl.strip()
    lbl = re.sub(r"\s+", "_", lbl)
    return lbl


def fid_to_band_id_ztf(fid: int) -> int:
    # ZTF fid: 1=g, 2=r. Map to 0..1
    if fid == 1:
        return 0  # g
    if fid == 2:
        return 1  # r
    # fallback: keep nonstandard fids as-is but shifted; or raise
    return int(fid)  # you may prefer: raise ValueError


# ---------------------------
# Ontology / label mapping (synonyms + parent propagation)
# ---------------------------

def normalize_label(lbl: str) -> str:
    lbl = lbl.strip()
    lbl = re.sub(r"\s+", "_", lbl)
    return lbl


_CAND_PREFIX = re.compile(r"^candidate[_\-\s]+(.+)$", re.IGNORECASE)
_CAND_SUFFIX = re.compile(r"^(.+?)[_\-\s]+candidate$", re.IGNORECASE)

def split_candidate(lbl: str):
    """
    Returns (base_label, is_candidate) from a normalized label.

    Handles:
      Candidate_X  / Candidate-X / Candidate X
      X_Candidate  / X-Candidate / X Candidate
    """
    lbl = lbl.strip().strip("_-")  # trim common separators

    m = _CAND_PREFIX.match(lbl)
    if m:
        base = m.group(1).strip().strip("_-")
        return base, True

    m = _CAND_SUFFIX.match(lbl)
    if m:
        base = m.group(1).strip().strip("_-")
        return base, True

    return lbl, False

def canonicalize_labels(raw_labels,
                        synonym_to_canon=SYNONYM_TO_CANON,
                        propagate_parents: bool = True,
                        drop_unknown: bool = True):
    out = set()

    for x in raw_labels:
        x = normalize_label(x)
        if not x:
            continue

        # apply non-candidate synonyms first
        x = synonym_to_canon.get(x, x)

        base, is_cand = split_candidate(x)
        if is_cand:
            cand = f"{base}_Candidate"
            out.add(cand)
            if propagate_parents and base:
                out.add(base)
        else:
            out.add(base)

    if drop_unknown:
        out = {x for x in out if x.lower() != "unknown"}

    return sorted(out)

# ---------------------------
# Build label vocabulary from CSV
# ---------------------------
# NOTE: Deprecated: kept for reference.
# def build_label_vocab_old(csv_path: str, label_col: str = "maxclass") -> dict[str, int]:
#     df = pd.read_csv(csv_path)
#     all_labels = set()
#     for s in df[label_col].fillna("").tolist():
#         raw = split_commas(s)
#         canon = canonicalize_labels(raw)
#         all_labels.update(canon)
#     labels = sorted(all_labels)
#     return {lbl: i for i, lbl in enumerate(labels)}


def build_label_vocab(
    csv_path: str, label_col: str = "maxclass", class_col: str = "collect_list(class)"
) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    all_labels = set()
    for _, r in df.iterrows():
        point_labels = split_semicol(r.get(class_col, ""))
        max_labels = split_commas(r.get(label_col, ""))
        canon = canonicalize_labels(point_labels + max_labels)
        all_labels.update(canon)
    labels = sorted(all_labels)
    return {lbl: i for i, lbl in enumerate(labels)}


# ---------------------------
# Convert CSV rows -> objects (dicts with arrays)
# ---------------------------
def load_objects_from_csv(
    csv_path: str,
    label2id: dict[str, int],
    use_label_col: str = "maxclass",
    class_col: str = "collect_list(class)",
    fid_col: str = "collect_list(fid)",
    mag_col: str = "collect_list(magpsf)",
    jd_col: str = "collect_list(jd)",
    object_id_col: str = "objectId",
    fid_to_band=fid_to_band_id_ztf,
) -> list[dict]:
    df = pd.read_csv(csv_path)

    objects = []
    L = len(label2id)

    for _, r in df.iterrows():
        obj_id = r[object_id_col]

        fids = list(map(int, split_semicol(r[fid_col])))
        mags = list(map(float, split_semicol(r[mag_col])))
        jds = list(map(float, split_semicol(r[jd_col])))

        if not (len(fids) == len(mags) == len(jds)):
            raise ValueError(
                f"Length mismatch for {obj_id}: "
                f"fid={len(fids)}, mag={len(mags)}, jd={len(jds)}"
            )

        band = np.array([fid_to_band(x) for x in fids], dtype=np.int32)
        t = np.array(jds, dtype=np.float32)
        mag = np.array(mags, dtype=np.float32)

        # Object-level multi-label target from per-point classes + maxclass.
        point_labels = split_semicol(r.get(class_col, ""))  # e.g. "EclBin;EB*;..."
        max_labels = split_commas(r.get(use_label_col, ""))  # e.g. "EB*,EclBin"

        raw_labels = point_labels + max_labels
        canon_labels = canonicalize_labels(raw_labels)

        y = np.zeros((L,), dtype=np.float32)
        for lbl in canon_labels:
            if lbl in label2id:
                y[label2id[lbl]] = 1.0

        objects.append({"objectId": obj_id, "t": t, "mag": mag, "band": band, "y": y})

    return objects


# ---------------------------
# tf.data.Dataset using the earlier prefix sampler
# (paste these in if you didn't keep them from the previous message)
# ---------------------------
TMAX = 256
MIN_PREFIX = 12
MAG_JITTER_STD = 0.02
POINT_DROPOUT_P = 0.05


def _make_dt_trel(t):
    t = tf.cast(t, tf.float32)
    t0 = t[0]
    trel = t - t0
    dt = tf.concat([tf.zeros([1], tf.float32), t[1:] - t[:-1]], axis=0)
    return dt, trel


def _augment(mag, keep_mask):
    mag = tf.cast(mag, tf.float32)
    if MAG_JITTER_STD and MAG_JITTER_STD > 0:
        mag = mag + tf.random.normal(tf.shape(mag), stddev=MAG_JITTER_STD)
    if POINT_DROPOUT_P and POINT_DROPOUT_P > 0:
        drop = tf.random.uniform(tf.shape(keep_mask)) < POINT_DROPOUT_P
        keep_mask = tf.logical_and(keep_mask, tf.logical_not(drop))
    return mag, keep_mask


def _pad_to_tmax(x, mask, tmax=TMAX):
    n = tf.shape(x)[0]
    pad_len = tf.maximum(0, tmax - n)
    x_pad = tf.pad(x, [[0, pad_len], [0, 0]])
    mask_pad = tf.pad(mask, [[0, pad_len]], constant_values=False)
    return x_pad[:tmax], mask_pad[:tmax]


def _sample_prefix(t, mag, band, y, training=True):
    t = tf.cast(t, tf.float32)
    mag = tf.cast(mag, tf.float32)
    band = tf.cast(band, tf.int32)
    y = tf.cast(y, tf.float32)

    n = tf.shape(t)[0]
    idx = tf.argsort(t)
    t = tf.gather(t, idx)
    mag = tf.gather(mag, idx)
    band = tf.gather(band, idx)

    if training:
        lo = tf.minimum(tf.cast(MIN_PREFIX, tf.int32), n)
        lo = tf.maximum(lo, 1)
        k = tf.random.uniform([], minval=lo, maxval=n + 1, dtype=tf.int32)
    else:
        k = n

    t = t[:k]
    mag = mag[:k]
    band = band[:k]

    dt, trel = _make_dt_trel(t)
    keep_mask = tf.ones([tf.shape(t)[0]], dtype=tf.bool)
    if training:
        mag, keep_mask = _augment(mag, keep_mask)

    x_num = tf.stack([dt, trel, mag], axis=-1)  # [k,3]
    x_num = tf.boolean_mask(x_num, keep_mask)
    band = tf.boolean_mask(band, keep_mask)

    k2 = tf.shape(x_num)[0]
    mask = tf.ones([k2], dtype=tf.bool)

    x_num_pad, mask_pad = _pad_to_tmax(x_num, mask, TMAX)
    band_pad, _ = _pad_to_tmax(
        tf.expand_dims(tf.cast(band, tf.float32), -1), mask, TMAX
    )
    band_pad = tf.cast(tf.squeeze(band_pad, axis=-1), tf.int32)

    return (x_num_pad, band_pad, mask_pad), y


def make_dataset(objects, num_labels: int, batch_size=64, training=True, shuffle=4096):
    def gen():
        for o in objects:
            yield (o["t"], o["mag"], o["band"], o["y"])

    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # t
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # mag
        tf.TensorSpec(shape=(None,), dtype=tf.int32),  # band
        tf.TensorSpec(shape=(num_labels,), dtype=tf.float32),  # y
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if training:
        ds = ds.shuffle(shuffle, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda t, mag, band, y: _sample_prefix(t, mag, band, y, training=training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.map(
        lambda x, y: ({"x_num": x[0], "band": x[1], "mask": x[2]}, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_objects_hash(objects, train_frac=0.8):
    train_objs, val_objs = [], []
    for o in objects:
        oid = o["objectId"]
        h = hashlib.md5(oid.encode("utf-8")).hexdigest()
        u = int(h[:8], 16) / 16**8  # uniform in [0,1)
        (train_objs if u < train_frac else val_objs).append(o)
    return train_objs, val_objs


def build_datasets(csv_path):
    label2id = build_label_vocab(csv_path, label_col="maxclass")
    objects = load_objects_from_csv(csv_path, label2id)
    train_objs, val_objs = split_objects_hash(objects, train_frac=0.8)

    train_ds = make_dataset(
        train_objs, num_labels=len(label2id), batch_size=32, training=True
    )
    val_ds = make_dataset(
        val_objs, num_labels=len(label2id), batch_size=32, training=False
    )
    return train_ds, val_ds, label2id


# ---------------------------
# Example: wire it all together
# ---------------------------
if __name__ == "__main__":
    csv_path = "example.csv"

    label2id = build_label_vocab(csv_path, label_col="maxclass")
    print("NUM_LABELS =", len(label2id))
    print("Some labels:", list(label2id.keys())[:20])

    objects = load_objects_from_csv(csv_path, label2id, use_label_col="maxclass")

    # naive split for demo; do *group split* by objectId in real life
    rng = np.random.default_rng(0)
    rng.shuffle(objects)
    cut = int(0.8 * len(objects))
    train_objs, val_objs = objects[:cut], objects[cut:]

    train_ds = make_dataset(
        train_objs, num_labels=len(label2id), batch_size=32, training=True
    )
    val_ds = make_dataset(
        val_objs, num_labels=len(label2id), batch_size=32, training=False
    )

    # Now plug train_ds / val_ds into your Keras model training.
    # (Use the build_model/compile_and_train functions from the previous skeleton.)
