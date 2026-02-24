# pretrain_lsst.py
from __future__ import annotations
import os, time, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from lsst_io import (
    NUM_BANDS_LSST, list_json_files, stable_hash_split,
    make_file_dataset, parse_json_to_tensors,
    _make_dt_trel, _pad_to_tmax
)

# -----------------
# Config (good CPU defaults)
# -----------------
TMAX = 96            # LSST often sparse per band; total points may vary; start with 96
MIN_PREFIX = 12
BATCH = 256
D_MODEL = 128
BAND_EMB = 16
DROPOUT = 0.2

MASK_PROB = 0.15     # fraction of points to mask (on observed mags only)
MAG_JITTER_STD = 0.01

# -----------------
# Custom layers (registered so saved models reload cleanly)
# -----------------
#@keras.saving.register_keras_serializable()
@tf.keras.utils.register_keras_serializable()
class MaskedAttentionPooling(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True  # <--- add this line
        self.d_model = int(d_model)
        self.query = layers.Dense(self.d_model, use_bias=False)
        self.score = layers.Dense(1, use_bias=False)

    def call(self, x, mask=None):
        h = tf.nn.tanh(self.query(x))
        logits = tf.squeeze(self.score(h), -1)
        if mask is not None:
            neg_inf = tf.constant(-1e9, dtype=logits.dtype)
            logits = tf.where(mask, logits, neg_inf)
        w = tf.nn.softmax(logits, axis=-1)
        z = tf.einsum("bt,btd->bd", w, x)
        return z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg
        
def compute_mask(self, inputs, mask=None):
        return None
# -----------------
# Dataset mapping: prefix sampling + masking + padding
# Outputs:
#   x: {"x_num": [T,4], "band":[T], "mask":[T]}
#   y: true magnitudes [T,1]
#   sample_weight: masked positions [T] (so loss is only on masked points)
# -----------------
def to_pretrain_example(t, mag, band, training=True):
    # sort by time (should already be sorted, but safe)
    idx = tf.argsort(t)
    t = tf.gather(t, idx)
    mag = tf.gather(mag, idx)
    band = tf.gather(band, idx)

    n = tf.shape(t)[0]
    # filter empties upstream, but safety:
    n = tf.maximum(n, 1)

    # prefix
    if training:
        lo = tf.minimum(tf.cast(MIN_PREFIX, tf.int32), n)
        lo = tf.maximum(lo, 1)
        k = tf.random.uniform([], minval=lo, maxval=n + 1, dtype=tf.int32)
    else:
        k = n

    t = t[:k]
    mag_true = tf.cast(mag[:k], tf.float32)
    band = tf.cast(band[:k], tf.int32)

    # time features
    dt, trel = _make_dt_trel(t)
    # optional jitter for robustness
    if training and MAG_JITTER_STD > 0:
        mag_true = mag_true + tf.random.normal(tf.shape(mag_true), stddev=MAG_JITTER_STD)

    # choose which points to mask (only real points)
    if training:
        m = tf.random.uniform(tf.shape(mag_true)) < MASK_PROB
    else:
        # for evaluation of pretrain loss you can still mask; but here keep none masked
        m = tf.zeros(tf.shape(mag_true), dtype=tf.bool)

    # create masked input magnitude and a mag_mask feature (1=masked, 0=visible)
    mag_in = tf.where(m, tf.zeros_like(mag_true), mag_true)
    mag_mask_feat = tf.cast(m, tf.float32)

    # numeric token features: dt, trel, mag_in, mag_mask_feat  => 4 dims
    x_num = tf.stack([dt, trel, mag_in, mag_mask_feat], axis=-1)  # [k,4]

    # padding mask for real tokens
    tok_mask = tf.ones([tf.shape(x_num)[0]], dtype=tf.bool)

    # pad to TMAX
    x_num_pad, tok_mask_pad = _pad_to_tmax(x_num, tok_mask, TMAX)
    band_pad, _ = _pad_to_tmax(tf.expand_dims(tf.cast(band, tf.float32), -1), tok_mask, TMAX)
    band_pad = tf.cast(tf.squeeze(band_pad, axis=-1), tf.int32)

    # targets: true magnitude, padded
    y = tf.expand_dims(mag_true, -1)  # [k,1]
    y_pad, _ = _pad_to_tmax(y, tok_mask, TMAX)  # [T,1]

    # sample weights: 1 only where masked AND token exists, else 0
    sw = tf.cast(m, tf.float32)  # [k]
    sw_pad, _ = _pad_to_tmax(tf.expand_dims(sw, -1), tok_mask, TMAX)  # [T,1]
    sw_pad = tf.squeeze(sw_pad, -1)  # [T]

    x = (x_num_pad, band_pad, tok_mask_pad)   # positional order MUST match model inputs
    return x, y_pad, sw_pad

def build_pretrain_dataset(paths, batch_size, training=True):
    ds = make_file_dataset(paths)
    if training:
        ds = ds.shuffle(20000, reshuffle_each_iteration=True)

    ds = ds.map(parse_json_to_tensors, num_parallel_calls=tf.data.AUTOTUNE)

    # filter empty objects (length>0)
    ds = ds.filter(lambda t, mag, band: tf.shape(t)[0] > 0)

    ds = ds.map(lambda t, mag, band: to_pretrain_example(t, mag, band, training=training),
                num_parallel_calls=tf.data.AUTOTUNE)

    # Keras expects (x, y, sample_weight)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------
# Model: encoder + per-step regression head
# -----------------
def build_encoder(num_bands=NUM_BANDS_LSST, tmax=TMAX, d_model=D_MODEL, band_emb=BAND_EMB, dropout=DROPOUT):
    x_num_in = keras.Input(shape=(tmax, 4), dtype=tf.float32, name="x_num")
    band_in  = keras.Input(shape=(tmax,), dtype=tf.int32, name="band")
    mask_in  = keras.Input(shape=(tmax,), dtype=tf.bool, name="mask")

    b = layers.Embedding(num_bands, band_emb, name="band_embedding")(band_in)
    x = layers.Concatenate()([x_num_in, b])
    x = layers.Dense(d_model)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.GRU(d_model, return_sequences=True, dropout=dropout)(x, mask=mask_in)
    x = layers.GRU(d_model, return_sequences=True, dropout=dropout)(x, mask=mask_in)

    # Pass mask explicitly (don’t rely on Keras mask propagation)
    z = MaskedAttentionPooling(d_model, name="pool")(x, mask=mask_in)

    # IMPORTANT: return tuple outputs (seq, emb)
    encoder = keras.Model(
        inputs=[x_num_in, band_in, mask_in],   # positional
        outputs=[x, z],
        name="lsst_encoder"
    )    
    return encoder

def build_pretrain_model(encoder: keras.Model):
    x_num_in, band_in, mask_in = encoder.inputs
    seq, emb = encoder([x_num_in, band_in, mask_in])
    pred = layers.Dense(1, name="pred_mag")(seq)
    return keras.Model(inputs=[x_num_in, band_in, mask_in], outputs=pred, name="lsst_pretrain_model")# -----------------
# Main
# -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory of LSST JSON files")
    ap.add_argument("--out_dir", default="runs/lsst_pretrain", help="Output directory")
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    paths = list_json_files(args.data_dir)
    train_paths, val_paths = stable_hash_split(paths, train_frac=args.train_frac)
    print("Files:", len(paths), "train:", len(train_paths), "val:", len(val_paths))

    train_ds = build_pretrain_dataset(train_paths, batch_size=args.batch, training=True)
    val_ds   = build_pretrain_dataset(val_paths,   batch_size=args.batch, training=False)

    encoder = build_encoder()
    model = build_pretrain_model(encoder)

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.MeanSquaredError(),
    )

    logdir = os.path.join(run_dir, "tb")
    tb = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False, profile_batch=0)

    ckpt = os.path.join(run_dir, "best_pretrain_model.keras")
    mc = keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", mode="min", save_best_only=True)

    es = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[tb, mc, es],
        verbose=1,
    )

    # Save final models
    pretrain_path = os.path.join(run_dir, "pretrain_model.keras")
    model.save(pretrain_path)

    encoder_path = os.path.join(run_dir, "encoder.keras")
    encoder.save(encoder_path)

    # Save config for later fine-tune
    cfg = {
        "tmax": TMAX,
        "num_bands": NUM_BANDS_LSST,
        "d_model": D_MODEL,
        "band_emb": BAND_EMB,
    }
    with open(os.path.join(run_dir, "pretrain_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)
        
    print("\nSaved:")
    print("  pretrain model:", pretrain_path)
    print("  encoder:", encoder_path)
    print("  tensorboard logdir:", logdir)
    print("  best checkpoint:", ckpt)

if __name__ == "__main__":
    main()
