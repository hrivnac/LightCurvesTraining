"""
Keras skeleton for streaming / early multi-label classification of multi-band lightcurves.

Core ideas:
- One object => variable-length sequence of observations (time, mag, band_id)
- Train on random prefixes (early classification)
- Multi-label output
- Optional "label-vector head" so you can add new labels cheaply later

Assumptions:
- You can provide per-object arrays:
    t:    float32 [N]   (JD or any monotonic time)
    mag:  float32 [N]
    band: int32   [N]   (ZTF: 0..1, LSST: 0..5)
    y:    float32 [L]   multi-hot over canonical labels (after ontology mapping/propagation)

Works for ZTF now and LSST later by setting NUM_BANDS.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # 0=all logs, 1=filter INFO, 2=filter WARNING, 3=filter ERROR

# ---------------------------
# Config
# ---------------------------
NUM_BANDS = 2        # ZTF: 2, LSST: 6
NUM_LABELS = 100     # set to current number of canonical labels
TMAX = 64           # max padded length (pick based on your distributions)
D_MODEL = 128
BAND_EMB = 16
DROPOUT = 0.1

MIN_PREFIX = 20      # minimum number of points in a prefix during training
MAG_JITTER_STD = 0.02
POINT_DROPOUT_P = 0.05

NUM_EPOCHS = 100 # 100
OUT_DIR = "runs/ztf_run1"


# ---------------------------
# Utilities: padding + feature building
# ---------------------------
def _make_dt_trel(t):
    """t: [N] -> dt [N], trel [N] (both float32)"""
    t = tf.cast(t, tf.float32)
    # relative time from first point
    t0 = t[0]
    trel = t - t0
    # dt between successive points
    dt = tf.concat([tf.zeros([1], tf.float32), t[1:] - t[:-1]], axis=0)
    return dt, trel

def _augment(mag, keep_mask):
    """Apply simple robustness augmentations on-the-fly."""
    mag = tf.cast(mag, tf.float32)

    # mag jitter
    if MAG_JITTER_STD and MAG_JITTER_STD > 0:
        mag = mag + tf.random.normal(tf.shape(mag), stddev=MAG_JITTER_STD)

    # point dropout: drop some points by masking them out
    if POINT_DROPOUT_P and POINT_DROPOUT_P > 0:
        drop = tf.random.uniform(tf.shape(keep_mask)) < POINT_DROPOUT_P
        keep_mask = tf.logical_and(keep_mask, tf.logical_not(drop))

    return mag, keep_mask

def _pad_to_tmax(x, mask, tmax=TMAX):
    """
    x: [N, F], mask: [N] bool
    -> x_pad: [TMAX, F], mask_pad: [TMAX] bool
    """
    n = tf.shape(x)[0]
    pad_len = tf.maximum(0, tmax - n)
    x_pad = tf.pad(x, [[0, pad_len], [0, 0]])
    mask_pad = tf.pad(mask, [[0, pad_len]], constant_values=False)

    # truncate if longer than TMAX
    x_pad = x_pad[:tmax]
    mask_pad = mask_pad[:tmax]
    return x_pad, mask_pad

def _sample_prefix(t, mag, band, y, training=True):
    """
    Random prefix sampling for early classification.
    Inputs are 1D arrays of length N. Output padded sequence.
    """
    t = tf.cast(t, tf.float32)
    mag = tf.cast(mag, tf.float32)
    band = tf.cast(band, tf.int32)
    y = tf.cast(y, tf.float32)

    n = tf.shape(t)[0]

    # Ensure sorted by time (do it upstream if you can; this is safe fallback)
    idx = tf.argsort(t)
    t = tf.gather(t, idx)
    mag = tf.gather(mag, idx)
    band = tf.gather(band, idx)

    # Choose prefix length
    if training:
        lo = tf.minimum(tf.cast(MIN_PREFIX, tf.int32), n)
        lo = tf.maximum(lo, 1)
        k = tf.random.uniform([], minval=lo, maxval=n + 1, dtype=tf.int32)  # inclusive end
    else:
        k = n

    t = t[:k]
    mag = mag[:k]
    band = band[:k]

    # Build features
    dt, trel = _make_dt_trel(t)  # [k]
    keep_mask = tf.ones([k], dtype=tf.bool)

    if training:
        mag, keep_mask = _augment(mag, keep_mask)

    # Numeric features: [dt, trel, mag]
    x_num = tf.stack([dt, trel, mag], axis=-1)  # [k, 3]

    # Keep mask for real points (also after dropout)
    x_num = tf.boolean_mask(x_num, keep_mask)
    band = tf.boolean_mask(band, keep_mask)

    # Recompute a clean mask after dropout
    k2 = tf.shape(x_num)[0]
    mask = tf.ones([k2], dtype=tf.bool)

    # Pad
    x_num_pad, mask_pad = _pad_to_tmax(x_num, mask, TMAX)
    band_pad, _ = _pad_to_tmax(tf.expand_dims(tf.cast(band, tf.float32), -1), mask, TMAX)
    band_pad = tf.cast(tf.squeeze(band_pad, axis=-1), tf.int32)

    return (x_num_pad, band_pad, mask_pad), y


# ---------------------------
# tf.data pipeline
# ---------------------------
def make_dataset(objects, batch_size=64, training=True, shuffle=4096):
    """
    objects: iterable of dicts with keys: t, mag, band, y
    Each value is a numpy array.
    """
    def gen():
        for o in objects:
            yield (o["t"], o["mag"], o["band"], o["y"])

    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # t
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # mag
        tf.TensorSpec(shape=(None,), dtype=tf.int32),    # band
        tf.TensorSpec(shape=(NUM_LABELS,), dtype=tf.float32),  # y
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if training:
        ds = ds.shuffle(shuffle, reshuffle_each_iteration=True)

    ds = ds.map(lambda t, mag, band, y: _sample_prefix(t, mag, band, y, training=training),
                num_parallel_calls=tf.data.AUTOTUNE)

    # (x_num, band, mask), y  -> model expects dict inputs
    ds = ds.map(lambda x, y: ({"x_num": x[0], "band": x[1], "mask": x[2]}, y),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------
# Model components
# ---------------------------
class MaskedAttentionPooling(layers.Layer):
    """Attention pooling over time with padding mask."""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.query = layers.Dense(d_model, use_bias=False)
        self.score = layers.Dense(1, use_bias=False)

    def call(self, x, mask=None, training=None):
        # x: [B, T, D]
        h = tf.nn.tanh(self.query(x))         # [B, T, D]
        logits = tf.squeeze(self.score(h), -1)  # [B, T]

        if mask is not None:
            # mask: [B, T] bool; set masked logits to very negative
            neg_inf = tf.constant(-1e9, dtype=logits.dtype)
            logits = tf.where(mask, logits, neg_inf)

        w = tf.nn.softmax(logits, axis=-1)    # [B, T]
        z = tf.einsum("bt,btd->bd", w, x)     # [B, D]
        return z


class LabelVectorHead(layers.Layer):
    """
    Flexible multi-label head: scores = z @ L^T + b
    You can later extend NUM_LABELS by creating a new head (or manage label vectors externally).
    For incremental training, freeze encoder and train only this layer (or even only some rows).
    """
    def __init__(self, num_labels, d_model, use_cosine=False, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.d_model = d_model
        self.use_cosine = use_cosine

    def build(self, input_shape):
        self.L = self.add_weight(
            shape=(self.num_labels, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="label_vectors"
        )
        self.b = self.add_weight(
            shape=(self.num_labels,),
            initializer="zeros",
            trainable=True,
            name="label_bias"
        )

    def call(self, z):
        # z: [B, D]
        if self.use_cosine:
            z_norm = tf.nn.l2_normalize(z, axis=-1)
            L_norm = tf.nn.l2_normalize(self.L, axis=-1)
            logits = tf.matmul(z_norm, L_norm, transpose_b=True) + self.b
        else:
            logits = tf.matmul(z, self.L, transpose_b=True) + self.b
        return logits


def transformer_block(x, mask, d_model=D_MODEL, num_heads=4, ff_mult=4, dropout=DROPOUT):
    # Self-attention with mask
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(x, x, attention_mask=mask[:, tf.newaxis, tf.newaxis, :])  # [B, 1, 1, T]
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(d_model * ff_mult, activation="gelu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    return x


def build_model(num_bands=NUM_BANDS, num_labels=NUM_LABELS, tmax=TMAX,
                d_model=D_MODEL, band_emb=BAND_EMB,
                use_label_vectors=True):
    # Inputs
    x_num_in = keras.Input(shape=(tmax, 3), dtype=tf.float32, name="x_num")
    band_in  = keras.Input(shape=(tmax,), dtype=tf.int32, name="band")
    mask_in  = keras.Input(shape=(tmax,), dtype=tf.bool, name="mask")

    # Band embedding
    b = layers.Embedding(num_bands, band_emb, name="band_embedding")(band_in)  # [B,T,band_emb]

    # Combine features
    x = layers.Concatenate()([x_num_in, b])  # [B,T,3+band_emb]
    x = layers.Dense(d_model)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)

    # Transformer encoder (small)
    for _ in range(4):
        x = transformer_block(x, mask_in, d_model=d_model, num_heads=4)

    # Pool to object embedding
    z = MaskedAttentionPooling(d_model)(x, mask=mask_in)  # [B,D]
    z = layers.Dropout(DROPOUT)(z)

    # Head
    if use_label_vectors:
        logits = LabelVectorHead(num_labels, d_model, use_cosine=True, name="label_head")(z)
    else:
        logits = layers.Dense(num_labels, name="dense_head")(z)

    out = layers.Activation("sigmoid", name="probs")(logits)
    model = keras.Model(inputs={"x_num": x_num_in, "band": band_in, "mask": mask_in}, outputs=out)
    return model

def build_model_gru(num_bands, num_labels, tmax,
                    d_model=128, band_emb=16, dropout=0.2,
                    use_label_vectors=True):

    x_num_in = keras.Input(shape=(tmax, 3), dtype=tf.float32, name="x_num")
    band_in  = keras.Input(shape=(tmax,), dtype=tf.int32, name="band")
    mask_in  = keras.Input(shape=(tmax,), dtype=tf.bool, name="mask")

    b = layers.Embedding(num_bands, band_emb, name="band_embedding")(band_in)
    x = layers.Concatenate()([x_num_in, b])

    x = layers.Dense(d_model)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # Keras masking: create a mask from mask_in
    # GRU supports mask=...
    x = layers.GRU(d_model, return_sequences=True, dropout=dropout)(x, mask=mask_in)
    x = layers.GRU(d_model, return_sequences=True, dropout=dropout)(x, mask=mask_in)

    # Attention pooling (works with your existing MaskedAttentionPooling layer)
    z = MaskedAttentionPooling(d_model)(x, mask=mask_in)
    z = layers.Dropout(dropout)(z)

    if use_label_vectors:
        logits = LabelVectorHead(num_labels, d_model, use_cosine=True, name="label_head")(z)
    else:
        logits = layers.Dense(num_labels, name="dense_head")(z)

    out = layers.Activation("sigmoid", name="probs")(logits)
    return keras.Model(inputs={"x_num": x_num_in, "band": band_in, "mask": mask_in}, outputs=out)

# ---------------------------
# Training
# ---------------------------
def compile_and_trainOld(train_ds, val_ds, num_labels=NUM_LABELS, lr=3e-4):
    model = build_model(num_bands=NUM_BANDS, num_labels=num_labels, use_label_vectors=True)

    # Multi-label loss
    loss = keras.losses.BinaryCrossentropy(from_logits=False)

    # Optional: class weights per label (vector), if you have strong imbalance.
    # You can implement weighted BCE by multiplying per-label weights inside a custom loss.

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[
            keras.metrics.AUC(curve="ROC", multi_label=True, name="auc_roc"),
            keras.metrics.AUC(curve="PR",  multi_label=True, name="auc_pr"),
        ],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=callbacks)
    return model, history

def compile_and_train(train_ds, val_ds, num_labels, lr=3e-4):
    #model = build_model(num_bands=NUM_BANDS, num_labels=num_labels, tmax=TMAX, use_label_vectors=True)
    model = build_model_gru(num_bands=2, num_labels=num_labels, tmax=TMAX)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR",  multi_label=True, name="auc_pr"),
        ],
    )

    #callbacks = [
    #    tf.keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=8, restore_best_weights=True),
    #    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=3, min_lr=1e-5),
    #]


    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, "best_model.keras")

    run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    #logdir = os.path.join(run_dir, "tb")
    #os.makedirs(logdir, exist_ok=True)

    #tb = keras.callbacks.TensorBoard(
    #    log_dir=logdir,
    #    profile_batch=0
    #)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_auc_pr",     # or "val_loss"
            mode="max",               # "min" if monitoring val_loss
            save_best_only=True,
            save_weights_only=False
        ),
        keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=3, min_lr=1e-5),
        #tb,
    ]


    #print("TensorBoard logs:", logdir)
    
    #history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=callbacks)
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=NUM_EPOCHS,
      steps_per_epoch=4000,          # ~4000*batch_size objects/epoch
      validation_steps=300,
      callbacks=callbacks
      )
    

    # Save the final (may equal best if restore_best_weights=True)
    model.save(os.path.join(out_dir, "final_model.keras"))
    print("Saved best:", ckpt_path)
    print("Saved final:", os.path.join(out_dir, "final_model.keras"))
    
    return model, history

# ---------------------------
# Incremental label addition (conceptual)
# ---------------------------
"""
When a new canonical label arrives:
1) Update ontology (synonyms/parent), update NUM_LABELS_NEW.
2) Rebuild model with num_labels=NUM_LABELS_NEW
3) Load old weights into encoder, and copy old label vectors into the new head.
4) Freeze encoder, train only new label vector row(s) on few-shot data.

In Keras, easiest path is:
- build new model
- set encoder layers trainable=False
- initialize new head weights with old weights where possible
- train with data that includes the new label column.
"""


# ---------------------------
# Example usage (pseudo)
# ---------------------------
if __name__ == "__main__":
    # objects = [
    #   {"t": np.array([...], np.float32),
    #    "mag": np.array([...], np.float32),
    #    "band": np.array([...], np.int32),
    #    "y": np.array([...], np.float32)},  # multi-hot length NUM_LABELS
    # ]
    objects_train = []
    objects_val = []

    #train_ds = make_dataset(objects_train, batch_size=64, training=True)
    #val_ds   = make_dataset(objects_val, batch_size=64, training=False)

    from parse_ztf import build_datasets
    train_ds, val_ds, label2id = build_datasets("all.csv")
    num_labels = len(label2id)
    

    
    model, history = compile_and_train(train_ds, val_ds, num_labels=num_labels)
    
    from plot_ztf import plot_history
    
    plot_history(history, OUT_DIR)
    
    pass
