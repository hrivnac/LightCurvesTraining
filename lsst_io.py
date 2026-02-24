# lsst_io.py
from __future__ import annotations
import os, json, glob, hashlib
import numpy as np
import tensorflow as tf

LSST_BAND_TO_ID = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Y": 5}
NUM_BANDS_LSST = 6

def list_json_files(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.json")))

def stable_hash_split(paths: list[str], train_frac: float = 0.9) -> tuple[list[str], list[str]]:
    train, val = [], []
    for p in paths:
        oid = os.path.splitext(os.path.basename(p))[0]
        h = hashlib.md5(oid.encode("utf-8")).hexdigest()
        u = int(h[:8], 16) / 16**8
        (train if u < train_frac else val).append(p)
    return train, val

def load_one_object(path: str) -> dict | None:
    """Return dict with arrays t, mag, band; filters mag==0.0; returns None if empty."""
    oid = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r") as f:
        d = json.load(f)

    t_all, mag_all, band_all = [], [], []

    for band_name, payload in d.items():
        if band_name not in LSST_BAND_TO_ID:
            continue
        times = payload.get("times", [])
        vals  = payload.get("values", [])
        if not times or not vals or len(times) != len(vals):
            continue

        bid = LSST_BAND_TO_ID[band_name]
        for t, v in zip(times, vals):
            # 0.0 means missing value -> skip
            if v is None:
                continue
            try:
                v = float(v)
                t = float(t)
            except Exception:
                continue
            if v == 0.0:
                continue
            t_all.append(t)
            mag_all.append(v)
            band_all.append(bid)

    if not t_all:
        return None

    t = np.asarray(t_all, dtype=np.float32)
    mag = np.asarray(mag_all, dtype=np.float32)
    band = np.asarray(band_all, dtype=np.int32)

    idx = np.argsort(t)
    return {"objectId": oid, "t": t[idx], "mag": mag[idx], "band": band[idx]}

def make_file_dataset(paths: list[str]) -> tf.data.Dataset:
    """Dataset of file paths."""
    return tf.data.Dataset.from_tensor_slices(paths)

# --- Feature construction (shared) ---

def _make_dt_trel(t: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    t = tf.cast(t, tf.float32)
    t0 = t[0]
    trel = t - t0
    dt = tf.concat([tf.zeros([1], tf.float32), t[1:] - t[:-1]], axis=0)
    return dt, trel

def _pad_to_tmax(x: tf.Tensor, mask: tf.Tensor, tmax: int) -> tuple[tf.Tensor, tf.Tensor]:
    n = tf.shape(x)[0]
    pad_len = tf.maximum(0, tmax - n)
    x_pad = tf.pad(x, [[0, pad_len], [0, 0]])
    mask_pad = tf.pad(mask, [[0, pad_len]], constant_values=False)
    return x_pad[:tmax], mask_pad[:tmax]


def parse_json_to_tensors(path: tf.Tensor):
    """tf.py_function wrapper: path -> (t, mag, band) variable length."""
    def _py(p_tensor):
        # p_tensor is an EagerTensor (dtype=string); convert to Python str
        p = p_tensor.numpy().decode("utf-8")

        obj = load_one_object(p)
        if obj is None:
            return (np.zeros((0,), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))
        return (obj["t"], obj["mag"], obj["band"])

    t, mag, band = tf.py_function(_py, [path], [tf.float32, tf.float32, tf.int32])
    t.set_shape([None]); mag.set_shape([None]); band.set_shape([None])
    return t, mag, band