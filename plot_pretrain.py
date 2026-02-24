# plot_pretrain.py
from __future__ import annotations
import os, json, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from lsst_io import load_one_object, LSST_BAND_TO_ID, NUM_BANDS_LSST
from pretrain_lsst import _make_dt_trel, _pad_to_tmax  # reuse same helpers


ID_TO_LSST_BAND = {v: k for k, v in LSST_BAND_TO_ID.items()}
BAND_ORDER = ["u", "g", "r", "i", "z", "Y"]  # fixed order for plots

def band_name_from_id(bid: int) -> str:
    return ID_TO_LSST_BAND.get(int(bid), str(bid)) 

def load_history(history_path: str):
    with open(history_path, "r") as f:
        return json.load(f)

def plot_loss(history: dict, out_png: str):
    plt.figure()
    if "loss" in history:
        plt.plot(history["loss"], label="train loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE (masked)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def build_one_input(obj, tmax: int):
    t = obj["t"].astype(np.float32)
    mag_true = obj["mag"].astype(np.float32)
    band = obj["band"].astype(np.int32)

    # sort by time
    idx = np.argsort(t)
    t, mag_true, band = t[idx], mag_true[idx], band[idx]

    # TRUNCATE to tmax so masks match predictions
    n = min(len(t), tmax)
    t = t[:n]
    mag_true = mag_true[:n]
    band = band[:n]

    # compute time feats
    dt, trel = _make_dt_trel(tf.constant(t))
    dt = dt.numpy()
    trel = trel.numpy()

    # random mask (for visualization)
    rng = np.random.default_rng(0)
    m = rng.random(n) < 0.15

    mag_in = mag_true.copy()
    mag_in[m] = 0.0
    mag_mask_feat = m.astype(np.float32)

    x_num = np.stack([dt, trel, mag_in, mag_mask_feat], axis=-1)  # [n,4]
    tok_mask = np.ones((n,), dtype=bool)

    # pad (here n<=tmax, so mostly padding only if n<tmax)
    x_num_pad, mask_pad = _pad_to_tmax(tf.constant(x_num), tf.constant(tok_mask), tmax)
    band_pad, _ = _pad_to_tmax(tf.constant(band[:, None], dtype=tf.float32), tf.constant(tok_mask), tmax)
    band_pad = tf.cast(tf.squeeze(band_pad, axis=-1), tf.int32)

    # add batch dim
    x_num_pad = x_num_pad[None, ...]
    band_pad = band_pad[None, ...]
    mask_pad = mask_pad[None, ...]

    return (x_num_pad, band_pad, mask_pad), (t, mag_true, mag_in, m, band)


def band_name_from_id(bid: int) -> str:
    # reverse mapping
    for k, v in LSST_BAND_TO_ID.items():
        if v == bid:
            return k
    return str(bid)
    
def plot_reconstruction(model, obj, tmax: int, out_png: str, connect_lines: bool = True):
    (x_num, band, mask), (t, mag_true, mag_in, m, band_arr) = build_one_input(obj, tmax)

    # Predict only for the unpadded part
    pred = model.predict([x_num, band, mask], verbose=0)[0, :len(t), 0]

    # Residuals on masked points only
    masked_idx = np.where(m)[0]
    resid = pred[masked_idx] - mag_true[masked_idx]  # pred - true

    # ---- Figure layout: 2 rows
    # Top: 6 band panels (2x3)
    # Bottom: residual histogram spanning full width
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.35, wspace=0.25)

    # Band panels
    for k, bname in enumerate(BAND_ORDER):
        r = 0 if k < 3 else 1
        c = k % 3
        ax = fig.add_subplot(gs[r, c])

        bid = LSST_BAND_TO_ID[bname]
        sel = (band_arr == bid)

        if sel.any():
            tt = t[sel]
            yy = mag_true[sel]

            # scatter true
            ax.scatter(tt, yy, s=18, label="true")

            # optional line connecting points (sorted by time)
            if connect_lines and len(tt) >= 2:
                o = np.argsort(tt)
                ax.plot(tt[o], yy[o], linewidth=1)

            # masked points in this band
            sel_m = sel & m
            if sel_m.any():
                ax.scatter(t[sel_m], mag_true[sel_m],
                           facecolors="none", edgecolors="k", s=70, label="masked true")

                # predicted values at masked points
                ax.scatter(t[sel_m], pred[sel_m], marker="*", s=90, label="pred@masked")

        ax.invert_yaxis()
        ax.set_title(f"Band {bname}")
        ax.set_xlabel("time")
        ax.set_ylabel("mag")

        # Only put legend on first subplot to reduce clutter
        if k == 0:
            ax.legend(loc="best", fontsize=8)

    # Residual histogram (bottom row spanning all columns)
    axr = fig.add_subplot(gs[2, :])
    if len(resid) > 0:
        axr.hist(resid, bins=40)
        axr.axvline(0.0, linewidth=1)
        axr.set_title("Residuals on masked points (pred − true)")
        axr.set_xlabel("mag residual")
        axr.set_ylabel("count")
    else:
        axr.text(0.5, 0.5, "No masked points in this example", ha="center", va="center")
        axr.set_axis_off()

    fig.suptitle(f"Reconstruction example objectId={obj['objectId']} (truncated to TMAX={tmax})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    
def plot_reconstructionOld(model, obj, tmax: int, out_png: str):
    (x_num, band, mask), (t, mag_true, mag_in, m, band_arr) = build_one_input(obj, tmax)

    pred = model.predict([x_num, band, mask], verbose=0)[0, :len(t), 0]    # unpad to original length
    n = len(t)
    pred = pred[:n]
    
    

    # Plot magnitude vs time (invert y axis)
    plt.figure(figsize=(10, 5))
    # show true points by band color group using markers
    # (no explicit colors as per your preference; use different markers)
    markers = {0:"o", 1:"s", 2:"^", 3:"v", 4:"D", 5:"x"}

    for bid in np.unique(band_arr):
        sel = (band_arr == bid)
        mk = markers.get(int(bid), "o")
        plt.scatter(t[sel], mag_true[sel], marker=mk, label=f"true {band_name_from_id(int(bid))}", s=18)

    # masked inputs (where mag_in==0) plot as open circles
    plt.scatter(t[m], mag_true[m], facecolors="none", edgecolors="k", label="masked points", s=60)

    # predictions at masked points
    plt.scatter(t[m], pred[m], marker="*", label="pred@masked", s=80)

    plt.gca().invert_yaxis()
    plt.xlabel("time")
    plt.ylabel("mag")
    plt.title(f"Reconstruction example objectId={obj['objectId']}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    
def plot_residuals_over_dataset(model, data_dir: str, tmax: int, out_png: str, n_objects: int = 200):
    import glob, os
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))[:n_objects]

    all_resid = []
    for p in files:
        obj = load_one_object(p)
        if obj is None:
            continue
        (x_num, band, mask), (t, mag_true, mag_in, m, band_arr) = build_one_input(obj, tmax)
        pred = model.predict([x_num, band, mask], verbose=0)[0, :len(t), 0]
        if m.any():
            all_resid.append(pred[m] - mag_true[m])

    if not all_resid:
        return

    all_resid = np.concatenate(all_resid, axis=0)

    plt.figure(figsize=(8, 4))
    plt.hist(all_resid, bins=60)
    plt.axvline(0.0, linewidth=1)
    plt.title(f"Residuals (pred − true) over {n_objects} objects (masked points)")
    plt.xlabel("mag residual")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="runs/lsst_pretrain/<timestamp> directory")
    ap.add_argument("--data_dir", required=True, help="Directory with LSST JSON files (for examples)")
    ap.add_argument("--tmax", type=int, default=96)
    ap.add_argument("--n_examples", type=int, default=3)
    args = ap.parse_args()

    history_path = os.path.join(args.run_dir, "history.json")
    model_path = os.path.join(args.run_dir, "pretrain_model.keras")

    os.makedirs(args.run_dir, exist_ok=True)

    # Loss curves
    hist = load_history(history_path)
    plot_loss(hist, os.path.join(args.run_dir, "pretrain_loss.png"))
    print("Wrote:", os.path.join(args.run_dir, "pretrain_loss.png"))

    # Load model (custom layer is registered via tf.keras.utils.register_keras_serializable)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Pick a few objects for reconstruction plots
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if not files:
        raise RuntimeError("No JSON files found in data_dir")

    picked = files[:args.n_examples]
    for i, p in enumerate(picked):
        obj = load_one_object(p)
        if obj is None:
            continue
        out_png = os.path.join(args.run_dir, f"recon_{i+1}.png")
        plot_reconstruction(model, obj, args.tmax, out_png)
        print("Wrote:", out_png)

    plot_residuals_over_dataset(
        model,
        data_dir=args.data_dir,
        tmax=args.tmax,
        out_png=os.path.join(args.run_dir, "residuals_over_dataset.png"),
        n_objects=200
    )
    print("Wrote:", os.path.join(args.run_dir, "residuals_over_dataset.png"))

if __name__ == "__main__":
    main()
