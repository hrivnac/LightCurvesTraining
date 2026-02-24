import os

import matplotlib.pyplot as plt


def plot_history(history, out_dir):
    hist = history.history

    # Loss
    plt.figure()
    plt.plot(hist.get("loss", []), label="train loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=150)

    # Metrics (whatever exists)
    for key in ["auc_pr", "auc_roc"]:
        if key in hist or f"val_{key}" in hist:
            plt.figure()
            if key in hist:
                plt.plot(hist[key], label=f"train {key}")
            if f"val_{key}" in hist:
                plt.plot(hist[f"val_{key}"], label=f"val {key}")
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{key}.png"), dpi=150)

    plt.close("all")

    print("Saved plots in:", out_dir)
