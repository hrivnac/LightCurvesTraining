import argparse
import json
import re

import pandas as pd

from parse_ztf import build_label_vocab

from conf import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="label2id.json")
    args = ap.parse_args()

    label2id = build_label_vocab(TRAIN_CSV)
    with open(args.out, "w") as f:
        json.dump(label2id, f, indent=2, sort_keys=True)

    print("Wrote", args.out, "with", len(label2id), "labels")


if __name__ == "__main__":
    main()
