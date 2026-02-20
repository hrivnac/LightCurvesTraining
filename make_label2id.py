import json
import argparse
import pandas as pd
import re

def split_semicol(s):
    if not isinstance(s, str) or s.strip() == "":
        return []
    return [x.strip() for x in s.split(";") if x.strip()]

def split_commas(s):
    if not isinstance(s, str) or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def normalize_label(lbl: str) -> str:
    lbl = lbl.strip()
    lbl = re.sub(r"\s+", "_", lbl)
    return lbl

# MUST MATCH TRAINING
SYNONYM_TO_CANON = {
    "Candidate_EB*": "EB*_Candidate",
    "RRLyr": "RRLyrae",
}
PARENT_OF = {
    "EB*_Candidate": "EB*",
}

def canonicalize_labels(raw_labels, propagate_parents=True):
    canon = []
    for x in raw_labels:
        x = normalize_label(x)
        x = SYNONYM_TO_CANON.get(x, x)
        canon.append(x)

    if propagate_parents:
        out = set(canon)
        for x in list(out):
            p = PARENT_OF.get(x)
            if p:
                out.add(p)
        canon = sorted(out)

    canon = [x for x in canon if x.lower() != "unknown"]
    return canon

def build_label_vocab(csv_path: str,
                      maxclass_col="maxclass",
                      class_col="collect_list(class)"):
    df = pd.read_csv(csv_path)
    all_labels = set()
    for _, r in df.iterrows():
        point_labels = split_semicol(r.get(class_col, ""))
        max_labels   = split_commas(r.get(maxclass_col, ""))
        canon = canonicalize_labels(point_labels + max_labels)
        all_labels.update(canon)
    labels = sorted(all_labels)  # deterministic ordering
    return {lbl: i for i, lbl in enumerate(labels)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", default="label2id.json")
    args = ap.parse_args()

    label2id = build_label_vocab(args.train_csv)
    with open(args.out, "w") as f:
        json.dump(label2id, f, indent=2, sort_keys=True)

    print("Wrote", args.out, "with", len(label2id), "labels")

if __name__ == "__main__":
    main()
