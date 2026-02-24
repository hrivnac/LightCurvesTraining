# Keep only "true" synonyms here (not Candidate_* patterns)
SYNONYM_TO_CANON = {
    "RRLyr": "RRLyrae",
    # add your non-candidate synonyms here
}

NUM_BANDS = 6  # ZTF: 2, LSST: 6
NUM_LABELS = 100  # set to current number of canonical labels
TMAX = 64  # max padded length (pick based on your distributions)
D_MODEL = 128
BAND_EMB = 16
DROPOUT = 0.1

MIN_PREFIX = 5  # minimum number of points in a prefix during training
MAG_JITTER_STD = 0.02
POINT_DROPOUT_P = 0.05

NUM_EPOCHS = 100
OUT_DIR = "runs/ztf_run1"

TRAIN_CSV = "latent.csv"
TEST_CSV = "latent.csv"

TRAIN_DIR = "LSST"
