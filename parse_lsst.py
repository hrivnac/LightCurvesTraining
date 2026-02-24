import os, json, glob
import numpy as np

from conf import *

LSST_BAND_TO_ID = {"u":0, "g":1, "r":2, "i":3, "z":4, "Y":5}

def load_objects_from_lsst_dir(data_dir):
    objects = []
    for path in glob.glob(os.path.join(data_dir, "*.json")):
        object_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r") as f:
            d = json.load(f)

        t_list, mag_list, band_list = [], [], []

        for band, payload in d.items():
            if band not in LSST_BAND_TO_ID:
                continue
            times = payload.get("times", [])
            vals  = payload.get("values", [])
            if not times or not vals:
                continue
            if len(times) != len(vals):
                continue

            bid = LSST_BAND_TO_ID[band]
            t_list.extend(times)
            mag_list.extend(vals)
            band_list.extend([bid] * len(times))

        if not t_list:
            continue

        t = np.asarray(t_list, dtype=np.float32)
        mag = np.asarray(mag_list, dtype=np.float32)
        band = np.asarray(band_list, dtype=np.int32)

        # sort by time
        idx = np.argsort(t)
        t, mag, band = t[idx], mag[idx], band[idx]

        objects.append({"objectId": object_id, "t": t, "mag": mag, "band": band})

    return objects