"""
Rebuilds everything in data/ and model/ from the research package.

Run ONCE on your own machine if you ever retrain the model:
    python prepare_artifacts.py path/to/phishing_detection_complete_package

The Streamlit app itself never needs this script or the original package.
"""
import json
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PKG = Path(sys.argv[1] if len(sys.argv) > 1 else "phishing_detection_complete_package")
OUT = Path(__file__).parent
RANDOM_STATE, TEST_SIZE = 42, 0.20  # must match the notebook

raw = pd.read_csv(PKG / "dataset" / "dataset_B_05_2020.csv")
enc = pd.read_csv(PKG / "model_outputs" / "preprocessed_dataset_encoded.csv")
meta = json.load(open(PKG / "model_outputs" / "model_metadata.json"))
FEATURES = meta["feature_columns"]
assert len(raw) == len(enc), "raw and preprocessed row counts differ"
assert (raw["length_url"].values == enc["length_url"].values).all(), "row order differs"

# ---- 1. one compact dataset file: url + 81 features + label + split flag -------------
X, y = enc[FEATURES], enc["target"]
idx_tr, idx_te = train_test_split(enc.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
ds = pd.concat([raw[["url"]], X], axis=1)
ds.insert(1, "label", y.map({0: "legitimate", 1: "phishing"}))
ds.insert(2, "split", "train")
ds.loc[idx_te, "split"] = "test"
ds.insert(0, "record_id", ds.index)
ds.to_csv(OUT / "data" / "phishing_dataset.csv", index=False)

# ---- 2. feature groups, from the original column order (56 URL / 24 content / 7 external)
cols = list(raw.columns)
i_url, i_content, i_ext = cols.index("statistical_report"), cols.index("nb_hyperlinks"), cols.index("whois_registered_domain")
group_of = {}
for i, c in enumerate(cols):
    if c in ("url", "status"):
        continue
    group_of[c] = "URL-based" if i <= i_url else ("Content-based" if i < i_ext else "External-service")
pd.DataFrame({"feature": list(group_of), "group": list(group_of.values()),
              "kept": [c in FEATURES for c in group_of]}).to_csv(OUT / "data" / "feature_groups.csv", index=False)

# ---- 3. model predictions on the held-out test set + curves --------------------------
models = {"Initial": "random_forest_initial.joblib", "Tuned": "random_forest_tuned.joblib"}
Xte, yte = X.loc[idx_te], y.loc[idx_te]
preds = pd.DataFrame({"record_id": idx_te, "true": yte.values})
curves = []
for name, fn in models.items():
    m = joblib.load(PKG / "model_outputs" / fn)
    p = m.predict_proba(Xte)[:, 1]
    preds[f"proba_{name.lower()}"] = p
    fpr, tpr, _ = roc_curve(yte, p)
    curves += [dict(model=name, curve="roc", x=a, y=b) for a, b in zip(fpr, tpr)]
    pr, rc, _ = precision_recall_curve(yte, p)
    curves += [dict(model=name, curve="pr", x=a, y=b) for a, b in zip(rc, pr)]
    shutil.copy(PKG / "model_outputs" / fn, OUT / "model" / fn)
preds.to_csv(OUT / "data" / "test_predictions.csv", index=False)
pd.DataFrame(curves).round(5).to_csv(OUT / "data" / "curves.csv", index=False)

# ---- 4. copy the tables / figures the app presents as-is ----------------------------
mo = PKG / "model_outputs"
for f in ["metrics_comparison.csv", "feature_importance_comparison.csv", "feature_correlation_matrix.csv",
          "random_forest_tuning_cv_results.csv"]:
    shutil.copy(mo / f, OUT / "data" / f)
for f in ["model_metadata.json", "feature_columns.json", "label_mapping.json"]:
    shutil.copy(mo / f, OUT / "model" / f)
shutil.copy(mo / "model_comparison_dashboard.png", OUT / "assets" / "model_comparison_dashboard.png")

# ---- 5. 300 unlabeled held-out records for the batch-upload demo -------------------
Xte.sample(300, random_state=RANDOM_STATE).to_csv(OUT / "data" / "sample_batch_upload.csv", index=False)
print("done:", len(ds), "records,", len(idx_te), "test records")
