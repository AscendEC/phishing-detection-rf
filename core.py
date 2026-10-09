"""
Model + data logic for the phishing-detection presentation app.
No Streamlit imports here, so it can be unit-tested on its own.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA, MODEL, ASSETS = ROOT / "data", ROOT / "model", ROOT / "assets"

MODEL_FILES = {
    "Tuned Random Forest": "random_forest_tuned.joblib",
    "Initial Random Forest": "random_forest_initial.joblib",
}
LABELS = {0: "legitimate", 1: "phishing"}

# Plain-language meaning of the features that matter most (dataset: Hannousse & Yahiouche, 2021).
FEATURE_DESC = {
    "google_index": "Whether Google has indexed the URL (1 = NOT indexed)",
    "page_rank": "Open PageRank score of the domain, 0-10 (higher = more reputable)",
    "web_traffic": "Traffic rank from an external service (0 = no traffic data)",
    "nb_hyperlinks": "Number of hyperlinks found in the page's HTML",
    "nb_www": "How many times 'www' appears in the URL",
    "ratio_extHyperlinks": "Share of the page's hyperlinks that point to other domains",
    "ratio_intHyperlinks": "Share of the page's hyperlinks that stay on the same domain",
    "phish_hints": "Count of typical phishing words in the URL (e.g. login, secure, update)",
    "domain_age": "Age of the domain in days from WHOIS (-1 = lookup failed / unavailable)",
    "safe_anchor": "Percentage of anchor tags considered 'safe' (not empty / not pointing elsewhere)",
    "longest_word_path": "Length of the longest word in the URL path",
    "length_url": "Total number of characters in the URL",
    "length_hostname": "Number of characters in the hostname",
    "ratio_digits_url": "Share of the URL's characters that are digits",
    "char_repeat": "Longest run of a repeated character in the URL words",
    "nb_dots": "Number of dots in the URL",
    "nb_slash": "Number of slashes in the URL",
    "nb_subdomains": "Number of subdomains in the hostname",
    "ip": "1 if the URL uses an IP address instead of a domain name",
    "login_form": "1 if the page contains a login form",
    "domain_registration_length": "Days until the domain registration expires",
    "dns_record": "1 if the domain has no DNS record",
    "shortening_service": "1 if the URL uses a URL-shortening service",
    "suspecious_tld": "1 if the top-level domain is commonly abused",
    "statistical_report": "1 if the URL/IP appears in public phishing statistics reports",
}


def load_model(name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence sklearn version-mismatch notices on unpickle
        return joblib.load(MODEL / MODEL_FILES[name])


def metadata() -> dict:
    return json.loads((MODEL / "model_metadata.json").read_text(encoding="utf-8"))


def feature_columns() -> list:
    return json.loads((MODEL / "feature_columns.json").read_text(encoding="utf-8"))


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA / "phishing_dataset.csv")


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Phishing probability for each row; columns are aligned to the training order."""
    return model.predict_proba(X[feature_columns()])[:, 1]


def tree_votes(model, X_row: pd.DataFrame) -> np.ndarray:
    """Class chosen by every individual tree (0 = legitimate, 1 = phishing) for one record."""
    Xi = model.named_steps["imputer"].transform(X_row[feature_columns()])
    rf = model.named_steps["model"]
    return np.array([int(rf.classes_[int(t.predict(Xi)[0])]) for t in rf.estimators_])


def validate_upload(df: pd.DataFrame):
    """Returns (missing_columns, extra_columns) relative to the 81 required predictors."""
    need = feature_columns()
    missing = [c for c in need if c not in df.columns]
    extra = [c for c in df.columns if c not in need]
    return missing, extra


def to_numeric_frame(df: pd.DataFrame):
    """Coerces the 81 predictors to numbers. Returns (frame, n_unparseable_cells)."""
    raw = df[feature_columns()]
    num = raw.apply(pd.to_numeric, errors="coerce")
    bad = int((num.isna() & raw.notna()).sum().sum())
    return num, bad


def is_phishing(p, thr: float = 0.5):
    """Same rule as sklearn's predict(): an exact tie (e.g. 50/50 tree votes) goes to 'legitimate'."""
    return np.asarray(p) > thr


def confusion_at(y_true: np.ndarray, p: np.ndarray, thr: float):
    pred = is_phishing(p, thr)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return tn, fp, fn, tp
