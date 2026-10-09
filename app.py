"""
Web Page Phishing Detection Using Random Forest Classification
Interactive research presentation  -  run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="Phishing Detection · Random Forest", page_icon="🎣", layout="wide")

# ------------------------------------------------------------------ design tokens
INK, CANVAS, SURFACE = "#14213d", "#f6f4ee", "#ffffff"
MUTED, BORDER = "#586174", "#e3dfd2"
LEGIT, PHISH, AMBER = "#0f7b6c", "#cf4129", "#e7a33b"
LEGIT_SOFT, PHISH_SOFT, AMBER_SOFT = "#dcefeb", "#f9e1db", "#fbefd5"
MODEL_COLORS = {"Initial": "#7c8aa5", "Tuned": "#14213d"}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp {{background:{CANVAS}; color:{INK};}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp button, .stApp input,
.stMarkdown {{font-family:'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{font-family:Fraunces, Georgia, serif; color:{INK}; letter-spacing:-0.01em;}}
header[data-testid="stHeader"] {{background:transparent;}}
.block-container {{max-width:1240px; padding-top:2.2rem; padding-bottom:3rem;}}
code {{font-family:'IBM Plex Mono', monospace; color:{INK}; background:#ece8da; border-radius:5px;}}
[data-testid="stSidebar"] {{background:{INK};}}
[data-testid="stSidebar"] * {{color:#e9ecf3;}}
[data-testid="stSidebar"] hr {{border-color:#2b3a5c;}}
[data-testid="stSidebar"] [role="radiogroup"] label {{padding:6px 10px; border-radius:10px;}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{background:#1f2f52;}}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {{color:#9fb0cf !important;}}

[class*="st-key-card_"] {{background:{SURFACE}; border:1px solid {BORDER}; border-radius:18px; padding:22px 26px; gap:12px;}}
[class*="st-key-hero_"] {{background:{INK}; border-radius:22px; padding:34px 38px;}}
[class*="st-key-hero_"] * {{color:#f6f4ee;}}
.eyebrow {{font:500 12px 'IBM Plex Mono', monospace; letter-spacing:.14em; text-transform:uppercase; color:{AMBER} !important;}}
.hero-title {{font:800 40px/1.08 Fraunces, Georgia, serif; margin:8px 0 10px 0; letter-spacing:-0.02em;}}
.hero-sub {{color:#c9d2e6 !important; font-size:16px; max-width:760px;}}
.page-head {{margin:0 0 20px 0;}}
.page-head .t {{font:800 36px/1.1 Fraunces, Georgia, serif; letter-spacing:-0.02em;}}
.page-head .s {{color:{MUTED}; font-size:16px; margin-top:6px; max-width:860px;}}
.card-title {{font:700 21px Fraunces, Georgia, serif; margin:0 0 2px 0;}}
.muted {{color:{MUTED};}} .small {{font-size:14px;}}

.kpis {{display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); gap:16px; margin:4px 0 18px 0;}}
.kpi {{background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:16px 20px;}}
.kpi .l {{font:500 11.5px 'IBM Plex Mono', monospace; letter-spacing:.1em; text-transform:uppercase; color:{MUTED};}}
.kpi .v {{font:800 34px/1.15 Fraunces, Georgia, serif; margin-top:4px;}}
.kpi .d {{font-size:13px; color:{MUTED}; margin-top:2px;}}
.kpi.legit {{border-top:4px solid {LEGIT};}} .kpi.phish {{border-top:4px solid {PHISH};}}
.kpi.amber {{border-top:4px solid {AMBER};}} .kpi.ink {{border-top:4px solid {INK};}}

.callout {{border-radius:14px; padding:14px 18px; margin:6px 0 10px 0; font-size:15px; line-height:1.5;}}
.callout.warn {{background:{AMBER_SOFT}; border-left:5px solid {AMBER};}}
.callout.info {{background:#e8ecf5; border-left:5px solid {INK};}}
.callout.good {{background:{LEGIT_SOFT}; border-left:5px solid {LEGIT};}}
.callout.bad  {{background:{PHISH_SOFT}; border-left:5px solid {PHISH};}}
.pill {{display:inline-block; padding:4px 14px; border-radius:99px; font:600 13px 'IBM Plex Mono', monospace; letter-spacing:.06em;}}
.pill.legit {{background:{LEGIT_SOFT}; color:{LEGIT};}} .pill.phish {{background:{PHISH_SOFT}; color:{PHISH};}}
.pill.neutral {{background:#e8ecf5; color:{INK};}}
.meter {{height:14px; background:#ebe8dc; border-radius:99px; overflow:hidden; margin:6px 0 2px 0;}}
.meter > div {{height:100%; border-radius:99px;}}
.steps {{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px;}}
@media (max-width: 900px) {{.steps {{grid-template-columns:repeat(2,minmax(0,1fr));}}}}
.step {{background:#faf8f2; border:1px solid {BORDER}; border-radius:14px; padding:14px 14px 12px 14px;}}
.step .n {{font:600 12px 'IBM Plex Mono', monospace; color:{AMBER};}}
.step .h {{font:700 16px Fraunces, Georgia, serif; margin:2px 0 4px 0;}}
.step .b {{font-size:13px; color:{MUTED}; line-height:1.4;}}
.url-box {{font:500 14px/1.5 'IBM Plex Mono', monospace; background:#f1eee3; border:1px dashed #cdc7b2;
  border-radius:12px; padding:12px 16px; word-break:break-all;}}
.stTabs [data-baseweb="tab-list"] {{gap:6px;}}
.stTabs [data-baseweb="tab"] {{border-radius:99px; padding:6px 16px; background:#ebe8dc;}}
.stTabs [aria-selected="true"] {{background:{INK}; color:#fff !important;}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display:none;}}
@media (max-width: 800px) {{.hero-title {{font-size:30px;}} .page-head .t {{font-size:28px;}}}}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ small UI helpers
_card_no = 0


def card():
    """A white rounded container. Keys must be unique per run, so we count them."""
    global _card_no
    _card_no += 1
    return st.container(key=f"card_{_card_no}")


def head(title, sub=""):
    st.markdown(f'<div class="page-head"><div class="t">{title}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def kpis(items):
    """items: (label, value, detail, tone)"""
    html = "".join(
        f'<div class="kpi {t}"><div class="l">{l}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
        for l, v, d, t in items
    )
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def callout(text, tone="info"):
    st.markdown(f'<div class="callout {tone}">{text}</div>', unsafe_allow_html=True)


def pill(text, tone):
    return f'<span class="pill {tone}">{text}</span>'


def meter(p):
    colour = PHISH if p > 0.5 else LEGIT
    return f'<div class="meter"><div style="width:{p * 100:.1f}%; background:{colour};"></div></div>'


def style(fig, h=380, legend=True):
    fig.update_layout(
        height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    )
    fig.update_xaxes(gridcolor="#ebe8dc", zeroline=False)
    fig.update_yaxes(gridcolor="#ebe8dc", zeroline=False)
    return fig


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


# ------------------------------------------------------------------ cached data / models
@st.cache_resource(show_spinner="Loading Random Forest model…")
def get_model(name):
    return core.load_model(name)


@st.cache_data
def dataset():
    return core.load_dataset()


@st.cache_data
def tables():
    d = core.DATA
    m = pd.read_csv(d / "metrics_comparison.csv").set_index("model")
    m.index = [i.replace(" Random Forest", "") for i in m.index]
    imp = pd.read_csv(d / "feature_importance_comparison.csv").rename(columns={"Unnamed: 0": "feature"})
    groups = pd.read_csv(d / "feature_groups.csv")
    corr = pd.read_csv(d / "feature_correlation_matrix.csv", index_col=0)
    cv = pd.read_csv(d / "random_forest_tuning_cv_results.csv")
    curves = pd.read_csv(d / "curves.csv")
    preds = pd.read_csv(d / "test_predictions.csv")
    return m, imp, groups, corr, cv, curves, preds


METRICS, IMP, GROUPS, CORR, CV, CURVES, PREDS = tables()
DS = dataset()
FEATURES = core.feature_columns()
META = core.metadata()
GROUP_OF = dict(zip(GROUPS.feature, GROUPS.group))
TEST = DS[DS.split == "test"].merge(PREDS[["record_id", "proba_initial", "proba_tuned"]], on="record_id").reset_index(drop=True)

# ------------------------------------------------------------------ sidebar / navigation
PAGES = ["Overview", "Background & Objectives", "Dataset", "Methodology", "Model Results",
         "Try a Record", "Batch Prediction", "Conclusion & References"]
with st.sidebar:
    st.image(str(core.ASSETS / "um_logo.jpg"), width=150)
    st.markdown("**Web Page Phishing Detection**  \nUsing Random Forest Classification")
    st.divider()
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    active = st.radio("Active model", list(core.MODEL_FILES), key="active_model",
                      help="Used on the results highlights, Try a Record, and Batch Prediction pages.")
    st.caption("The tuned model is the documentation's recommended demo candidate; the initial model is kept for comparison.")
    st.divider()
    st.caption("Academic demonstration prototype. Not a security product and not a guarantee that any website is safe.")

SHORT = active.split()[0]  # "Tuned" / "Initial"
PROBA_COL = f"proba_{SHORT.lower()}"
M = METRICS.loc[SHORT]


# =================================================================== PAGE: OVERVIEW
def page_overview():
    with st.container(key="hero_1"):
        st.markdown(
            '<div class="eyebrow">CST9/L · University of Mindanao · College of Computing Education</div>'
            '<div class="hero-title">Web Page Phishing Detection Using Random Forest Classification</div>'
            '<div class="hero-sub">Can a Random Forest tell phishing pages from legitimate ones using 81 prepared features '
            'about the URL, the page content, and external services? This app walks through the data, the method, '
            'the measured results, and a live demo of the trained model.</div>'
            '<div class="hero-sub" style="margin-top:14px;font-size:14px;">Ascend Earn L. Cañete · 1st Semester, 1st Term, S.Y. 2026–2027</div>',
            unsafe_allow_html=True,
        )
    st.write("")
    kpis([
        ("Records", f"{len(DS):,}", "5,715 phishing · 5,715 legitimate", "ink"),
        ("Predictors", "81", "87 supplied − 6 constant columns", "ink"),
        (f"Accuracy ({SHORT})", pct(M.accuracy), "held-out test set, n = 2,286", "legit"),
        (f"Phishing recall ({SHORT})", pct(M.recall_phishing), f"{int(M.false_negative)} phishing pages missed", "phish"),
        (f"ROC-AUC ({SHORT})", f"{M.roc_auc:.4f}", "class separation", "amber"),
    ])
    c1, c2 = st.columns([3, 2])
    with c1, card():
        st.markdown('<div class="card-title">Abstract</div>', unsafe_allow_html=True)
        st.write(
            "This project develops and evaluates supervised Random Forest models for classifying prepared website-feature "
            "records as legitimate or phishing, using the *Web page phishing detection* dataset (Version 3) from Mendeley Data. "
            "The dataset has 11,430 records with no missing values or exact duplicates and a perfectly balanced class split. "
            "After excluding the raw URL and six constant predictors, 81 numeric features remain. An initial Random Forest is "
            "compared with a tuned one on a stratified 80:20 split, using accuracy, precision, recall, F1-score, ROC-AUC, "
            "confusion matrices, feature importance, and correlation analysis."
        )
    with c2:
        callout("<b>Scope reminder.</b> The model reads <b>prepared feature records</b>. It does not take a raw URL, crawl a live "
                "site, or query DNS/WHOIS services. Feature extraction from arbitrary URLs is future work.", "warn")
        callout("<b>Honest headline.</b> Both models score ≈ 96.5% accuracy. Tuning did <i>not</i> improve every metric — it "
                "shifted the balance toward catching more phishing pages.", "info")
    st.write("")
    st.markdown("#### What's in this presentation")
    tour = [
        ("Background & Objectives", "Why phishing detection matters, what the study set out to do, and where its limits are."),
        ("Dataset", "Where the 11,430 records come from, what the features measure, and how the classes compare."),
        ("Methodology", "Cleaning, transformation, the 80:20 split, the Random Forest, and hyperparameter tuning."),
        ("Model Results", "Metrics, confusion matrices, ROC/PR curves, a threshold explorer, and feature importance."),
        ("Try a Record", "Pick a held-out web page, see the model's verdict, how the trees voted, and test what-ifs."),
        ("Batch Prediction", "Upload a CSV of prepared features and download predictions with phishing probabilities."),
    ]
    cols = st.columns(3)
    for i, (t, d) in enumerate(tour):
        with cols[i % 3], card():
            st.markdown(f'<div class="card-title">{t}</div><div class="muted small">{d}</div>', unsafe_allow_html=True)


# =================================================================== PAGE: BACKGROUND
def page_background():
    head("Background & Objectives", "The problem, the purpose of the study, and its boundaries.")
    t1, t2, t3 = st.tabs(["Background", "Objectives", "Scope & limitations"])
    with t1, card():
        st.markdown(
            "Phishing is a cybersecurity threat in which attackers use deceptive websites or URLs to imitate legitimate online "
            "services and induce users to disclose sensitive information. As online services grow, detecting fraudulent web "
            "pages has become an important security concern."
        )
        st.markdown(
            "**Related work.** Sahingoz et al. (2019) studied phishing detection from URLs using lexical, natural-language and "
            "hybrid features. Hannousse and Yahiouche (2021) built the benchmark dataset used here, examined 87 commonly "
            "recognised website-phishing features, and reported Random Forest as the most predictive classifier in their experiments."
        )
        st.markdown(
            "**This project** applies supervised binary classification to that dataset, developing an initial and a tuned "
            "Random Forest, and demonstrating them through this Streamlit interface."
        )
    with t2:
        with card():
            st.markdown('<div class="card-title">General objective</div>', unsafe_allow_html=True)
            st.write("To develop and evaluate a Random Forest classification model for prepared website-feature records and "
                     "demonstrate its predictions through a simple Streamlit interface.")
        with card():
            st.markdown('<div class="card-title">Specific objectives</div>', unsafe_allow_html=True)
            objs = [
                ("Obtain and inspect", "the Mendeley dataset (Version 3): structure, data quality, and class distribution."),
                ("Clean and transform", "by excluding the raw URL, checking missing and duplicate records, and removing constant predictors."),
                ("Examine and split", "feature relevance and correlation, then create stratified training and testing subsets."),
                ("Train and present", "initial and tuned Random Forests with scikit-learn, and design a Streamlit interface that shows predictions."),
                ("Evaluate and compare", "using accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and feature-importance plots."),
            ]
            for i, (a, b) in enumerate(objs, 1):
                st.markdown(f"**{i}. {a}** {b}")
    with t3:
        c1, c2 = st.columns(2)
        with c1, card():
            st.markdown('<div class="card-title">In scope</div>', unsafe_allow_html=True)
            st.markdown("- The Mendeley *Web page phishing detection* dataset, Version 3\n- Inspection, cleaning, transformation, feature analysis\n"
                        "- Training, hyperparameter tuning, evaluation, model saving\n"
                        "- A prediction interface for CSVs of **prepared** features, with downloadable results")
        with c2, card():
            st.markdown('<div class="card-title">Out of scope</div>', unsafe_allow_html=True)
            st.markdown("- Extracting features from a plain URL\n- Live crawling, DNS/WHOIS lookups, or traffic queries\n"
                        "- Continuous monitoring, large-scale scanning, automatic blocking\n- Browser extensions or enterprise deployment")
        callout("Evaluation uses <b>one dataset and one held-out split</b>, so performance may differ on newer websites or newer "
                "phishing techniques. Predictions are a demonstration, not a guarantee of website safety.", "warn")


# =================================================================== PAGE: DATASET
def page_dataset():
    head("Dataset", "Web page phishing detection, Version 3 · Hannousse & Yahiouche · Mendeley Data (doi:10.17632/c2gw7fy2j4.3)")
    kpis([
        ("URLs / records", "11,430", "no missing values · no exact duplicates", "ink"),
        ("Supplied features", "87", "56 URL · 24 content · 7 external", "ink"),
        ("Used for training", "81", "after removing 6 constant columns", "legit"),
        ("Class balance", "50 / 50", "5,715 legitimate · 5,715 phishing", "amber"),
    ])
    t1, t2, t3 = st.tabs(["Composition", "Feature explorer", "Browse records"])
    with t1:
        c1, c2 = st.columns(2)
        with c1, card():
            st.markdown('<div class="card-title">Features kept, by family</div>', unsafe_allow_html=True)
            g = GROUPS.groupby(["group", "kept"]).size().unstack(fill_value=0).reindex(["URL-based", "Content-based", "External-service"])
            g = g.rename(columns={True: "Kept", False: "Removed (constant)"})
            fig = go.Figure()
            fig.add_bar(x=g.index, y=g.get("Kept", 0), name="Kept", marker_color=INK, text=g.get("Kept", 0), textposition="inside")
            if "Removed (constant)" in g:
                fig.add_bar(x=g.index, y=g["Removed (constant)"], name="Removed (constant)", marker_color=PHISH,
                            text=g["Removed (constant)"].replace(0, ""), textposition="outside")
            fig.update_layout(barmode="stack")
            st.plotly_chart(style(fig, 320))
            st.caption("URL-based: syntax of the address · Content-based: the page's HTML · External-service: WHOIS, traffic, Google index, PageRank.")
        with c2, card():
            st.markdown('<div class="card-title">The six constant columns that were removed</div>', unsafe_allow_html=True)
            st.write("These hold a single value in every record, so they carry no information for a classifier:")
            st.code(", ".join(META["dropped_constant_features"]), language=None)
            st.markdown('<div class="card-title" style="margin-top:14px">Train / test split</div>', unsafe_allow_html=True)
            sp = DS.groupby(["split", "label"]).size().unstack()
            sp = sp.loc[["train", "test"]]
            sp["total"] = sp.sum(axis=1)
            st.dataframe(sp.rename(index={"train": "Training (80%)", "test": "Test (20%)"}))
            st.caption("Stratified split, random_state = 42. Every demo on the 'Try a Record' page uses test records the model never saw.")
    with t2, card():
        st.markdown('<div class="card-title">How does a feature differ between the two classes?</div>', unsafe_allow_html=True)
        ranked = IMP.sort_values("max_importance", ascending=False).feature.tolist()
        c1, c2 = st.columns([2, 1])
        feat = c1.selectbox("Feature (sorted by importance)", ranked,
                            format_func=lambda f: f"{f}  ·  {GROUP_OF[f]}")
        clip = c2.checkbox("Clip long tails at the 99th percentile", value=True)
        st.caption(core.FEATURE_DESC.get(feat, f"{GROUP_OF[feat]} feature from the original dataset."))
        s = DS[["label", feat]].copy()
        if s[feat].nunique() <= 2:
            share = s.groupby(["label", feat]).size().reset_index(name="n")
            share["share"] = share.n / share.groupby("label").n.transform("sum")
            share[feat] = share[feat].astype(str)
            fig = px.bar(share, x=feat, y="share", color="label", barmode="group",
                         color_discrete_map={"legitimate": LEGIT, "phishing": PHISH}, text=share.share.map(lambda v: f"{v:.0%}"))
            fig.update_yaxes(tickformat=".0%", title="share of class")
        else:
            if clip:
                s = s[s[feat] <= s[feat].quantile(0.99)]
            fig = px.histogram(s, x=feat, color="label", nbins=45, barmode="overlay", opacity=0.65,
                               color_discrete_map={"legitimate": LEGIT, "phishing": PHISH})
        st.plotly_chart(style(fig, 360))
        stats = DS.groupby("label")[feat].agg(["mean", "median"]).T
        st.dataframe(stats.style.format("{:.3f}"))
    with t3, card():
        st.markdown('<div class="card-title">Browse the records</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        lab = c1.selectbox("Class", ["both", "legitimate", "phishing"])
        spl = c2.selectbox("Split", ["both", "train", "test"])
        v = DS
        if lab != "both":
            v = v[v.label == lab]
        if spl != "both":
            v = v[v.split == spl]
        st.caption(f"{len(v):,} records. URLs are shown as plain text only — please don't visit phishing addresses.")
        st.dataframe(v[["record_id", "url", "label", "split"] + FEATURES[:6]].head(500), hide_index=True)


# =================================================================== PAGE: METHODOLOGY
def page_method():
    head("Methodology", "From the raw CSV to a tuned, evaluated Random Forest — every step is recorded in the accompanying notebook.")
    steps = [
        ("01", "Inspect", "Column types, missing values, duplicates, label consistency."),
        ("02", "Clean", "Drop the raw <code>url</code> identifier and 6 constant predictors → 81 features."),
        ("03", "Transform", "Map legitimate → 0, phishing → 1. Median imputer lives inside the pipeline."),
        ("04", "Split", "Stratified 80 : 20 → 9,144 train / 2,286 test, random_state 42."),
        ("05", "Train", "Initial Random Forest baseline (100 trees, default settings)."),
        ("06", "Tune", "Randomized search, 8 candidates, 3-fold stratified CV, scored by ROC-AUC."),
        ("07", "Evaluate", "Both models scored on the <i>same</i> held-out test set."),
        ("08", "Demonstrate", "This Streamlit app predicts from prepared feature CSVs."),
    ]
    html = "".join(f'<div class="step"><div class="n">STEP {n}</div><div class="h">{h}</div><div class="b">{b}</div></div>' for n, h, b in steps)
    with card():
        st.markdown(f'<div class="steps">{html}</div>', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Random Forest", "Data preparation", "Hyperparameter tuning", "Feature analysis"])
    with t1:
        c1, c2 = st.columns([3, 2])
        with c1, card():
            st.markdown('<div class="card-title">How a Random Forest decides</div>', unsafe_allow_html=True)
            st.write("A Random Forest trains many decision trees on random resamples of the data. Each tree votes for a class, "
                     "and the forest returns the majority vote:")
            st.latex(r"\hat{y}(x) = \mathrm{mode}\{\,T_1(x),\,T_2(x),\,\dots,\,T_B(x)\,\}")
            st.write("Here ŷ(x) is the predicted class for feature record x, T_b(x) is the b-th tree's prediction, and B is the "
                     "number of trees. The share of trees voting 'phishing' is what the app reports as the *phishing probability*.")
            st.write("It suits this dataset because the features are structured and numeric, and trees can capture non-linear "
                     "relationships among URL, page, and external-service characteristics without scaling.")
        with c2, card():
            st.markdown('<div class="card-title">Why compare two models?</div>', unsafe_allow_html=True)
            st.write("The study does not assume tuning helps. An **initial** baseline is compared with a **tuned** model chosen by "
                     "cross-validation on the training data only, then both are scored on the same untouched test split.")
            callout("The test set is never used to pick hyperparameters, so the final numbers are an honest estimate for this split.", "good")
    with t2:
        c1, c2 = st.columns(2)
        with c1, card():
            st.markdown('<div class="card-title">Cleaning</div>', unsafe_allow_html=True)
            st.markdown("- 11,430 records inspected; labels are only *legitimate* and *phishing*\n- **0** missing values, **0** exact duplicate rows\n"
                        "- `url` excluded: it is an identifier, not a numeric predictor\n- 6 constant predictors removed (no variation)")
        with c2, card():
            st.markdown('<div class="card-title">Transformation</div>', unsafe_allow_html=True)
            st.markdown("- Target mapped to **0 = legitimate**, **1 = phishing**\n- All 81 predictors are numeric → no encoding or scaling needed for tree models\n"
                        "- Median imputation is stored *inside* the saved pipeline, so prediction-time gaps are handled the same way as in training\n"
                        "- **No engineered features** were added in this version")
        callout("Generating the 81 features straight from an arbitrary URL would need a separate extraction pipeline (page content, WHOIS, "
                "traffic, search-index lookups). That is outside this prototype's scope.", "warn")
    with t3:
        init, tuned = (core.load_model(n).named_steps["model"].get_params() for n in ("Initial Random Forest", "Tuned Random Forest"))
        space = {"n_estimators": "50, 100, 150", "max_depth": "None, 15, 30", "min_samples_split": "2, 5",
                 "min_samples_leaf": "1, 2", "max_features": "sqrt, log2", "class_weight": "None, balanced"}
        tbl = pd.DataFrame({"Search space": space, "Initial model": {k: init[k] for k in space}, "Tuned model": {k: tuned[k] for k in space}}).astype(str)
        c1, c2 = st.columns([1, 1])
        with c1, card():
            st.markdown('<div class="card-title">Search setup</div>', unsafe_allow_html=True)
            st.markdown("- Method: `RandomizedSearchCV`\n- Candidates tried: **8** of 144 possible combinations\n- Validation: **3-fold** stratified CV on the training set\n"
                        f"- Score: mean ROC-AUC · best = **{META['best_cross_validation_roc_auc']:.4f}**")
            st.dataframe(tbl)
        with c2, card():
            st.markdown('<div class="card-title">Cross-validation results (8 candidates)</div>', unsafe_allow_html=True)
            cv = CV.sort_values("rank_test_score").reset_index(drop=True)
            depth = lambda d: "None" if pd.isna(d) else str(int(float(d)))
            cv["cfg"] = [f"#{i + 1} · {r.param_model__n_estimators} trees, depth {depth(r.param_model__max_depth)}, {r.param_model__max_features}"
                         for i, r in cv.iterrows()]
            fig = go.Figure(go.Bar(x=cv.mean_test_score, y=cv.cfg, orientation="h", marker_color=[INK] + ["#a9b3c9"] * (len(cv) - 1),
                                   error_x=dict(type="data", array=cv.std_test_score, color=AMBER),
                                   text=cv.mean_test_score.map("{:.4f}".format), textposition="inside", insidetextanchor="start", textfont=dict(color="#ffffff")))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[0.988, 0.994], title="mean CV ROC-AUC (bars: ±1 SD)")
            st.plotly_chart(style(fig, 360, legend=False))
        st.caption("All 8 candidates fall within about 0.001 ROC-AUC of each other — tuning moved the model very little on this dataset. "
                   "Training ROC-AUC is ≈ 1.0 because deep forests fit the training data almost perfectly; generalization is judged on the held-out test set.")
    with t4, card():
        st.markdown('<div class="card-title">Two complementary views of the features</div>', unsafe_allow_html=True)
        st.markdown("- **Pearson correlation heatmap** — exploratory: shows pairs of predictors that may carry overlapping information. "
                    "It is not, by itself, proof a feature should be removed.\n"
                    "- **Impurity-based feature importance** — shows which variables the fitted trees rely on most. "
                    "It describes *model reliance*, not causal effects.")
        st.write("Both are shown on the **Model Results** page. The top predictors are "
                 + ", ".join(f"`{f}`" for f in IMP.sort_values("max_importance", ascending=False).feature[:5]) + ".")


# =================================================================== PAGE: RESULTS
def page_results():
    head("Model Results", f"Held-out test set: 2,286 records (1,143 legitimate · 1,143 phishing). Highlights follow the active model: <b>{active}</b>.")
    kpis([
        ("Accuracy", pct(M.accuracy), f"{int(M.true_negative + M.true_positive):,} of 2,286 correct", "legit"),
        ("Precision (phishing)", pct(M.precision_phishing), f"{int(M.false_positive)} false alarms", "ink"),
        ("Recall (phishing)", pct(M.recall_phishing), f"{int(M.false_negative)} phishing pages missed", "phish"),
        ("F1 (phishing)", pct(M.f1_phishing), "precision/recall balance", "ink"),
        ("ROC-AUC", f"{M.roc_auc:.4f}", f"avg. precision {M.average_precision:.4f}", "amber"),
    ])
    tabs = st.tabs(["Metrics", "Confusion matrices", "ROC & PR", "Threshold explorer", "Feature importance", "Correlation"])

    with tabs[0]:
        c1, c2 = st.columns([5, 4])
        names = {"accuracy": "Accuracy", "precision_phishing": "Precision (phishing)", "recall_phishing": "Recall (phishing)",
                 "f1_phishing": "F1 (phishing)", "specificity_legitimate": "Specificity (legitimate)",
                 "roc_auc": "ROC-AUC", "average_precision": "Average precision"}
        tbl = pd.DataFrame({names[k]: METRICS.loc[["Initial", "Tuned"], k] for k in names}).T
        tbl["Δ tuned−initial"] = (tbl["Tuned"] - tbl["Initial"]) * 100
        with c1, card():
            st.markdown('<div class="card-title">Table 1 · Initial vs tuned</div>', unsafe_allow_html=True)
            st.dataframe((tbl.style.format({"Initial": "{:.2%}", "Tuned": "{:.2%}", "Δ tuned−initial": "{:+.2f} pts"})
                          .highlight_max(axis=1, subset=["Initial", "Tuned"], color=LEGIT_SOFT)))
            st.caption("Green = higher of the two. The tuned model is higher only on phishing recall; the initial model is higher on every other measure shown.")
        with c2, card():
            long = tbl.drop(columns="Δ tuned−initial").reset_index(names="metric").melt("metric", var_name="model", value_name="v")
            fig = px.bar(long, x="metric", y="v", color="model", barmode="group", color_discrete_map=MODEL_COLORS)
            fig.update_yaxes(range=[0.94, 1.0], tickformat=".0%", title=None)
            fig.update_xaxes(title=None)
            st.plotly_chart(style(fig, 380))
            st.caption("Y-axis starts at 94% to make small differences visible — the models are very close.")
        callout("<b>Reading the trade-off.</b> The initial model is marginally stronger on accuracy, precision, F1 and ROC-AUC. "
                "The tuned model catches slightly more phishing pages (higher recall). Tuning changed the <i>error balance</i>; it did not improve every measure.", "info")

    with tabs[1]:
        c1, c2 = st.columns(2)
        for col, nm, scale in ((c1, "Initial", [[0, "#f3f1ea"], [1, "#7c8aa5"]]), (c2, "Tuned", [[0, "#f3f1ea"], [1, INK]])):
            r = METRICS.loc[nm]
            z = [[r.true_negative, r.false_positive], [r.false_negative, r.true_positive]]
            with col, card():
                st.markdown(f'<div class="card-title">{nm} Random Forest</div>', unsafe_allow_html=True)
                fig = go.Figure(go.Heatmap(z=z, x=["Pred. legitimate", "Pred. phishing"], y=["Actual legitimate", "Actual phishing"],
                                           colorscale=scale, showscale=False, text=[[f"{int(v):,}" for v in row] for row in z],
                                           texttemplate="%{text}", textfont=dict(size=22), hoverinfo="skip"))
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(style(fig, 330, legend=False))
                st.markdown(f"False positives (legit flagged): **{int(r.false_positive)}** · False negatives (phishing missed): **{int(r.false_negative)}**")
        di, dt = METRICS.loc["Initial"], METRICS.loc["Tuned"]
        callout(f"Tuning detected <b>{int(di.false_negative - dt.false_negative)}</b> more phishing records but incorrectly flagged "
                f"<b>{int(dt.false_positive - di.false_positive)}</b> more legitimate ones. In phishing detection a false negative "
                "(a phishing page called legitimate) is usually the costlier mistake — but the right balance depends on the cost of false alarms.", "warn")

    with tabs[2]:
        c1, c2 = st.columns(2)
        for col, kind, xt, yt in ((c1, "roc", "False-positive rate", "True-positive rate (recall)"), (c2, "pr", "Recall", "Precision")):
            with col, card():
                st.markdown(f'<div class="card-title">{"ROC curve" if kind == "roc" else "Precision–recall curve"}</div>', unsafe_allow_html=True)
                fig = go.Figure()
                for nm in ("Initial", "Tuned"):
                    c = CURVES[(CURVES.model == nm) & (CURVES.curve == kind)]
                    val = METRICS.loc[nm, "roc_auc" if kind == "roc" else "average_precision"]
                    fig.add_scatter(x=c.x, y=c.y, mode="lines", name=f"{nm} ({'AUC' if kind == 'roc' else 'AP'} {val:.4f})",
                                    line=dict(color=MODEL_COLORS[nm], width=3))
                if kind == "roc":
                    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(color="#b8b3a1", dash="dash"))
                fig.update_xaxes(title=xt, range=[0, 1])
                fig.update_yaxes(title=yt, range=[0.85 if kind == "pr" else 0, 1.01])
                st.plotly_chart(style(fig, 400))
        st.caption("Both curves hug the top-left corner: strong class separation on this split. The gap between the two models is tiny and "
                   "should not be read as proof that either will perform equally well on new datasets or future phishing techniques.")

    with tabs[3], card():
        st.markdown('<div class="card-title">Move the decision threshold</div>', unsafe_allow_html=True)
        st.write("The saved model flags a page as phishing when more than **50%** of trees vote for it. Raising or lowering that cut-off "
                 "trades missed phishing pages against false alarms. *(Exploration only — the documented results use 0.50.)*")
        thr = st.slider("Flag as phishing when phishing probability is above", 0.05, 0.95, 0.50, 0.01)
        y = (TEST.label == "phishing").values
        tn, fp, fn, tp = core.confusion_at(y, TEST[PROBA_COL].values, thr)
        prec = tp / (tp + fp) if tp + fp else 0
        kpis([("Phishing missed (FN)", f"{fn}", f"recall {tp / (tp + fn):.2%}", "phish"), ("False alarms (FP)", f"{fp}", f"specificity {tn / (tn + fp):.2%}", "amber"),
              ("Precision", f"{prec:.2%}", f"{active}", "ink"), ("Accuracy", f"{(tn + tp) / len(y):.2%}", f"threshold {thr:.2f}", "legit")])
        grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
        sweep = pd.DataFrame([(t, *core.confusion_at(y, TEST[PROBA_COL].values, t)) for t in grid], columns=["t", "tn", "fp", "fn", "tp"])
        fig = go.Figure()
        fig.add_scatter(x=sweep.t, y=sweep.fn, name="Phishing missed (FN)", line=dict(color=PHISH, width=3))
        fig.add_scatter(x=sweep.t, y=sweep.fp, name="False alarms (FP)", line=dict(color=AMBER, width=3))
        fig.add_vline(x=thr, line_color=INK, line_dash="dot")
        fig.update_xaxes(title="threshold")
        fig.update_yaxes(title="records (out of 2,286)")
        st.plotly_chart(style(fig, 340))

    with tabs[4]:
        n = st.slider("Number of top features", 5, 30, 15)
        top = IMP.sort_values("max_importance", ascending=False).head(n).iloc[::-1]
        c1, c2 = st.columns([3, 2])
        with c1, card():
            st.markdown('<div class="card-title">Figure 4 · Impurity-based importance</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_bar(y=top.feature, x=top.initial_importance, name="Initial", orientation="h", marker_color=MODEL_COLORS["Initial"])
            fig.add_bar(y=top.feature, x=top.tuned_importance, name="Tuned", orientation="h", marker_color=MODEL_COLORS["Tuned"])
            fig.update_layout(barmode="group")
            fig.update_xaxes(title="importance")
            st.plotly_chart(style(fig, 120 + 28 * n))
        with c2, card():
            st.markdown('<div class="card-title">By feature family</div>', unsafe_allow_html=True)
            fam = IMP.assign(group=IMP.feature.map(GROUP_OF)).groupby("group")[["initial_importance", "tuned_importance"]].sum()
            fam = fam.reindex(["URL-based", "Content-based", "External-service"])
            nfeat = GROUPS[GROUPS.kept].groupby("group").size()
            fig = go.Figure()
            fig.add_bar(x=fam.index, y=fam.initial_importance, name="Initial", marker_color=MODEL_COLORS["Initial"])
            fig.add_bar(x=fam.index, y=fam.tuned_importance, name="Tuned", marker_color=MODEL_COLORS["Tuned"])
            fig.update_yaxes(tickformat=".0%", title="share of total importance")
            st.plotly_chart(style(fig, 330))
            st.caption(f"Only {nfeat.get('External-service', 0)} of 81 predictors are external-service features, yet they carry a large share "
                       "of the model's reliance — including google_index, page_rank and web_traffic.")
        callout("Importance rankings describe how the <i>trained forests</i> use predictors. They do not show that any feature <i>causes</i> a page to be phishing.", "info")
        with card():
            st.markdown('<div class="card-title">What the top features mean</div>', unsafe_allow_html=True)
            top10 = IMP.sort_values("max_importance", ascending=False).head(10).feature
            med = DS.groupby("label")[list(top10)].median().T
            tbl = pd.DataFrame({"Feature": top10.values, "Family": [GROUP_OF[f] for f in top10],
                                "Meaning": [core.FEATURE_DESC.get(f, "") for f in top10],
                                "Median · legitimate": med.loc[top10, "legitimate"].values, "Median · phishing": med.loc[top10, "phishing"].values})
            st.dataframe(tbl, hide_index=True)
    with tabs[5], card():
        st.markdown('<div class="card-title">Figure 5 · Pearson correlation: top-ranked features and the target</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Heatmap(z=CORR.values, x=CORR.columns, y=CORR.index, zmin=-1, zmax=1, colorscale="RdBu_r",
                                   text=np.round(CORR.values, 2), texttemplate="%{text}", textfont=dict(size=10)))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(style(fig, 640, legend=False))
        st.caption("Diagonal = 1.0 (each variable with itself). Off-diagonal cells are pairwise linear associations. "
                   "The plot helps spot possible redundancy but does not prove causation or that correlated predictors should be dropped.")
    with st.expander("Static dashboard exported from the notebook"):
        st.image(str(core.ASSETS / "model_comparison_dashboard.png"))


# =================================================================== PAGE: TRY A RECORD
WHATIF_STEP = {"google_index": 1, "page_rank": 1, "web_traffic": 1000, "nb_hyperlinks": 1, "nb_www": 1, "domain_age": 100, "phish_hints": 1}


def page_try():
    head("Try a Record", "Pick a web page from the held-out test set, see the model's verdict, how its trees voted, and what changes if you edit a feature.")
    model = get_model(active)
    pool_filter = st.radio("Show", ["Any record", "Phishing only", "Legitimate only", "Only records the active model gets wrong"], horizontal=True)
    pool = TEST
    if pool_filter == "Phishing only":
        pool = TEST[TEST.label == "phishing"]
    elif pool_filter == "Legitimate only":
        pool = TEST[TEST.label == "legitimate"]
    elif pool_filter.startswith("Only"):
        pool = TEST[core.is_phishing(TEST[PROBA_COL]) != (TEST.label == "phishing")]
    pool = pool.reset_index(drop=True)
    sig = (pool_filter, active)
    if st.session_state.get("pool_sig") != sig:
        st.session_state.pool_sig, st.session_state.pos = sig, int(np.random.default_rng().integers(len(pool)))
    b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
    if b1.button("◀ Previous"):
        st.session_state.pos = (st.session_state.pos - 1) % len(pool)
    if b2.button("Next ▶"):
        st.session_state.pos = (st.session_state.pos + 1) % len(pool)
    if b3.button("🎲 Random"):
        st.session_state.pos = int(np.random.default_rng().integers(len(pool)))
    b4.caption(f"Record {st.session_state.pos + 1:,} of {len(pool):,} in this view")

    row = pool.iloc[[st.session_state.pos]]
    rid = int(row.record_id.iloc[0])
    X = row[FEATURES]
    p = float(core.predict_proba(model, X)[0])
    is_p = bool(core.is_phishing(p))
    truth = row.label.iloc[0]
    correct = (is_p == (truth == "phishing"))
    votes = core.tree_votes(model, X)
    n_trees, n_phish = len(votes), int(votes.sum())

    c1, c2 = st.columns([3, 2])
    with c1, card():
        st.markdown('<div class="card-title">The web page (shown as text — do not visit)</div>', unsafe_allow_html=True)
        url = str(row.url.iloc[0]).replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="url-box">{url}</div>', unsafe_allow_html=True)
        st.markdown(f"True label: {pill(truth.upper(), 'phish' if truth == 'phishing' else 'legit')} &nbsp; "
                    f"Model says: {pill('PHISHING' if is_p else 'LEGITIMATE', 'phish' if is_p else 'legit')} &nbsp; "
                    f"{pill('✓ CORRECT' if correct else '✗ WRONG', 'legit' if correct else 'phish')}", unsafe_allow_html=True)
        st.caption(f"Test record #{rid} · {active}")
    with c2, card():
        st.markdown('<div class="card-title">Estimated phishing probability</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi" style="border:0;padding:0"><div class="v" style="color:{PHISH if is_p else LEGIT}">{p:.1%}</div></div>{meter(p)}', unsafe_allow_html=True)
        st.caption(f"{n_phish} of {n_trees} trees vote phishing, {n_trees - n_phish} vote legitimate. The majority wins.")
        fig = go.Figure()
        fig.add_bar(x=[n_trees - n_phish], y=[""], orientation="h", name="legitimate", marker_color=LEGIT, text=[n_trees - n_phish], textposition="inside")
        fig.add_bar(x=[n_phish], y=[""], orientation="h", name="phishing", marker_color=PHISH, text=[n_phish], textposition="inside")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(visible=False, range=[0, n_trees])
        st.plotly_chart(style(fig, 110))

    with card():
        st.markdown('<div class="card-title">Where this record sits on the most important features</div>', unsafe_allow_html=True)
        top = IMP.sort_values(f"{SHORT.lower()}_importance", ascending=False).feature.head(10).tolist()
        med = DS.groupby("label")[top].median().T
        vals = row[top].iloc[0]
        rows = []
        for f in top:
            v, ml, mp = vals[f], med.loc[f, "legitimate"], med.loc[f, "phishing"]
            closer = "—" if ml == mp else ("phishing" if abs(v - mp) < abs(v - ml) else "legitimate")
            rows.append({"Feature": f, "This record": v, "Typical legitimate": ml, "Typical phishing": mp, "Closer to": closer})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Features are ranked by the model's *global* importance; this table is a plain comparison with class medians, "
                   "not a per-record explanation of the forest's decision.")

    with st.expander("🔧 What-if: edit the top features and re-run the model"):
        st.write("Change values below and watch the phishing probability move. This shows how sensitive the model is to each feature.")
        edit_feats = top[:8]
        cols = st.columns(4)
        new = {}
        for i, f in enumerate(edit_feats):
            v = float(vals[f])
            with cols[i % 4]:
                if f == "google_index":
                    new[f] = st.selectbox(f"{f} (1 = not indexed)", [0, 1], index=int(v), key=f"wi_{rid}_{f}")
                else:
                    step = WHATIF_STEP.get(f, 0.05 if "ratio" in f else 1.0)
                    new[f] = st.number_input(f, min_value=float(min(0.0, DS[f].min())), value=v, step=float(step), key=f"wi_{rid}_{f}")
        X2 = X.copy()
        for f, v in new.items():
            X2[f] = v
        p2 = float(core.predict_proba(model, X2)[0])
        d1, d2, d3 = st.columns(3)
        d1.metric("Original probability", f"{p:.1%}")
        d2.metric("Edited probability", f"{p2:.1%}", f"{(p2 - p) * 100:+.1f} pts", delta_color="inverse")
        d3.markdown(f"Edited verdict: {pill('PHISHING' if core.is_phishing(p2) else 'LEGITIMATE', 'phish' if core.is_phishing(p2) else 'legit')}", unsafe_allow_html=True)
        st.caption("The remaining 73 features keep their original values, so an edited record may be unrealistic. For demonstration only.")


# =================================================================== PAGE: BATCH
def page_batch():
    head("Batch Prediction", "Upload a CSV of prepared website-feature records and get a prediction and phishing probability for each row.")
    model = get_model(active)
    c1, c2 = st.columns([3, 2])
    with c1, card():
        st.markdown('<div class="card-title">Requirements</div>', unsafe_allow_html=True)
        st.markdown("- One row per website, with the **81 numeric predictor columns** (names must match exactly; order doesn't matter)\n"
                    "- Extra columns such as `url` or `status` are allowed and ignored by the model\n"
                    "- Blank cells are filled with the training median (the pipeline's imputer)\n"
                    "- **Raw URLs are not accepted** — this prototype doesn't extract features from addresses")
        with st.expander("Show the 81 required column names"):
            st.code(", ".join(FEATURES), language=None)
    with c2, card():
        st.markdown('<div class="card-title">No file handy?</div>', unsafe_allow_html=True)
        sample = (core.DATA / "sample_batch_upload.csv")
        st.download_button("⬇ Download sample CSV (300 held-out records)", sample.read_bytes(), "sample_batch_upload.csv", "text/csv")
        use_sample = st.checkbox("Or just run the built-in sample now")

    up = st.file_uploader("Upload your CSV", type=["csv"])
    df = None
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file as CSV: {e}")
            return
    elif use_sample:
        df = pd.read_csv(sample)
    if df is None:
        return

    missing, extra = core.validate_upload(df)
    if missing:
        st.error(f"This file is missing {len(missing)} of the 81 required columns.")
        st.code(", ".join(missing[:40]) + (" …" if len(missing) > 40 else ""), language=None)
        return
    num, bad = core.to_numeric_frame(df)
    filled = int(num.isna().sum().sum())
    if len(df) > 20000:
        st.warning("Only the first 20,000 rows are scored to keep the app responsive.")
        df, num = df.head(20000), num.head(20000)
    proba = core.predict_proba(model, num)
    out = pd.DataFrame({"row": np.arange(1, len(df) + 1)})
    if "url" in df.columns:
        out["url"] = df["url"].values
    out["prediction"] = np.where(core.is_phishing(proba), "phishing", "legitimate")
    out["phishing_probability"] = proba.round(4)

    label_col = next((c for c in ("status", "label", "target") if c in df.columns), None)
    acc_note = ""
    if label_col:
        truth = df[label_col].astype(str).str.lower().map({"phishing": "phishing", "1": "phishing", "legitimate": "legitimate", "0": "legitimate"})
        if truth.notna().all():
            out["true_label"] = truth.values
            acc_note = f"{(out.true_label == out.prediction).mean():.2%}"
    n_ph = int((out.prediction == "phishing").sum())
    items = [("Rows scored", f"{len(out):,}", f"model: {active}", "ink"),
             ("Flagged phishing", f"{n_ph:,}", f"{n_ph / len(out):.1%} of rows", "phish"),
             ("Predicted legitimate", f"{len(out) - n_ph:,}", f"{(len(out) - n_ph) / len(out):.1%} of rows", "legit")]
    items.append(("Accuracy vs. your labels", acc_note, f"from the {label_col} column", "amber") if acc_note else ("Cells auto-filled", f"{filled:,}", "median imputation", "amber"))
    kpis(items)
    if bad:
        st.warning(f"{bad:,} non-numeric cells were treated as missing and median-filled.")
    if extra:
        st.caption(f"Ignored {len(extra)} extra column(s): {', '.join(extra[:8])}{' …' if len(extra) > 8 else ''}")

    c1, c2 = st.columns([2, 3])
    with c1, card():
        fig = px.histogram(out, x="phishing_probability", nbins=20, color="prediction",
                           color_discrete_map={"legitimate": LEGIT, "phishing": PHISH})
        fig.update_xaxes(title="phishing probability", range=[0, 1])
        st.plotly_chart(style(fig, 320))
    with c2, card():
        show = st.radio("Show rows", ["All", "Phishing", "Legitimate"] + (["Wrong vs. true label"] if acc_note else []), horizontal=True)
        v = out
        if show in ("Phishing", "Legitimate"):
            v = out[out.prediction == show.lower()]
        elif show.startswith("Wrong"):
            v = out[out.prediction != out.true_label]
        st.dataframe(v.head(1000), hide_index=True)
    st.download_button("⬇ Download predictions (CSV)", out.to_csv(index=False).encode("utf-8"), "phishing_predictions.csv", "text/csv")


# =================================================================== PAGE: CONCLUSION
def page_conclusion():
    head("Conclusion & References", "What the study found, what it does not show, and where it could go next.")
    c1, c2, c3 = st.columns(3)
    with c1, card():
        st.markdown('<div class="card-title">What was found</div>', unsafe_allow_html=True)
        st.write("A structured workflow — quality checks, removal of the raw URL and constant predictors, and a fair comparison on one "
                 "held-out set — gave two Random Forests that perform very similarly (≈ 96.5% accuracy, ROC-AUC > 0.99).")
    with c2, card():
        st.markdown('<div class="card-title">The trade-off</div>', unsafe_allow_html=True)
        st.write("The tuned model raised phishing recall and cut false negatives from 37 to 33, at the cost of 5 more false alarms. "
                 "The initial model is marginally ahead on accuracy, precision, F1 and ROC-AUC.")
    with c3, card():
        st.markdown('<div class="card-title">Recommendation</div>', unsafe_allow_html=True)
        st.write("Use the **tuned model** for the demonstration if missing fewer phishing pages is the priority. The choice should "
                 "follow the project's error-cost priorities, not accuracy alone.")
    st.write("")
    c1, c2 = st.columns(2)
    with c1, card():
        st.markdown('<div class="card-title">Limitations to keep in view</div>', unsafe_allow_html=True)
        st.markdown("- One dataset and one 80:20 split; results may differ on newer sites or tactics\n"
                    "- The model needs **prepared features**; it cannot score a raw URL\n"
                    "- Several top predictors (`google_index`, `page_rank`, `web_traffic`) depend on external services that a live system would have to query\n"
                    "- Feature importance shows model reliance, not cause\n- A demonstration — not a guarantee that a site is safe")
    with c2, card():
        st.markdown('<div class="card-title">Future work</div>', unsafe_allow_html=True)
        st.markdown("- A validated feature-extraction component for raw URLs\n- Testing on newer, independent phishing data\n"
                    "- Reviewing false-positive and false-negative behaviour before any real-world use\n"
                    "- Comparing against other classifiers and using repeated cross-validation")
    with card():
        st.markdown('<div class="card-title">References</div>', unsafe_allow_html=True)
        st.markdown(
            "1. Hannousse, A., & Yahiouche, S. (2021). *Web page phishing detection* [Data set] (Version 3). Mendeley Data. https://doi.org/10.17632/c2gw7fy2j4.3\n"
            "2. Hannousse, A., & Yahiouche, S. (2021). Towards benchmark datasets for machine learning based website phishing detection: An experimental study. "
            "*Engineering Applications of Artificial Intelligence, 104*, 104347. https://doi.org/10.1016/j.engappai.2021.104347\n"
            "3. Sahingoz, O. K., Buber, E., Demir, O., & Diri, B. (2019). Machine learning based phishing detection from URLs. "
            "*Expert Systems with Applications, 117*, 345–357. https://doi.org/10.1016/j.eswa.2018.09.029\n"
            "4. Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324\n"
            "5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html"
        )
        st.markdown("**Appendix** · [Google Colab notebook](https://colab.research.google.com/drive/1oedtkHdOBYBSnOZbMTt05yTn-OBZ2IoQ?usp=sharing)")


# =================================================================== ROUTER
{
    "Overview": page_overview,
    "Background & Objectives": page_background,
    "Dataset": page_dataset,
    "Methodology": page_method,
    "Model Results": page_results,
    "Try a Record": page_try,
    "Batch Prediction": page_batch,
    "Conclusion & References": page_conclusion,
}[page]()
