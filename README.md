# Web Page Phishing Detection Using Random Forest Classification

Interactive research-presentation app (Streamlit) for the CST9/L project of **Ascend Earn L. Cañete**,
College of Computing Education, University of Mindanao, S.Y. 2026-2027.

## Run locally
Needs Python 3.11 or newer.
```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. `scikit-learn` is pinned to **1.8.0**, the version the models were saved with.

## Pages
| Page | What it shows |
|---|---|
| Overview | Title, headline numbers for the active model, abstract, scope reminder |
| Background & Objectives | Background, general/specific objectives, scope and limitations |
| Dataset | 11,430 records, feature families, removed constant columns, feature explorer, record browser |
| Methodology | 8-step pipeline, Random Forest formula, data preparation, tuning search and CV results |
| Model Results | Metrics, confusion matrices, ROC/PR, threshold explorer, feature importance, correlation |
| Try a Record | Pick a held-out test page: verdict, tree votes, feature comparison, what-if editing |
| Analyze a URL | Enter a web address, derive available URL features, optionally inspect limited public-page HTML, and view an experimental prediction. |
| Batch Prediction | Upload a CSV of the 81 prepared features, download predictions |
| Conclusion & References | Findings, trade-off, limitations, future work, references |

The sidebar **Active model** switch (Tuned / Initial Random Forest) changes the highlights, Try a Record, Analyze a URL, and Batch Prediction.

## Suggested defense demo (about 6 minutes)
1. **Overview** - introduce the project and explain that live URL predictions are experimental.
2. **Dataset -> Feature explorer** - pick `google_index`, then `nb_hyperlinks`: show how the classes differ.
3. **Model Results -> Metrics** - tuning did not improve every metric; tuned only wins on recall.
4. **Confusion matrices** - 37 -> 33 missed phishing pages, but 42 -> 47 false alarms.
5. **Threshold explorer** - slide the threshold to show the miss/false-alarm trade-off.
6. **Try a Record** - *Random* a few times, then *Only records the active model gets wrong* to be upfront about errors.
7. **What-if** - set `google_index` to 0 on a phishing page and watch the probability fall.
8. **Analyze a URL** - enter a URL, review extracted features and unavailable fields, and explain the limitations.
9. **Batch Prediction** - run the built-in sample, download the results.

## Files
```
app.py                  Streamlit app (all pages)
core.py                 Model loading, prediction, tree votes, upload validation
url_features.py         URL lexical and limited HTML feature extraction for live demo
prepare_artifacts.py    Rebuilds data/ and model/ from the research package (only needed if you retrain)
model/                  random_forest_tuned.joblib, random_forest_initial.joblib, metadata
data/                   dataset with test-split flag, curves, importances, CV results, sample upload CSV
assets/                 University logo, notebook dashboard image
.streamlit/config.toml  Theme
```

## Reproducibility check
On the 2,286 held-out records the saved models reproduce the documentation exactly:
Initial 1101/42/37/1106 and Tuned 1096/47/33/1110 (TN/FP/FN/TP). Test records come from the stratified 80:20 split, `random_state=42`.

## Limitations
The URL demo derives URL-syntax features and can inspect a limited HTML snapshot from a public HTTP(S) address. It does not query WHOIS, traffic, Google indexing, PageRank, or phishing-reputation services; these features are median-imputed. Live URL results are experimental and are not covered by the reported held-out test metrics.
Results are for one dataset and one split. A demonstration, not a guarantee that a website is safe.
