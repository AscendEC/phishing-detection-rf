# Deploying to Streamlit Community Cloud via GitHub

## 0. Before you start
- A free GitHub account (github.com) and a free Streamlit account (share.streamlit.io - sign in *with GitHub*).
- This folder, unzipped. Total size is about 24 MB; the largest file is 12 MB, so no Git LFS is needed.
- Optional but recommended: run `pip install -r requirements.txt` then `streamlit run app.py` locally first.

## 1. Create the GitHub repository
1. GitHub -> **+** (top right) -> **New repository**.
2. Name it e.g. `phishing-detection-rf`. Choose **Public** (the free Streamlit tier deploys from public repos; private works if you grant access).
3. Do **not** tick "Add a README" (you already have one). Click **Create repository**.

## 2a. Upload the files - web browser (easiest)
1. On the empty repo page click **uploading an existing file**.
2. Drag in the *contents* of the folder: `app.py`, `core.py`, `requirements.txt`, `README.md`, `prepare_artifacts.py`, `.gitignore`, and the folders `model/`, `data/`, `assets/`, `.streamlit/`.
   Keep `app.py` at the **top level** of the repo, not inside a sub-folder.
   If hidden folders like `.streamlit` don't drag in, use 2b instead (or create `.streamlit/config.toml` with **Add file -> Create new file**).
3. Commit message "Initial commit" -> **Commit changes**.

## 2b. Upload the files - Git command line
```
cd phishing-detection-rf          # the unzipped folder
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/phishing-detection-rf.git
git push -u origin main
```

## 3. Deploy on Streamlit
1. Go to **share.streamlit.io**, sign in with GitHub, and authorize access if asked.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Fill in: Repository `<you>/phishing-detection-rf` · Branch `main` · Main file path `app.py`.
4. Open **Advanced settings** and choose **Python 3.12** (scikit-learn 1.8.0 needs 3.11+).
5. Optionally edit the app URL to something short (e.g. `phishing-rf-canete`).
6. Click **Deploy**. The first build takes a few minutes while packages install.

## 4. After it's live
- Open the URL and click through every page once, including Batch Prediction -> built-in sample.
- Put the URL in your documentation under **APPENDICES -> Streamlit URL**.
- Free apps go to sleep after inactivity. Open the link a few minutes before your defense to wake it, and keep the local copy as a backup.
- Updating: edit files on GitHub (or `git push`); the app redeploys automatically.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `No matching distribution found for scikit-learn==1.8.0` | Python version too old. Redeploy with Python 3.12 in Advanced settings. |
| `FileNotFoundError: model/...joblib` | Folders weren't uploaded or `app.py` is in a sub-folder. Check the repo layout on GitHub. |
| Theme looks plain | `.streamlit/config.toml` didn't upload. Add it. |
| Fonts differ from screenshots | Fonts load from Google Fonts; they fall back to system fonts if blocked. Layout still works. |
| App shows "Oh no" error | Click **Manage app** (bottom right) -> logs. |
