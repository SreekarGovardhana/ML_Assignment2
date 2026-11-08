import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_TEST_CSV = os.path.join(BASE_DIR, "test_data.csv")
TARGET = "Target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

CLASS_COLOURS = {"Dropout": "#d1495b", "Enrolled": "#edae49", "Graduate": "#00798c"}

st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .hero {padding: 1.2rem 1.5rem; border-radius: 12px;
             background: linear-gradient(90deg, #003d5b 0%, #00798c 100%); color: #fff;}
      .hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
      .hero p  {margin: .35rem 0 0 0; opacity: .9;}
      div[data-testid="stMetric"] {background: rgba(0,121,140,.07); border-radius: 10px;
                                   padding: .6rem .9rem; border: 1px solid rgba(0,121,140,.25);}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Loading helpers (cached so models load once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models…")
def load_artifacts():
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        features = json.load(f)
    return models, label_encoder, features


@st.cache_data
def load_reference_metrics():
    path = os.path.join(MODEL_DIR, "metrics_comparison.csv")
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None


def read_uploaded_csv(file) -> pd.DataFrame:
    """Accept both comma- and semicolon-separated files; clean header whitespace."""
    df = pd.read_csv(file)
    if df.shape[1] == 1:
        file.seek(0)
        df = pd.read_csv(file, sep=";")
    df.columns = df.columns.str.strip()
    return df


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    try:
        if len(np.unique(y_true)) < n_classes:
            auc = np.nan  # AUC undefined if a class is absent from the upload
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion(cm, labels, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 13})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🎓 Student Outcome Predictor</h1>
      <p>Will a student <b>drop out</b>, stay <b>enrolled</b>, or <b>graduate</b>?
      Evaluate five pre-trained classifiers on your own test data
      (UCI “Predict Students' Dropout and Academic Success”).</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

try:
    models, le, FEATURES = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model artifacts not found in `model/` — run the training notebook first. ({e})")
    st.stop()

if not models:
    st.error("No trained model files found in `model/`.")
    st.stop()

CLASS_NAMES = list(le.classes_)

# --------------------------------------------------------------------------
# Sidebar — upload + model selection
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("1 · Upload test data")
    uploaded = st.file_uploader(
        "CSV with all feature columns and the `Target` column",
        type=["csv"],
    )
    if os.path.exists(SAMPLE_TEST_CSV):
        with open(SAMPLE_TEST_CSV, "rb") as f:
            st.download_button("⬇️ Download sample test_data.csv", f,
                               file_name="test_data.csv", mime="text/csv",
                               width="stretch")
        use_sample = st.checkbox("Use bundled test_data.csv instead", value=False)
    else:
        use_sample = False

    st.subheader("2 · Choose a model")
    model_name = st.selectbox("Classifier", list(models.keys()))

    st.divider()
    st.caption(f"Models loaded: {len(models)} / {len(MODEL_FILES)}")
    st.caption(f"Expected features: {len(FEATURES)} · Classes: {', '.join(CLASS_NAMES)}")

# --------------------------------------------------------------------------
# Data intake + validation
# --------------------------------------------------------------------------
if uploaded is not None:
    data = read_uploaded_csv(uploaded)
    source = uploaded.name
elif use_sample:
    data = pd.read_csv(SAMPLE_TEST_CSV)
    data.columns = data.columns.str.strip()
    source = "test_data.csv (bundled)"
else:
    st.info("👈 Upload a test CSV in the sidebar (or tick **Use bundled test_data.csv**) to begin.")
    ref = load_reference_metrics()
    if ref is not None:
        st.subheader("📊 Reference results on the held-out test split")
        st.dataframe(ref.style.format("{:.4f}").highlight_max(axis=0, color="#b7e4c7"),
                     width="stretch")
    with st.expander("Which columns does my CSV need?"):
        st.write(pd.DataFrame({"Required columns": FEATURES + [TARGET]}))
    st.stop()

missing = [c for c in FEATURES if c not in data.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} required feature column(s):")
    st.code("\n".join(missing))
    st.stop()
if TARGET not in data.columns:
    st.error(f"Uploaded file must contain the `{TARGET}` column for evaluation.")
    st.stop()

unknown = sorted(set(data[TARGET].astype(str)) - set(CLASS_NAMES))
if unknown:
    st.error(f"`{TARGET}` contains unknown labels: {unknown}. Expected: {CLASS_NAMES}")
    st.stop()

data = data.dropna(subset=FEATURES + [TARGET])
X = data[FEATURES]
y_true = le.transform(data[TARGET].astype(str))

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_eval, tab_compare, tab_pred, tab_data = st.tabs(
    ["📈 Model evaluation", "🏆 Compare all models", "🔮 Predictions", "🗂️ Uploaded data"]
)

# ---- Tab 1: selected model ------------------------------------------------
with tab_eval:
    model = models[model_name]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    m = compute_metrics(y_true, y_pred, y_proba, len(CLASS_NAMES))

    st.subheader(f"{model_name} — evaluation on `{source}` ({len(X)} rows)")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")
    st.caption("Precision, Recall and F1 are support-weighted averages; "
               "AUC is one-vs-rest weighted ROC-AUC; MCC is the multi-class Matthews coefficient.")

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("##### Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
        st.pyplot(plot_confusion(cm, CLASS_NAMES, model_name))
    with right:
        st.markdown("##### Classification report")
        rep = classification_report(y_true, y_pred, labels=range(len(CLASS_NAMES)),
                                    target_names=CLASS_NAMES, output_dict=True,
                                    zero_division=0)
        rep_df = pd.DataFrame(rep).T
        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(rep_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                          "f1-score": "{:.4f}", "support": "{:d}"}),
                     width="stretch")

        st.markdown("##### Per-class recall")
        per_class = pd.Series(np.diag(cm) / cm.sum(axis=1).clip(min=1), index=CLASS_NAMES)
        fig, ax = plt.subplots(figsize=(5.5, 2.4))
        ax.barh(per_class.index, per_class.values,
                color=[CLASS_COLOURS.get(c, "#888") for c in per_class.index])
        ax.set_xlim(0, 1)
        for i, v in enumerate(per_class.values):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

# ---- Tab 2: all models side by side ---------------------------------------
with tab_compare:
    st.subheader("All five models on the uploaded data")
    rows = {}
    for name, mdl in models.items():
        rows[name] = compute_metrics(y_true, mdl.predict(X), mdl.predict_proba(X),
                                     len(CLASS_NAMES))
    comp = pd.DataFrame(rows).T
    comp.index.name = "ML Model Name"
    st.dataframe(comp.style.format("{:.4f}", na_rep="n/a")
                 .highlight_max(axis=0, color="#b7e4c7"),
                 width="stretch")

    best = comp["MCC"].idxmax()
    st.success(f"Best model on this data by MCC: **{best}** ({comp.loc[best, 'MCC']:.4f})")

    metric_pick = st.radio("Metric to chart", list(comp.columns), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    order = comp[metric_pick].sort_values()
    ax.barh(order.index, order.values,
            color=["#00798c" if i == order.idxmax() else "#9fc5cc" for i in order.index])
    for i, v in enumerate(order.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_pick)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    st.download_button("⬇️ Download comparison table (CSV)",
                       comp.to_csv().encode(), "model_comparison.csv", "text/csv")

# ---- Tab 3: predictions ---------------------------------------------------
with tab_pred:
    st.subheader(f"Row-level predictions — {model_name}")
    out = data[[TARGET]].copy().rename(columns={TARGET: "Actual"})
    out["Predicted"] = le.inverse_transform(y_pred)
    out["Correct"] = out["Actual"] == out["Predicted"]
    for i, c in enumerate(CLASS_NAMES):
        out[f"P({c})"] = y_proba[:, i].round(4)

    only_wrong = st.toggle("Show misclassified rows only")
    view = out[~out["Correct"]] if only_wrong else out
    st.caption(f"{int(out['Correct'].sum())} of {len(out)} rows predicted correctly.")
    st.dataframe(view, width="stretch", height=420)
    st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(),
                       f"predictions_{model_name.lower().replace(' ', '_')}.csv", "text/csv")

# ---- Tab 4: data preview --------------------------------------------------
with tab_data:
    st.subheader("Uploaded data preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(data))
    c2.metric("Feature columns", len(FEATURES))
    c3.metric("Classes present", data[TARGET].nunique())
    st.dataframe(data.head(50), width="stretch")

    dist = data[TARGET].value_counts().reindex(CLASS_NAMES).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.bar(dist.index, dist.values, color=[CLASS_COLOURS.get(c, "#888") for c in dist.index])
    ax.set_title("Target distribution in uploaded file")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

st.divider()
st.caption("ML Assignment 2 · M.Tech AIML · Models trained offline in "
           "`ML_Assignment2_Step2_Models.ipynb`; this app only loads and evaluates them.")
