"""
app.py - Streamlit front end.  Run with:  streamlit run app.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import ids_core as core

st.set_page_config(page_title="Student Habits vs Performance", page_icon="🎓", layout="wide")
DEFAULT_CSV = os.path.join("data", "student_habits_performance.csv")


# ------------------------------------------------------------------ data loading
@st.cache_data
def get_data(source):
    raw = core.load_raw(source)
    cleaned, log = core.clean(raw)
    return raw, cleaned, log


@st.cache_resource
def get_models(df):
    return core.train_models(df)


st.sidebar.title("🎓 Student Habits vs Academic Performance")
upload = st.sidebar.file_uploader("Upload dataset CSV (optional)", type="csv")
source = upload if upload is not None else (DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else None)
if source is None:
    st.error("Dataset not found. Put the Kaggle CSV at data/student_habits_performance.csv "
             "or upload it from the sidebar.")
    st.stop()

raw, df, log = get_data(source)
if core.TARGET not in df.columns:
    st.error(f"Column '{core.TARGET}' not found. Columns present: {list(raw.columns)}")
    st.stop()

page = st.sidebar.radio("Go to", ["1. Dataset overview", "2. Preprocessing", "3. Visualisation (EDA)",
                                  "4. Model comparison", "5. Predict my score", "6. Student clusters"])
num_cols, cat_cols = core.feature_types(df)

# ----------------------------------------------------------------------- pages
if page.startswith("1"):
    st.title("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(raw))
    c2.metric("Features", raw.shape[1] - 1)
    c3.metric("Target", core.TARGET)
    st.dataframe(raw.head(20), width='stretch')
    st.subheader("Summary statistics")
    st.dataframe(raw.describe(include="all").T, width='stretch')

elif page.startswith("2"):
    st.title("Data preprocessing")
    st.write("Steps applied automatically before analysis and modelling:")
    for line in log:
        st.markdown(f"- {line}")
    st.markdown("- Numeric features: **StandardScaler**  \n- Categorical features: **One-Hot Encoding**  \n"
                "- Split: **80% train / 20% test** (random_state=42)")
    c1, c2 = st.columns(2)
    c1.write("Missing values BEFORE cleaning"); c1.dataframe(raw.isna().sum().rename("missing"))
    c2.write("Missing values AFTER cleaning"); c2.dataframe(df.isna().sum().rename("missing"))

elif page.startswith("3"):
    st.title("Exploratory data analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Target distribution", "Correlation heatmap", "Feature vs score", "Categorical"])
    with tab1:
        fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(df[core.TARGET], kde=True, ax=ax)
        ax.set_title("Distribution of exam score"); st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(df[num_cols + [core.TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        corr = df[num_cols + [core.TARGET]].corr()[core.TARGET].drop(core.TARGET).sort_values()
        st.write("Correlation with exam score:"); st.bar_chart(corr)
    with tab3:
        x = st.selectbox("Numeric feature", num_cols)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(data=df, x=x, y=core.TARGET, scatter_kws={"alpha": .4}, line_kws={"color": "red"}, ax=ax)
        st.pyplot(fig)
    with tab4:
        if cat_cols:
            c = st.selectbox("Categorical feature", cat_cols)
            fig, ax = plt.subplots(figsize=(7, 4)); sns.boxplot(data=df, x=c, y=core.TARGET, ax=ax)
            plt.xticks(rotation=30); st.pyplot(fig)

elif page.startswith("4"):
    st.title("Model comparison (predicting exam score)")
    res = get_models(df)
    st.dataframe(res["metrics"].style.format("{:.3f}"), width='stretch')
    best = res["metrics"]["R2"].idxmax()
    st.success(f"Best model on test data: **{best}** (R² = {res['metrics'].loc[best, 'R2']:.3f})")
    name = st.selectbox("Model for plots", list(res["models"]), index=list(res["models"]).index(best))
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(res["y_test"], res["preds"][name], alpha=.5)
        lim = [res["y_test"].min(), res["y_test"].max()]; ax.plot(lim, lim, "r--")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs predicted")
        st.pyplot(fig)
    with c2:
        imp = core.feature_importance(res, name)
        fig, ax = plt.subplots(figsize=(5, 4)); imp.head(10).iloc[::-1].plot.barh(ax=ax)
        ax.set_title("Top features (share of importance)"); st.pyplot(fig)

elif page.startswith("5"):
    st.title("Predict exam score from habits")
    res = get_models(df)
    model_name = st.selectbox("Model", list(res["models"]), index=1)
    row = {}
    cols = st.columns(3)
    for i, c in enumerate(res["num"]):
        lo, hi, med = float(df[c].min()), float(df[c].max()), float(df[c].median())
        is_int = pd.api.types.is_integer_dtype(df[c])
        with cols[i % 3]:
            row[c] = st.slider(c.replace("_", " ").title(), int(lo) if is_int else lo,
                               int(hi) if is_int else hi, int(med) if is_int else med)
    for i, c in enumerate(res["cat"]):
        with cols[(len(res["num"]) + i) % 3]:
            row[c] = st.selectbox(c.replace("_", " ").title(), sorted(df[c].unique()))
    if st.button("Predict exam score", type="primary"):
        score = core.predict_one(res, model_name, row)
        st.metric("Predicted exam score", f"{score:.1f} / 100")
        st.progress(int(score))
        st.caption("Estimate learned from the dataset; it shows association, not a guarantee.")

elif page.startswith("6"):
    st.title("Student clusters (K-Means on habits)")
    ks = core.elbow(df)
    fig, ax = plt.subplots(figsize=(5, 3)); ax.plot(*zip(*ks), marker="o")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow method"); st.pyplot(fig)
    k = st.slider("Number of clusters (k)", 2, 8, 3)
    if st.button("Run clustering", type="primary"):
        clustered, profile, cols = core.cluster_students(df, k)
        st.subheader("Cluster profiles (mean values)")
        st.dataframe(profile, width='stretch')
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=clustered, x=cols[0], y=cols[1], hue="cluster", palette="tab10", ax=ax)
        st.pyplot(fig)
        st.bar_chart(profile[core.TARGET])
