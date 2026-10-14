"""
ids_core.py - Data science logic for the Student Habits vs Academic Performance mini project.
The Streamlit front end (app.py) only calls the functions defined here.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "exam_score"
ID_COLS = ["student_id"]
RANDOM_STATE = 42


# ---------------------------------------------------------------- preprocessing
def load_raw(path_or_buffer) -> pd.DataFrame:
    """Read the CSV and normalise column names (lower case, underscores)."""
    df = pd.read_csv(path_or_buffer)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def clean(df: pd.DataFrame):
    """Clean the raw data. Returns (clean_df, log) where log lists every step performed."""
    log = []
    out = df.copy()

    n_dup = int(out.duplicated().sum())
    out = out.drop_duplicates()
    log.append(f"Duplicate rows removed: {n_dup}")

    drop = [c for c in ID_COLS if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
        log.append(f"Identifier column(s) dropped (no predictive value): {drop}")

    out = out.dropna(subset=[TARGET])  # cannot learn from rows with no label

    for c in out.columns:
        n_miss = int(out[c].isna().sum())
        if n_miss == 0:
            continue
        if out[c].dtype == object or str(out[c].dtype).startswith("str"):
            out[c] = out[c].fillna("Unknown")
            log.append(f"'{c}': {n_miss} missing -> filled with 'Unknown' (categorical)")
        else:
            med = out[c].median()
            out[c] = out[c].fillna(med)
            log.append(f"'{c}': {n_miss} missing -> filled with median ({med:.2f})")

    if not log[1:] or all("missing" not in l for l in log):
        log.append("No missing values found in remaining columns")

    out[TARGET] = out[TARGET].clip(0, 100)
    log.append("Target 'exam_score' clipped to the valid range 0-100")
    return out.reset_index(drop=True), log


def feature_types(df: pd.DataFrame):
    feats = [c for c in df.columns if c != TARGET]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    return num, cat


# ---------------------------------------------------------------------- models
def _pipeline(model, num, cat):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),                       # scaling
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),  # encoding
    ])
    return Pipeline([("prep", pre), ("model", model)])


def train_models(df: pd.DataFrame):
    """Train Linear Regression and Random Forest; return fitted models + metrics."""
    num, cat = feature_types(df)
    X, y = df[num + cat], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }
    fitted, rows, preds = {}, [], {}
    for name, m in models.items():
        pipe = _pipeline(m, num, cat).fit(Xtr, ytr)
        p = pipe.predict(Xte)
        fitted[name], preds[name] = pipe, p
        rows.append({
            "Model": name,
            "R2": r2_score(yte, p),
            "MAE": mean_absolute_error(yte, p),
            "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
        })
    metrics = pd.DataFrame(rows).set_index("Model")
    return {"models": fitted, "metrics": metrics, "y_test": yte.reset_index(drop=True),
            "preds": preds, "num": num, "cat": cat}


def feature_importance(res, model_name="Linear Regression") -> pd.Series:
    """Standardised coefficients (LR) or impurity importances (RF), grouped back to original columns."""
    pipe = res["models"][model_name]
    names = pipe.named_steps["prep"].get_feature_names_out()
    m = pipe.named_steps["model"]
    vals = np.abs(m.coef_) if hasattr(m, "coef_") else m.feature_importances_
    s = pd.Series(vals, index=names)
    grouped = {}
    for n, v in s.items():
        kind, col = n.split("__", 1)
        if kind == "cat":  # strip the one-hot suffix
            col = next(c for c in res["cat"] if col.startswith(c))
        grouped[col] = grouped.get(col, 0) + v
    out = pd.Series(grouped).sort_values(ascending=False)
    return out / out.sum()


def predict_one(res, model_name, row: dict) -> float:
    X = pd.DataFrame([row])[res["num"] + res["cat"]]
    return float(np.clip(res["models"][model_name].predict(X)[0], 0, 100))


# ------------------------------------------------------------------ clustering
CLUSTER_FEATURES = ["study_hours_per_day", "social_media_hours", "netflix_hours",
                    "sleep_hours", "attendance_percentage", "exercise_frequency"]


def cluster_students(df: pd.DataFrame, k: int):
    """K-Means on lifestyle/study habits. Returns df with 'cluster' column + profile table."""
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    Z = StandardScaler().fit_transform(df[cols])
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    out = df.copy()
    out["cluster"] = km.labels_
    profile = out.groupby("cluster")[cols + [TARGET]].mean().round(2)
    profile["students"] = out["cluster"].value_counts().sort_index()
    return out, profile, cols


def elbow(df: pd.DataFrame, kmax=8):
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    Z = StandardScaler().fit_transform(df[cols])
    return [(k, KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z).inertia_)
            for k in range(1, kmax + 1)]
