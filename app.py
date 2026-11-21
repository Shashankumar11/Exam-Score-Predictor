"""Exam Score Predictor.

Loads the existing XGBoost regressor and label encoders. The model is never
retrained here. On Windows, the Linux pickle's random-number snapshot is
rewritten so the same trees can be deserialized; predictions still come from
those trees.
"""

from __future__ import annotations

import math
import pickle
import warnings
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Exam Score Predictor",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_xgb_model.pkl"
DATA_PATH = BASE_DIR / "Exam_Score_Prediction.csv"
ENCODER_CANDIDATES = (
    BASE_DIR / "label_encoders.pkl",
    BASE_DIR / "label_encoders(1).pkl",
)

FEATURES = [
    "study_hours",
    "class_attendance",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
]
CATEGORICAL_FEATURES = ("sleep_quality", "study_method", "facility_rating")
NUMERIC_FEATURES = ("study_hours", "class_attendance", "sleep_hours")

FEATURE_LABELS = {
    "study_hours": "Study Hours",
    "class_attendance": "Class Attendance",
    "sleep_hours": "Sleep Hours",
    "sleep_quality": "Sleep Quality",
    "study_method": "Study Method",
    "facility_rating": "Facility Rating",
}
CHOICE_LABELS = {
    "average": "Average",
    "good": "Good",
    "poor": "Poor",
    "coaching": "Coaching",
    "group study": "Group Study",
    "mixed": "Mixed",
    "online videos": "Online Videos",
    "self-study": "Self-study",
    "high": "High",
    "low": "Low",
    "medium": "Medium",
}
CHOICE_ORDER = {
    "sleep_quality": ["average", "good", "poor"],
    "study_method": ["coaching", "group study", "mixed", "online videos", "self-study"],
    "facility_rating": ["high", "low", "medium"],
}


def _extract_rng_text(payload: bytes) -> str:
    marker = b"rng_stateSL"
    position = payload.find(marker)
    if position < 0:
        raise RuntimeError("The serialized booster does not contain an rng_state field.")
    length_at = position + len(marker)
    length = int.from_bytes(payload[length_at : length_at + 8], "big")
    start = length_at + 8
    return payload[start : start + length].decode("ascii")


def _windows_rng_text() -> str:
    """Read the RNG text format of the installed Windows XGBoost build.

    A one-tree probe is fit only to expose that text. It is discarded and is
    not the exam-score model.
    """
    probe = xgb.XGBRegressor(n_estimators=1, max_depth=1, random_state=0)
    probe.fit(
        np.array([[0.0], [1.0]], dtype=float),
        np.array([0.0, 1.0], dtype=float),
    )
    raw = pickle.dumps(probe)
    header = raw.find(b"{L")
    opcode = raw.rfind(bytes([0x96]), 0, header)
    if opcode < 0 or header < 0:
        raise RuntimeError("Could not read the local XGBoost serialization format.")
    size = int.from_bytes(raw[opcode + 1 : opcode + 9], "little")
    payload = raw[opcode + 9 : opcode + 9 + size]
    return _extract_rng_text(payload)


def _repair_linux_rng(blob: bytes, windows_rng: str) -> bytearray:
    marker = b"rng_stateSL"
    position = blob.find(marker)
    if position < 0:
        return bytearray(blob)
    length_at = position + len(marker)
    length = int.from_bytes(blob[length_at : length_at + 8], "big")
    start = length_at + 8
    encoded = windows_rng.encode("ascii")
    repaired = bytearray()
    repaired += blob[:length_at]
    repaired += len(encoded).to_bytes(8, "big")
    repaired += encoded
    repaired += blob[start + length :]
    return repaired


class PortableBooster(xgb.Booster):
    """Booster loader that swaps the non-portable Linux RNG snapshot."""

    windows_rng = ""

    def __setstate__(self, state):
        handle = state.get("handle")
        if isinstance(handle, (bytes, bytearray)):
            state = dict(state)
            state["handle"] = _repair_linux_rng(bytes(handle), type(self).windows_rng)
        super().__setstate__(state)


class ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "xgboost.core" and name == "Booster":
            return PortableBooster
        return super().find_class(module, name)


def resolve_encoder_path() -> Path | None:
    for path in ENCODER_CANDIDATES:
        if path.is_file():
            return path
    return None


def load_trained_model(path: Path) -> xgb.XGBRegressor:
    try:
        with path.open("rb") as handle:
            model = pickle.load(handle)
    except Exception as exc:
        message = str(exc).lower()
        repairable = "input stream corrupted" in message or "not writable" in message
        if not repairable:
            raise RuntimeError(f"The XGBoost model could not be loaded: {exc}") from exc
        PortableBooster.windows_rng = _windows_rng_text()
        with path.open("rb") as handle:
            try:
                model = ModelUnpickler(handle).load()
            except Exception as repair_error:
                raise RuntimeError(
                    "The saved model could not be deserialized. "
                    "Install xgboost==3.4.1 and keep best_xgb_model.pkl unchanged."
                ) from repair_error

    if not isinstance(model, xgb.XGBRegressor):
        raise RuntimeError("best_xgb_model.pkl does not contain an XGBRegressor.")

    names = list(model.get_booster().feature_names or [])
    if names != FEATURES:
        found = ", ".join(names) if names else "none"
        raise RuntimeError(
            "The model features do not match the expected six inputs. "
            f"Found: {found}."
        )
    return model


def load_encoder_map(path: Path) -> dict:
    with path.open("rb") as handle:
        encoders = pickle.load(handle)
    if not isinstance(encoders, dict):
        raise RuntimeError("The encoder file does not contain a dictionary of LabelEncoders.")
    missing = [name for name in CATEGORICAL_FEATURES if name not in encoders]
    if missing:
        raise RuntimeError("Missing label encoders: " + ", ".join(missing))
    for name in CATEGORICAL_FEATURES:
        classes = getattr(encoders[name], "classes_", None)
        if classes is None or len(classes) == 0:
            raise RuntimeError(f"The {name} encoder has no classes.")
    return encoders


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in FEATURES + ["exam_score"] if column not in frame.columns]
    if missing:
        raise RuntimeError("The dataset is missing columns: " + ", ".join(missing))
    for column in NUMERIC_FEATURES + ("exam_score",):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


@st.cache_resource(show_spinner=False)
def cached_model() -> xgb.XGBRegressor:
    return load_trained_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def cached_encoders() -> dict:
    path = resolve_encoder_path()
    if path is None:
        raise FileNotFoundError("Label encoder pickle")
    return load_encoder_map(path)


@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


def missing_project_files() -> list[str]:
    missing = []
    if not MODEL_PATH.is_file():
        missing.append(MODEL_PATH.name)
    if resolve_encoder_path() is None:
        missing.append("label_encoders.pkl")
    if not DATA_PATH.is_file():
        missing.append(DATA_PATH.name)
    return missing


def choice_options(encoders: dict, feature: str) -> list[str]:
    available = {str(value) for value in encoders[feature].classes_}
    ordered = [value for value in CHOICE_ORDER[feature] if value in available]
    leftovers = sorted(available.difference(ordered))
    return ordered + leftovers


def choice_label(value: str) -> str:
    return CHOICE_LABELS.get(value, value.replace("_", " ").title())


def dataset_bounds(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds = {}
    for column in NUMERIC_FEATURES:
        series = frame[column].dropna()
        if series.empty:
            raise RuntimeError(f"The dataset has no values for {column}.")
        low = float(series.min())
        high = float(series.max())
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise RuntimeError(f"The dataset range for {column} is not usable.")
        bounds[column] = (low, high)
    return bounds


def snap_to_step(value: float, low: float, high: float, step: float) -> float:
    if step <= 0:
        return min(max(value, low), high)
    steps = round((value - low) / step)
    snapped = low + steps * step
    snapped = min(max(snapped, low), high)
    decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(snapped, decimals)


def score_band(score: float) -> tuple[str, str]:
    if score >= 90:
        return "Outstanding", "This sits in the highest band of examination performance."
    if score >= 75:
        return "Strong", "This is a strong result, above most of the range students reach."
    if score >= 55:
        return "On track", "This is close to a typical result in the dataset."
    if score >= 40:
        return "Developing", "This is below most results in the dataset."
    return "Low", "This is near the lower end of scores observed in the dataset."


def format_score(value: float) -> str:
    text = f"{value:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def format_metric(value: float) -> str:
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def prepare_features(encoders: dict, values: dict[str, object]) -> pd.DataFrame:
    encoded = {
        "study_hours": float(values["study_hours"]),
        "class_attendance": float(values["class_attendance"]),
        "sleep_hours": float(values["sleep_hours"]),
        "sleep_quality": values["sleep_quality"],
        "study_method": values["study_method"],
        "facility_rating": values["facility_rating"],
    }
    for name in NUMERIC_FEATURES:
        number = encoded[name]
        if not math.isfinite(number):
            raise ValueError(f"{FEATURE_LABELS[name]} must be a finite number.")
    for name in CATEGORICAL_FEATURES:
        raw = str(encoded[name])
        classes = {str(item) for item in encoders[name].classes_}
        if raw not in classes:
            allowed = ", ".join(choice_label(item) for item in choice_options(encoders, name))
            raise ValueError(f"{FEATURE_LABELS[name]} must be one of: {allowed}.")
        encoded[name] = int(encoders[name].transform([raw])[0])

    frame = pd.DataFrame([encoded], columns=FEATURES)
    return frame.astype(
        {
            "study_hours": "float64",
            "class_attendance": "float64",
            "sleep_hours": "float64",
            "sleep_quality": "int64",
            "study_method": "int64",
            "facility_rating": "int64",
        }
    )


def predict_score(model: xgb.XGBRegressor, features: pd.DataFrame) -> float:
    prediction = model.predict(features)
    if len(prediction) != 1:
        raise RuntimeError("The model did not return a single score.")
    score = float(prediction[0])
    if not math.isfinite(score):
        raise RuntimeError("The model returned a non-numeric score.")
    return score


def inject_css() -> None:
    st.markdown(
        """
        <style>
          @import url("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,520;9..144,650&family=Outfit:wght@360;470;560;650&display=swap");

          :root {
            --ink: #1c1915;
            --muted: #5f584e;
            --line: rgba(28, 25, 21, 0.08);
            --card: rgba(255, 252, 247, 0.94);
            --teal: #0f6e56;
            --copper: #b8613a;
          }

          html, body, [class*="css"] {
            font-family: Outfit, "Segoe UI", sans-serif;
            color: var(--ink);
          }

          .stApp {
            background:
              radial-gradient(1100px 480px at 8% -10%, rgba(15, 110, 86, 0.14), transparent 55%),
              radial-gradient(900px 420px at 100% 0%, rgba(184, 97, 58, 0.12), transparent 50%),
              #f3efe6;
          }

          h1, h2, h3, .serif { font-family: Fraunces, Georgia, "Times New Roman", serif; }

          #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
          header[data-testid="stHeader"] { background: transparent; }
          .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px; }

          section[data-testid="stSidebar"] {
            background: #1c1915;
            border-right: 0;
          }
          section[data-testid="stSidebar"] * { color: #f6f1e7; }
          section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            background: rgba(246, 241, 231, 0.04);
            border: 1px solid rgba(246, 241, 231, 0.08);
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            margin: 0.28rem 0;
            transition: background 0.18s ease, transform 0.18s ease;
          }
          section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(246, 241, 231, 0.12);
            transform: translateX(2px);
          }
          section[data-testid="stSidebar"] hr { border-color: rgba(246, 241, 231, 0.12); }

          .panel, .kpi, .hero, .result-card, .empty-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: 0 16px 40px rgba(48, 36, 18, 0.06);
          }
          .hero, .panel, .result-card, .empty-card { padding: 26px 28px; }
          .kpi {
            padding: 18px 18px 16px;
            min-height: 132px;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
          }
          .kpi:hover, .panel:hover, .result-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 44px rgba(48, 36, 18, 0.1);
          }
          .eyebrow {
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.74rem;
            font-weight: 650;
            color: var(--teal);
            margin-bottom: 8px;
          }
          .hero h1 {
            font-size: clamp(2.3rem, 4vw, 4rem);
            line-height: 0.98;
            font-weight: 650;
            margin: 0 0 12px;
            letter-spacing: -0.03em;
          }
          .lede { font-size: 1.08rem; color: var(--muted); max-width: 640px; margin: 0; }
          .hero-grid { display: flex; justify-content: space-between; gap: 28px; flex-wrap: wrap; }
          .plaque {
            min-width: 220px;
            background: #1c1915;
            color: #f6f1e7;
            border-radius: 18px;
            padding: 18px 18px 14px;
          }
          .plaque span, .plaque strong { color: #f6f1e7; display: block; }
          .plaque span { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.66; }
          .plaque strong { font-family: Fraunces, Georgia, serif; font-size: 1.35rem; font-weight: 560; margin: 4px 0 12px; }
          .plaque small { color: rgba(246, 241, 231, 0.78); display: block; margin-top: 4px; }
          .kpi .label { color: var(--muted); font-size: 0.84rem; }
          .kpi .value {
            font-family: Fraunces, Georgia, serif;
            font-size: 2rem;
            line-height: 1.1;
            margin: 8px 0 4px;
          }
          .kpi .note { color: var(--muted); font-size: 0.82rem; }
          .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
          .chip {
            border-radius: 999px;
            padding: 6px 11px;
            background: rgba(15, 110, 86, 0.1);
            color: #0c5644;
            font-size: 0.84rem;
            font-weight: 560;
          }
          .result-layout { display: flex; gap: 22px; align-items: center; flex-wrap: wrap; }
          .score-number {
            font-family: Fraunces, Georgia, serif;
            font-size: 4.4rem;
            line-height: 0.9;
            letter-spacing: -0.04em;
            margin: 4px 0;
          }
          .band {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 11px;
            background: rgba(15, 110, 86, 0.12);
            color: #0c5644;
            font-weight: 650;
            font-size: 0.82rem;
          }
          .band.warm { background: rgba(184, 97, 58, 0.14); color: #8a4324; }
          .muted { color: var(--muted); }
          div[data-testid="stForm"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 8px 18px 18px;
            box-shadow: 0 16px 40px rgba(48, 36, 18, 0.06);
          }
          div[data-testid="stFormSubmitButton"] button,
          .stButton > button[kind="primary"] {
            background: #0f6e56;
            color: white;
            border: 0;
            border-radius: 999px;
            min-height: 3rem;
            font-weight: 650;
            letter-spacing: 0.01em;
            transition: transform 0.16s ease, background 0.16s ease;
          }
          div[data-testid="stFormSubmitButton"] button:hover,
          .stButton > button[kind="primary"]:hover {
            background: #0c5a46;
            color: white;
            transform: translateY(-1px);
          }
          div[data-baseweb="slider"] { padding-top: 0.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(figure: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    figure.update_layout(
        title=dict(text=title, font=dict(family="Fraunces, Georgia, serif", size=22, color="#1c1915")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,252,247,0.55)",
        font=dict(family="Outfit, Segoe UI, sans-serif", color="#5f584e", size=13),
        margin=dict(l=8, r=8, t=64, b=8),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#1c1915", font=dict(color="#f6f1e7", size=13)),
        xaxis=dict(title=x_title, gridcolor="rgba(28,25,21,0.06)", zeroline=False),
        yaxis=dict(title=y_title, gridcolor="rgba(28,25,21,0.06)", zeroline=False),
    )
    return figure


def chart_config() -> dict:
    return {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def relationship_chart(frame: pd.DataFrame, feature: str, title: str, x_title: str) -> go.Figure:
    valid = frame[[feature, "exam_score"]].dropna()
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=valid[feature],
            y=valid["exam_score"],
            mode="markers",
            name="Students",
            marker=dict(size=5, color="#0f6e56", opacity=0.28),
            hovertemplate=f"{x_title}: %{{x:.2f}}<br>Exam score: %{{y:.2f}}<extra></extra>",
        )
    )
    cuts = pd.cut(valid[feature], bins=14)
    trend = valid.groupby(cuts, observed=True)["exam_score"].mean()
    if len(trend) > 0:
        figure.add_trace(
            go.Scatter(
                x=[interval.mid for interval in trend.index],
                y=trend.to_numpy(),
                mode="lines",
                name="Binned average",
                line=dict(color="#b8613a", width=3),
                hovertemplate="Bin center: %{x:.2f}<br>Average score: %{y:.2f}<extra></extra>",
            )
        )
    return plot_layout(figure, title, x_title, "Exam score")


def distribution_chart(scores: pd.Series, average: float) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=36,
            name="Students",
            marker=dict(color="#0f6e56", line=dict(color="#f3efe6", width=1)),
            hovertemplate="Score: %{x:.1f}<br>Students: %{y}<extra></extra>",
        )
    )
    figure.add_vline(
        x=average,
        line_dash="dash",
        line_color="#b8613a",
        annotation_text="Average",
        annotation_font_color="#8a4324",
    )
    return plot_layout(figure, "Distribution of exam scores", "Exam score", "Students")


def importance_chart(model: xgb.XGBRegressor) -> go.Figure:
    pairs = sorted(
        zip(FEATURES, model.feature_importances_.astype(float)),
        key=lambda item: item[1],
    )
    figure = go.Figure(
        go.Bar(
            x=[value * 100 for _, value in pairs],
            y=[FEATURE_LABELS[name] for name, _ in pairs],
            orientation="h",
            marker=dict(color="#0f6e56", cornerradius=6),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title="Share of gain (%)", yaxis_title="", showlegend=False, height=380)
    return plot_layout(figure, "What the model relies on", "Share of gain (%)", "")


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="eyebrow" style="color:#d7b08a;">Examination intelligence</div>
            <div class="serif" style="font-size:1.7rem;line-height:1.05;margin-bottom:6px;">Exam Score Predictor</div>
            <div style="opacity:.72;font-size:.92rem;margin-bottom:18px;">XGBoost Regressor · six study factors</div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            ["Dashboard", "Predict Score", "Dataset Insights"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            """
            <div style="font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;opacity:.55;">Model inputs</div>
            """,
            unsafe_allow_html=True,
        )
        for index, feature in enumerate(FEATURES, start=1):
            st.markdown(f"{index}. {FEATURE_LABELS[feature]}")
        st.caption("Saved model and encoders. Nothing is retrained in this app.")
    return page


def render_kpis(frame: pd.DataFrame) -> dict[str, float]:
    scores = frame["exam_score"].dropna()
    stats = {
        "records": float(len(frame)),
        "average": float(scores.mean()),
        "highest": float(scores.max()),
        "lowest": float(scores.min()),
    }
    cards = [
        ("Records", f"{int(stats['records']):,}", "Rows in the original dataset"),
        ("Average score", format_metric(stats["average"]), "Mean exam_score"),
        ("Highest score", format_metric(stats["highest"]), "Best observed result"),
        ("Lowest score", format_metric(stats["lowest"]), "Lowest observed result"),
    ]
    columns = st.columns(4)
    for column, (label, value, note) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="kpi">
                  <div class="label">{escape(label)}</div>
                  <div class="value">{escape(value)}</div>
                  <div class="note">{escape(note)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    return stats


def render_dashboard(model: xgb.XGBRegressor, frame: pd.DataFrame) -> None:
    trees = model.n_estimators
    depth = model.max_depth
    rate = model.learning_rate
    tree_line = f"{int(trees)} trees" if trees else "Gradient-boosted trees"
    depth_line = f"Depth {int(depth)}" if depth is not None else "Shallow trees"
    rate_line = f"Learning rate {rate:g}" if rate is not None else "Squared-error objective"
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-grid">
            <div>
              <div class="eyebrow">Home</div>
              <h1>Exam Score Predictor</h1>
              <p class="lede">Predict student examination performance using Machine Learning.</p>
              <div class="chips">
                <span class="chip">Algorithm · XGBoost Regressor</span>
                <span class="chip">Task · Regression</span>
                <span class="chip">Target · exam_score</span>
              </div>
            </div>
            <aside class="plaque">
              <span>Algorithm</span>
              <strong>XGBoost Regressor</strong>
              <small>{escape(tree_line)}</small>
              <small>{escape(depth_line)}</small>
              <small>{escape(rate_line)}</small>
            </aside>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    render_kpis(frame)
    st.write("")
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown(
            """
            <div class="panel">
              <div class="eyebrow">About this model</div>
              <h3 style="margin:0 0 8px;">A regressor trained on study habits</h3>
              <p class="muted" style="margin-top:0;">
                The dashboard uses the saved XGBoost model to estimate an exam score from
                study hours, class attendance, sleep hours, sleep quality, study method,
                and facility rating. Categorical answers are encoded with the original
                label encoders. Student ID and the other unused columns never enter the model.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(importance_chart(model), width="stretch", config=chart_config())


def render_prediction(model: xgb.XGBRegressor, encoders: dict, frame: pd.DataFrame) -> None:
    bounds = dataset_bounds(frame)
    steps = {"study_hours": 0.01, "class_attendance": 0.1, "sleep_hours": 0.1}
    st.markdown(
        """
        <div class="panel" style="margin-bottom:16px;">
          <div class="eyebrow">Prediction</div>
          <h2 style="margin:0 0 6px;">Score a student profile</h2>
          <p class="muted" style="margin:0;">Sliders use the minimum and maximum values present in the dataset. The score shown after you submit is the model output, unmodified.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("exam-score-form"):
        left, right = st.columns(2)
        numeric_values = {}
        with left:
            for feature in ("study_hours", "class_attendance"):
                low, high = bounds[feature]
                default = snap_to_step(float(frame[feature].median()), low, high, steps[feature])
                numeric_values[feature] = st.slider(
                    FEATURE_LABELS[feature],
                    min_value=low,
                    max_value=high,
                    value=default,
                    step=steps[feature],
                    help=f"Dataset range: {format_score(low)} to {format_score(high)}.",
                )
            sleep_options = choice_options(encoders, "sleep_quality")
            sleep_quality = st.selectbox(
                "Sleep Quality",
                sleep_options,
                format_func=choice_label,
                index=sleep_options.index("average") if "average" in sleep_options else 0,
            )
        with right:
            low, high = bounds["sleep_hours"]
            default = snap_to_step(float(frame["sleep_hours"].median()), low, high, steps["sleep_hours"])
            numeric_values["sleep_hours"] = st.slider(
                "Sleep Hours",
                min_value=low,
                max_value=high,
                value=default,
                step=steps["sleep_hours"],
                help=f"Dataset range: {format_score(low)} to {format_score(high)}.",
            )
            method_options = choice_options(encoders, "study_method")
            study_method = st.selectbox(
                "Study Method",
                method_options,
                format_func=choice_label,
                index=method_options.index("self-study") if "self-study" in method_options else 0,
            )
            facility_options = choice_options(encoders, "facility_rating")
            facility_rating = st.selectbox(
                "Facility Rating",
                facility_options,
                format_func=choice_label,
                index=facility_options.index("medium") if "medium" in facility_options else 0,
            )
        submitted = st.form_submit_button("Predict Exam Score", type="primary", width="stretch")

    if submitted:
        raw_inputs = {
            "study_hours": numeric_values["study_hours"],
            "class_attendance": numeric_values["class_attendance"],
            "sleep_hours": numeric_values["sleep_hours"],
            "sleep_quality": sleep_quality,
            "study_method": study_method,
            "facility_rating": facility_rating,
        }
        try:
            for feature in NUMERIC_FEATURES:
                low, high = bounds[feature]
                value = float(raw_inputs[feature])
                if value < low or value > high:
                    raise ValueError(
                        f"{FEATURE_LABELS[feature]} must stay between {format_score(low)} and {format_score(high)}."
                    )
            features = prepare_features(encoders, raw_inputs)
            score = predict_score(model, features)
            st.session_state["result"] = {
                "score": score,
                "inputs": raw_inputs,
                "average": float(frame["exam_score"].dropna().mean()),
            }
            st.session_state["result_error"] = None
        except Exception as exc:
            st.session_state["result"] = None
            st.session_state["result_error"] = str(exc)

    error = st.session_state.get("result_error")
    result = st.session_state.get("result")
    if error:
        st.error(error)
    elif result:
        render_result(result)
    else:
        st.markdown(
            """
            <div class="empty-card">
              <div class="eyebrow">Result</div>
              <h3 style="margin:0 0 6px;">The score will appear here</h3>
              <p class="muted" style="margin:0;">Choose the six inputs and press Predict Exam Score.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result(result: dict) -> None:
    score = float(result["score"])
    average = float(result["average"])
    band, interpretation = score_band(score)
    delta = score - average
    direction = "above" if delta >= 0 else "below"
    band_class = "band" if score >= 60 else "band warm"
    visual = min(max(score, 0.0), 100.0)
    radius = 70
    circumference = 2 * math.pi * radius
    dash = circumference * (visual / 100.0)
    ring_color = "#0f6e56" if score >= 75 else "#b8613a" if score >= 45 else "#8c3a32"
    outside = ""
    if score < 0 or score > 100:
        outside = " This regressor can score outside 0–100. The number is still the raw prediction."
    inputs = result["inputs"]
    chips = "".join(
        f'<span class="chip">{escape(FEATURE_LABELS[name])}: {escape(display_input(name, inputs[name]))}</span>'
        for name in FEATURES
    )
    st.markdown(
        f"""
        <section class="result-card">
          <div class="result-layout">
            <svg width="180" height="180" viewBox="0 0 180 180" aria-hidden="true">
              <circle cx="90" cy="90" r="{radius}" fill="none" stroke="rgba(28,25,21,0.08)" stroke-width="12"/>
              <circle cx="90" cy="90" r="{radius}" fill="none" stroke="{ring_color}" stroke-width="12"
                stroke-linecap="round"
                stroke-dasharray="{dash:.2f} {circumference:.2f}"
                transform="rotate(-90 90 90)"/>
              <text x="90" y="86" text-anchor="middle" font-family="Fraunces, Georgia, serif" font-size="32" fill="#1c1915">{escape(format_score(score))}</text>
              <text x="90" y="108" text-anchor="middle" font-family="Outfit, Segoe UI, sans-serif" font-size="12" fill="#5f584e">predicted</text>
            </svg>
            <div style="flex:1;min-width:240px;">
              <div class="eyebrow">Prediction result</div>
              <div class="{band_class}">{escape(band)}</div>
              <div class="score-number">{escape(format_score(score))}</div>
              <p class="muted" style="margin:0 0 8px;">Predicted exam score. {escape(interpretation)}</p>
              <p class="muted" style="margin:0;">
                {escape(format_score(abs(delta)))} points {direction} the dataset average of {escape(format_score(average))}.{escape(outside)}
              </p>
            </div>
          </div>
          <div class="chips">{chips}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption("The bar represents the prediction against a 0–100 scale. The score above is not adjusted.")
    st.progress(min(max(score / 100.0, 0.0), 1.0))


def display_input(feature: str, value: object) -> str:
    if feature in CATEGORICAL_FEATURES:
        return choice_label(str(value))
    if feature == "class_attendance":
        return f"{float(value):.1f}%"
    return format_score(float(value))


def render_insights(frame: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="panel" style="margin-bottom:16px;">
          <div class="eyebrow">Dataset insights</div>
          <h2 style="margin:0 0 6px;">What the training records show</h2>
          <p class="muted" style="margin:0;">Every figure below is calculated from Exam_Score_Prediction.csv. The copper line is the average exam score inside each bin of the actual records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stats = render_kpis(frame)
    st.write("")
    scores = frame["exam_score"].dropna()
    st.plotly_chart(
        relationship_chart(frame, "study_hours", "Study hours and exam score", "Study hours"),
        width="stretch",
        config=chart_config(),
    )
    st.plotly_chart(
        relationship_chart(frame, "class_attendance", "Class attendance and exam score", "Class attendance"),
        width="stretch",
        config=chart_config(),
    )
    st.plotly_chart(
        distribution_chart(scores, stats["average"]),
        width="stretch",
        config=chart_config(),
    )


def main() -> None:
    inject_css()
    page = render_sidebar()
    missing = missing_project_files()
    if missing:
        st.error(
            "These project files need to sit next to app.py before the dashboard can start: "
            + ", ".join(missing)
            + "."
        )
        st.stop()
    try:
        with st.spinner("Loading the saved model and dataset..."):
            model = cached_model()
            encoders = cached_encoders()
            frame = cached_dataset()
    except Exception as exc:
        st.error("The saved model, encoders, or dataset could not be loaded.")
        st.caption(str(exc))
        st.stop()

    if page == "Dashboard":
        render_dashboard(model, frame)
    elif page == "Predict Score":
        render_prediction(model, encoders, frame)
    else:
        render_insights(frame)


if __name__ == "__main__":
    main()
