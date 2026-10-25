
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AutoMetric | Mileage Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("car_mileage_model.pkl")

try:
    saved = load_model()
    model = saved["model"]
    features = saved["features"]
except Exception:
    st.error(
        "Model not found. Run `python train_model.py` first."
    )
    st.stop()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background: #0b1120;
    color: #e5edf7;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

[data-testid="stSidebar"] {
    background: #101a2b;
    border-right: 1px solid #243148;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f3f7ff !important;
}

.eyebrow {
    color: #38d9c4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 700;
    line-height: 1.12;
    color: #f7faff;
    margin: 12px 0;
}

.hero-subtitle {
    color: #93a4bb;
    font-size: 15px;
    line-height: 1.7;
}

.panel {
    background: linear-gradient(145deg, #152238, #101a2a);
    border: 1px solid #263751;
    border-radius: 18px;
    padding: 23px;
    min-height: 120px;
}

.panel-label {
    color: #98a9c0;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.1px;
}

.panel-value {
    color: #f5f8ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    margin-top: 12px;
}

.panel-note {
    color: #8ea1ba;
    font-size: 12px;
    margin-top: 7px;
}

.result-panel {
    background: linear-gradient(135deg, #123c42, #102b3c);
    border: 1px solid #287b7a;
    border-radius: 22px;
    padding: 28px;
}

.result-number {
    font-family: 'Space Grotesk', sans-serif;
    color: #65f0d7;
    font-size: clamp(38px, 5vw, 58px);
    font-weight: 700;
    line-height: 1.2;
    margin: 14px 0 5px;
}

.result-caption {
    color: #b7d8da;
    font-size: 13px;
}

.section-label {
    color: #38d9c4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 7px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #38d9c4;
    color: #071821;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 46px;
    transition: 0.2s;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #65f0d7;
    color: #071821;
    border: none;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #111e31;
    border-color: #30415b;
    border-radius: 9px;
}

div[data-testid="stMetric"] {
    background: #121e30;
    border: 1px solid #263751;
    padding: 17px;
    border-radius: 13px;
}

div[data-testid="stMetricLabel"] {
    color: #9aacc3;
}

hr {
    border-color: #263751;
}

.footer {
    color: #75869e;
    font-size: 12px;
    text-align: center;
    padding: 22px 0 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def predict_mileage(values):
    car = pd.DataFrame([values])
    car = car[features]
    mpg = float(model.predict(car)[0])
    return mpg, mpg * 0.4251437

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<div class="eyebrow">AUTOMETRIC / ML SYSTEM</div>',
        unsafe_allow_html=True
    )

    st.markdown("## 🚘 AutoMetric")
    st.caption("Vehicle intelligence platform")

    st.divider()

    st.markdown("### Vehicle configuration")
    st.caption("Adjust the specifications below.")

    with st.form("vehicle_form"):
        cylinders = st.selectbox(
            "Engine cylinders",
            [3, 4, 5, 6, 8],
            index=1
        )

        displacement = st.slider(
            "Displacement (cu. in.)",
            60, 455, 140
        )

        horsepower = st.slider(
            "Horsepower (hp)",
            40, 230, 90
        )

        weight = st.slider(
            "Vehicle weight (lb)",
            1500, 5500, 2500, step=100
        )

        acceleration = st.slider(
            "Acceleration (0–60 proxy)",
            8.0, 25.0, 15.0, step=0.5
        )

        model_year = st.selectbox(
            "Model year",
            list(range(1970, 1983)),
            index=12
        )

        origin = st.selectbox(
            "Manufacturing origin",
            [1, 2, 3],
            format_func=lambda x: {
                1: "USA",
                2: "Europe",
                3: "Japan"
            }[x]
        )

        submitted = st.form_submit_button(
            "✦  Analyze vehicle",
            use_container_width=True
        )

    st.divider()
    st.caption("MODEL  /  Random Forest Regression")
    st.caption("DATASET  /  UCI Auto MPG")
    st.caption("UNITS  /  Kilometres per litre")

# ---------------- HEADER ----------------
st.markdown(
    '<div class="eyebrow">AUTOMOTIVE ANALYTICS  /  DASHBOARD</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Mileage intelligence.<br>'
    'Driven by data.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Estimate vehicle fuel efficiency using a trained machine learning '
    'model. Configure your vehicle and explore the predicted result.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- DEFAULT STATE ----------------
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if submitted:
    values = {
        "cylinders": cylinders,
        "displacement": displacement,
        "horsepower": horsepower,
        "weight": weight,
        "acceleration": acceleration,
        "model_year": model_year - 1900,
        "origin": origin
    }

    # The original dataset uses year codes such as 70, 71, ..., 82.
    st.session_state.last_prediction = {
        "values": values,
        "mpg": predict_mileage(values)[0],
        "km_l": predict_mileage(values)[1]
    }

# ---------------- KPI CARDS ----------------
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown("""
    <div class="panel">
      <div class="panel-label">ML algorithm</div>
      <div class="panel-value" style="font-size:23px">
        Random Forest
      </div>
      <div class="panel-note">Ensemble regression model</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class="panel">
      <div class="panel-label">Prediction target</div>
      <div class="panel-value">km/L</div>
      <div class="panel-note">Fuel efficiency estimate</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="panel">
      <div class="panel-label">Model status</div>
      <div class="panel-value"
           style="font-size:23px;color:#65f0d7">
        Loaded
      </div>
      <div class="panel-note">Ready for inference</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown("---")

# ---------------- RESULTS ----------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown(
        '<div class="section-label">01 / PREDICTION OUTPUT</div>',
        unsafe_allow_html=True
    )
    st.subheader("Estimated fuel efficiency")

    result = st.session_state.last_prediction

    if result is None:
        st.markdown("""
        <div class="result-panel">
          <div class="panel-label">AWAITING VEHICLE ANALYSIS</div>
          <div class="result-number">--.-- <span
            style="font-size:22px">km/L</span></div>
          <div class="result-caption">
            Configure your vehicle in the sidebar and click
            Analyze vehicle to generate a prediction.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-panel">
          <div class="panel-label">PREDICTED MILEAGE</div>
          <div class="result-number">
            {result['km_l']:.2f}
            <span style="font-size:22px">km/L</span>
          </div>
          <div class="result-caption">
            Equivalent to {result['mpg']:.2f} miles per US gallon
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.metric(
            "Predicted MPG",
            f"{result['mpg']:.2f} MPG"
        )

    st.caption(
        "Estimates are based on historical training data. "
        "Actual mileage varies with driving conditions and maintenance."
    )

with right:
    st.markdown(
        '<div class="section-label">02 / VEHICLE PROFILE</div>',
        unsafe_allow_html=True
    )
    st.subheader("Configuration overview")

    if result is not None:
        v = result["values"]

        specs = pd.DataFrame({
            "Specification": [
                "Cylinders",
                "Displacement",
                "Horsepower",
                "Weight",
                "Acceleration",
                "Model year",
                "Origin"
            ],
            "Value": [
                f"{v['cylinders']} cylinders",
                f"{v['displacement']} cu. in.",
                f"{v['horsepower']} hp",
                f"{v['weight']:,} lb",
                f"{v['acceleration']:.1f}",
                str(v["model_year"] + 1900),
                {1: "USA", 2: "Europe", 3: "Japan"}[v["origin"]]
            ]
        })

        st.dataframe(
            specs,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info(
            "Your selected specifications will appear here "
            "after the first prediction."
        )

# ---------------- MODEL INSIGHTS ----------------
st.write("")
st.markdown("---")
st.markdown(
    '<div class="section-label">03 / MODEL INSIGHTS</div>',
    unsafe_allow_html=True
)
st.subheader("What influences the prediction?")

if hasattr(model, "feature_importances_"):
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=importance["Importance"],
            y=importance["Feature"],
            orientation="h",
            marker=dict(
                color=importance["Importance"],
                colorscale=[
                    [0, "#24515f"],
                    [1, "#38d9c4"]
                ]
            ),
            hovertemplate=(
                "%{y}<br>Importance: %{x:.3f}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b9c7d9", family="DM Sans"),
        xaxis=dict(
            title="Relative feature importance",
            gridcolor="#263751",
            zeroline=False
        ),
        yaxis=dict(
            title="",
            gridcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=10, r=15, t=15, b=45),
        height=340
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Feature importance reflects how the fitted Random Forest "
        "uses features across its trees. It does not establish causation."
    )
else:
    st.info("Feature importance is unavailable for this model.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div class="footer">
AUTOMETRIC · CAR MILEAGE PREDICTION SYSTEM<br>
Built with Python · Scikit-learn · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)
