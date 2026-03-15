import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from backend.backend import *
import plotly.graph_objects as go
import os



# -------------------- 1. SETUP --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "database", "myopia.csv")

st.set_page_config(
    page_title="MyoTrack – Myopia Progression Predictor",
    page_icon="👁️",
    layout="wide"
)

# -------------------- 2. STYLING --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

header { visibility: hidden; }
.stApp { margin-top: -60px; background-color: #0a0f1e; color: #e2e8f0; }

/* Card blocks */
div[data-testid="stVerticalBlock"] > div {
    background-color: #111827;
    border-radius: 12px;
    border: 1px solid #1e2d45;
    padding: 24px;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
    padding: 14px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    border-radius: 8px;
    width: 100%;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }

/* Hero */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 52px;
    font-style: italic;
    background: linear-gradient(120deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 4px;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    color: #64748b;
    text-align: center;
    margin-bottom: 32px;
    font-weight: 300;
    letter-spacing: 0.3px;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.5px;
}

/* Section labels */
h3, .stSubheader {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #94a3b8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- 3. HEADER --------------------
st.markdown('<p class="hero-title">MyoTrack</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Myopia Progression Predictor · Evidence-Based · ML-Powered</p>', unsafe_allow_html=True)

# -------------------- 4. DATA LOAD CHECK --------------------
df = load_data(DATA_PATH)
if df is None or df.empty:
    st.error("Could not load myopia.csv — make sure it's in the same folder as this file.")
    st.stop()

# -------------------- 5. INPUT UI --------------------
left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Patient Profile")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Current Age", min_value=5, max_value=24, value=10, step=1)
    with c2:
        gender = st.selectbox("Gender", options=[("Male", 1), ("Female", 0)],
                               format_func=lambda x: x[0])
        gender_val = gender[1]

    st.subheader("Family History")
    c3, c4 = st.columns(2)
    with c3:
        mommy = st.selectbox("Mother myopic?", options=[("No", 0), ("Yes", 1)],
                              format_func=lambda x: x[0])
        mommy_val = mommy[1]
    with c4:
        dadmy = st.selectbox("Father myopic?", options=[("No", 0), ("Yes", 1)],
                              format_func=lambda x: x[0])
        dadmy_val = dadmy[1]

    st.subheader("Daily Habits (hours/day)")
    tvhr = st.slider("TV watching", 0.0, 8.0, 2.0, 0.5)
    comphr = st.slider("Computer / Phone", 0.0, 10.0, 2.0, 0.5)
    readhr = st.slider("Reading / Studying", 0.0, 8.0, 2.0, 0.5)
    sporthr = st.slider("Outdoor / Sports", 0.0, 8.0, 1.0, 0.5)

    screen_time = tvhr + comphr + 0.5 * readhr
    outdoor_time = sporthr

    run_btn = st.button("Predict Progression →")

with right:
    st.subheader("Projection")

    if not run_btn:
        st.markdown("""
        <div style="text-align:center; padding: 80px 20px; color: #334155;">
            <div style="font-size: 56px; margin-bottom: 12px;">👁️</div>
            <div style="font-family: 'DM Sans', sans-serif; font-size: 15px;">
                Fill in the profile and click <strong>Predict Progression</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if age >= 25:
            st.warning("Already at or past age 25 — no projection range available.")
            st.stop()

        result = progression_tracker(
            age=age,
            gender=gender_val,
            mommy=mommy_val,
            dadmy=dadmy_val,
            screen_time=screen_time,
            outdoor_time=outdoor_time,
            data_path=DATA_PATH
        )

        ages = result["ages"]
        spheq = result["spheq_pred"]
        delta = result["delta"]
        baseline = result["baseline_spheq"]
        final_delta = delta[-1]

        risk_label, risk_color = get_risk_label(
            mommy_val + dadmy_val, screen_time, outdoor_time
        )

        # --- Metrics row ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline SPHEQ", f"{baseline:.2f} D")
        m2.metric("Projected Δ by 25", f"{final_delta:+.2f} D")
        m3.metric("Risk Level", risk_label)

        # --- Chart ---
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ages, y=spheq,
            mode="lines+markers",
            name="Predicted SPHEQ",
            line=dict(color="#60a5fa", width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.08)"
        ))

        fig.add_trace(go.Scatter(
            x=ages, y=delta,
            mode="lines",
            name="Δ from baseline",
            line=dict(color="#a78bfa", width=2, dash="dot"),
            yaxis="y2"
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#94a3b8"),
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)"
            ),
            xaxis=dict(
                title="Age", gridcolor="#1e293b",
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title="SPHEQ (Diopters)",
                gridcolor="#1e293b",
                tickfont=dict(size=12)
            ),
            yaxis2=dict(
                title="Δ SPHEQ",
                overlaying="y",
                side="right",
                showgrid=False,
                tickfont=dict(size=11, color="#a78bfa")
            ),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Interpretation ---
        st.markdown("#### Clinical Notes")

        notes = []
        if mommy_val + dadmy_val == 2:
            notes.append("Both parents myopic — significantly elevated genetic risk.")
        elif mommy_val + dadmy_val == 1:
            notes.append("One parent myopic — moderate genetic predisposition.")

        if screen_time >= 6:
            notes.append("High near-work exposure (≥6 hrs/day) accelerates progression.")
        elif screen_time >= 3:
            notes.append("Moderate screen/near-work time. Reducing it may help slow progression.")

        if outdoor_time < 1:
            notes.append("Very low outdoor time — studies show <1 hr/day is a risk factor.")
        elif outdoor_time >= 2:
            notes.append("Good outdoor time — protective against progression.")

        if final_delta < -0.5:
            notes.append(f"Model projects ~{abs(final_delta):.1f}D worsening by age 25 under current habits.")

        if not notes:
            notes.append("Risk profile looks favourable under current habits.")

        for note in notes:
            st.markdown(f"- {note}")

        st.caption("This tool is for educational/research purposes only and is not a clinical diagnostic instrument.")