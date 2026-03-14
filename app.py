"""
SafeStage Analytics Dashboard
==============================
Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from ml_functions import (
    run_classification, predict_user_type,
    run_clustering,
    run_regression, predict_max_price,
    run_association_rules,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeStage Analytics",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d0d1a 0%, #12122a 100%);
    border-right: 1px solid #2a2a4a;
  }
  [data-testid="stSidebar"] * { color: #e0e0f0 !important; }

  /* Main background */
  .stApp { background: #07070f; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #12122a, #1a1a35);
    border: 1px solid #2d2d5a;
    border-radius: 12px;
    padding: 1rem;
  }
  [data-testid="stMetricValue"] { color: #e94560 !important; font-family: 'Syne', sans-serif !important; font-size: 2rem !important; }
  [data-testid="stMetricLabel"] { color: #8888bb !important; }

  /* Section headers */
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #e94560;
    letter-spacing: -0.5px;
    margin: 1.5rem 0 0.5rem;
    border-left: 4px solid #e94560;
    padding-left: 12px;
  }
  .page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
  }
  .subtitle {
    font-size: 1rem;
    color: #8888bb;
    margin-bottom: 2rem;
  }
  .insight-box {
    background: linear-gradient(135deg, #1a1a35, #0f0f25);
    border: 1px solid #2d2d5a;
    border-left: 4px solid #e94560;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #c0c0e0;
    font-size: 0.92rem;
    line-height: 1.6;
    margin: 0.8rem 0;
  }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
  }
  .badge-red   { background:#e9456022; color:#e94560; border:1px solid #e94560; }
  .badge-blue  { background:#3d6fe922; color:#6a9ef5; border:1px solid #6a9ef5; }
  .badge-green { background:#27c77f22; color:#27c77f; border:1px solid #27c77f; }
  .divider { border: none; border-top: 1px solid #2a2a4a; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

PALETTE   = ["#e94560", "#6a9ef5", "#27c77f", "#f5a623", "#b06ef5"]
DARK_BG   = "#07070f"
CARD_BG   = "#12122a"
GRID_CLR  = "#2a2a4a"

def dark_fig(fig, height=420):
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color="#c0c0e0", family="DM Sans"),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    fig.update_yaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Data + model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading SafeStage dataset…")
def load_data():
    return pd.read_csv("safestage_cleaned.csv")

@st.cache_resource(show_spinner="Training ML models…")
def load_models(df):
    clf_r  = run_classification(df)
    clus_r = run_clustering(df)
    reg_r  = run_regression(df)
    ar_r   = run_association_rules(df)
    return clf_r, clus_r, reg_r, ar_r

df = load_data()
clf_results, clus_results, reg_results, ar_results = load_models(df)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem;'>
      <span style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:#e94560;'>🎫 SafeStage</span><br>
      <span style='font-size:0.75rem;color:#5555aa;letter-spacing:2px;'>ANALYTICS PLATFORM</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2a2a4a;margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

    page = st.selectbox(
        "Navigate",
        ["📊 Overview & EDA", "🤖 Scalper Detection (ML)", "👥 User Personas", "📈 Market Insights"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#2a2a4a;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem;color:#5555aa;'>DATASET STATS</p>", unsafe_allow_html=True)

    vc = df["User_Type"].value_counts()
    for label, color in [("Fan","#27c77f"),("Scalper","#f5a623"),("Bot","#e94560")]:
        pct = vc.get(label, 0) / len(df) * 100
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.85rem;color:{color};'>"
                    f"<span>{label}s</span><span>{vc.get(label,0)} ({pct:.0f}%)</span></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2a2a4a;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem;color:#3a3a6a;text-align:center;'>Built for the SafeStage<br>Scalper-Proof Platform</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & EDA
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown("<div class='page-title'>Overview & Exploratory Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Understanding the battlefield — who are the actors, and how do they behave?</div>", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", f"{len(df):,}")
    c2.metric("Genuine Fans", f"{(df.User_Type=='Fan').sum():,}", f"{(df.User_Type=='Fan').mean()*100:.0f}%")
    c3.metric("Scalpers Detected", f"{(df.User_Type=='Scalper').sum():,}", f"{(df.User_Type=='Scalper').mean()*100:.0f}%")
    c4.metric("Bots Flagged", f"{(df.User_Type=='Bot').sum():,}", f"{(df.User_Type=='Bot').mean()*100:.0f}%")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Behavioural Fingerprints</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Time to click buy histogram — the bot detection centrepiece
        fig = go.Figure()
        for utype, color in [("Fan","#27c77f"),("Scalper","#f5a623"),("Bot","#e94560")]:
            sub = df[df.User_Type==utype]["Time_to_Click_Buy_ms"]
            fig.add_trace(go.Histogram(
                x=sub, name=utype, nbinsx=50, opacity=0.75,
                marker_color=color, showlegend=True
            ))
        fig.update_layout(barmode="overlay", title="⚡ Time to Click 'Buy' (ms) — Bot Behaviour Exposed")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='insight-box'>🔍 <b>Insight:</b> Bots respond in under 500ms — physically impossible for humans. This single feature drives 40%+ of classification accuracy.</div>", unsafe_allow_html=True)

    with col2:
        # Session count box plot
        fig = px.box(
            df, x="User_Type", y="Session_Count_Last_24h",
            color="User_Type", color_discrete_sequence=["#27c77f","#e94560","#f5a623"],
            title="📡 Session Count (Last 24h) by User Type",
            category_orders={"User_Type": ["Fan","Scalper","Bot"]},
        )
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='insight-box'>📊 <b>Insight:</b> Scalpers refresh event pages 10–35× per day, revealing their inventory-stalking behaviour. Bots go even higher (25–80×).</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Correlation Heatmap</div>", unsafe_allow_html=True)

    num_cols = [
        "Age","Account_Age_Days","Liveness_Check_Duration_sec",
        "Time_to_Click_Buy_ms","Session_Count_Last_24h",
        "Historical_Spend_USD","Max_Price_Willing_to_Pay_USD",
        "Events_Attended_Last_Year","User_Type_Code",
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        linewidths=0.4, linecolor="#2a2a4a",
        annot_kws={"size": 8, "color": "white"},
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    plt.setp(ax.get_xticklabels(), color="white", fontsize=8, rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), color="white", fontsize=8)
    ax.set_title("Feature Correlation Matrix", color="white", fontsize=12, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("<div class='insight-box'>📌 <b>Key correlation:</b> <code>Historical_Spend_USD</code> and <code>Max_Price_Willing_to_Pay_USD</code> have a strong positive relationship (r ≈ 0.84), validating our regression model. <code>User_Type_Code</code> correlates strongly (negatively) with <code>Time_to_Click_Buy_ms</code> — bots are fastest.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Spend Distribution by User Type</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.violin(
            df, x="User_Type", y="Historical_Spend_USD",
            color="User_Type", box=True, points=False,
            color_discrete_sequence=["#27c77f","#e94560","#f5a623"],
            title="💰 Historical Spend Distribution",
            category_orders={"User_Type": ["Fan","Scalper","Bot"]},
        )
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(
            df.sample(500, random_state=1),
            x="Historical_Spend_USD", y="Max_Price_Willing_to_Pay_USD",
            color="User_Type", opacity=0.65,
            color_discrete_sequence=["#27c77f","#e94560","#f5a623"],
            title="🎯 Spend vs Max Willingness to Pay",
            trendline="ols",
        )
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Scalper Detection (ML)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Scalper Detection (ML)":
    st.markdown("<div class='page-title'>Scalper Detection Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Random Forest classification + Linear Regression pricing model — the twin engines of SafeStage's scalper-proof checkout.</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌲 Classification Model", "📉 Regression Model"])

    # ── Tab 1: Classification ─────────────────────────────────────────────────
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Accuracy", f"{clf_results['accuracy']*100:.1f}%")
        report = clf_results["classification_report"]
        c2.metric("Fan F1-Score",   f"{report['Fan']['f1-score']:.3f}")
        c3.metric("Bot F1-Score",   f"{report['Bot']['f1-score']:.3f}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
            fi = clf_results["feature_importance"].reset_index()
            fi.columns = ["Feature", "Importance"]
            fig = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=["#3d3d6a","#e94560"],
                title="What gives scalpers & bots away?",
            )
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
            cm = clf_results["confusion_matrix"]
            labels = ["Bot","Scalper","Fan"]
            fig = px.imshow(
                cm, x=labels, y=labels,
                color_continuous_scale=["#07070f","#e94560"],
                text_auto=True, aspect="auto",
                title="Predicted vs Actual",
            )
            dark_fig(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>🕹️ Live Classifier — Try It Yourself</div>", unsafe_allow_html=True)
        st.markdown("<div class='insight-box'>Adjust the sliders below to simulate a user's behaviour at checkout. The model will predict in real time whether they are a Fan, Scalper, or Bot.</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            ttc  = st.slider("Time to Click Buy (ms)", 50, 18000, 4000, step=50)
            mmp  = st.selectbox("Mouse Movement", ["Natural→2", "Erratic→1", "Linear→0"])
            mmp_code = int(mmp.split("→")[1])
        with col2:
            sess = st.slider("Sessions Last 24h", 1, 80, 5)
            live = st.slider("Liveness Check (sec)", 0.1, 9.0, 4.5, step=0.1)
        with col3:
            ver  = st.selectbox("Verified?", ["Yes→1","No→0"])
            ver_code = int(ver.split("→")[1])
            acct = st.slider("Account Age (days)", 1, 2000, 365)
            hist = st.slider("Historical Spend ($)", 50, 7500, 500, step=50)

        pred = predict_user_type(
            clf_results["model"],
            clf_results["features"],
            clf_results["label_map"],
            {
                "Time_to_Click_Buy_ms": ttc,
                "Mouse_Movement_Pattern_Code": mmp_code,
                "Session_Count_Last_24h": sess,
                "Liveness_Check_Duration_sec": live,
                "Verified_Status_Code": ver_code,
                "Account_Age_Days": acct,
                "Historical_Spend_USD": hist,
            }
        )
        label = pred["predicted_label"]
        color_map = {"Fan":"#27c77f","Scalper":"#f5a623","Bot":"#e94560"}
        color = color_map[label]
        probs = pred["probabilities"]

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{color}22,#0d0d1a);border:2px solid {color};
             border-radius:12px;padding:1.2rem 1.5rem;margin-top:1rem;'>
          <span style='font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:{color};'>
            Prediction: {label}
          </span><br>
          <span style='color:#8888bb;font-size:0.85rem;'>
            Fan {probs['Fan']*100:.1f}%  ·  Scalper {probs['Scalper']*100:.1f}%  ·  Bot {probs['Bot']*100:.1f}%
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: Regression ──────────────────────────────────────────────────────
    with tab2:
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score",  f"{reg_results['r2']:.4f}")
        c2.metric("MAE",       f"${reg_results['mae']:,.0f}")
        c3.metric("RMSE",      f"${reg_results['rmse']:,.0f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>Predicted vs Actual</div>", unsafe_allow_html=True)
            y_test = reg_results["y_test"]
            y_pred = reg_results["y_pred"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_pred, mode="markers",
                marker=dict(color="#6a9ef5", opacity=0.4, size=4),
                name="Predictions",
            ))
            max_val = float(max(y_test.max(), y_pred.max()))
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line=dict(color="#e94560", dash="dash", width=1),
                name="Perfect Fit",
            ))
            fig.update_layout(title="Predicted vs Actual Max Price")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<div class='section-title'>Regression Coefficients</div>", unsafe_allow_html=True)
            coeff = reg_results["coefficients"]
            fig = px.bar(
                coeff, x="Coefficient", y="Feature", orientation="h",
                color="Coefficient",
                color_continuous_scale=["#e94560","#07070f","#6a9ef5"],
                title="Feature impact on Max Price",
            )
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>🕹️ Price Predictor</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            r_hist  = st.slider("Historical Spend ($)", 50, 7500, 600, step=50, key="r_hist")
            r_age   = st.slider("User Age", 18, 60, 28, key="r_age")
        with col2:
            r_acct  = st.slider("Account Age (days)", 1, 2000, 400, key="r_acct")
            r_sess  = st.slider("Sessions Last 24h", 1, 80, 4, key="r_sess")

        predicted_price = predict_max_price(
            reg_results["model"], reg_results["features"],
            {"Historical_Spend_USD": r_hist, "Age": r_age,
             "Account_Age_Days": r_acct, "Session_Count_Last_24h": r_sess}
        )
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#6a9ef522,#0d0d1a);border:2px solid #6a9ef5;
             border-radius:12px;padding:1.2rem 1.5rem;margin-top:1rem;'>
          <span style='font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:#6a9ef5;'>
            Predicted Max Price: ${max(0, predicted_price):,.0f}
          </span><br>
          <span style='color:#8888bb;font-size:0.85rem;'>
            SafeStage can use this to enforce a personalised face-value ceiling at checkout.
          </span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — User Personas (Clustering)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 User Personas":
    st.markdown("<div class='page-title'>User Persona Segments</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>K-Means clustering reveals three behavioural archetypes hidden in your audience data.</div>", unsafe_allow_html=True)

    df_c = clus_results["df_clustered"]
    summary = clus_results["summary"]

    # Persona KPI cards
    persona_colors = {"🎸 The Superfan":"#e94560","🎟️ The Casual":"#6a9ef5","🤖 The Bot/Scalper":"#f5a623"}
    cols = st.columns(3)
    for i, (persona, color) in enumerate(persona_colors.items()):
        count = (df_c["Persona"]==persona).sum()
        pct   = count / len(df_c) * 100
        cols[i].markdown(f"""
        <div style='background:linear-gradient(135deg,{color}18,#0d0d1a);border:1px solid {color}55;
             border-radius:12px;padding:1.2rem;text-align:center;'>
          <div style='font-size:1.6rem;'>{persona}</div>
          <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:{color};'>{count:,}</div>
          <div style='color:#8888bb;font-size:0.85rem;'>{pct:.0f}% of users</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='section-title'>2D Persona Scatter Plot</div>", unsafe_allow_html=True)
        persona_color_map = {
            "🎸 The Superfan": "#e94560",
            "🎟️ The Casual":   "#6a9ef5",
            "🤖 The Bot/Scalper": "#f5a623",
        }
        fig = px.scatter(
            df_c.sample(800, random_state=7),
            x="Historical_Spend_USD",
            y="Session_Count_Last_24h",
            color="Persona",
            color_discrete_map=persona_color_map,
            size="Account_Age_Days",
            size_max=14,
            hover_data=["Age","Liveness_Check_Duration_sec","User_Type"],
            title="Spend vs Session Activity — Persona Clusters",
            opacity=0.72,
        )
        dark_fig(fig, 480)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Cluster Summary</div>", unsafe_allow_html=True)
        for _, row in summary.iterrows():
            persona = row["Persona"]
            color   = persona_colors.get(persona, "#aaaacc")
            st.markdown(f"""
            <div style='background:#12122a;border:1px solid {color}55;border-left:3px solid {color};
                 border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.8rem;'>
              <b style='color:{color};'>{persona}</b><br>
              <span style='font-size:0.82rem;color:#9090c0;'>
                Avg Age: {row['Age']:.0f}  ·  Spend: ${row['Historical_Spend_USD']:,.0f}<br>
                Sessions: {row['Session_Count_Last_24h']:.0f}  ·  Liveness: {row['Liveness_Check_Duration_sec']:.1f}s<br>
                Acct Age: {row['Account_Age_Days']:.0f} days
              </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>3D Persona View</div>", unsafe_allow_html=True)
    sample_3d = df_c.sample(600, random_state=3)
    fig3d = px.scatter_3d(
        sample_3d,
        x="Historical_Spend_USD", y="Session_Count_Last_24h", z="Age",
        color="Persona",
        color_discrete_map=persona_color_map,
        opacity=0.75, size_max=4,
        title="3D Persona Space: Spend × Sessions × Age",
    )
    fig3d.update_traces(marker_size=3)
    fig3d.update_layout(
        paper_bgcolor=DARK_BG,
        scene=dict(
            bgcolor=CARD_BG,
            xaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_CLR, color="#8888bb"),
            yaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_CLR, color="#8888bb"),
            zaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_CLR, color="#8888bb"),
        ),
        font=dict(color="#c0c0e0"),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("<div class='section-title'>Elbow Curve — Optimal K</div>", unsafe_allow_html=True)
    inertias = clus_results["inertias"]
    fig = go.Figure(go.Scatter(
        x=list(inertias.keys()), y=list(inertias.values()),
        mode="lines+markers",
        line=dict(color="#e94560", width=2),
        marker=dict(color="#e94560", size=8),
    ))
    fig.update_layout(title="Inertia vs Number of Clusters (Elbow Method)",
                      xaxis_title="k", yaxis_title="Inertia")
    dark_fig(fig, 340)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Market Insights (Association Rules)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Market Insights":
    st.markdown("<div class='page-title'>Market Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Association rule mining uncovers what users want most — and why they need SafeStage.</div>", unsafe_allow_html=True)

    # Live parameter controls
    with st.expander("⚙️ Adjust Apriori Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        min_sup  = col1.slider("Min Support",    0.05, 0.50, 0.10, 0.01)
        min_conf = col2.slider("Min Confidence", 0.30, 0.90, 0.50, 0.05)
        min_lift = col3.slider("Min Lift",        1.0,  3.0,  1.0, 0.1)

    # Recompute on slider change
    ar = run_association_rules(df, min_sup, min_conf, min_lift)
    rules = ar["rules"]

    if rules.empty:
        st.warning("No rules found with these parameters. Try lowering support or confidence.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rules Found",    len(rules))
        c2.metric("Max Lift",       f"{rules['lift'].max():.2f}")
        c3.metric("Avg Confidence", f"{rules['confidence'].mean()*100:.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-title'>Top Association Rules</div>", unsafe_allow_html=True)
            display_rules = rules[["antecedents_str","consequents_str","support","confidence","lift"]].head(15).copy()
            display_rules.columns = ["If User Has…","They Also Want…","Support","Confidence","Lift"]
            display_rules["Support"]    = display_rules["Support"].map("{:.3f}".format)
            display_rules["Confidence"] = display_rules["Confidence"].map("{:.2f}".format)
            display_rules["Lift"]       = display_rules["Lift"].map("{:.2f}".format)
            st.dataframe(
                display_rules,
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            st.markdown("<div class='section-title'>Lift vs Confidence</div>", unsafe_allow_html=True)
            fig = px.scatter(
                rules,
                x="confidence", y="lift",
                size="support", color="lift",
                color_continuous_scale=["#3d3d6a","#e94560"],
                hover_data=["antecedents_str","consequents_str"],
                title="Rule Quality: Lift vs Confidence",
                labels={"confidence":"Confidence","lift":"Lift"},
            )
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>📌 Key SafeStage Insights</div>", unsafe_allow_html=True)

        # Insight 1: Bot Competition → Face-value Resale
        highlight = ar.get("highlight_rules", pd.DataFrame())
        if not highlight.empty:
            h = highlight.iloc[0]
            st.markdown(f"""
            <div class='insight-box'>
            🎯 <b>Core Platform Validation:</b><br>
            Users who cite <b>"Bot Competition"</b> as their biggest challenge are highly likely
            to prefer <b>"Face-value Resale"</b> as their #1 feature.<br>
            <span style='color:#e94560;font-weight:600;'>Confidence: {h['confidence']*100:.0f}%  ·  Lift: {h['lift']:.2f}×</span><br>
            <span style='font-size:0.82rem;color:#6666aa;'>This directly validates SafeStage's core identity-linked resale proposition.</span>
            </div>
            """, unsafe_allow_html=True)

        # Feature frequency bar
        st.markdown("<div class='section-title'>Feature & Challenge Demand</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            feat_counts = df["Preferred_Feature"].value_counts().reset_index()
            feat_counts.columns = ["Feature","Count"]
            fig = px.bar(feat_counts, x="Count", y="Feature", orientation="h",
                         color="Count", color_continuous_scale=["#3d3d6a","#6a9ef5"],
                         title="Most Wanted Platform Features")
            dark_fig(fig, 320)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            chal_counts = df["Challenges_Faced"].value_counts().reset_index()
            chal_counts.columns = ["Challenge","Count"]
            fig = px.bar(chal_counts, x="Count", y="Challenge", orientation="h",
                         color="Count", color_continuous_scale=["#3d3d6a","#e94560"],
                         title="Top Challenges Driving Demand")
            dark_fig(fig, 320)
            st.plotly_chart(fig, use_container_width=True)

        # Interest × Feature heatmap
        st.markdown("<div class='section-title'>Interest × Feature Cross-Analysis</div>", unsafe_allow_html=True)
        cross = pd.crosstab(df["Primary_Interest"], df["Preferred_Feature"])
        fig = px.imshow(
            cross, text_auto=True, aspect="auto",
            color_continuous_scale=["#07070f","#e94560"],
            title="Who wants what? (Interest × Preferred Feature)",
        )
        dark_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

        # App adoption by challenge
        st.markdown("<div class='section-title'>SafeStage Adoption Likelihood by Challenge</div>", unsafe_allow_html=True)
        adoption = pd.crosstab(df["Challenges_Faced"], df["App_Inclination"], normalize="index") * 100
        fig = go.Figure()
        for col_name, color in [("Yes","#27c77f"),("Maybe","#f5a623"),("No","#e94560")]:
            if col_name in adoption.columns:
                fig.add_trace(go.Bar(
                    name=col_name, x=adoption.index,
                    y=adoption[col_name].round(1),
                    marker_color=color,
                ))
        fig.update_layout(barmode="stack", title="App Inclination % by Challenge Group",
                          yaxis_title="% of users")
        dark_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='insight-box'>💡 <b>Takeaway for product roadmap:</b> Users citing 'Bot Competition' show the highest 'Yes' rate for SafeStage adoption — they are your primary early adopters. Lead all marketing with anti-bot messaging.</div>", unsafe_allow_html=True)
