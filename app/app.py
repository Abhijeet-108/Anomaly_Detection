from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import razorpay as rp

from auth import (
    init_session, 
    logout, 
    render_auth_page,
    _load_users,
    _save_users
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR.parent / "notebook"

st.set_page_config(
    page_title="KAVACAPay · Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: radial-gradient(ellipse 100% 80% at 50% -20%, #1a2744 0%, #0b0f19 45%, #060810 100%);
    }
    

    # .stApp {
    #     background:
    #         radial-gradient(circle at top left, #ff0080 0%, transparent 35%),
    #         radial-gradient(circle at bottom right, #00e5ff 0%, transparent 35%),
    #         #0a0a0f;
    # }
    
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    .hero h1 {
        background: linear-gradient(90deg,#ff4ecd,#00e5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p { color: #94a3b8; margin: 0; line-height: 1.65; max-width: 720px; }

    .pipe-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 1.1rem 1.2rem;
        height: 100%; transition: border-color 0.2s;
    }
    .pipe-card:hover { border-color: rgba(59,130,246,0.35); }
    .pipe-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
    .pipe-card h4 { color: #e2e8f0; font-size: 0.88rem; font-weight: 700; margin: 0 0 0.3rem; }
    .pipe-card p { color: #64748b; font-size: 0.76rem; margin: 0; line-height: 1.45; }

    .section-title {
        color: #cbd5e1; font-size: 0.95rem; font-weight: 700;
        margin: 1.75rem 0 0.85rem;
        display: flex; align-items: center; gap: 8px;
    }
    .section-title::before {
        content: ''; width: 4px; height: 18px;
        background: linear-gradient(180deg, #3b82f6, #8b5cf6);
        border-radius: 2px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 1rem 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important; font-size: 0.72rem !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important; font-weight: 800 !important; font-size: 1.5rem !important;
    }

    .alert-card {
        border-radius: 12px; padding: 1rem 1.15rem; margin-bottom: 0.75rem;
        border-left: 4px solid;
    }
    .alert-danger  { background: rgba(239,68,68,0.1);  border-color: #ef4444; color: #fca5a5; }
    .alert-warning { background: rgba(245,158,11,0.1); border-color: #f59e0b; color: #fcd34d; }
    .alert-success { background: rgba(34,197,94,0.1);  border-color: #22c55e; color: #86efac; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1222 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0,0,0,0.25); border-radius: 12px; padding: 5px; gap: 6px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(37,99,235,0.35), rgba(79,70,229,0.35)) !important;
        color: #bfdbfe !important; border-radius: 8px !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.02);
        border: 2px dashed rgba(56,189,248,0.25);
        border-radius: 14px;
    }

    .model-stat {
        display: flex; justify-content: space-between;
        padding: 0.45rem 0; border-bottom: 1px solid rgba(255,255,255,0.06);
        font-size: 0.82rem;
    }
    .model-stat span:first-child { color: #64748b; }
    .model-stat span:last-child { color: #e2e8f0; font-weight: 600; }

    .auth-wrap { max-width: 720px; margin: 2rem auto 1rem; }
    .auth-header {
        background: linear-gradient(135deg, rgba(30,58,95,0.9) 0%, rgba(15,23,42,0.95) 100%);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 20px; padding: 2rem 2.25rem; text-align: center;
    }
    .auth-header h1 { color: #f8fafc; font-size: 1.75rem; font-weight: 800; margin: 0 0 0.5rem; }
    .auth-header p { color: #94a3b8; margin: 0; font-size: 0.95rem; }
    
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    iso = joblib.load(MODEL_DIR / "isolation_forest_model.pkl")
    rf = joblib.load(MODEL_DIR / "random_forest_fraud_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    meta_path = MODEL_DIR / "model_metadata.pkl"
    metadata = joblib.load(meta_path) if meta_path.exists() else {}
    return iso, rf, scaler, metadata


def risk_label(anomaly_pct, fraud_pct):
    if fraud_pct > 1 or anomaly_pct > 8:
        return "CRITICAL", "#ef4444"
    if fraud_pct > 0.2 or anomaly_pct > 3:
        return "HIGH", "#f97316"
    if fraud_pct > 0.05 or anomaly_pct > 1:
        return "MEDIUM", "#eab308"
    return "LOW", "#22c55e"


def prepare_data(raw_df, expected_features):
    data = raw_df.copy()
    for col in ("Class", "id"):
        if col in data.columns:
            data = data.drop(columns=[col])
    data = data.select_dtypes(include=[np.number])

    missing = set(expected_features) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    extra = set(data.columns) - set(expected_features)
    if extra:
        data = data[expected_features]
    return data


def run_detection(data, iso_model, rf_model, scaler, fraud_threshold):
    scaled = scaler.transform(data)
    iso_pred = iso_model.predict(scaled)
    anomaly_scores = iso_model.decision_function(scaled)

    result = data.copy()
    result["anomaly_score"] = anomaly_scores
    result["Anomaly"] = np.where(iso_pred == -1, 1, 0)

    fraud_probs = rf_model.predict_proba(scaled)[:, 1]
    result["fraud_probability"] = fraud_probs
    result["fraud"] = (fraud_probs >= fraud_threshold).astype(int)

    score_pct = result["anomaly_score"].rank(pct=True)
    result["risk_tier"] = np.select(
        [
            (result["fraud"] == 1),
            (result["Anomaly"] == 1) & (result["fraud"] == 0),
            score_pct <= 0.05,
        ],
        ["Confirmed Fraud", "Suspicious", "Low Risk"],
        default="Normal",
    )
    return result


MAX_TABLE_ROWS = 500

RESULT_COLUMN_CONFIG = {
    "fraud_probability": st.column_config.ProgressColumn(
        "Fraud Score", format="%.1f%%", min_value=0, max_value=1
    ),
    "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
    "Anomaly": st.column_config.NumberColumn("Anomaly", format="%d"),
    "fraud": st.column_config.NumberColumn("Fraud", format="%d"),
}


def prepare_display_df(df, max_rows=MAX_TABLE_ROWS):
    display_cols = [c for c in df.columns if c != "anomaly_score"]
    view = df[display_cols].head(max_rows)
    return view


def show_results_table(df, max_rows=MAX_TABLE_ROWS):
    total = len(df)
    view = prepare_display_df(df, max_rows)
    if total > max_rows:
        st.caption(f"Showing top **{max_rows:,}** of **{total:,}** rows. Download CSV for the full report.")
    st.dataframe(
        view,
        use_container_width=True,
        height=420,
        column_config=RESULT_COLUMN_CONFIG,
    )


def build_charts(result, total, anomaly_count, fraud_count):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#0b0f19")
    bg = "#0b0f19"
    tick = "#64748b"
    title_c = "#e2e8f0"

    # Donut — overall split
    ax = axes[0]
    ax.set_facecolor(bg)
    sizes = [total - anomaly_count, anomaly_count]
    if anomaly_count == 0:
        sizes, labels, cols = [total], ["Normal"], ["#22c55e"]
    else:
        labels, cols = ["Normal", "Anomaly"], ["#22c55e", "#f97316"]
    ax.pie(
        sizes, labels=labels, colors=cols, autopct="%1.1f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor=bg, linewidth=2),
        textprops={"color": title_c, "fontsize": 9},
    )
    ax.set_title("Anomaly Split", color=title_c, fontsize=11, fontweight="bold", pad=10)

    # Fraud breakdown
    ax = axes[1]
    ax.set_facecolor(bg)
    safe_susp = max(anomaly_count - fraud_count, 0)
    bars = ax.bar(
        ["Fraud", "Suspicious", "Clean"],
        [fraud_count, safe_susp, total - anomaly_count],
        color=["#dc2626", "#f59e0b", "#16a34a"],
        width=0.55, edgecolor=bg,
    )
    ax.set_title("Risk Breakdown", color=title_c, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=tick)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(total * 0.004, 0.3),
                str(int(h)), ha="center", color=title_c, fontweight="bold", fontsize=10)

    # Fraud probability histogram
    ax = axes[2]
    ax.set_facecolor(bg)
    ax.hist(result["fraud_probability"], bins=30, color="#6366f1", edgecolor=bg, alpha=0.85)
    ax.axvline(result["fraud_probability"].median(), color="#f59e0b", linestyle="--", linewidth=1.5, label="Median")
    ax.set_title("Fraud Score Distribution", color=title_c, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Fraud Probability", color=tick, fontsize=9)
    ax.set_ylabel("Count", color=tick, fontsize=9)
    ax.tick_params(colors=tick)
    ax.legend(facecolor=bg, edgecolor="#334155", labelcolor=title_c, fontsize=8)

    plt.tight_layout()
    return fig

init_session()

if not st.session_state.authenticated:
    render_auth_page()
    st.stop()

try:
    model, model_rf, scaler, metadata = load_model()
except FileNotFoundError as exc:
    st.error(f"Model files not found. Run `python notebook/train_models.py` first.\n\n{exc}")
    st.stop()

default_threshold = float(metadata.get("fraud_threshold", 0.5))
metrics = metadata.get("metrics", {})
expected_features = list(metadata.get("features", scaler.feature_names_in_))

# ── payment client ───────────────────────────────────────────────────────────────────

client = rp.Client(
    auth = ("", "")
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ KAVACA Pay")
    st.caption("Enterprise fraud screening")
    st.markdown(
        f"**{st.session_state.user_name}**  \n"
        f"<span style='color:#64748b;font-size:0.8rem;'>{st.session_state.user_email}</span>",
        unsafe_allow_html=True,
    )
    if st.button("Logout", use_container_width=True):
        logout()
        st.rerun()
    st.divider()
    
    users = _load_users()
    current_user = users.get(st.session_state.user_email, {})
    
    st.markdown("**Subscription Plan**")
    if current_user.get("plan") == "premium":
        st.markdown("You are on the **Premium** plan.")
    else:
        remaining = (
            current_user.get("max_free_predictions", 3)
            - current_user.get("prediction_count", 0)
        )

        st.warning(f"{remaining} free scans left")

        if st.button("Upgrade to Premium"):

            payment_link = client.payment_link.create({
                "amount": 9900,      # ₹99
                "currency": "INR",
                "description": "Premium Subscription",
                "customer": {
                    "name": current_user["name"]
                }
            })

            current_user["payment_link_id"] = payment_link["id"]
            current_user["payment_status"] = "pending"

            users[st.session_state.user_email] = current_user
            _save_users(users)

            st.link_button(
                "Pay Now",
                payment_link["short_url"]
            )
        
            if st.button("verify Payment"):
                link_id = current_user.get("payment_link_id")

                if not link_id:
                    st.error("No payment found.")
                    st.stop()

                payment = client.payment_link.fetch(link_id)

                status = payment["status"]
                if status == "paid":
                    current_user["plan"] = "premium"
                    current_user["payment_status"] = "paid"

                    users[st.session_state.user_email] = current_user
                    _save_users(users)

                    st.success("Premium Activated!")

                else:
                    st.warning(f"Payment Status: {status}")

        
    st.divider()

    st.markdown("**Detection Settings**")
    fraud_threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.05, max_value=0.95, value=default_threshold, step=0.01,
        help="Transactions above this score are flagged as confirmed fraud.",
    )
    st.caption(f"Default optimal threshold: **{default_threshold:.2f}**")

    st.divider()
    st.markdown("**Model Performance**")
    if metrics:
        st.markdown(
            f"""
            <div class="model-stat"><span>RF ROC-AUC</span><span>{metrics.get('random_forest_auc', 0):.4f}</span></div>
            <div class="model-stat"><span>RF F1 Score</span><span>{metrics.get('random_forest_f1', 0):.4f}</span></div>
            <div class="model-stat"><span>IF F1 (fraud)</span><span>{metrics.get('isolation_forest_f1', 0):.4f}</span></div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Retrain models to see performance metrics.")

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(
        """
        1. Upload transaction CSV
        2. **Isolation Forest** — outlier detection (normal-class trained)
        3. **Random Forest** — fraud probability scoring
        4. Review, filter & export
        """
    )
    st.divider()
    st.markdown("**Required columns**")
    st.code("V1 – V28, Amount", language=None)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge"><span>●</span> AI Fraud Detection</div>
        <h1>Credit Card Anomaly &amp; Fraud Detection</h1>
        <p>
            Screen thousands of transactions in seconds. Our improved two-stage pipeline
            uses Isolation Forest (trained on normal behavior) plus a tuned Random Forest
            classifier for high-accuracy fraud detection.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(
        '<div class="pipe-card"><div class="pipe-icon">🌲</div>'
        "<h4>Isolation Forest</h4>"
        "<p>Learns normal patterns, flags statistical outliers as anomalies</p></div>",
        unsafe_allow_html=True,
    )
with p2:
    st.markdown(
        '<div class="pipe-card"><div class="pipe-icon">🎯</div>'
        "<h4>Random Forest</h4>"
        "<p>Scores every transaction with calibrated fraud probability</p></div>",
        unsafe_allow_html=True,
    )
with p3:
    st.markdown(
        '<div class="pipe-card"><div class="pipe-icon">📊</div>'
        "<h4>Smart Dashboard</h4>"
        "<p>Interactive charts, risk tiers, and exportable reports</p></div>",
        unsafe_allow_html=True,
    )

users = _load_users()
current_user = users.get(st.session_state.user_email, {})

plan = current_user.get("plan", "free")

if plan == "premium":
    upload_label = "Drop your CSV here (Max 200 MB)"
else:
    upload_label = "Drop your CSV here (Max 100 MB)"


st.markdown('<div class="section-title">Upload Data</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    upload_label,
    type=["csv"],
    help="Must include V1–V28 and Amount. Class/id columns are ignored.",
)

# users = _load_users()
# current_user = users.get(st.session_state.user_email, {})

free_limit = current_user.get("max_free_predictions", 3)
used = current_user.get("prediction_count", 0)

if (
    current_user.get("plan") != "premium"
    and used >= free_limit
):
    st.error("Your free trial has ended.")

    st.link_button(
        "Upgrade to Premium",
        payment_link = client.payment_link.create({
                'amount': 100,
                'currency':'INR',
                'accept_partial':False,
                'description':'Premium Subscription',
                'customer':{
                    'name':'Customer',
                    
                },
                'notify':{
                    'sms':False,
                    'email':False
                }
            }) 
    )

    st.stop()

if uploaded is not None:

    file_size = uploaded.size/(1024 * 1024)
    
    if plan == "premium":
        max_size = 200
    else:
        max_size = 100
        
    if file_size > max_size:
        st.error(
            f"Your {plan} plan allows files up to {max_size} MB. "
            f"Uploaded file is {file_size:.2f} MB."
        )
        st.stop()

    users = _load_users()
    current_user = users.get(
        st.session_state.user_email,
        {}
    )

    if current_user.get("plan") != "premium":

        current_user["prediction_count"] = (
            current_user.get(
                "prediction_count",
                0
            ) + 1
        )

        users[
            st.session_state.user_email
        ] = current_user

        _save_users(users)
    with st.spinner("Analyzing transactions…"):
        try:
            raw = pd.read_csv(uploaded)
            data = prepare_data(raw, expected_features)
            result = run_detection(data, model, model_rf, scaler, fraud_threshold)
        except ValueError as err:
            st.error(str(err))
            st.stop()

    total = len(result)
    anomaly_count = int(result["Anomaly"].sum())
    fraud_count = int(result["fraud"].sum())
    anomaly_pct = anomaly_count / total * 100
    fraud_pct = fraud_count / total * 100
    risk, risk_color = risk_label(anomaly_pct, fraud_pct)

    st.markdown('<div class="section-title">Live Results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Transactions", f"{total:,}")
    c2.metric("Anomalies", f"{anomaly_count:,}", delta=f"{anomaly_pct:.2f}%", delta_color="inverse")
    c3.metric("Confirmed Fraud", f"{fraud_count:,}", delta=f"{fraud_pct:.3f}%", delta_color="inverse")
    c4.metric("Clean", f"{total - anomaly_count:,}")
    c5.metric("Avg Fraud Score", f"{result['fraud_probability'].mean():.1%}")
    c6.metric("Risk Level", risk)

    left, right = st.columns([5, 3])

    with left:
        fig = build_charts(result, total, anomaly_count, fraud_count)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("**Risk Assessment**")
        if fraud_count > 0:
            st.markdown(
                f'<div class="alert-card alert-danger">🚨 <b>{fraud_count}</b> confirmed fraud '
                f"transaction(s) detected — immediate review required.</div>",
                unsafe_allow_html=True,
            )
        elif anomaly_count > 0:
            st.markdown(
                f'<div class="alert-card alert-warning">⚠️ <b>{anomaly_count}</b> suspicious '
                f"transaction(s) — none confirmed as fraud at current threshold.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="alert-card alert-success">✅ No anomalies detected. '
                "All transactions appear within normal patterns.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Risk Rates**")
        st.progress(min(anomaly_pct / 10, 1.0), text=f"Anomaly rate: {anomaly_pct:.2f}%")
        st.progress(min(fraud_pct * 20, 1.0), text=f"Fraud rate: {fraud_pct:.4f}%")

        if fraud_count and anomaly_count:
            precision = fraud_count / anomaly_count * 100
            st.caption(f"Fraud precision among flagged anomalies: **{precision:.1f}%**")

        st.markdown("**Top 5 Highest Risk**")
        top5 = result.nlargest(5, "fraud_probability")[["fraud_probability", "Anomaly", "risk_tier"]]
        if "Amount" in result.columns:
            top5 = result.nlargest(5, "fraud_probability")[
                ["Amount", "fraud_probability", "Anomaly", "risk_tier"]
            ]
        st.dataframe(
            top5.reset_index(drop=True),
            use_container_width=True,
            column_config={
                "fraud_probability": st.column_config.ProgressColumn(
                    "Fraud Score", format="%.1f%%", min_value=0, max_value=1
                ),
                "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            },
        )

    tab_all, tab_anomaly, tab_fraud, tab_preview = st.tabs(
        ["🔍 All Results", "⚠️ Anomalies", "🚨 Confirmed Fraud", "📋 Preview"]
    )

    sorted_result = result.sort_values(["fraud", "fraud_probability", "Anomaly"], ascending=False)

    with tab_all:
        show_results_table(sorted_result)

    with tab_anomaly:
        anom = sorted_result[sorted_result["Anomaly"] == 1]
        if len(anom):
            show_results_table(anom)
        else:
            st.info("No anomalies detected.")

    with tab_fraud:
        fraud_df = sorted_result[sorted_result["fraud"] == 1]
        if len(fraud_df):
            show_results_table(fraud_df)
        else:
            st.info("No confirmed fraud at the current threshold. Try lowering the slider in the sidebar.")

    with tab_preview:
        st.dataframe(raw.head(15), use_container_width=True, height=320)

    st.markdown('<div class="section-title">Export Report</div>', unsafe_allow_html=True)
    export_cols = [c for c in result.columns if c != "anomaly_score"]
    csv_bytes = result[export_cols].to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Full Analysis (CSV)",
        data=csv_bytes,
        file_name="fraud_detection_report.csv",
        mime="text/csv",
        type="primary",
    )

else:
    st.info("Upload a CSV file to start analysis.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **Getting started**
            - CSV with columns `V1`–`V28` and `Amount`
            - Optional `Class` column (ignored during prediction)
            - Upload and review the dashboard
            """
        )
    with col_b:
        st.markdown(
            """
            **Tips**
            - Adjust the fraud threshold in the sidebar
            - Lower threshold → more fraud flagged (higher recall)
            - Higher threshold → fewer false alarms (higher precision)
            """
        )