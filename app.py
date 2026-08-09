"""
AI Fraud Detection — Fraud Analyst Dashboard
=============================================
Streamlit Cloud deployment for the AI Fraud Detection System.

IMPORTANT: This file does NOT modify any existing project file.
It imports from rag_module.py and loads saved artefacts from outputs/.
"""

import os, sys, json, time, io, warnings, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from PIL import Image
warnings.filterwarnings("ignore")
import traceback

#  Resolve project root so relative imports work on Streamlit Cloud
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

#  Import and patch rag_module AFTER ROOT is defined
os.environ["ANONYMIZED_TELEMETRY"] = "False"   # suppress ChromaDB telemetry noise
os.environ["CHROMA_TELEMETRY"] = "False"
import rag_module as _rm
_rm.RAG_CONFIG["CHROMA_DB_PATH"] = os.path.join(ROOT, "chroma_db")

def _patch_rag_module():
    """Returns the already-patched rag_module instance."""
    return _rm

import theme
import landing

# Initialize page selection state
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Landing Page"

# PAGE CONFIG
st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject theme
theme.inject_theme()


# CONSTANTS / PATHS
DEPLOY_DIR  = os.path.join(ROOT, "outputs", "deployment")
MODELS_DIR  = os.path.join(ROOT, "outputs", "models")
METRICS_DIR = os.path.join(ROOT, "outputs", "metrics")
PLOTS_DIR   = os.path.join(ROOT, "outputs", "plots")
ABLATION_DIR= os.path.join(ROOT, "outputs", "ablation")

FEATURES = ["step","amount","oldbalanceOrg","newbalanceOrig",
            "oldbalanceDest","newbalanceDest","balanceDiff","amount_ratio",
            "type_CASH_OUT","type_DEBIT","type_PAYMENT","type_TRANSFER"]

FEATURE_LABELS = {
    "step": "Time Step", "amount": "Amount (PKR)",
    "oldbalanceOrg": "Sender Old Balance", "newbalanceOrig": "Sender New Balance",
    "oldbalanceDest": "Recipient Old Balance", "newbalanceDest": "Recipient New Balance",
    "balanceDiff": "Balance Difference", "amount_ratio": "Amount Ratio",
    "type_CASH_OUT": "Type: CASH_OUT", "type_DEBIT": "Type: DEBIT",
    "type_PAYMENT": "Type: PAYMENT", "type_TRANSFER": "Type: TRANSFER",
}

RISK_COLORS = {
    "CRITICAL": "#DC2626", "HIGH": "#DC2626",
    "MEDIUM": "#D97706", "LOW": "#2563EB"
}

# HELPERS — LOAD ARTEFACTS

@st.cache_resource(show_spinner="Loading embedding models...")
def load_embedding_models():
    # pyrefly: ignore [missing-import]
    from sentence_transformers import SentenceTransformer, CrossEncoder
    em = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except:
        reranker = None
    return em, reranker

em, reranker = load_embedding_models()
_rm.set_shared_models(em, reranker)
# Call once at top level so embed_model and reranker are available globally
embed_model, reranker = load_embedding_models()

@st.cache_resource(show_spinner="Loading model artefacts…")
def load_deployment_model():
    import joblib
    model  = joblib.load(os.path.join(DEPLOY_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(DEPLOY_DIR, "scaler.pkl"))
    with open(os.path.join(DEPLOY_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return model, scaler, meta

@st.cache_resource(show_spinner="Loading selected model…")
def load_chosen_model(model_name):
    import joblib
    # Map selection names to file names
    mapping = {
        "XGBoost (Deployed)": ("xgboost.pkl", "model.pkl"),
        "Random Forest": ("random_forest.pkl", "random_forest.pkl"),
        "Neural Network": ("neural_network.pkl", "neural_network.pkl"),
        "Logistic Regression": ("logistic_regression.pkl", "logistic_regression.pkl")
    }
    fname, fallback = mapping.get(model_name, ("xgboost.pkl", "model.pkl"))
    
    # Try models folder first, then fallback to deployment folder
    m_path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(m_path):
        m_path = os.path.join(DEPLOY_DIR, fallback)
        
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(DEPLOY_DIR, "scaler.pkl")
        
    model = joblib.load(m_path)
    scaler = joblib.load(scaler_path)
    
    # Load meta
    meta_path = os.path.join(DEPLOY_DIR, "model_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return model, scaler, meta

import base64
def show_img(path, width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<img src="data:image/png;base64,{b64}" style="width:{width}; border-radius:8px;">', unsafe_allow_html=True)
    else:
        st.warning(f" Missing: `{path}`")

@st.cache_resource(show_spinner="Loading all model artefacts…")
def load_all_models():
    import joblib
    models = {}
    for name in ["xgboost", "random_forest", "neural_network", "logistic_regression"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return models, scaler

@st.cache_data(show_spinner=False)
def load_model_comparison():
    path = os.path.join(METRICS_DIR, "model_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    return None

@st.cache_data(show_spinner=False)
def load_ablation():
    path = os.path.join(ABLATION_DIR, "ablation_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

def get_cohere_api_key():
    """Try st.secrets first, then session_state (sidebar input)."""
    try:
        key = st.secrets.get("COHERE_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return st.session_state.get("cohere_api_key", "")

def risk_badge(tier: str) -> str:
    return f'<span class="badge badge-{tier.lower()}">{tier}</span>'

def risk_color(tier: str) -> str:
    return RISK_COLORS.get(tier, "#263D5B")

def get_risk_tier(prob: float) -> str:
    if prob >= 0.85: return "CRITICAL"
    if prob >= 0.65: return "HIGH"
    if prob >= 0.50: return "MEDIUM"
    return "LOW"

def render_circular_gauge(prob, risk_tier, color, target_threshold):
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            number = {'suffix': "%", 'font': {'size': 38, 'family': 'JetBrains Mono', 'color': color}},
            title = {'text': f"Risk: {risk_tier}", 'font': {'size': 18, 'family': 'Inter', 'color': color}, 'align': 'center'},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.08)",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(37,99,235,0.05)'},
                    {'range': [50, 65], 'color': 'rgba(217,119,6,0.06)'},
                    {'range': [65, 85], 'color': 'rgba(220,38,38,0.06)'},
                    {'range': [85, 100], 'color': 'rgba(220,38,38,0.12)'}
                ],
                'threshold': {
                    'line': {'color': "rgba(255,255,255,0.4)", 'width': 2},
                    'thickness': 0.75,
                    'value': target_threshold * 100
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            height=140,
        )
        return fig
    except Exception:
        return None



# XAI — Plotly Interactive SHAP + fallback
def build_interactive_xai(model, X_scaled: np.ndarray, feature_names: list, threshold: float = 0.5):
    """
    Returns (plotly_fig, method_used).
    """
    try:
        import plotly.graph_objects as go
        import shap
        
        # Explain using independent masker
        masker = shap.maskers.Independent(X_scaled, max_samples=50)
        explainer = shap.Explainer(model, masker)
        model_type = type(model).__name__
        if "XGB" in model_type or "GBM" in model_type:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, feature_perturbation="interventional")
            
        shap_values = explainer(X_scaled)
        sv = shap_values[0]
        vals = sv.values if hasattr(sv, "values") else sv
        base = float(sv.base_values if hasattr(sv, "base_values") else explainer.expected_value)
        
        order = np.argsort(np.abs(vals))[::-1]
        top_n = min(12, len(order))
        idx = order[:top_n][::-1]
        fnames = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in idx]
        fvals = vals[idx]
        
        colors = ["#DC2626" if v > 0 else "#2563EB" for v in fvals]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=fnames,
            x=fvals,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.3f}" for v in fvals],
            textposition='auto',
            hoverinfo="x+y"
        ))
        fig.update_layout(
            title=dict(text=f"SHAP Explanation (Base Value: {base:.3f})", font=dict(color="#F0F0F2", size=13, family="Inter")),
            xaxis_title="SHAP Value (impact on fraud probability)",
            xaxis=dict(color="#8B8B9A", gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
            yaxis=dict(color="#F0F0F2", tickfont=dict(size=11, family="Inter")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10),
            height=350,
            showlegend=False,
            font=dict(family="Inter", color="#F0F0F2")
        )
        return fig, "SHAP Waterfall Explainer (Plotly)"
    except Exception as e:
        # Fallback: simple Plotly Feature Importance
        try:
            import plotly.graph_objects as go
            try:
                importances = model.feature_importances_
            except AttributeError:
                importances = np.ones(len(feature_names)) / len(feature_names)
            order = np.argsort(importances)
            fnames = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
            fimps = importances[order]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=fnames,
                x=fimps,
                orientation='h',
                marker_color="#2563EB",
                text=[f"{v:.3f}" for v in fimps],
                textposition='auto'
            ))
            fig.update_layout(
                title=dict(text="Feature Importance (Fallback)", font=dict(color="#F0F0F2", size=13, family="Inter")),
                xaxis_title="Gini Gain",
                xaxis=dict(color="#8B8B9A", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(color="#F0F0F2", tickfont=dict(size=11, family="Inter")),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10),
                height=350,
                showlegend=False,
                font=dict(family="Inter", color="#F0F0F2")
            )
            return fig, "Feature Importance Fallback (Plotly)"
        except Exception:
            return None, "No XAI figure available"


# INFERENCE
def predict_transaction(model, scaler, row: dict, threshold: float = 0.5):
    """
    row: dict with keys = FEATURES
    Returns (fraud_prob, is_fraud, scaled_array)
    """
    X = np.array([[row.get(f, 0.0) for f in FEATURES]], dtype=float)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    return prob, prob >= threshold, X_scaled

# SIDEBAR
with st.sidebar:
    st.markdown("## AI Fraud Detection")
    st.markdown("**Fraud Detection Dashboard**")
    st.markdown("---")

    page = st.radio("Navigate", [
        "Landing Page",
        " Predict Transaction",
        " Batch CSV Analysis",
        " Dataset & Imbalance",  
        " Model Performance",
        " Ablation Study",
        " About"
    ], key="nav_selection")

    st.markdown("---")
    
    # ── Interactive Sidebar Guide ──
    with st.expander("Tester & Checker Guide", expanded=True):
        st.markdown("""
        Follow these steps to evaluate the system:
        
        1. **Select a Model**: Choose from the dropdown below (XGBoost is the most accurate).
        2. **Quick Test**: Click one of the **Quick-Fill Scenarios** at the top of the page.
        3. **Play with Numbers**: Adjust the amount or balances. Toggle *Linked Controls* to auto-calculate math.
        4. **ML Prediction**: The fraud probability, risk badge, and SHAP explanation update instantly.
        5. **Compliance Report**: Click **Generate SBP Report** to query regulations and view STR/CTR filing statuses.
        """)

    st.markdown("---")
    
    # ── Model Selection Dropdown ──
    st.markdown("#### Classifier Model")
    model_choice = st.selectbox(
        "Select Active Model",
        ["XGBoost (Deployed)", "Random Forest", "Neural Network", "Logistic Regression"],
        index=0,
        help="Switch classification engines dynamically to see how different algorithms classify the same transaction."
    )
    
    st.markdown("---")
    st.markdown("#### Cohere API Key")
    st.caption("For RAG regulatory justification. Tries `st.secrets` first.")
    api_input = st.text_input("Paste key (optional)", type="password",
                               placeholder="cxx...", key="cohere_api_key")
    if api_input:
        st.success("Key stored for this session.")
    elif get_cohere_api_key():
        st.success("Key loaded from Secrets")
    else:
        st.warning("No key — RAG will be skipped.")

    st.markdown("---")
    st.markdown("#### Decision Threshold")
    threshold = st.slider("Fraud threshold", 0.1, 0.9, 0.5, 0.01,
                          help="Probability cutoff to classify as fraud")

    st.markdown("---")
    st.caption("AI Fraud Detection v1.0")

# TOP BAR
if page != "Landing Page":
    st.markdown("""
    <div class="topbar">
        <div>
            <h1> AI Fraud Detection Dashboard</h1>
                    <span>Fraud detection with SHAP explanations and SBP regulatory justification</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# PAGE ROUTING
if page == "Landing Page":
    landing.render_landing_page("nav_selection")

elif page == " Predict Transaction":
    # Live Model Swapping
    model, scaler, meta = load_chosen_model(model_choice)

    st.markdown(theme.section_header("Predict a Single Transaction", "· INFERENCE ENGINE ·", "red"), unsafe_allow_html=True)

    #  How It Works banner 
    with st.expander("How does this work? (Click to read — recommended for first-time users)", expanded=False):
        st.markdown("""
        <div style="font-size:0.95rem; line-height:1.7">

        <b>Welcome to the AI Fraud Detection Dashboard.</b> Here's what happens when you submit a transaction:

        <ol>
        <li> <b>You enter a transaction</b> — the type (e.g. CASH_OUT), amount in PKR, and the sender/recipient balances before and after.</li>
        <li> <b>Our AI model analyses it</b> — the selected model (e.g., XGBoost) instantly scores the transaction for fraud probability.</li>
        <li> <b>XAI explains the verdict</b> — an interactive SHAP bar chart shows <i>which features</i> pushed the score up or down.</li>
        <li> <b>SBP regulations are retrieved</b> — the RAG engine searches our SBP regulatory knowledge base and generates a compliance report with STR/CTR obligations.</li>
        </ol>

        <b>Key terms:</b><br>
        &bull; <b>CASH_OUT</b>: Withdrawing cash from a mobile account.<br>
        &bull; <b>TRANSFER</b>: Sending money to another account.<br>
        &bull; <b>Fraud Probability</b>: 0% = definitely legitimate, 100% = definitely fraud.<br>
        &bull; <b>STR</b>: Suspicious Transaction Report — mandatory filing with SBP.<br>
        &bull; <b>CTR</b>: Currency Transaction Report — required for large cash transactions ≥ PKR 2,500,000.<br>
        &bull; <b>SHAP</b>: A method that explains AI decisions by showing each feature's contribution.

        </div>
        """, unsafe_allow_html=True)

    #  Quick-fill Scenario Presets 
    st.markdown("#### Quick-Fill Scenarios")
    st.caption("New here? Click a scenario below to auto-fill the form with a realistic example.")

    SCENARIOS = {
        "High-Risk Fraud": {
            "tx_type": "CASH_OUT", "step": 312, "amount": 920000.0,
            "old_orig": 920000.0, "new_orig": 0.0,
            "old_dest": 0.0,      "new_dest": 0.0,
            "tx_id": "TXN-FRAUD-01",
            "desc": "Sender empties entire balance in one CASH_OUT — a classic fraud pattern."
        },
        "Legitimate Payment": {
            "tx_type": "PAYMENT", "step": 50, "amount": 3500.0,
            "old_orig": 80000.0, "new_orig": 76500.0,
            "old_dest": 15000.0, "new_dest": 18500.0,
            "tx_id": "TXN-LEGIT-01",
            "desc": "Small routine payment — balances update normally, low fraud risk."
        },
        "Suspicious Transfer": {
            "tx_type": "TRANSFER", "step": 180, "amount": 450000.0,
            "old_orig": 450000.0, "new_orig": 0.0,
            "old_dest": 0.0,      "new_dest": 0.0,
            "tx_id": "TXN-SUSP-01",
            "desc": "Full account drain via TRANSFER to a zero-balance recipient — medium-high risk."
        },
        "High-Value Legitimate": {
            "tx_type": "TRANSFER", "step": 100, "amount": 1500000.0,
            "old_orig": 5000000.0, "new_orig": 3500000.0,
            "old_dest": 2000000.0, "new_dest": 3500000.0,
            "tx_id": "TXN-HV-01",
            "desc": "Large transfer between well-funded accounts — triggers EDD but likely legitimate."
        },
    }

    preset_cols = st.columns(len(SCENARIOS))
    chosen_preset = None
    for col, (label, data) in zip(preset_cols, SCENARIOS.items()):
        with col:
            if st.button(label, use_container_width=True, help=data["desc"]):
                chosen_preset = data

    if chosen_preset:
        st.session_state["_preset"] = chosen_preset
    preset = st.session_state.get("_preset", None)

    # Sync Session State
    if preset:
        st.session_state["tx_type_val"] = preset["tx_type"]
        st.session_state["step_val"] = preset["step"]
        st.session_state["tx_amount_val"] = preset["amount"]
        st.session_state["old_orig_val"] = preset["old_orig"]
        st.session_state["new_orig_val"] = preset["new_orig"]
        st.session_state["old_dest_val"] = preset["old_dest"]
        st.session_state["new_dest_val"] = preset["new_dest"]
        st.session_state["tx_id_val"] = preset["tx_id"]
        # Clear preset
        st.session_state["_preset"] = None

    if "tx_type_val" not in st.session_state: st.session_state["tx_type_val"] = "CASH_OUT"
    if "step_val" not in st.session_state: st.session_state["step_val"] = 200
    if "tx_amount_val" not in st.session_state: st.session_state["tx_amount_val"] = 180000.0
    if "old_orig_val" not in st.session_state: st.session_state["old_orig_val"] = 200000.0
    if "new_orig_val" not in st.session_state: st.session_state["new_orig_val"] = 20000.0
    if "old_dest_val" not in st.session_state: st.session_state["old_dest_val"] = 0.0
    if "new_dest_val" not in st.session_state: st.session_state["new_dest_val"] = 0.0
    if "tx_id_val" not in st.session_state: st.session_state["tx_id_val"] = "TXN-001"

    st.markdown("---")
    st.markdown("#### Enter Transaction Details")
    st.caption("Fill in the fields below, or use a quick-fill scenario above. All amounts are in Pakistani Rupees (PKR).")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)

        tx_type = st.selectbox(
            "Transaction Type",
            ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"],
            index=["CASH_OUT","TRANSFER","PAYMENT","DEBIT","CASH_IN"].index(st.session_state["tx_type_val"]),
            key="tx_type_selector",
            help="CASH_OUT = cash withdrawal | TRANSFER = send to another account | PAYMENT = bill/merchant payment | DEBIT = direct debit | CASH_IN = deposit"
        )
        st.session_state["tx_type_val"] = tx_type

        step = st.number_input(
            "Time Step (hour)",
            min_value=1, max_value=744,
            value=int(st.session_state["step_val"]),
            key="step_input",
            help="Hour of the simulation (1 = first hour, 744 = last hour of a 31-day month). Most frauds cluster at late hours."
        )
        st.session_state["step_val"] = step

        amount = st.number_input(
            "Transaction Amount (PKR)",
            min_value=0.0,
            value=float(st.session_state["tx_amount_val"]),
            step=1000.0, format="%.2f",
            key="amount_input",
            help="The total PKR value being moved. Transactions >= PKR 2,500,000 trigger a Currency Transaction Report (CTR)."
        )
        st.session_state["tx_amount_val"] = amount

        # Play with Numbers Linked Controls option
        link_balances = st.checkbox("Link Balance to Transaction Amount (Auto-update Remaining Balance)", value=True)

        st.markdown("**Sender Balances** — the account sending money")
        c1, c2 = st.columns(2)
        with c1:
            old_orig = st.number_input(
                "Before Transfer (PKR)",
                min_value=0.0,
                value=float(st.session_state["old_orig_val"]),
                step=1000.0, format="%.2f",
                key="old_orig_input",
                help="Sender's account balance BEFORE this transaction."
            )
            st.session_state["old_orig_val"] = old_orig
        with c2:
            if link_balances:
                computed_new_orig = max(0.0, old_orig - amount)
                st.session_state["new_orig_val"] = computed_new_orig
                new_orig = st.number_input(
                    "After Transfer (PKR)",
                    min_value=0.0,
                    value=float(computed_new_orig),
                    step=1000.0, format="%.2f",
                    key="new_orig_input_disabled",
                    disabled=True,
                    help="Auto-calculated sender's balance AFTER transaction."
                )
            else:
                new_orig = st.number_input(
                    "After Transfer (PKR)",
                    min_value=0.0,
                    value=float(st.session_state["new_orig_val"]),
                    step=1000.0, format="%.2f",
                    key="new_orig_input",
                    help="Sender's balance AFTER transaction."
                )
                st.session_state["new_orig_val"] = new_orig

        # Math Helper Buttons UI
        st.caption("Math helpers for Sender balance:")
        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            if st.button("Empty Account", key="empty_acc_btn", use_container_width=True):
                st.session_state["tx_amount_val"] = old_orig
                st.session_state["new_orig_val"] = 0.0
                st.rerun()
        with hcol2:
            if st.button("Transfer Half", key="transfer_half_btn", use_container_width=True):
                st.session_state["tx_amount_val"] = old_orig / 2
                st.session_state["new_orig_val"] = old_orig / 2
                st.rerun()
        with hcol3:
            if st.button("Clear Recipient", key="clear_recip_btn", use_container_width=True):
                st.session_state["old_dest_val"] = 0.0
                st.session_state["new_dest_val"] = 0.0
                st.rerun()

        st.markdown("**Recipient Balances** — the account receiving money")
        c3, c4 = st.columns(2)
        with c3:
            old_dest = st.number_input(
                "Before Receipt (PKR)",
                min_value=0.0,
                value=float(st.session_state["old_dest_val"]),
                step=1000.0, format="%.2f",
                key="old_dest_input",
                help="Recipient's balance BEFORE. A balance of 0 before and after receipt is suspicious (mule account)."
            )
            st.session_state["old_dest_val"] = old_dest
        with c4:
            if link_balances:
                computed_new_dest = old_dest + amount
                st.session_state["new_dest_val"] = computed_new_dest
                new_dest = st.number_input(
                    "After Receipt (PKR)",
                    min_value=0.0,
                    value=float(computed_new_dest),
                    step=1000.0, format="%.2f",
                    key="new_dest_input_disabled",
                    disabled=True,
                    help="Auto-calculated recipient's balance AFTER transaction."
                )
            else:
                new_dest = st.number_input(
                    "After Receipt (PKR)",
                    min_value=0.0,
                    value=float(st.session_state["new_dest_val"]),
                    step=1000.0, format="%.2f",
                    key="new_dest_input",
                    help="Recipient's balance AFTER transaction."
                )
                st.session_state["new_dest_val"] = new_dest

        tx_id = st.text_input(
            "Transaction ID (optional)",
            value=st.session_state["tx_id_val"],
            placeholder="e.g. TXN-001",
            key="tx_id_input",
            help="A reference ID for the report. Appears in the exported compliance document."
        )
        st.session_state["tx_id_val"] = tx_id

    # ── Build Feature Dict ──
    balance_diff  = old_orig - new_orig
    amount_ratio  = (amount / old_orig) if old_orig > 0 else 0.0
    row = {
        "step": step, "amount": amount,
        "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "balanceDiff": balance_diff, "amount_ratio": amount_ratio,
        "type_CASH_OUT": 1 if tx_type == "CASH_OUT" else 0,
        "type_DEBIT":    1 if tx_type == "DEBIT"    else 0,
        "type_PAYMENT":  1 if tx_type == "PAYMENT"  else 0,
        "type_TRANSFER": 1 if tx_type == "TRANSFER" else 0,
    }

    with col_right:
        st.markdown('<div class="section-title">Feature Preview & Risk Signals</div>', unsafe_allow_html=True)
        st.caption("These are the values the AI model will use. Red flags indicate known fraud patterns.")

        # Compute risk signals
        signals = []
        if amount_ratio >= 0.9:
            signals.append(("DRAIN", "Full account drain", f"Amount is {amount_ratio:.0%} of sender balance"))
        if old_dest == 0 and new_dest == 0 and amount > 0:
            signals.append(("MULE", "Mule account pattern", "Recipient balance unchanged despite receiving funds"))
        if amount >= 2_500_000:
            signals.append(("CTR", "CTR threshold exceeded", f"PKR {amount:,.0f} >= PKR 2,500,000 — CTR required"))
        if amount >= 1_000_000:
            signals.append(("EDD", "EDD trigger", f"PKR {amount:,.0f} >= PKR 1,000,000 — Enhanced Due Diligence needed"))
        if tx_type in ("CASH_OUT", "TRANSFER") and new_orig == 0:
            signals.append(("ZERO", "Zero remaining balance", "Sender drained entire account"))

        if signals:
            for flag, title, detail in signals:
                # choose color keyword for helper
                color_kw = "red" if flag in ("DRAIN","ZERO","MULE") else "amber"
                st.markdown(theme.signal_alert(flag, title, detail, color_kw), unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='background:var(--bg-elevated);border-left:4px solid #16A34A;"
                "padding:8px 12px;border-radius:6px;font-size:0.85rem;color:var(--text-primary)'>"
                "<b>No obvious fraud signals</b> detected from input values.</div>",
                unsafe_allow_html=True
            )

        st.markdown("")
        preview_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(k, k) for k in FEATURES],
            "Value":   [f"{row[k]:,.4f}" for k in FEATURES],
        })
        st.dataframe(preview_df, use_container_width=True, hide_index=True, height=240)

    st.markdown("---")

    # ── INSTANT ML INFERENCE (no button click needed for ML/XAI!)
    prob, is_fraud, X_scaled = predict_transaction(model, scaler, row, threshold)
    risk_tier = get_risk_tier(prob)

    # Display Results Row
    r1, r2, r3, r4 = st.columns(4)
    card_class = "tg-card-danger" if is_fraud else "tg-card-success"
    
    with r1:
        # Plotly Donut Gauge
        fig_gauge = render_circular_gauge(prob, risk_tier, risk_color(risk_tier), threshold)
        if fig_gauge is not None:
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="tg-card {card_class}">
                <div class="gauge-wrap">
                    <div class="gauge-pct" style="color:{risk_color(risk_tier)}">
                        {prob:.1%}
                    </div>
                    <div class="gauge-label">Fraud Probability</div>
                </div>
            </div>""", unsafe_allow_html=True)
            
    with r2:
        verdict = "FRAUD" if is_fraud else "LEGITIMATE"
        v_color = "#DC2626" if is_fraud else "#16A34A"
        st.markdown(f"""
        <div class="tg-card {card_class}" style="height:140px">
            <div class="gauge-wrap" style="padding-top:1.2rem">
                <div class="gauge-pct" style="color:{v_color};font-size:1.8rem">
                    {verdict}
                </div>
                <div class="gauge-label">Verdict (Threshold {threshold:.0%})</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
    with r3:
        st.markdown(f"""
        <div class="tg-card" style="height:140px">
            <div class="gauge-wrap" style="padding-top:1.2rem">
                <div class="gauge-pct" style="color:{risk_color(risk_tier)};font-size:1.8rem">
                    {risk_badge(risk_tier)}
                </div>
                <div class="gauge-label">Risk Level</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
    with r4:
        confidence = abs(prob - 0.5) * 2
        st.markdown(f"""
        <div class="tg-card" style="height:140px">
            <div class="gauge-wrap" style="padding-top:1.2rem">
                <div class="gauge-pct" style="color:var(--text-secondary);font-size:1.8rem">
                    {confidence:.1%}
                </div>
                <div class="gauge-label">Model Confidence</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION A — SBP REGULATORY REPORT  (shown first)
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### SBP Regulatory Compliance Report")
    st.caption("The AI verdict triggers an automated search of the SBP regulatory knowledge base. Click below to generate a full compliance report for this transaction.")

    cohere_key = get_cohere_api_key()
    if not cohere_key:
        st.warning("No Cohere API key — RAG skipped. Enter key in the sidebar.")
    else:
        generate_rag = st.button("Generate SBP Regulatory Report", type="primary", use_container_width=True)

        if generate_rag:
            with st.spinner("Retrieving SBP regulatory guidelines..."):
                try:
                    rag_mod = _patch_rag_module()
                    rag_res = rag_mod.rag_pipeline_for_streamlit(
                        fraud_probability=prob,
                        features=row,
                        transaction_id=tx_id or "TXN-STREAMLIT",
                        cohere_api_key=cohere_key,
                        embed_model=embed_model,
                        reranker=reranker,
                    )
                    st.session_state["rag_result"] = rag_res
                    st.session_state["rag_context_str"] = "\n\n".join(
                        [f"{c['citation']}: {c['text']}" for c in rag_res.retrieved_chunks]
                    )
                    st.session_state["rag_chat"] = []
                except Exception as e:
                    # Log full traceback to terminal for diagnostics
                    tb = traceback.format_exc()
                    print("[RAG ERROR TRACEBACK]", tb, flush=True)
                    # Classify error for concise user-facing message
                    msg = str(e).lower()
                    if isinstance(e, ValueError) and "query builder returned none" in msg:
                        st.error("RAG error: internal query builder returned no text. (internal bug)")
                    elif "cohere" in msg or "clientv2" in msg or "cohere" in tb.lower():
                        st.error("RAG error: Cohere API error — check API key, network, or model availability.")
                    elif "chroma" in msg or "collection" in msg or "chromadb" in msg:
                        st.error("RAG error: ChromaDB error — verify chroma_db path and that the DB is initialized.")
                    elif "json" in msg or "decode" in msg or "expecting value" in msg:
                        st.error("RAG error: response parsing failed — model output couldn't be parsed as JSON.")
                    else:
                        st.error(f"RAG error: {e}")
                    st.info("See terminal logs for full traceback.")

        rag_result = st.session_state.get("rag_result", None)
        if rag_result:
            s = rag_result.structured
            if s:
                # STR / CTR badges
                str_req = s.get("str_required", False)
                ctr_req = s.get("ctr_required", False)
                b1, b2 = st.columns(2)
                with b1:
                    if str_req:
                        st.error("STR Filing Required")
                    else:
                        st.success("STR Not Required")
                    st.caption(s.get("str_reason", ""))
                with b2:
                    if ctr_req:
                        st.warning("CTR Filing Required")
                    else:
                        st.success("CTR Not Required")
                    st.caption(s.get("ctr_reason", ""))

                st.markdown("---")

                # Regulations + Compliance side-by-side
                reg_col, act_col = st.columns([1, 1], gap="large")
                with reg_col:
                    regs = s.get("regulations_triggered", [])
                    if regs:
                        st.markdown("**Regulations Triggered**")
                        for r in regs:
                            st.markdown(
                                f"<div class='tg-card' style='padding:0.6rem 1rem;margin-bottom:0.5rem'>"
                                f"<b>{r.get('name','')}</b> &nbsp;"
                                f"<code style='font-size:0.75rem'>{r.get('citation','')}</code>"
                                f"<br><span style='font-size:0.85rem;color:var(--text-secondary)'>{r.get('description','')}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                with act_col:
                    actions = s.get("compliance_actions", [])
                    if actions:
                        st.markdown("**Compliance Actions**")
                        for i, act in enumerate(actions, 1):
                            st.markdown(f"{i}. {act}")

                    st.markdown("")
                    rj = s.get("risk_justification", "")
                    if rj:
                        st.markdown("**Risk Justification**")
                        st.info(rj)

                st.markdown("---")

                rs_text = s.get("regulatory_summary", "")
                if rs_text:
                    st.markdown("**Regulatory Summary**")
                    st.markdown(rs_text)

                # Recommended Next Steps (new section)
                rns = s.get("recommended_next_steps", "")
                if rns:
                    st.markdown("**Recommended Next Steps**")
                    st.info(rns)

                # Workflow submission
                st.markdown("---")
                st.markdown("##### Workflow Submission")
                if str_req:
                    if st.button("Transmit STR to State Bank of Pakistan FMU", type="primary"):
                        st.success(f"STR Transmitted! SBP Reference ID: FMU-{tx_id or 'TXN'}-99X")
                elif ctr_req:
                    if st.button("Transmit CTR to State Bank of Pakistan FMU", type="primary"):
                        st.success(f"CTR Transmitted! SBP Reference ID: FMU-{tx_id or 'TXN'}-99C")
                else:
                    st.info("No mandatory regulatory filing triggered. Standard archive saved.")

                # Conversational Q&A
                st.markdown("---")
                st.markdown("**Conversational Q&A on SBP Rules**")
                st.caption("Ask questions about this transaction or relevant SBP guidelines based on retrieved context:")
                chat_input = st.text_input("Ask SBP Compliance Assistant:", placeholder="e.g. Why is a CTR required for this transaction?")
                ask_chat = st.button("Submit Question")
                if "rag_chat" not in st.session_state:
                    st.session_state["rag_chat"] = []
                if ask_chat and chat_input:
                    with st.spinner("Compliance agent typing..."):
                        try:
                            import cohere
                            client = cohere.ClientV2(api_key=cohere_key)

                            # Perform a fresh high-quality retrieval for this question
                            try:
                                rag_mod = _patch_rag_module()
                                retrieval = rag_mod.rag_retrieve_for_question(
                                    chat_input,
                                    cohere_api_key=cohere_key,
                                    embed_model=embed_model,
                                    reranker=reranker,
                                )
                                context_str = retrieval.get("context_str", "")
                                st.session_state["rag_context_str"] = context_str
                                st.session_state["rag_retrieved_chunks"] = retrieval.get("chunks_display", [])
                            except Exception:
                                # Fall back to previously-stored context if retrieval fails
                                context_str = st.session_state.get("rag_context_str", "")

                            # Strong system prompt: require checking context first, cite documents, handle ambiguity
                            system_p = (
                                "You are a senior SBP regulatory compliance officer. "
                                "You have already been given retrieved SBP regulatory context below. "
                                "You MUST check the provided context first and base your answer ONLY on that context when relevant. "
                                "If the context contains the answer, cite specific regulations/pages inline using the format [Document, Section, Page X]. "
                                "Include at least one inline citation for factual claims (preferably 1-3 citations). "
                                "Only suggest consulting external sources if you have confirmed the provided context does not contain the answer. "
                                "If the user's question is ambiguous, unclear, or appears to contain typos, first paraphrase your interpretation (one sentence) before answering. "
                                "Do not hallucinate; when unsure, state that the context is insufficient and identify what additional information would help."
                            )

                            user_p = (
                                f"Retrieved context:\n{context_str}\n\n"
                                f"User Question: {chat_input}\n\n"
                                "Answer using only the provided context when possible, and include inline citations in [Document, Section, Page X] format. "
                                "If the context is insufficient, say so explicitly."
                            )

                            res = client.chat(
                                model="command-r-plus-08-2024",
                                messages=[
                                    {"role": "system", "content": system_p},
                                    {"role": "user", "content": user_p}
                                ]
                            )
                            ans = res.message.content[0].text.strip()
                            st.session_state["rag_chat"].append((chat_input, ans))
                        except Exception as ce:
                            st.error(f"Chat error: {ce}")
                if st.session_state["rag_chat"]:
                    for q, a in st.session_state["rag_chat"]:
                        st.markdown(f"**Q**: {q}")
                        st.markdown(f"**A**: {a}")
                        # Show sources consulted for the most recent answer (if retrieval ran)
                        chunks = st.session_state.get("rag_retrieved_chunks", [])
                        if chunks:
                            with st.expander(f"Sources consulted for this answer ({len(chunks)})"):
                                for c in chunks:
                                    st.markdown(f"**{c.get('citation','')}**")
                                    st.markdown(f"<div style='font-size:0.85rem; color:var(--text-secondary)'>{c.get('text','')}</div>", unsafe_allow_html=True)
                                    st.markdown("---")
                        st.markdown("---")

                # Grounding badge (tiered) + tooltip with exact percentage
                gs = float(rag_result.grounding_score)
                no_ev = bool(rag_result.no_evidence_flag)
                if no_ev:
                    badge_html = '<span style="background:#6B7280;color:#fff;padding:4px 8px;border-radius:6px;font-size:0.9rem">No Direct Match Found</span>'
                else:
                    if gs >= 0.6:
                        badge_html = '<span style="background:#059669;color:#fff;padding:4px 8px;border-radius:6px;font-size:0.9rem">Well Grounded</span>'
                    elif gs >= 0.3:
                        badge_html = '<span style="background:#D97706;color:#fff;padding:4px 8px;border-radius:6px;font-size:0.9rem">Partially Grounded</span>'
                    else:
                        badge_html = '<span style="background:#DC2626;color:#fff;padding:4px 8px;border-radius:6px;font-size:0.9rem">Limited Grounding — Verify Independently</span>'

                st.markdown(
                    f'⏱ {rag_result.latency_seconds:.1f}s  |  {badge_html} '
                    f'<span title="Exact grounding score: {gs:.0%}" style="color:var(--text-secondary);margin-left:8px;font-size:0.9rem">(details)</span>  |  {len(regs)} regulation(s) cited',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(rag_result.response_text)

            if rag_result.sources:
                with st.expander("Retrieved SBP Sources (Click to view texts)"):
                    for c in rag_result.retrieved_chunks:
                        raw_text = c["text"]
                        for term in ["AML", "CFT", "STR", "CTR", "CDD", "EDD", "limit"]:
                            raw_text = re.sub(
                                rf"\b({term})\b",
                                r'<b><mark style="background-color: yellow">\1</mark></b>',
                                raw_text,
                                flags=re.IGNORECASE
                            )
                        st.markdown(f"**{c['citation']}**")
                        st.markdown(f"<div style='font-size:0.85rem; color:var(--text-secondary)'>{raw_text}</div>", unsafe_allow_html=True)
                        st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # SECTION B — EXPLAINABILITY HEATMAP + EXPORT  (side by side)
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Technical Evidence & Export")
    xai_col, export_col = st.columns([1, 1], gap="large")

    with xai_col:
        st.markdown('<div class="section-title">Explainability Heatmap (SHAP)</div>', unsafe_allow_html=True)
        st.caption("Each bar shows how much that feature pushed the fraud score up (red) or down (blue).")
        xai_fig, xai_method = build_interactive_xai(model, X_scaled, FEATURES, threshold)
        if xai_fig is not None:
            st.plotly_chart(xai_fig, use_container_width=True)
            st.caption(f"Method: **{xai_method}**")

            # Download SHAP chart as PNG
            try:
                import plotly.io as pio
                img_bytes = pio.to_image(xai_fig, format="png", width=800, height=500)
                st.download_button(
                    "Download SHAP Chart (.png)",
                    data=img_bytes,
                    file_name=f"shap_chart_{tx_id or 'TXN'}_{risk_tier}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception:
                pass   # kaleido not installed — skip silently
        else:
            st.warning("Could not generate explainability plot.")

    with export_col:
        st.markdown('<div class="section-title">Export Report</div>', unsafe_allow_html=True)
        st.caption("Download the full analysis as a plain-text compliance document.")
        report_lines = [
            f"AI Fraud Detection Report",
            f"=" * 40,
            f"Transaction ID  : {tx_id}",
            f"Type            : {tx_type}",
            f"Amount (PKR)    : {amount:,.2f}",
            f"Fraud Probability: {prob:.4f} ({prob:.1%})",
            f"Risk Tier       : {risk_tier}",
            f"Verdict         : {'FRAUD' if is_fraud else 'LEGITIMATE'}",
            f"Threshold Used  : {threshold}",
            f"Model           : {model_choice} | Deployed XGBoost AUC 0.9995",
            f"",
            f"Features:",
            *[f"  {FEATURE_LABELS.get(k,k):35s}: {row[k]:>12.4f}" for k in FEATURES],
        ]
        rag_result = st.session_state.get("rag_result", None)
        if rag_result and rag_result.structured:
            s = rag_result.structured
            report_lines += [
                f"",
                f"SBP Compliance",
                f"-" * 40,
                f"STR Required    : {s.get('str_required', 'N/A')}",
                f"CTR Required    : {s.get('ctr_required', 'N/A')}",
                f"Summary         : {s.get('regulatory_summary', '')}",
                f"Recommended Next Steps: {s.get('recommended_next_steps', '')}",
            ]
        report_text = "\n".join(report_lines)
        st.code(report_text, language=None)
        st.download_button(
            "Download Report (.txt)",
            data=report_text,
            file_name=f"fraud_report_{tx_id}_{risk_tier}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# PAGE 2 — BATCH CSV ANALYSIS
elif page == " Batch CSV Analysis":
    model, scaler, meta = load_deployment_model()

    st.markdown(theme.section_header("Batch Transaction Analysis", "· BULK SCORING ·", "blue"), unsafe_allow_html=True)
    st.markdown("""
    Upload a CSV with transaction data. Required columns:
    `step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type`

    Optional: `isFraud` for ground-truth comparison.
    """)

    # CSV template download
    template_df = pd.DataFrame([{
        "step": 200, "amount": 180000, "oldbalanceOrg": 200000,
        "newbalanceOrig": 20000, "oldbalanceDest": 0, "newbalanceDest": 0,
        "type": "CASH_OUT", "isFraud": 1,
    },{
        "step": 50, "amount": 1500, "oldbalanceOrg": 50000,
        "newbalanceOrig": 48500, "oldbalanceDest": 10000, "newbalanceDest": 11500,
        "type": "PAYMENT", "isFraud": 0,
    }])
    
    st.download_button("Download CSV Template", template_df.to_csv(index=False),
                       "template.csv", "text/csv")
    
    st.markdown("---")
    
    # ── Preset Batch Demos Dropdown ──
    preset_batch = st.selectbox(
        "Select Preset Demo Dataset (Zero Uploads Required)",
        ["Select a preset...", "Normal Payment Traffic", "Active Fraud Campaign", "Mixed Portfolio"],
        help="Select a preset to instantly test the batch processing charts and confusion matrices without needing to locate a CSV."
    )

    uploaded = st.file_uploader("Or upload your own transactions CSV", type=["csv"])

    raw = None
    if uploaded:
        ROW_LIMIT = 500_000
        raw = pd.read_csv(uploaded, nrows=ROW_LIMIT)
        st.info(f"Loaded **{len(raw):,}** rows × {len(raw.columns)} columns from file.")
    elif preset_batch == "Normal Payment Traffic":
        raw = pd.DataFrame([
            {"step": 10, "amount": 1200.0, "oldbalanceOrg": 5000.0, "newbalanceOrig": 3800.0, "oldbalanceDest": 100.0, "newbalanceDest": 1300.0, "type": "PAYMENT", "isFraud": 0},
            {"step": 12, "amount": 3400.0, "oldbalanceOrg": 12000.0, "newbalanceOrig": 8600.0, "oldbalanceDest": 4000.0, "newbalanceDest": 7400.0, "type": "PAYMENT", "isFraud": 0},
            {"step": 15, "amount": 800.0, "oldbalanceOrg": 2500.0, "newbalanceOrig": 1700.0, "oldbalanceDest": 0.0, "newbalanceDest": 800.0, "type": "PAYMENT", "isFraud": 0},
            {"step": 20, "amount": 5000.0, "oldbalanceOrg": 50000.0, "newbalanceOrig": 45000.0, "oldbalanceDest": 12000.0, "newbalanceDest": 17000.0, "type": "TRANSFER", "isFraud": 0},
            {"step": 25, "amount": 1500.0, "oldbalanceOrg": 3000.0, "newbalanceOrig": 1500.0, "oldbalanceDest": 500.0, "newbalanceDest": 2000.0, "type": "PAYMENT", "isFraud": 0},
            {"step": 30, "amount": 2200.0, "oldbalanceOrg": 15000.0, "newbalanceOrig": 12800.0, "oldbalanceDest": 3000.0, "newbalanceDest": 5200.0, "type": "DEBIT", "isFraud": 0},
        ])
        st.info("Loaded **Normal Payment Traffic** preset dataset (6 rows).")
    elif preset_batch == "Active Fraud Campaign":
        raw = pd.DataFrame([
            {"step": 110, "amount": 450000.0, "oldbalanceOrg": 450000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "TRANSFER", "isFraud": 1},
            {"step": 112, "amount": 980000.0, "oldbalanceOrg": 980000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "CASH_OUT", "isFraud": 1},
            {"step": 115, "amount": 350000.0, "oldbalanceOrg": 350000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "TRANSFER", "isFraud": 1},
            {"step": 120, "amount": 120000.0, "oldbalanceOrg": 120000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "CASH_OUT", "isFraud": 1},
            {"step": 125, "amount": 670000.0, "oldbalanceOrg": 670000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "TRANSFER", "isFraud": 1},
        ])
        st.info("Loaded **Active Fraud Campaign** preset dataset (5 rows).")
    elif preset_batch == "Mixed Portfolio":
        raw = pd.DataFrame([
            {"step": 10, "amount": 1200.0, "oldbalanceOrg": 5000.0, "newbalanceOrig": 3800.0, "oldbalanceDest": 100.0, "newbalanceDest": 1300.0, "type": "PAYMENT", "isFraud": 0},
            {"step": 110, "amount": 450000.0, "oldbalanceOrg": 450000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "TRANSFER", "isFraud": 1},
            {"step": 12, "amount": 3400.0, "oldbalanceOrg": 12000.0, "newbalanceOrig": 8600.0, "oldbalanceDest": 4000.0, "newbalanceDest": 7400.0, "type": "PAYMENT", "isFraud": 0},
            {"step": 112, "amount": 980000.0, "oldbalanceOrg": 980000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "CASH_OUT", "isFraud": 1},
            {"step": 15, "amount": 800.0, "oldbalanceOrg": 2500.0, "newbalanceOrig": 1700.0, "oldbalanceDest": 0.0, "newbalanceDest": 800.0, "type": "PAYMENT", "isFraud": 0},
            {"step": 115, "amount": 350000.0, "oldbalanceOrg": 350000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "TRANSFER", "isFraud": 1},
            {"step": 20, "amount": 5000.0, "oldbalanceOrg": 50000.0, "newbalanceOrig": 45000.0, "oldbalanceDest": 12000.0, "newbalanceDest": 17000.0, "type": "TRANSFER", "isFraud": 0},
            {"step": 120, "amount": 120000.0, "oldbalanceOrg": 120000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "type": "CASH_OUT", "isFraud": 1},
        ])
        st.info("Loaded **Mixed Portfolio** preset dataset (8 rows).")

    if raw is not None:
        # Preview
        with st.expander("Preview raw data (first 5 rows)"):
            st.dataframe(raw.head(), use_container_width=True)

        # Feature engineering
        def prep_batch(df):
            d = df.copy()
            for t in ["CASH_OUT","DEBIT","PAYMENT","TRANSFER"]:
                d[f"type_{t}"] = (d["type"].str.upper() == t).astype(int)
            d["balanceDiff"]  = d["oldbalanceOrg"]  - d["newbalanceOrig"]
            d["amount_ratio"] = d["amount"] / d["oldbalanceOrg"].replace(0, np.nan)
            d["amount_ratio"] = d["amount_ratio"].fillna(0)
            missing = [f for f in FEATURES if f not in d.columns]
            for m in missing: d[m] = 0
            return d

        with st.spinner("Running batch inference…"):
            df_prep = prep_batch(raw)
             # Process in chunks for very large files:
            CHUNK = 50_000
            probs = np.concatenate([
                model.predict_proba(scaler.transform(
                    df_prep[FEATURES].iloc[i:i+CHUNK].values.astype(float)
                ))[:,1]
                for i in range(0, len(df_prep), CHUNK)
            ])
            preds = (probs >= threshold).astype(int)   # add this

        df_out = raw.copy()
        df_out["fraud_probability"] = np.round(probs, 4)
        df_out["prediction"]        = preds
        df_out["risk_tier"]         = [get_risk_tier(p) for p in probs]

        # Summary stats
        n_fraud = int(preds.sum())
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Transactions", f"{len(df_out):,}")
        s2.metric("Flagged as Fraud", f"{n_fraud:,}", delta=f"{n_fraud/len(df_out):.1%}")
        s3.metric("Avg Fraud Prob", f"{probs.mean():.3f}")
        s4.metric("Max Fraud Prob", f"{probs.max():.3f}")

        # Risk tier chart
        tier_counts = df_out["risk_tier"].value_counts().reindex(
            ["CRITICAL","HIGH","MEDIUM","LOW"], fill_value=0)

        fig_t, ax_t = plt.subplots(figsize=(8, 3))
        fig_t.patch.set_facecolor("#0a0a0a")
        ax_t.set_facecolor("#0a0a0a")
        colors = [RISK_COLORS[t] for t in tier_counts.index]
        ax_t.bar(tier_counts.index, tier_counts.values, color=colors,
                 edgecolor="#141416", linewidth=0.8, width=0.6)
        ax_t.set_title("Transactions by Risk Tier", fontsize=11, color="#F0F0F2", fontweight="bold", fontfamily="DejaVu Sans")
        ax_t.tick_params(colors="#8B8B9A", labelsize=9)
        for sp in ax_t.spines.values(): sp.set_visible(False)
        for i, (idx, v) in enumerate(tier_counts.items()):
            ax_t.text(i, v + 0.3, str(v), ha="center", va="bottom",
                      fontsize=9, color="#F0F0F2", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)

        # Results table
        st.markdown("#### Results Table (first 100 rows)")
        display_cols = [c for c in ["step","amount","type","fraud_probability","prediction","risk_tier","isFraud"]
                        if c in df_out.columns]
        st.dataframe(df_out[display_cols].head(100), use_container_width=True, hide_index=True)

        # Accuracy if ground truth present
        if "isFraud" in df_out.columns:
            from sklearn.metrics import (classification_report, roc_auc_score,
                                         confusion_matrix)
            gt = df_out["isFraud"].values
            try:
                auc = roc_auc_score(gt, probs)
                report = classification_report(gt, preds, output_dict=True)
                st.markdown("#### Ground Truth Comparison")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AUC-ROC", f"{auc:.4f}")
                m2.metric("Precision (fraud)", f"{report['1']['precision']:.4f}")
                m3.metric("Recall (fraud)",    f"{report['1']['recall']:.4f}")
                m4.metric("F1 (fraud)",        f"{report['1']['f1-score']:.4f}")

                # Confusion matrix
                cm = confusion_matrix(gt, preds)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                fig_cm.patch.set_facecolor("#0a0a0a")
                ax_cm.set_facecolor("#0a0a0a")
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt="d",
                            cmap=sns.light_palette("#2563EB", as_cmap=True),
                            xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"],
                            ax=ax_cm, linewidths=0.5, linecolor="#0a0a0a",
                            annot_kws={"color": "#F0F0F2", "size": 10})
                ax_cm.set_title("Confusion Matrix", fontsize=10, color="#F0F0F2")
                ax_cm.tick_params(colors="#8B8B9A")
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            except Exception:
                pass

        # Download results
        csv_out = df_out.to_csv(index=False)
        st.download_button(" Download Results CSV", csv_out,
                           "fraud_batch_results.csv", "text/csv",
                           use_container_width=True)


# PAGE 3 — DATA & IMBALANCE
elif page == " Dataset & Imbalance":
    st.markdown(theme.section_header("Data Overview & Imbalance Handling", "· TRAINING DATA ·", ""), unsafe_allow_html=True)

    PLOTS    = PLOTS_DIR
    ABLATION = ABLATION_DIR

    def show_img(path, caption=None):
        """Display a pre-generated PNG plot, or show a clear warning if missing."""
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.warning(f"Plot not found: `{os.path.basename(path)}`")

    #  Section 1: Dataset Summary
    st.markdown("####  Dataset at a Glance")
    st.markdown("""
    <div class="tg-card">
    The model was trained on the <b>PaySim</b> synthetic financial transaction dataset —
    6.3 million rows simulating mobile money transactions over 30 days.
    Features include transaction type, amount, origin/destination balances, and engineered
    ratio features. The dataset is severely imbalanced: <b>only ~0.13% of transactions are fraudulent</b>.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", "6,362,620")
    c2.metric("Fraud Cases", "8,213")
    c3.metric("Fraud Rate", "0.13%")
    c4.metric("Features", "18")

    st.markdown("---")

    #  Section 2: EDA Visualizations
    st.markdown("####  Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Transaction Types", "Amount Distributions",
        "Correlations & Heatmap", "Fraud Patterns", "Feature Importance"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transaction Type Distribution**")
            show_img(os.path.join(PLOTS, "transaction_types_distribution.png"))
        with c2:
            st.markdown("**Fraud by Transaction Type**")
            show_img(os.path.join(PLOTS, "fraud_by_transaction_type.png"))
        st.markdown("**Detailed Fraud by Type**")
        show_img(os.path.join(PLOTS, "fraud_by_type_detailed.png"))

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transaction Amount Distribution**")
            show_img(os.path.join(PLOTS, "transaction_amount_distribution.png"))
        with c2:
            st.markdown("**Fraud vs Normal Amounts**")
            show_img(os.path.join(PLOTS, "fraud_vs_normal_transaction_amounts.png"))
        st.markdown("**Amount Distribution by Type & Fraud Status**")
        show_img(os.path.join(PLOTS, "amount_distribution_by_type_fraud.png"))

    with tab3:
        st.markdown("**Correlation Heatmap**")
        show_img(os.path.join(PLOTS, "correlation_heatmap.png"))
        st.markdown("**Models Comparison Heatmap**")
        show_img(os.path.join(PLOTS, "models_comparison_heatmap.png"))
        st.markdown("**EDA Feature Distributions**")
        show_img(os.path.join(PLOTS, "eda_feature_distributions.png"))

    with tab4:
        st.markdown("**Top Fraud Patterns — Mean Feature Comparison**")
        show_img(os.path.join(PLOTS, "top_fraud_patterns.png"))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transaction Volume Over Time**")
            show_img(os.path.join(PLOTS, "transaction_volume_over_time.png"))
        with c2:
            st.markdown("**EDA Class Distribution**")
            show_img(os.path.join(PLOTS, "eda_class_distribution.png"))

    with tab5:
        st.markdown("**Feature Importance (XGBoost)**")
        show_img(os.path.join(PLOTS, "feature_importance_1.png"))
        st.markdown("**Feature Importance Comparison — All Models**")
        show_img(os.path.join(PLOTS, "feature_importance_comparison.png"))

    st.markdown("---")

    #  Section 3: Class Imbalance
    st.markdown("####  Handling Class Imbalance")

    st.markdown("""
    <div class="tg-card">
    With only <b>0.13% fraud</b>, a naive model achieves 99.87% accuracy by predicting everything
    as legitimate — making accuracy useless. The pipeline uses a two-step strategy:
    <b>Fraud Simulation</b> followed by <b>SMOTE oversampling</b>, evaluated via Precision-Recall AUC.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Class Distribution (Before Balancing)**")
        show_img(os.path.join(PLOTS, "class_imbalance.png"))
    with c2:
        st.markdown("**Fraud Distribution**")
        show_img(os.path.join(PLOTS, "fraud_distribution.png"))

    st.markdown("**Pipeline stages — fraud ratio at each step:**")
    s1, s2, s3 = st.columns(3)
    s1.metric("① Original Train Split", "~0.13%", help="Raw PaySim fraud rate")
    s2.metric("② After Fraud Simulation", "~1.26%", help="5% of TRANSFER/CASH_OUT injected as fraud")
    s3.metric("③ After SMOTE (Final Train)", "~23.07%", help="sampling_strategy=0.3 applied on augmented set")

    st.markdown("####  Techniques Applied")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **① Fraud Simulation Engine**
        - Samples **5%** of legitimate TRANSFER & CASH_OUT transactions
        - Simulates full account drain: `amount = oldbalanceOrg`, `newbalanceOrig = 0`
        - Recomputes `balanceDiff` and `amount_ratio` to reflect drain behaviour
        - Labels injected rows as fraud — adds realistic, behaviour-grounded samples
        - Applied **before** SMOTE so synthetic fraud is also oversampled
        """)
    with col2:
        st.markdown("""
        **② SMOTE Oversampling**
        - `sampling_strategy = 0.3` — minority class reaches **30% of majority class size**
        - Reduced from 0.5 → 0.3 to avoid 40–60% training set bloat, We also try with 0.1 but it was very small and had a very minor affect on dataset.
        - We also tried 0.1 but it had very minor effect on the dataset
        - Applied **only to training set** — test set always stays real data only
        - Inside K-Fold CV, SMOTE runs per fold via `ImbPipeline` to prevent leakage
        - Three ratios tested in ablation: `0.0`, `0.3` (Best), `0.5`
        """)

    st.markdown("**SMOTE Ratio Ablation — AUPRC, Recall, Precision vs ratio (0.0 → 0.3 → 0.5)**")
    show_img(os.path.join(ABLATION, "ablation_smote_trend.png"))
    st.markdown("**Cost-Benefit Analysis**")
    show_img(os.path.join(PLOTS, "cost_benefit_analysis.png"))

# PAGE 4 — MODEL PERFORMANCE
elif page == " Model Performance":
    st.markdown(theme.section_header("Model Performance Dashboard", "· METRICS ·", "blue"), unsafe_allow_html=True)

    comp_df = load_model_comparison()
    if comp_df is not None:
        # Summary card: pull deployed model metrics (XGBoost preferred) or fall back to first row
        deployed_row = None
        try:
            deployed_row = comp_df[comp_df['Model'].str.contains('xgboost', case=False)].iloc[0]
        except Exception:
            try:
                deployed_row = comp_df.iloc[0]
            except Exception:
                deployed_row = None

        if deployed_row is not None:
            try:
                auc_v = float(deployed_row.get('Test AUC-ROC', deployed_row.get('Test AUC', 0)))
            except Exception:
                auc_v = 0.0
            try:
                recall_v = float(deployed_row.get('Test Recall', 0.0))
            except Exception:
                recall_v = 0.0

            # Render a short interpretive summary as a themed card
            summary_html = f"""
            <div class="fd-card" style="margin-bottom:0.8rem;">
                {theme.eyebrow('· MODEL SUMMARY ·','blue')}
                <div style="font-size:0.95rem; color:var(--text-primary);">
                    Deployed model: <b>{deployed_row.get('Model','XGBoost')}</b>.
                    Key metrics: AUC-ROC {auc_v:.4f}, Recall {recall_v:.4f}.
                </div>
                <div style="font-size:0.9rem; color:var(--text-secondary); margin-top:0.5rem;">
                    Given the strong class imbalance, optimize operating thresholds to balance precision and recall — high recall reduces missed fraud but increases analyst workload from false alarms.
                </div>
            </div>
            """
            st.markdown(summary_html, unsafe_allow_html=True)

        st.markdown("#### All Models — Test Metrics")
        disp_cols = ["Model","Test Precision","Test Recall","Test F1","Test AUC-ROC","Test Avg Prec"]
        st.dataframe(comp_df[[c for c in disp_cols if c in comp_df.columns]],
                     use_container_width=True, hide_index=True)

        # ── Business Value Metrics (Translate Math to Money) ──
        st.markdown("#### Financial ROI Calculator (Business Impact)")
        st.caption("Translate mathematical recall/precision into actual savings. Simulates a mock PKR 1B portfolio containing PKR 10M in actual fraud.")
        
        # Calculate dynamic metrics per model
        savings_data = []
        for _, row_m in comp_df.iterrows():
            m_name = row_m["Model"]
            # Extract precision & recall
            prec = float(row_m.get("Test Precision", 0.9))
            rec = float(row_m.get("Test Recall", 0.9))
            
            # Math translation
            actual_fraud = 10_000_000 # PKR 10M
            prevented_losses = actual_fraud * rec
            missed_fraud = actual_fraud * (1 - rec)
            
            # Legitimate transactions blocked (False alarms)
            # Assume 10,000 total alerts generated. False alarm rate = (1 - precision)
            false_alarms = int(1000 * (1 - prec)) if prec < 1.0 else 0
            operational_cost = false_alarms * 500  # PKR 500 overhead per false alert verification
            net_saved = prevented_losses - operational_cost
            
            savings_data.append({
                "Model": m_name,
                "Fraud Losses Prevented": f"PKR {prevented_losses:,.0f}",
                "Missed Fraud (Loss)": f"PKR {missed_fraud:,.0f}",
                "False Alarm Disruptions": f"{false_alarms} accounts",
                "Net Financial Savings": f"PKR {net_saved:,.0f}"
            })
            
        st.table(pd.DataFrame(savings_data))
        st.markdown("---")

        # Radar / grouped bar
        metrics = ["Test Precision","Test Recall","Test F1","Test AUC-ROC"]
        avail   = [m for m in metrics if m in comp_df.columns]
        fig_b, ax_b = plt.subplots(figsize=(10, 4))
        fig_b.patch.set_facecolor("#0a0a0a")
        ax_b.set_facecolor("#0a0a0a")
        x     = np.arange(len(avail))
        width = 0.18
        colors_m = ["#2563EB","#3B82F6","#D97706","#DC2626"]
        for i, (_, row_m) in enumerate(comp_df.iterrows()):
            vals = [row_m[m] for m in avail]
            ax_b.bar(x + i*width, vals, width, label=row_m["Model"],
                     color=colors_m[i % len(colors_m)], edgecolor="#0a0a0a", linewidth=0.6)
        ax_b.set_xticks(x + width*1.5)
        ax_b.set_xticklabels([m.replace("Test ","") for m in avail], fontsize=9, color="#F0F0F2")
        ax_b.set_ylim(0, 1.08)
        ax_b.set_title("Model Comparison — Test Metrics", fontsize=12, color="#F0F0F2", fontweight="bold")
        ax_b.legend(fontsize=8, framealpha=0.2, labelcolor="#F0F0F2", facecolor="#141416", edgecolor=(1, 1, 1, 0.08))
        ax_b.tick_params(colors="#8B8B9A", labelsize=9)
        ax_b.yaxis.set_tick_params(labelcolor="#8B8B9A")
        for sp in ax_b.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    st.markdown("---")
    st.markdown("#### Key Saved Visualisations")

    plot_map = {
        "ROC & PR Curves":           "roc_pr_curves.png",
        "Confusion Matrices":         "confusion_matrices_all_models.png",
        "Feature Importance":         "feature_importance_comparison.png",
        "Fraud Probability Dist.":    "fraud_prob_distribution_all_models.png",
        "Cost-Benefit Analysis":      "cost_benefit_analysis.png",
        "Error Analysis per Model":    "error_analysis.png",
        "Models Comparison of All Merics":    "model_comparison_all_metrics.png",
        "Classification Report Heatmap": "classification_report_heatmap.png",
        "Model Metrics Lineplot":     "model_metrics_lineplot.png",
        "Models Comparison Heatmap":  "models_comparison_heatmap.png",
    }
    # Show the two decision-relevant visuals by default, hide others in an expander
    key_vis = ["ROC & PR Curves", "Confusion Matrices"]
    for title in key_vis:
        fname = plot_map.get(title)
        fpath = os.path.join(PLOTS_DIR, fname) if fname else None
        if fpath and os.path.exists(fpath):
            st.markdown(f"**{title}**")
            show_img(fpath)

    with st.expander("Show additional visualisations"):
        cols = st.columns(2)
        extra = [t for t in plot_map.keys() if t not in key_vis]
        for i, title in enumerate(extra):
            fname = plot_map[title]
            fpath = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(fpath):
                with cols[i % 2]:
                    st.markdown(f"**{title}**")
                    show_img(fpath)

    # XGBoost metrics detail
    xgb_path = os.path.join(METRICS_DIR, "xgboost_metrics.json")
    if os.path.exists(xgb_path):
        with open(xgb_path) as f:
            xgb_m = json.load(f)
        st.markdown("---")
        st.markdown("#### XGBoost (Deployed Model) — Full Metrics")
        m_cols = st.columns(5)
        keys = [("test_auc_roc","AUC-ROC"),("test_recall","Recall"),
                ("test_precision","Precision"),("test_f1","F1"),("test_avg_prec","Avg Prec")]
        for j, (k, lbl) in enumerate(keys):
            m_cols[j].metric(lbl, f"{xgb_m.get(k, 0):.4f}")


# PAGE 5 — ABLATION STUDY
elif page == " Ablation Study":
    st.markdown(theme.section_header("Ablation Study", "· COMPONENT ANALYSIS ·", "blue"), unsafe_allow_html=True)
    st.markdown("Component-level analysis of the pipeline — showing which parts contribute most.")

    ab_df = load_ablation()
    if ab_df is not None:
        st.dataframe(ab_df, use_container_width=True, hide_index=True)

        # Ablation summary: infer main contributor(s) from table values
        try:
            # prefer CV F1 if present, else Test AUC
            if 'CV F1' in ab_df.columns:
                best_idx = ab_df['CV F1'].idxmax()
                best_cond = ab_df.loc[best_idx, 'Condition']
                best_val = float(ab_df.loc[best_idx, 'CV F1'])
                metric_label = 'CV F1'
            elif 'Test AUC' in ab_df.columns:
                best_idx = ab_df['Test AUC'].idxmax()
                best_cond = ab_df.loc[best_idx, 'Condition']
                best_val = float(ab_df.loc[best_idx, 'Test AUC'])
                metric_label = 'Test AUC'
            else:
                best_cond = None
        except Exception:
            best_cond = None

        if best_cond:
            ab_summ = f"""
            <div class="fd-card" style="margin-bottom:0.8rem;">
                {theme.eyebrow('· ABLATION SUMMARY ·','blue')}
                <div style="font-size:0.95rem; color:var(--text-primary);">
                    The ablation indicates <b>{best_cond}</b> achieves the highest {metric_label} ({best_val:.4f}).
                </div>
                <div style="font-size:0.9rem; color:var(--text-secondary); margin-top:0.5rem;">
                    This suggests that the components present in this configuration (e.g., full feature set, SMOTE/augmentation, or simulation) contribute most to model performance relative to variants in the table.
                </div>
            </div>
            """
            st.markdown(ab_summ, unsafe_allow_html=True)

        # Bar chart — F1 comparison
        fig_a, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig_a.patch.set_facecolor("#0a0a0a")
        for ax in axes: ax.set_facecolor("#0a0a0a")

        pal = ["#2563EB" if "full" in c.lower() or "current" in c.lower()
               else "#505060" for c in ab_df["Condition"]]

        # CV F1
        axes[0].barh(ab_df["Condition"], ab_df["CV F1"], color=pal,
                     edgecolor="#0a0a0a", linewidth=0.6)
        axes[0].set_title("CV F1 by Condition", fontsize=10, color="#F0F0F2", fontweight="bold")
        axes[0].set_xlim(0, 1.1)
        axes[0].tick_params(labelsize=7, colors="#8B8B9A")
        for sp in axes[0].spines.values(): sp.set_visible(False)

        # Test AUC
        axes[1].barh(ab_df["Condition"], ab_df["Test AUC"], color=pal,
                     edgecolor="#0a0a0a", linewidth=0.6)
        axes[1].set_title("Test AUC by Condition", fontsize=10, color="#F0F0F2", fontweight="bold")
        min_auc = ab_df["Test AUC"].min()
        axes[1].set_xlim(max(0, min_auc - 0.01), 1.002)
        axes[1].tick_params(labelsize=7, colors="#8B8B9A")
        for sp in axes[1].spines.values(): sp.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_a, use_container_width=True)
        plt.close(fig_a)

    st.markdown("---")
    st.markdown("#### Ablation Plots")
    ab_plots = {
        "Ablation Summary":     "ablation_study.png",
        "Ablation Heatmap":     "ablation_heatmap.png",
        "SMOTE Trend":          "ablation_smote_trend.png",
        "PR Scatter":           "ablation_pr_scatter.png",
        "Delta (Δ) Chart":      "ablation_delta.png",
    }
    # Keep primary ablation summary visible; hide other ablation plots behind an expander
    primary = "Ablation Summary"
    if primary in ab_plots:
        fpath = os.path.join(ABLATION_DIR, ab_plots[primary])
        if os.path.exists(fpath):
            st.markdown(f"**{primary}**")
            show_img(fpath)

    with st.expander("Show additional ablation plots"):
        ab_cols = st.columns(2)
        extras = [t for t in ab_plots.keys() if t != primary]
        for i, title in enumerate(extras):
            fname = ab_plots[title]
            fpath = os.path.join(ABLATION_DIR, fname)
            if os.path.exists(fpath):
                with ab_cols[i % 2]:
                    st.markdown(f"**{title}**")
                    show_img(fpath)
                #st.image(fpath, use_container_width=True)

# PAGE 6 — ABOUT
elif page == " About":
    import re as _re
    def _clean(s): return _re.sub(r'\s+',' ', s.replace('\n',' ')).strip()

    st.markdown(theme.section_header("AI Fraud Detection", "· ABOUT THIS SYSTEM ·", "red"), unsafe_allow_html=True)

    # ── Overview ──────────────────────────────────────────────────────────────
    st.markdown(_clean(f"""
    <div class="fd-card" style="margin-bottom:2rem;">
        {theme.eyebrow("· OVERVIEW ·")}
        <h3 style="margin:0.2rem 0 0.75rem 0; font-size:1.15rem;">
            High-Precision Fraud Intelligence for Pakistan's Financial Sector
        </h3>
        <p style="color:var(--text-secondary); font-size:0.9rem; line-height:1.7; margin:0;">
            This system demonstrates a complete, production-ready fraud detection pipeline built on the
            PaySim synthetic mobile money dataset (6.3 million transactions). It combines classical and
            gradient-boosted machine learning, per-prediction SHAP explainability, and a Retrieval-Augmented
            Generation (RAG) layer grounded in real State Bank of Pakistan (SBP) regulatory documents —
            so every fraud verdict comes with a human-readable compliance justification, not just a score.
        </p>
    </div>
    """), unsafe_allow_html=True)

    # ── Technical System ──────────────────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(_clean(f"""
        <div class="fd-card fd-card-red fd-card-sharp" style="height:100%;">
            {theme.eyebrow("· DETECTION ENGINE ·", "red")}
            <h4 style="margin:0.2rem 0 0.75rem 0;">ML Classification Stack</h4>
            <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6; margin-bottom:1rem;">
                Four classifiers trained and benchmarked in a rigorous cross-validation + out-of-time test split.
                XGBoost is the deployed model — it achieves <b style="color:var(--text-primary);">99.95% AUC-ROC</b>
                and <b style="color:var(--text-primary);">99.76% recall</b> on the held-out test set.
            </p>
            <div style="font-family:var(--font-mono); font-size:0.78rem; color:var(--text-secondary); line-height:2;">
                XGBoost (Deployed) &nbsp;·&nbsp; AUC 0.9995<br>
                Random Forest &nbsp;·&nbsp; Neural Network (MLP)<br>
                Logistic Regression (baseline)<br>
                SMOTE + Fraud Simulation augmentation<br>
                K-Fold CV with ImbPipeline (no leakage)
            </div>
        </div>
        """), unsafe_allow_html=True)

    with col_b:
        st.markdown(_clean(f"""
        <div class="fd-card fd-card-blue" style="height:100%;">
            {theme.eyebrow("· EXPLAINABILITY & COMPLIANCE ·", "blue")}
            <h4 style="margin:0.2rem 0 0.75rem 0;">XAI + RAG Layer</h4>
            <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6; margin-bottom:1rem;">
                Every prediction is accompanied by a SHAP waterfall showing each feature's contribution
                to the fraud score. A separate RAG pipeline retrieves the most relevant SBP regulatory
                clauses and generates a structured compliance report via Cohere Command R+.
            </p>
            <div style="font-family:var(--font-mono); font-size:0.78rem; color:var(--text-secondary); line-height:2;">
                SHAP TreeExplainer (primary)<br>
                Feature importance fallback (Plotly)<br>
                ChromaDB vector store + BM25 hybrid retrieval<br>
                Sentence-Transformers (all-MiniLM-L6-v2)<br>
                Cohere Command R+ (command-r-plus-08-2024)
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Capability List ───────────────────────────────────────────────────────
    st.markdown(_clean(f"""
    <div class="fd-card" style="margin-bottom:2rem;">
        {theme.eyebrow("· CAPABILITIES ·")}
        <h4 style="margin:0.2rem 0 1rem 0;">What This Dashboard Does</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.65rem;">
            <div style="display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="font-family:var(--font-mono); color:var(--blue-light); font-size:0.78rem; margin-top:0.1rem; flex-shrink:0;">01</span>
                <span style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    <b style="color:var(--text-primary);">Real-time single transaction scoring</b> — fill in transaction fields or use Quick-Fill scenarios, get an instant probability and risk tier.
                </span>
            </div>
            <div style="display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="font-family:var(--font-mono); color:var(--blue-light); font-size:0.78rem; margin-top:0.1rem; flex-shrink:0;">02</span>
                <span style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    <b style="color:var(--text-primary);">SHAP interactive explainability</b> — see exactly which features drove the model's verdict, with red/blue directional bars.
                </span>
            </div>
            <div style="display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="font-family:var(--font-mono); color:var(--blue-light); font-size:0.78rem; margin-top:0.1rem; flex-shrink:0;">03</span>
                <span style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    <b style="color:var(--text-primary);">SBP RAG compliance report</b> — auto-generates STR/CTR filing recommendations grounded in SBP AML/CFT regulations.
                </span>
            </div>
            <div style="display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="font-family:var(--font-mono); color:var(--blue-light); font-size:0.78rem; margin-top:0.1rem; flex-shrink:0;">04</span>
                <span style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    <b style="color:var(--text-primary);">Batch CSV inference</b> — upload up to 500K transactions, get scored results with risk tiers and a ground-truth confusion matrix if labels are present.
                </span>
            </div>
            <div style="display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="font-family:var(--font-mono); color:var(--blue-light); font-size:0.78rem; margin-top:0.1rem; flex-shrink:0;">05</span>
                <span style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    <b style="color:var(--text-primary);">Model performance dashboard</b> — full AUC-ROC, precision, recall, F1, financial ROI calculations, and saved training plots.
                </span>
            </div>
            <div style="display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="font-family:var(--font-mono); color:var(--blue-light); font-size:0.78rem; margin-top:0.1rem; flex-shrink:0;">06</span>
                <span style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    <b style="color:var(--text-primary);">Ablation study</b> — visualises the isolated contribution of SMOTE ratios, fraud simulation, and model choice on detection quality.
                </span>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # ── Stack + Metadata ──────────────────────────────────────────────────────
    col_s1, col_s2 = st.columns([1.2, 1.0], gap="large")

    with col_s1:
        st.markdown(_clean(f"""
        <div class="fd-card" style="height:100%;">
            {theme.eyebrow("· STACK ·")}
            <h4 style="margin:0.2rem 0 0.75rem 0;">Technology Used</h4>
            <div style="font-size:0.85rem; line-height:2; color:var(--text-secondary);">
                <div><span style="font-family:var(--font-mono); color:var(--text-primary); font-size:0.78rem;">TRAINING DATA &nbsp;</span>PaySim — 6.3M synthetic mobile money txns</div>
                <div><span style="font-family:var(--font-mono); color:var(--text-primary); font-size:0.78rem;">ML LIBRARY &nbsp;&nbsp;&nbsp;&nbsp;</span>Scikit-Learn, XGBoost, imbalanced-learn</div>
                <div><span style="font-family:var(--font-mono); color:var(--text-primary); font-size:0.78rem;">EXPLAINABILITY &nbsp;</span>SHAP (TreeExplainer + Explainer fallback)</div>
                <div><span style="font-family:var(--font-mono); color:var(--text-primary); font-size:0.78rem;">VECTOR DB &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>ChromaDB with hybrid BM25 retrieval</div>
                <div><span style="font-family:var(--font-mono); color:var(--text-primary); font-size:0.78rem;">EMBEDDINGS &nbsp;&nbsp;&nbsp;&nbsp;</span>sentence-transformers/all-MiniLM-L6-v2</div>
                <div><span style="font-family:var(--font-mono); color:var(--text-primary); font-size:0.78rem;">LLM &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>Cohere command-r-plus-08-2024</div>
                <div><span style="font-family:var(--font-mono); color:var(--text-primary); font-size:0.78rem;">REGULATORY CORPUS</span>SBP AML/CFT, Branchless Banking PRs, SME PRs</div>
                <div><span style="font-family:var(--font-mono); color:var(--text-primary); font-size:0.78rem;">FRONTEND &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>Streamlit + Plotly + Matplotlib + custom CSS</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with col_s2:
        st.markdown(_clean(f"""
        <div class="fd-card fd-card-blue" style="margin-bottom:1rem;">
            {theme.eyebrow("· DEPLOYED MODEL METADATA ·", "blue")}
            <h4 style="margin:0.2rem 0 0.75rem 0;">XGBoost (Active)</h4>
        </div>
        """), unsafe_allow_html=True)
        _, _, meta = load_deployment_model()
        st.json(meta)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Author ────────────────────────────────────────────────────────────────
    st.markdown(_clean(f"""
    <div class="fd-card fd-card-red">
        {theme.eyebrow("· AUTHOR ·", "red")}
        <h4 style="margin:0.2rem 0 0.5rem 0;">Solo Hackathon Build</h4>
        <p style="font-size:0.88rem; color:var(--text-secondary); line-height:1.7; margin:0 0 0.75rem 0;">
            Designed and built end-to-end as a solo hackathon project — from dataset preprocessing,
            model training and ablation, to the RAG regulatory pipeline and this Streamlit dashboard.
            No pre-built starter templates were used; the entire system — ML, XAI, RAG, and UI — was
            authored from scratch.
        </p>
        <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
            <div>
                <div style="font-family:var(--font-mono); font-size:0.68rem; color:var(--text-dim); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.15rem;">Dataset</div>
                <div style="font-size:0.85rem; color:var(--text-primary);">PaySim (Kaggle)</div>
            </div>
            <div>
                <div style="font-family:var(--font-mono); font-size:0.68rem; color:var(--text-dim); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.15rem;">Regulatory Scope</div>
                <div style="font-size:0.85rem; color:var(--text-primary);">State Bank of Pakistan</div>
            </div>
            <div>
                <div style="font-family:var(--font-mono); font-size:0.68rem; color:var(--text-dim); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.15rem;">Build Type</div>
                <div style="font-size:0.85rem; color:var(--text-primary);">Hackathon — Solo Entry</div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ── RAG Limitations ───────────────────────────────────────────────────────
    st.markdown(_clean(f"""
    <div style="margin-bottom:0.5rem;">{theme.eyebrow("· KNOWN LIMITATIONS ·")}</div>
    """), unsafe_allow_html=True)
    st.warning("""
    - Requires a valid **Cohere API key** (command-r-plus-08-2024). Without it, the regulatory report section is skipped.
    - ChromaDB stores ~100 SBP document chunks. For best results, ensure `chroma_db/` is committed to the repo.
    - If Cohere rate-limits are hit, the RAG section will show a rate-limit error.
    - F1 score (0.57) reflects the severe class imbalance of the dataset — AUC-ROC and Recall are the primary metrics.
    """)
