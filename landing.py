import streamlit as st
import theme
import textwrap
import re

def clean_html(html_str: str) -> str:
    """Minifies HTML code to prevent Streamlit's markdown engine from treating indented HTML as code blocks."""
    # Remove HTML comments to prevent smartypants symbol translation issues
    html_str = re.sub(r'<!--.*?-->', '', html_str)
    # Convert all newlines to spaces
    html_str = html_str.replace("\n", " ")
    # Collapse multiple whitespaces
    html_str = re.sub(r'\s+', ' ', html_str)
    return html_str.strip()

def render_landing_page(nav_key: str):
    """Renders the complete landing page.
    
    Args:
        nav_key: the session state key for the sidebar radio selector,
                 so we can programmatically navigate into the dashboard.
    """
    # Callback function to handle navigation changes safely before script rerun
    def navigate_to_dashboard():
        st.session_state[nav_key] = " Predict Transaction"

    # 1. Hide Sidebar via CSS override when landing page is active
    st.markdown(clean_html("""
    <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="collapsedSidebar"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] {
            width: 0px !important;
            margin-left: -336px !important;
        }
        .main .block-container {
            max-width: 1080px !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
    </style>
    """), unsafe_allow_html=True)

    # 2. Hero Section
    st.markdown(clean_html(f"""
    <div style="text-align: center; padding: 4rem 0 3rem 0; max-width: 800px; margin: 0 auto;">
        {theme.eyebrow("· AI FRAUD OPERATIONS ·", "red")}
            <h1 style="font-size:2.2rem; margin-bottom:0.6rem;">Monitor transactions and surface useful alerts.</h1>
            <div style="font-size:1.05rem; color:var(--text-secondary);">Provides per-transaction risk scores, SHAP explanations, and regulatory references to help review decisions.</div>
    </div>
    """), unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        st.button("Launch Detection Console", type="primary", use_container_width=True, key="hero_cta", on_click=navigate_to_dashboard)

    # Marquee/Tag Strip below hero (no emojis)
    st.markdown(clean_html(f"""
    <div style="text-align: center; margin-top: 1.8rem; margin-bottom: 5rem;">
        <span style="font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.1em; color: var(--text-dim); text-transform: uppercase; background: var(--bg-surface); border: 1px solid var(--border); padding: 6px 16px; border-radius: 99px;">
            REAL-TIME SCORING &nbsp;·&nbsp; EXPLAINABLE SHAP &nbsp;·&nbsp; SBP-GROUNDED RAG &nbsp;·&nbsp; ABLATION VERIFIED
        </span>
    </div>
    """), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # 3. "The Problem" Section (Red Accent)
    st.markdown(clean_html(f"""
    <div style="margin-bottom: 3.5rem;">
        {theme.section_header("The Fragility of Legacy Fraud Systems", "· THE PROBLEM ·", "red")}
        <p style="color: var(--text-secondary); max-width: 650px; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 2rem;">
            Traditional fraud detection relies either on rigid, easily bypassed rule systems or complete black-box machine learning models that lack context and trigger excessive false alerts.
        </p>
    </div>
    """), unsafe_allow_html=True)

    # 4 numbered pain point cards (using 2x2 grid columns)
    col_p1, col_p2 = st.columns(2, gap="medium")
    with col_p1:
        st.markdown(clean_html("""
        <div class="fd-card fd-card-red" style="height: 180px;">
            <div style="font-family: var(--font-mono); font-size: 0.9rem; font-weight: 700; color: var(--red-light); margin-bottom: 0.6rem;">[ 01 ] BLACK-BOX VERDICTS</div>
            <h4 style="margin: 0 0 0.4rem 0; font-size: 1.05rem;">Alerts Without Explanation</h4>
            <p style="margin: 0; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">
                ML classifiers score risk but leave analysts in the dark. Without clear feature contributions, verifying a transaction feels like guesswork.
            </p>
        </div>
        """), unsafe_allow_html=True)
        st.markdown(clean_html("""
        <div class="fd-card fd-card-red" style="height: 180px;">
            <div style="font-family: var(--font-mono); font-size: 0.9rem; font-weight: 700; color: var(--red-light); margin-bottom: 0.6rem;">[ 03 ] REGULATORY SILOS</div>
            <h4 style="margin: 0 0 0.4rem 0; font-size: 1.05rem;">Regulatory Disconnection</h4>
            <p style="margin: 0; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">
                Scoring algorithms are blind to local guidelines. Compliance teams must manually cross-reference SBP handbooks to draft justification summaries.
            </p>
        </div>
        """), unsafe_allow_html=True)

    with col_p2:
        st.markdown(clean_html("""
        <div class="fd-card fd-card-red" style="height: 180px;">
            <div style="font-family: var(--font-mono); font-size: 0.9rem; font-weight: 700; color: var(--red-light); margin-bottom: 0.6rem;">[ 02 ] SLOW MANUAL REVIEW</div>
            <h4 style="margin: 0 0 0.4rem 0; font-size: 1.05rem;">Delayed Transaction Clearance</h4>
            <p style="margin: 0; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">
                Legitimate accounts face false holds. Compliance analysts spend up to an hour manually auditing balances and calculating transfer ratios.
            </p>
        </div>
        """), unsafe_allow_html=True)
        st.markdown(clean_html("""
        <div class="fd-card fd-card-red" style="height: 180px;">
            <div style="font-family: var(--font-mono); font-size: 0.9rem; font-weight: 700; color: var(--red-light); margin-bottom: 0.6rem;">[ 04 ] CLASS IMBALANCE DRIFT</div>
            <h4 style="margin: 0 0 0.4rem 0; font-size: 1.05rem;">Imbalance Blind Spots</h4>
            <p style="margin: 0; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">
                Models trained on typical datasets drift quickly or suffer from massive false alarm rates unless robust synthetic balancing techniques are verified.
            </p>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<hr style='margin: 4rem 0 !important;'>", unsafe_allow_html=True)

    # 4. "The Solution" Section (Blue Accent)
    st.markdown(clean_html(f"""
    <div style="margin-bottom: 3rem;">
        {theme.section_header("Explainable Scoring & SBP Grounding", "· THE SOLUTION ·", "blue")}
        <p style="color: var(--text-secondary); max-width: 650px; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 2rem;">
            A synchronized intelligence pipeline connecting transaction telemetry, high-precision classifiers, SHAP interactive local explanations, and SBP regulatory guidelines via vector search (RAG).
        </p>
    </div>
    """), unsafe_allow_html=True)

    # Beautiful HTML/CSS pipeline diagram (completely minified, comments removed by clean_html)
    st.markdown(clean_html("""
    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 2rem; margin-bottom: 4rem;">
        <div style="display: flex; flex-direction: row; align-items: center; justify-content: space-between; gap: 0.5rem; overflow-x: auto; min-width: 650px; padding: 0.5rem 0;">
            
            <!-- Step 1 -->
            <div style="flex: 1; text-align: center; background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 0.75rem 0.5rem;">
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-secondary); margin-bottom: 0.25rem;">STEP 01</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-primary);">Transaction</div>
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-dim); margin-top: 0.15rem;">PKR/Type/Balances</div>
            </div>
            
            <div style="color: var(--text-dim); font-size: 1.2rem; font-weight: bold; padding: 0 0.2rem;">→</div>
            
            <!-- Step 2 -->
            <div style="flex: 1; text-align: center; background: var(--bg-surface); border: 1px solid var(--border-blue); border-radius: var(--radius-md); padding: 0.75rem 0.5rem; box-shadow: 0 0 10px rgba(37,99,235,0.05);">
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--blue-light); margin-bottom: 0.25rem;">STEP 02</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-primary);">XGBoost Engine</div>
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-dim); margin-top: 0.15rem;">Probability Score</div>
            </div>
            
            <div style="color: var(--text-dim); font-size: 1.2rem; font-weight: bold; padding: 0 0.2rem;">→</div>
            
            <!-- Step 3 -->
            <div style="flex: 1; text-align: center; background: var(--bg-surface); border: 1px solid var(--border-blue); border-radius: var(--radius-md); padding: 0.75rem 0.5rem;">
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--blue-light); margin-bottom: 0.25rem;">STEP 03</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-primary);">SHAP Waterfall</div>
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-dim); margin-top: 0.15rem;">Feature Attribution</div>
            </div>
            
            <div style="color: var(--text-dim); font-size: 1.2rem; font-weight: bold; padding: 0 0.2rem;">→</div>
            
            <!-- Step 4 -->
            <div style="flex: 1; text-align: center; background: var(--bg-surface); border: 1px solid var(--border-blue); border-radius: var(--radius-md); padding: 0.75rem 0.5rem;">
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--blue-light); margin-bottom: 0.25rem;">STEP 04</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-primary);">Regulatory RAG</div>
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-dim); margin-top: 0.15rem;">SBP Guidelines</div>
            </div>
            
            <div style="color: var(--text-dim); font-size: 1.2rem; font-weight: bold; padding: 0 0.2rem;">→</div>
            
            <!-- Step 5 -->
            <div style="flex: 1; text-align: center; background: var(--red-dim); border: 1px solid var(--border-red); border-radius: var(--radius-md); padding: 0.75rem 0.5rem;">
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--red-light); margin-bottom: 0.25rem;">VERDICT</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-primary);">STR / CTR Filing</div>
                <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-dim); margin-top: 0.15rem;">Justified Case File</div>
            </div>
            
        </div>
    </div>
    """), unsafe_allow_html=True)

    # 5. Stats Band (real metrics retrieved from model_comparison.json)
    st.markdown(clean_html(f"""
    <div style="margin-bottom: 1rem;">
        {theme.eyebrow("· OPERATION METRICS ·", "blue")}
    </div>
    """), unsafe_allow_html=True)

    # Display 4 real metrics
    st.markdown(
        theme.stat_strip([
            {"value": "99.95%", "label": "Test AUC-ROC (XGBoost)", "color": "blue"},
            {"value": "99.76%", "label": "Test Recall (Catch Rate)", "color": "red"},
            {"value": "93.58%", "label": "Avg Precision (PR-AUC)", "color": "amber"},
            {"value": "< 150ms", "label": "Inference Latency", "color": ""},
        ]),
        unsafe_allow_html=True
    )

    st.markdown("<hr style='margin: 4rem 0 !important;'>", unsafe_allow_html=True)

    # 6. Capabilities & Comparison Section (Side by Side)
    col_c1, col_c2 = st.columns([1.2, 1.0], gap="large")

    with col_c1:
        st.markdown(clean_html(f"""
        <div style="margin-bottom: 1.5rem;">
            {theme.section_header("Operational Console Capabilities", "· SYSTEM DETAILS ·", "blue")}
        </div>
        """), unsafe_allow_html=True)

        # Numbered two-column / listing capabilities
        st.markdown(clean_html("""
        <div style="display: flex; flex-direction: column; gap: 1rem; padding-right: 1rem;">
            <div style="display: flex; gap: 0.75rem;">
                <span style="font-family: var(--font-mono); font-size: 0.82rem; font-weight: 700; color: var(--blue-light); margin-top: 0.1rem;">01</span>
                <div>
                    <h5 style="margin: 0 0 0.15rem 0; font-size: 0.95rem;">Real-Time Transaction Scoring</h5>
                    <p style="margin: 0; font-size: 0.82rem; color: var(--text-secondary); line-height: 1.4;">
                        Evaluate simulated mobile transactions instantly and adjust sliders dynamically to watch scores update in real time.
                    </p>
                </div>
            </div>
            <div style="display: flex; gap: 0.75rem;">
                <span style="font-family: var(--font-mono); font-size: 0.82rem; font-weight: 700; color: var(--blue-light); margin-top: 0.1rem;">02</span>
                <div>
                    <h5 style="margin: 0 0 0.15rem 0; font-size: 0.95rem;">SHAP Interactive Local Explanations</h5>
                    <p style="margin: 0; font-size: 0.82rem; color: var(--text-secondary); line-height: 1.4;">
                        Understand the math behind predictions. View top positive and negative features contributing to the risk scoring.
                    </p>
                </div>
            </div>
            <div style="display: flex; gap: 0.75rem;">
                <span style="font-family: var(--font-mono); font-size: 0.82rem; font-weight: 700; color: var(--blue-light); margin-top: 0.1rem;">03</span>
                <div>
                    <h5 style="margin: 0 0 0.15rem 0; font-size: 0.95rem;">State Bank of Pakistan (SBP) RAG</h5>
                    <p style="margin: 0; font-size: 0.82rem; color: var(--text-secondary); line-height: 1.4;">
                        Generate regulatory justification summaries using an LLM grounded in real, local compliance documents.
                    </p>
                </div>
            </div>
            <div style="display: flex; gap: 0.75rem;">
                <span style="font-family: var(--font-mono); font-size: 0.82rem; font-weight: 700; color: var(--blue-light); margin-top: 0.1rem;">04</span>
                <div>
                    <h5 style="margin: 0 0 0.15rem 0; font-size: 0.95rem;">Ablation & Model Benchmark Sweeps</h5>
                    <p style="margin: 0; font-size: 0.82rem; color: var(--text-secondary); line-height: 1.4;">
                        Audit model parameters, class balancing techniques (SMOTE vs downsampling), and side-by-side performance indicators.
                    </p>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with col_c2:
        st.markdown(clean_html(f"""
        <div style="margin-bottom: 1.5rem;">
            {theme.section_header("Model Performance (Test F1)", "· BENCHMARK ·", "")}
        </div>
        """), unsafe_allow_html=True)

        # Injected comparison bars representing Test F1 score across 4 models
        # XGBoost: 0.5691, Neural Network: 0.2505, Random Forest: 0.1875, Logistic Regression: 0.0425
        st.markdown(
            theme.comp_bar_group([
                {"label": "XGBoost (Deployed)", "value": 0.5691, "display": "0.569 F1", "color": "red"},
                {"label": "Neural Network", "value": 0.2505, "display": "0.251 F1", "color": "amber"},
                {"label": "Random Forest", "value": 0.1875, "display": "0.188 F1", "color": "blue"},
                {"label": "Logistic Regression", "value": 0.0425, "display": "0.043 F1", "color": "blue"},
            ], max_value=0.60),
            unsafe_allow_html=True
        )
        
        st.markdown(clean_html(f"""
        <div style="font-size: 0.76rem; color: var(--text-dim); line-height: 1.4; padding-left: 0.5rem; margin-top: 0.8rem;">
            *Scores evaluate the models on the highly imbalanced out-of-time test dataset, representing real-world production drift. XGBoost exhibits the strongest precision-recall balance.
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<hr style='margin: 4rem 0 !important;'>", unsafe_allow_html=True)

    # 7. Bottom CTA Section
    st.markdown(clean_html(f"""
    <div style="text-align: center; padding: 2.5rem 0 3rem 0; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); margin-bottom: 4rem; position: relative; overflow: hidden;">
        <div style="position: absolute; left: 0; right: 0; top: 0; bottom: 0; background: radial-gradient(circle at center, rgba(37,99,235,0.04) 0%, transparent 70%); pointer-events: none;"></div>
        {theme.eyebrow("· SECURE THE SYSTEM ·", "blue")}
        <h2 style="font-size: 1.8rem; font-weight: 700; margin: 0.25rem 0 0.75rem 0; letter-spacing: -0.01em;">
            Ready to secure your transaction ledger?
        </h2>
        <p style="font-size: 0.92rem; color: var(--text-secondary); line-height: 1.5; max-width: 480px; margin: 0 auto 1.8rem auto;">
            Enter the active classifier console to audit simulated transfers, test batch data, and generate compliance reports.
        </p>
    </div>
    """), unsafe_allow_html=True)

    # Reuse CTA button with custom key
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        st.button("Enter Dashboard Console", type="primary", use_container_width=True, key="bottom_cta", on_click=navigate_to_dashboard)

    st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)

    # 8. Footer
    st.markdown(clean_html("""
    <div style="border-top: 1px solid var(--border); padding: 2rem 0; text-align: center; font-family: var(--font-sans); font-size: 0.8rem; color: var(--text-dim);">
        <div style="font-weight: 600; color: var(--text-secondary); margin-bottom: 0.4rem;">AI Fraud Detection & Operations Console</div>
        <div style="margin-bottom: 0.6rem;">
            Stack: Streamlit · Scikit-Learn · XGBoost · SHAP · ChromaDB · Vector Search
        </div>
        <div>
            <a href="https://github.com" target="_blank" style="color: var(--blue-light); text-decoration: none; font-weight: 500; font-family: var(--font-mono); font-size: 0.75rem;">[ PROJECT REPOSITORY ]</a>
        </div>
    </div>
    """), unsafe_allow_html=True)
