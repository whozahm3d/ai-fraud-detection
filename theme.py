"""
theme.py — Shared Design System
================================
rig.ai-inspired dark design language, adapted for fraud detection.

Usage
-----
    from theme import inject_theme
    inject_theme()          # call once, right after st.set_page_config()

CSS Custom Properties (--var) are available to all inline HTML blocks
injected anywhere in the app via st.markdown(..., unsafe_allow_html=True).

Design Tokens
-------------
Backgrounds : --bg-base  #0a0a0a  | --bg-surface #0d0d10 | --bg-card #141416
              --bg-card-hover #1a1a1e
Accents     : --red  #DC2626  (risk / fraud / alert)
              --blue #2563EB  (trust / verified / safe)
Borders     : --border rgba(255,255,255,0.08)
              --border-red  rgba(220,38,38,0.35)
              --border-blue rgba(37,99,235,0.35)
Glows       : --glow-red  rgba(220,38,38,0.18)
              --glow-blue rgba(37,99,235,0.18)
Text        : --text-primary #F0F0F2 | --text-secondary #8B8B9A | --text-dim #505060
Typography  : --font-sans  (Inter → system sans)
              --font-mono  (JetBrains Mono)
Radius      : --radius-sm 6px | --radius-md 10px | --radius-lg 16px
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Master CSS string
# ---------------------------------------------------------------------------

_CSS = """
/* =========================================================
   FONT IMPORTS
   ========================================================= */
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* =========================================================
   DESIGN TOKENS — CSS Custom Properties
   ========================================================= */
:root {
    /* Backgrounds */
    --bg-base:        #0a0a0a;
    --bg-surface:     #0d0d10;
    --bg-card:        #141416;
    --bg-card-hover:  #1a1a1e;
    --bg-elevated:    #1e1e24;

    /* Accent: RED — risk / fraud / alert / critical (slightly desaturated) */
    --red:            #C94A45;
    --red-light:      #E35A4F;
    --red-dim:        rgba(201, 74, 69, 0.10);
    --border-red:     rgba(201, 74, 69, 0.28);
    --glow-red:       rgba(201, 74, 69, 0.12);

    /* Accent: BLUE — trust / verified / safe / low-risk (slightly desaturated) */
    --blue:           #2B61D9;
    --blue-light:     #3A7BE6;
    --blue-dim:       rgba(43, 97, 217, 0.10);
    --border-blue:    rgba(43, 97, 217, 0.28);
    --glow-blue:      rgba(43, 97, 217, 0.10);

    /* Accent: AMBER — medium/warning */
    --amber:          #D97706;
    --amber-dim:      rgba(217, 119, 6, 0.12);
    --border-amber:   rgba(217, 119, 6, 0.35);

    /* Borders */
    --border:         rgba(255, 255, 255, 0.08);
    --border-strong:  rgba(255, 255, 255, 0.14);

    /* Text */
    --text-primary:   #F0F0F2;
    --text-secondary: #8B8B9A;
    --text-dim:       #505060;

    /* Typography */
    --font-sans: 'Public Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;

    /* Spacing / Radius (reduced for a more minimal, sharp aesthetic) */
    --radius-sm:  4px;
    --radius-md:  6px;
    --radius-lg:  8px;
    --radius-xl: 12px;

    /* Transitions */
    --transition: 0.18s ease;
}

/* =========================================================
   GLOBAL RESETS & BASE
   ========================================================= */
html, body {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Streamlit root overrides */
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
.main,
section.main,
.block-container {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* Hide Streamlit default decoration */
[data-testid="stDecoration"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }
header    { visibility: hidden !important; }

/* =========================================================
   TYPOGRAPHY
   ========================================================= */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--font-sans) !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

p, li, label, span, div {
    font-family: var(--font-sans) !important;
}

code, pre, .mono, kbd,
[data-testid="stCode"],
[data-testid="stCodeBlock"] {
    font-family: var(--font-mono) !important;
    font-size: 0.875rem;
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}

/* Eyebrow labels — small uppercase tracking label (e.g. "· THE PROBLEM ·") */
.eyebrow {
    font-family: var(--font-sans);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    display: block;
}
.eyebrow-red  { color: var(--red-light); }
.eyebrow-blue { color: var(--blue-light); }

/* Mono stat number */
.stat-mono {
    font-family: var(--font-mono);
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1;
    color: var(--text-primary);
}

/* =========================================================
   SIDEBAR
   ========================================================= */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem;
    padding: 5px 0;
    color: var(--text-secondary) !important;
    transition: color var(--transition);
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}

/* =========================================================
   CARDS
   ========================================================= */

/* Base card */
.fd-card, .tg-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color var(--transition), box-shadow var(--transition),
                background var(--transition);
    position: relative;
    overflow: hidden;
}

.fd-card:hover, .tg-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-strong);
}

/* Red-accented card — HIGH/CRITICAL risk, fraud, alert */
.fd-card-red, .tg-card-danger {
    border-color: var(--border-red);
}
.fd-card-red:hover, .tg-card-danger:hover {
    border-color: var(--red);
    box-shadow: 0 0 8px var(--glow-red), 0 0 0.8px var(--red);
}

/* Blue-accented card — LOW risk, verified, safe, trust */
.fd-card-blue, .tg-card-success {
    border-color: var(--border-blue);
}
.fd-card-blue:hover, .tg-card-success:hover {
    border-color: var(--blue);
    box-shadow: 0 0 8px var(--glow-blue), 0 0 0.8px var(--blue);
}

/* Amber card — MEDIUM risk, warning */
.fd-card-amber, .tg-card-warning {
    border-color: var(--border-amber);
}
.fd-card-amber:hover, .tg-card-warning:hover {
    border-color: var(--amber);
    box-shadow: 0 0 8px var(--amber-dim), 0 0 0.8px var(--amber);
}

/* Inline accent stripe on left edge */
.fd-card-red::before, .tg-card-danger::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--red);
    border-radius: var(--radius-lg) 0 0 var(--radius-lg);
}
.fd-card-blue::before, .tg-card-success::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--blue);
    border-radius: var(--radius-lg) 0 0 var(--radius-lg);
}
.fd-card-amber::before, .tg-card-warning::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--amber);
    border-radius: var(--radius-lg) 0 0 var(--radius-lg);
}

/* =========================================================
   STAT STRIP — horizontal band of 4 big numbers
   ========================================================= */
.stat-strip {
    display: flex;
    gap: 0;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-bottom: 1rem;
}

.stat-strip-item {
    flex: 1;
    padding: 1.4rem 1rem 1.2rem;
    text-align: center;
    border-right: 1px solid var(--border);
    transition: background var(--transition);
    position: relative;
}
.stat-strip-item:last-child { border-right: none; }
.stat-strip-item:hover { background: var(--bg-card-hover); }

.stat-strip-value {
    font-family: var(--font-mono);
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.04em;
    line-height: 1;
    color: var(--text-primary);
    display: block;
    margin-bottom: 0.35rem;
}
.stat-strip-value.red  { color: var(--red-light); }
.stat-strip-value.blue { color: var(--blue-light); }
.stat-strip-value.amber{ color: var(--amber); }

.stat-strip-label {
    font-family: var(--font-sans);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-secondary);
    display: block;
}

/* =========================================================
   COMPARISON BAR — label + thin bar + value (horizontal)
   ========================================================= */
.comp-bar-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
    padding: 0.4rem 0;
}

.comp-bar-label {
    font-family: var(--font-sans);
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-secondary);
    min-width: 140px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.comp-bar-track {
    flex: 1;
    height: 4px;
    background: var(--bg-elevated);
    border-radius: 2px;
    overflow: hidden;
    position: relative;
}

.comp-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
    background: var(--blue);
}
.comp-bar-fill.red   { background: var(--red); }
.comp-bar-fill.blue  { background: var(--blue); }
.comp-bar-fill.amber { background: var(--amber); }

.comp-bar-value {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-primary);
    min-width: 56px;
    text-align: right;
    flex-shrink: 0;
}

/* =========================================================
   RISK BADGES
   ========================================================= */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3em;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    line-height: 1.6;
}

.badge-critical {
    background: rgba(220, 38, 38, 0.18);
    color: #F87171;
    border: 1px solid rgba(220, 38, 38, 0.4);
}
.badge-high {
    background: rgba(217, 119, 6, 0.18);
    color: #FCA34D;
    border: 1px solid rgba(217, 119, 6, 0.4);
}
.badge-medium {
    background: rgba(37, 99, 235, 0.18);
    color: #60A5FA;
    border: 1px solid rgba(37, 99, 235, 0.4);
}
.badge-low {
    background: rgba(22, 163, 74, 0.15);
    color: #4ADE80;
    border: 1px solid rgba(22, 163, 74, 0.35);
}

/* =========================================================
   SIGNAL ALERTS (inline flag chips in risk panel)
   ========================================================= */
.signal-alert {
    display: flex;
    gap: 0.65rem;
    align-items: flex-start;
    background: var(--bg-elevated);
    border-left: 3px solid var(--red);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: var(--text-primary);
}
.signal-alert.amber { border-left-color: var(--amber); }
.signal-alert.blue  { border-left-color: var(--blue);  }
.signal-alert.safe  {
    border-left-color: #16A34A;
    background: rgba(22, 163, 74, 0.06);
}
.signal-alert-tag {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--red-light);
    background: var(--red-dim);
    padding: 1px 6px;
    border-radius: 3px;
    flex-shrink: 0;
    margin-top: 1px;
}
.signal-alert.amber .signal-alert-tag { color: var(--amber); background: var(--amber-dim); }
.signal-alert.blue  .signal-alert-tag { color: var(--blue-light); background: var(--blue-dim); }
.signal-alert.safe  .signal-alert-tag { color: #4ADE80; background: rgba(22,163,74,0.12); }

/* =========================================================
   TOP HEADER BAR
   ========================================================= */
.topbar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 0.85rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.topbar-left {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.topbar-icon {
    width: 36px; height: 36px;
    background: var(--red-dim);
    border: 1px solid var(--border-red);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

.topbar-title {
    font-family: var(--font-sans);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.2;
}

.topbar-subtitle {
    font-family: var(--font-sans);
    font-size: 0.76rem;
    color: var(--text-secondary);
    line-height: 1.3;
}

.topbar-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.topbar-pill {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--blue-light);
    background: var(--blue-dim);
    border: 1px solid var(--border-blue);
    padding: 3px 10px;
    border-radius: 999px;
    text-transform: uppercase;
}
.topbar-pill.red {
    color: var(--red-light);
    background: var(--red-dim);
    border-color: var(--border-red);
}

/* =========================================================
   SECTION HEADER (eyebrow + title combo)
   ========================================================= */
.section-header {
    margin-bottom: 1rem;
}
.section-header .eyebrow {
    margin-bottom: 0.2rem;
}
.section-header h3 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    padding: 0 !important;
    letter-spacing: -0.02em;
}

/* Legacy .section-title compat */
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* =========================================================
   PROBABILITY GAUGE (fallback text display)
   ========================================================= */
.gauge-wrap  { text-align: center; padding: 1rem 0; }
.gauge-pct   {
    font-family: var(--font-mono);
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.04em;
    color: var(--text-primary);
}
.gauge-label {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 6px;
    font-family: var(--font-sans);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* =========================================================
   STREAMLIT NATIVE WIDGET OVERRIDES
   ========================================================= */

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    transition: border-color var(--transition) !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px var(--blue-dim) !important;
    outline: none !important;
}

/* Labels */
[data-testid="stWidgetLabel"] p,
label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    font-family: var(--font-sans) !important;
}

/* Buttons: broaden selectors to capture various Streamlit DOM wrappers
   (Streamlit sometimes nests the native <button> inside extra <div>s for layout)
*/
.stButton > button,
.stButton button,
.stButton > div > button,
div.stButton button {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    transition: background var(--transition), border-color var(--transition),
                box-shadow var(--transition) !important;
    letter-spacing: 0.01em;
}
.stButton > button:hover,
.stButton button:hover,
.stButton > div > button:hover,
div.stButton button:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--blue) !important;
    /* remove heavy glow on non-primary buttons */
    box-shadow: none !important;
    color: var(--text-primary) !important;
}
.stButton > button:active,
.stButton button:active,
.stButton > div > button:active,
div.stButton button:active {
    transform: translateY(1px);
}

/* Primary/type="primary" button — red accent */
.stButton > button[kind="primary"],
.stButton button[kind="primary"],
.stButton > div > button[kind="primary"],
div.stButton button[kind="primary"] {
    background: var(--red-dim) !important;
    border-color: var(--border-red) !important;
    color: var(--red-light) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton button[kind="primary"]:hover,
.stButton > div > button[kind="primary"]:hover,
div.stButton button[kind="primary"]:hover {
    background: var(--red) !important;
    color: #fff !important;
    box-shadow: 0 0 6px var(--glow-red) !important;
}

/* App-scoped fallback: force any button inside the Streamlit app root to use
   the dark button styling. This catches edge cases where Streamlit renders
   buttons without the `.stButton` wrapper or with inline styles.
*/
[data-testid="stApp"] button {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-sans) !important;
    padding: 0.4rem 0.9rem !important;
    transition: background var(--transition), box-shadow var(--transition) !important;
}
[data-testid="stApp"] button:hover {
    background: var(--bg-card-hover) !important;
    box-shadow: none !important;
}
[data-testid="stApp"] button[kind="primary"], [data-testid="stApp"] button[data-kind="primary"] {
    background: var(--red-dim) !important;
    border-color: var(--border-red) !important;
    color: var(--red-light) !important;
}
[data-testid="stApp"] button[kind="primary"]:hover, [data-testid="stApp"] button[data-kind="primary"]:hover {
    background: var(--red) !important;
    color: #fff !important;
    box-shadow: 0 0 6px var(--glow-red) !important;
}

/* Sharp / asymmetric card variant to introduce layout variety */
.fd-card-sharp {
    border-radius: 6px 6px 2px 2px !important;
    padding: 1rem 1rem !important;
}

/* Sliders */
[data-testid="stSlider"] [role="slider"] {
    background: var(--blue) !important;
    border-color: var(--blue) !important;
}
[data-testid="stSlider"] > div > div > div {
    background: var(--border) !important;
}

/* Checkboxes */
[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: var(--blue) !important;
}

/* Selectbox dropdown */
[data-testid="stSelectbox"] {
    color: var(--text-primary) !important;
}

/* DataFrames / Tables */
[data-testid="stDataFrame"],
.stDataFrame {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-family: var(--font-sans) !important;
}

/* Metric widget */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    color: var(--text-primary) !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg-surface) !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    gap: 0.1rem;
    padding: 0 0.5rem !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    transition: color var(--transition), background var(--transition) !important;
    padding: 0.5rem 0.85rem !important;
    border: none !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
    border-bottom: 2px solid var(--blue) !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: var(--text-primary) !important;
}

/* Info / Success / Warning / Error alerts */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    font-family: var(--font-sans) !important;
}

/* Dividers */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1rem 0 !important;
}

/* Caption text */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: var(--text-dim) !important;
    font-size: 0.78rem !important;
}

/* Plotly chart backgrounds (match dark theme) */
[data-testid="stPlotlyChart"] {
    background: transparent !important;
}

/* Radio buttons in sidebar */
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-surface); }
::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
/* NUMBER INPUTS — ensure numeric text and steppers are high-contrast */
[data-testid="stNumberInput"] input,
.stNumberInput input {
    color: var(--text-primary) !important;
    background: var(--bg-elevated) !important;
}
[data-testid="stNumberInput"] input:disabled,
.stNumberInput input:disabled {
    color: var(--text-primary) !important;
    opacity: 1 !important; /* override browsers that dim disabled inputs */
}
[data-testid="stNumberInput"] button,
.stNumberInput button,
[data-testid="stNumberInput"] svg,
.stNumberInput svg {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
}

/* DATAFRAME / AG-GRID descendants — ensure interactive dataframes inherit dark theme */
[data-testid="stDataFrame"] .ag-root,
.stDataFrame .ag-root,
[data-testid="stDataFrame"] .ag-body-viewport,
.stDataFrame .ag-body-viewport,
[data-testid="stDataFrame"] .ag-center-cols-viewport,
.stDataFrame .ag-center-cols-viewport,
[data-testid="stDataFrame"] .ag-cell,
.stDataFrame .ag-cell,
[data-testid="stDataFrame"] .ag-header,
.stDataFrame .ag-header,
[data-testid="stDataFrame"] .ag-header-cell,
.stDataFrame .ag-header-cell {
    background: transparent !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}
[data-testid="stDataFrame"] .ag-header, .stDataFrame .ag-header {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
}
[data-testid="stDataFrame"] .ag-row, .stDataFrame .ag-row {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}
[data-testid="stDataFrame"] *, .stDataFrame * {
    font-family: var(--font-sans) !important;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
/* NUMBER INPUTS — ensure numeric text and steppers are high-contrast */
[data-testid="stNumberInput"] input,
.stNumberInput input {
    color: var(--text-primary) !important;
    background: var(--bg-elevated) !important;
}
[data-testid="stNumberInput"] input:disabled,
.stNumberInput input:disabled {
    color: var(--text-primary) !important;
    opacity: 1 !important; /* override browsers that dim disabled inputs */
}
[data-testid="stNumberInput"] button,
.stNumberInput button,
[data-testid="stNumberInput"] svg,
.stNumberInput svg {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
}

/* DATAFRAME / AG-GRID descendants — ensure interactive dataframes inherit dark theme */
[data-testid="stDataFrame"] .ag-root,
.stDataFrame .ag-root,
[data-testid="stDataFrame"] .ag-body-viewport,
.stDataFrame .ag-body-viewport,
[data-testid="stDataFrame"] .ag-center-cols-viewport,
.stDataFrame .ag-center-cols-viewport,
[data-testid="stDataFrame"] .ag-cell,
.stDataFrame .ag-cell,
[data-testid="stDataFrame"] .ag-header,
.stDataFrame .ag-header,
[data-testid="stDataFrame"] .ag-header-cell,
.stDataFrame .ag-header-cell {
    background: transparent !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}
[data-testid="stDataFrame"] .ag-header, .stDataFrame .ag-header {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
}
[data-testid="stDataFrame"] .ag-row, .stDataFrame .ag-row {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}
[data-testid="stDataFrame"] *, .stDataFrame * {
    font-family: var(--font-sans) !important;
}
/* -------------------------------------------------------------------------
   Native Streamlit widget dark-theme overrides (fix unstyled white bars)
   - stDownloadButton
   - stFileUploader (dropzone + inner button)
   - stJson (JSON viewer container)
   - st.table / [data-testid="stTable"]
   ------------------------------------------------------------------------*/

[data-testid="stDownloadButton"] > button,
.stDownloadButton > button {
    background: linear-gradient(var(--blue-dim), var(--blue-dim)) , var(--bg-elevated) !important;
    background-blend-mode: overlay;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.45rem 0.9rem !important;
    font-family: var(--font-sans) !important;
    transition: box-shadow var(--transition), background var(--transition) !important;
}
[data-testid="stDownloadButton"] > button:hover,
.stDownloadButton > button:hover {
    background: var(--bg-card-hover) !important;
    box-shadow: none !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    padding: 0.6rem !important;
}
[data-testid="stFileUploader"] * { color: inherit !important; font-family: var(--font-sans) !important; }
[data-testid="stFileUploader"] button {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.35rem 0.7rem !important;
}

[data-testid="stJson"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    padding: 0.5rem !important;
}
[data-testid="stJson"] pre, [data-testid="stJson"] code {
    background: transparent !important;
    color: var(--text-primary) !important;
}

/* st.table / stTable dark styling for pandas tables */
.stTable, [data-testid="stTable"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.15rem !important;
}
.stTable table, [data-testid="stTable"] table {
    border-collapse: collapse !important;
    width: 100% !important;
}
.stTable th, .stTable td, [data-testid="stTable"] th, [data-testid="stTable"] td {
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    padding: 0.45rem 0.6rem !important;
    background: transparent !important;
    font-family: var(--font-sans) !important;
}
.stTable th, [data-testid="stTable"] th {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* Image frame to visually contain white-background PNGs */
.img-frame {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}
"""

# ---------------------------------------------------------------------------
# Python helpers for building HTML components
# ---------------------------------------------------------------------------

def eyebrow(text: str, color: str = "") -> str:
    """Return an eyebrow label HTML string.
    
    Args:
        text:  label text, e.g. "· THE PROBLEM ·"
        color: "" | "red" | "blue"
    """
    cls = f"eyebrow eyebrow-{color}" if color else "eyebrow"
    return f'<span class="{cls}">{text}</span>'


def stat_strip(items: list[dict]) -> str:
    """Return a horizontal stat-strip HTML block.
    
    Args:
        items: list of dicts with keys:
               value (str), label (str), color (str: ""/"red"/"blue"/"amber")
    
    Example:
        stat_strip([
            {"value": "98.7%", "label": "Precision", "color": "blue"},
            {"value": "4,251", "label": "Fraud Blocked", "color": "red"},
            {"value": "0.3s",  "label": "Avg Latency",  "color": ""},
            {"value": "PKR 2.1B", "label": "Value Protected", "color": "amber"},
        ])
    """
    cells = []
    for item in items:
        color_cls = item.get("color", "")
        val_cls = f"stat-strip-value {color_cls}".strip()
        cells.append(
            f'<div class="stat-strip-item">'
            f'<span class="{val_cls}">{item["value"]}</span>'
            f'<span class="stat-strip-label">{item["label"]}</span>'
            f'</div>'
        )
    return f'<div class="stat-strip">{"".join(cells)}</div>'


def comp_bar(label: str, value: float, max_value: float,
             display: str = "", color: str = "blue") -> str:
    """Return a single comparison-bar row HTML.
    
    Args:
        label:     left label text
        value:     numeric value for bar width
        max_value: maximum value (100% width)
        display:   right-side text override (defaults to str(value))
        color:     "red" | "blue" | "amber"
    """
    pct = min(100.0, (value / max_value * 100) if max_value else 0)
    display = display or f"{value:,.2f}"
    return (
        f'<div class="comp-bar-row">'
        f'  <span class="comp-bar-label">{label}</span>'
        f'  <div class="comp-bar-track">'
        f'    <div class="comp-bar-fill {color}" style="width:{pct:.1f}%"></div>'
        f'  </div>'
        f'  <span class="comp-bar-value">{display}</span>'
        f'</div>'
    )


def comp_bar_group(rows: list[dict], max_value: float = None) -> str:
    """Return a group of comparison-bar rows as an HTML block.
    
    Args:
        rows: list of dicts: label, value, display (opt), color (opt)
        max_value: shared max; if None, uses the max value in rows
    """
    if max_value is None and rows:
        max_value = max(r.get("value", 0) for r in rows) or 1
    html = '<div class="fd-card" style="padding:1rem 1.25rem">'
    for r in rows:
        html += comp_bar(
            label=r.get("label", ""),
            value=r.get("value", 0),
            max_value=max_value,
            display=r.get("display", ""),
            color=r.get("color", "blue"),
        )
    html += '</div>'
    return html


def section_header(title: str, eyebrow_text: str = "", color: str = "") -> str:
    """Return a section header HTML with optional eyebrow label."""
    ew = ""
    if eyebrow_text:
        ew = eyebrow(eyebrow_text, color)
    return (
        f'<div class="section-header">'
        f'{ew}'
        f'<h3>{title}</h3>'
        f'</div>'
    )


def topbar_html(title: str, subtitle: str, pills: list[dict] = None) -> str:
    """Return the top header bar HTML.
    
    Args:
        title:    main title
        subtitle: smaller subtitle line
        pills:    list of {"text": "...", "color": "blue"|"red"} dicts
    """
    pill_html = ""
    if pills:
        for p in pills:
            cls = f'topbar-pill {p.get("color","")}'.strip()
            pill_html += f'<span class="{cls}">{p["text"]}</span>'

    return (
        f'<div class="topbar">'
        f'  <div class="topbar-left">'
        f'    <div class="topbar-icon" style="font-family: var(--font-mono); font-size: 0.82rem; font-weight: 700; color: var(--red-light);">FD</div>'
        f'    <div>'
        f'      <div class="topbar-title">{title}</div>'
        f'      <div class="topbar-subtitle">{subtitle}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="topbar-right">{pill_html}</div>'
        f'</div>'
    )


def signal_alert(tag: str, title: str, detail: str,
                 color: str = "") -> str:
    """Return a signal/flag alert row HTML.
    
    Args:
        tag:    short tag (e.g. "DRAIN", "CTR", "MULE")
        title:  bold label
        detail: description text
        color:  "" (red) | "amber" | "blue" | "safe"
    """
    cls = f"signal-alert {color}".strip()
    return (
        f'<div class="{cls}">'
        f'  <span class="signal-alert-tag">{tag}</span>'
        f'  <span><b>{title}</b>: {detail}</span>'
        f'</div>'
    )


def risk_badge(tier: str) -> str:
    """Return a risk badge span HTML."""
    return f'<span class="badge badge-{tier.lower()}">{tier}</span>'


# ---------------------------------------------------------------------------
# Main inject function
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    """Inject the full design-system CSS into the Streamlit app.
    
    Call this once, immediately after st.set_page_config(), before any
    other st.* calls that render visible content.
    """
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
