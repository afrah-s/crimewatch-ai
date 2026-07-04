from pathlib import Path
from html import escape
from io import StringIO
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

try:
    import streamlit_shadcn_ui as ui
except Exception:
    ui = None

# -----------------------------
# App configuration
# -----------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

DISTRICT_FILE = DATA_DIR / "ka-district-wise-2025(1).csv"
CRIME_REVIEW_FILE = DATA_DIR / "crime_review_for_the_month_of_december_2025_9(1).csv"
STATION_FILE = DATA_DIR / "police_stations_processed.csv"
WOMEN_CHILD_SCST_FILE = DATA_DIR / "ka-crimes-women-children-scssts.csv"

st.set_page_config(
    page_title="CrimeWatch AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Authentication state
# -----------------------------

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "role" not in st.session_state:
    st.session_state["role"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None

DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"

# -----------------------------
# Visual theme
# -----------------------------

st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --cw-primary: #43B0F8;
    --cw-primary-dim: rgba(67,176,248,0.16);
    --cw-secondary: #8A1224;
    --cw-secondary-bright: #E63946;
    --cw-tertiary: #020617;
    --cw-tertiary-soft: #0B1220;
    --cw-neutral: #727877;
    --cw-text: #E8EBEE;
    --cw-text-dim: #9BA3AE;
    --font-body: 'Inter', -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-body);
}

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(67,176,248,0.10), transparent 30%),
        radial-gradient(circle at 90% 100%, rgba(138,18,36,0.14), transparent 32%),
        linear-gradient(160deg, #020617 0%, #05070D 55%, #020617 100%);
    color: var(--cw-text);
    font-family: var(--font-body);
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0F172A 100%);
    border-right: 1px solid rgba(148,163,184,0.15);
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #CBD5E1 !important;
    font-weight: 700;
}

/* Hero — styled as a top app bar, matching the CrimeWatch AI header pattern */
.hero-card {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(135deg, rgba(8,12,22,0.97), rgba(11,18,32,0.94)),
        radial-gradient(circle at top right, rgba(67,176,248,0.20), transparent 38%);
    border: 1px solid rgba(67,176,248,0.22);
    border-radius: 22px;
    padding: 26px 32px;
    margin-bottom: 22px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}

.hero-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
}

.hero-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.hero-brand-icon {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: var(--cw-primary-dim);
    border: 1px solid rgba(67,176,248,0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
}

.hero-brand-name {
    font-family: var(--font-body);
    font-size: 19px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: 0.2px;
}

.hero-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--cw-primary), var(--cw-secondary));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    border: 1px solid rgba(255,255,255,0.15);
}

.hero-topline {
    color: var(--cw-primary);
    font-family: var(--font-mono);
    font-size: 12px;
    letter-spacing: 1.6px;
    font-weight: 700;
    text-transform: uppercase;
}

.hero-title {
    font-size: 40px;
    font-weight: 900;
    color: #F8FAFC;
    margin-top: 6px;
    margin-bottom: 8px;
    line-height: 1.08;
}

.hero-subtitle {
    font-size: 15px;
    color: var(--cw-text-dim);
    max-width: 980px;
    line-height: 1.65;
}

.hero-footer {
    margin-top: 20px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.badge-soft {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 12px;
    border-radius: 999px;
    background: var(--cw-primary-dim);
    border: 1px solid rgba(67,176,248,0.3);
    color: #BEE3FB;
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 700;
}

/* Headings */
.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #F8FAFC;
    margin-top: 22px;
    margin-bottom: 5px;
}

.section-subtitle {
    font-size: 13.5px;
    color: var(--cw-text-dim);
    margin-bottom: 18px;
    line-height: 1.5;
}

/* KPI cards */
.kpi-card {
    background:
        linear-gradient(145deg, rgba(9,13,22,0.98), rgba(14,20,32,0.9)),
        radial-gradient(circle at top right, rgba(67,176,248,0.14), transparent 35%);
    border: 1px solid rgba(114,120,119,0.22);
    border-radius: 18px;
    padding: 20px;
    min-height: 154px;
    box-shadow: 0 13px 38px rgba(0,0,0,0.4);
    transition: 0.18s ease-in-out;
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(67,176,248,0.5);
    box-shadow: 0 20px 52px rgba(2,6,23,0.6);
}

.kpi-label {
    color: var(--cw-text-dim);
    font-family: var(--font-mono);
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.1px;
}

.kpi-value {
    color: #F8FAFC;
    font-size: 30px;
    font-weight: 900;
    margin-top: 12px;
    margin-bottom: 8px;
    line-height: 1.1;
}

.kpi-desc {
    color: var(--cw-primary);
    font-size: 12.5px;
    font-weight: 600;
    line-height: 1.35;
}

.kpi-delta-up { color: #FCA5A5; font-size: 12px; font-weight: 800; margin-left: 6px; }
.kpi-delta-down { color: #86EFAC; font-size: 12px; font-weight: 800; margin-left: 6px; }

/* Glass panels */
.glass-panel {
    background: rgba(15,23,42,0.74);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 22px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 12px 34px rgba(0,0,0,0.32);
}

/* NL search box — styled like the "Ask CrimeWatch AI" command bar */
.nl-box {
    background: linear-gradient(135deg, rgba(9,13,22,0.95), rgba(11,18,32,0.85));
    border: 1px solid rgba(67,176,248,0.22);
    border-radius: 18px;
    padding: 16px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.nl-icon {
    width: 30px;
    height: 30px;
    min-width: 30px;
    border-radius: 9px;
    background: var(--cw-primary-dim);
    border: 1px solid rgba(67,176,248,0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}

.nl-title {
    color: #F1F5F9;
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 2px;
}

.nl-subtitle {
    color: var(--cw-text-dim);
    font-size: 12.5px;
}

/* Insights — chat-bubble style AI callout */
.ai-insight {
    background: linear-gradient(135deg, rgba(9,13,22,0.96), rgba(15,23,20,0.6));
    border: 1px solid rgba(67,176,248,0.3);
    border-radius: 16px;
    padding: 18px 20px;
    color: var(--cw-text);
    margin: 16px 0;
    box-shadow: 0 12px 32px rgba(2,6,23,0.4);
    display: flex;
    gap: 12px;
    align-items: flex-start;
}

.ai-insight-icon {
    width: 26px;
    height: 26px;
    min-width: 26px;
    border-radius: 50%;
    background: var(--cw-primary-dim);
    border: 1px solid rgba(67,176,248,0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    margin-top: 1px;
}

.ai-insight-body { flex: 1; }

.ai-insight-title {
    color: var(--cw-primary);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    margin-bottom: 6px;
}

.ai-insight-text {
    color: #DCE3EA;
    line-height: 1.6;
    font-size: 14px;
}

/* Alerts — spike-alert style with severity badge, matches prototype's "HIGH RISK" cards */
.alert-card {
    position: relative;
    background: linear-gradient(135deg, rgba(23,5,8,0.94), rgba(138,18,36,0.42));
    border: 1px solid rgba(230,57,70,0.4);
    border-left: 6px solid var(--cw-secondary-bright);
    border-radius: 16px;
    padding: 18px 20px;
    color: #FBE4E6;
    margin-bottom: 14px;
    box-shadow: 0 10px 28px rgba(69,10,10,0.3);
}

.alert-badge {
    position: absolute;
    top: 16px;
    right: 18px;
    background: rgba(230,57,70,0.18);
    border: 1px solid rgba(230,57,70,0.5);
    color: #FCA5AD;
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding: 4px 10px;
    border-radius: 999px;
    text-transform: uppercase;
}

.alert-title {
    font-size: 16px;
    font-weight: 800;
    color: #FFF1F2;
    margin-bottom: 6px;
    padding-right: 90px;
}

.alert-text {
    font-size: 13.5px;
    color: #F3D3D6;
    line-height: 1.55;
}

.info-card {
    background: linear-gradient(135deg, rgba(9,13,22,0.94), rgba(11,24,42,0.5));
    border: 1px solid rgba(67,176,248,0.3);
    border-left: 6px solid var(--cw-primary);
    border-radius: 16px;
    padding: 18px 20px;
    color: #DDEEFB;
    margin-bottom: 14px;
}

/* Risk panels */
.risk-panel {
    border-radius: 18px;
    padding: 18px;
    margin: 16px 0;
    box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}

.risk-high-panel {
    background: linear-gradient(135deg, rgba(127,29,29,0.74), rgba(69,10,10,0.62));
    border: 1px solid rgba(248,113,113,0.44);
    border-left: 6px solid #EF4444;
}

.risk-medium-panel {
    background: linear-gradient(135deg, rgba(120,53,15,0.72), rgba(69,26,3,0.58));
    border: 1px solid rgba(251,191,36,0.42);
    border-left: 6px solid #F59E0B;
}

.risk-low-panel {
    background: linear-gradient(135deg, rgba(20,83,45,0.72), rgba(5,46,22,0.58));
    border: 1px solid rgba(74,222,128,0.36);
    border-left: 6px solid #22C55E;
}

.risk-title {
    color: #F8FAFC;
    font-weight: 900;
    font-size: 17px;
    margin-bottom: 6px;
}

.risk-text {
    color: #E5E7EB;
    font-size: 14px;
    line-height: 1.5;
}

.risk-pill-high, .risk-pill-medium, .risk-pill-low {
    display: inline-block;
    border-radius: 999px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .5px;
}

.risk-pill-high {
    background: rgba(239,68,68,0.16);
    color: #FCA5A5;
    border: 1px solid rgba(239,68,68,0.42);
}

.risk-pill-medium {
    background: rgba(245,158,11,0.16);
    color: #FCD34D;
    border: 1px solid rgba(245,158,11,0.42);
}

.risk-pill-low {
    background: rgba(34,197,94,0.16);
    color: #86EFAC;
    border: 1px solid rgba(34,197,94,0.42);
}

/* Streamlit element polish */
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 18px;
    padding: 16px;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: none;
    background: rgba(9,13,22,0.6);
    padding: 6px;
    border-radius: 16px;
    border: 1px solid rgba(114,120,119,0.18);
}

button[data-baseweb="tab"] {
    background: transparent;
    border-radius: 11px;
    margin-right: 0;
    border: 1px solid transparent;
    color: var(--cw-text-dim);
    font-weight: 600;
    font-size: 13.5px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--cw-primary-dim);
    border: 1px solid rgba(67,176,248,0.5);
    color: #F8FAFC;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none;
}

.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
    background-color: rgba(9,13,22,0.9) !important;
    border-color: rgba(114,120,119,0.3) !important;
    color: #F8FAFC !important;
    border-radius: 10px !important;
}

.stTextInput input:focus {
    border-color: var(--cw-primary) !important;
    box-shadow: 0 0 0 1px rgba(67,176,248,0.4) !important;
}

.stButton button, .stDownloadButton button {
    background: var(--cw-primary-dim) !important;
    border: 1px solid rgba(67,176,248,0.4) !important;
    color: #DFF1FE !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton button:hover, .stDownloadButton button:hover {
    background: rgba(67,176,248,0.28) !important;
    border-color: var(--cw-primary) !important;
    color: #FFFFFF !important;
}

hr {
    border-color: rgba(114,120,119,0.2);
}

.footer-note {
    color: var(--cw-neutral);
    font-family: var(--font-mono);
    font-size: 11.5px;
    text-align: center;
    margin-top: 30px;
    padding-top: 16px;
    border-top: 1px solid rgba(114,120,119,0.18);
}

/* Geospatial legend card */
.legend-card {
    background: rgba(9,13,22,0.85);
    border: 1px solid rgba(114,120,119,0.25);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 16px;
}

.legend-heading {
    color: #F1F5F9;
    font-size: 13.5px;
    font-weight: 700;
    margin-bottom: 12px;
}

.legend-row {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
}

.legend-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(15,20,30,0.6);
    border: 1px solid rgba(114,120,119,0.2);
    border-radius: 10px;
    padding: 8px 12px;
    min-width: 190px;
}

.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

.legend-label {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--cw-text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.legend-value {
    font-size: 13px;
    color: #F1F5F9;
    font-weight: 700;
}
</style>
""")

# -----------------------------
# Helper UI functions
# -----------------------------

def hero_header(status_label="Nominal", is_critical=False):
    status_color = "#FCA5AD" if is_critical else "#86EFAC"
    status_bg = "rgba(230,57,70,0.14)" if is_critical else "rgba(34,197,94,0.12)"
    status_border = "rgba(230,57,70,0.4)" if is_critical else "rgba(34,197,94,0.35)"
    dot_color = "#E63946" if is_critical else "#22C55E"
    st.html(f"""
    <div class="hero-card">
        <div class="hero-topbar">
            <div class="hero-brand">
                <div class="hero-brand-icon">🛡️</div>
                <div class="hero-brand-name">CrimeWatch AI</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="display:inline-flex; align-items:center; gap:6px; background:{status_bg}; border:1px solid {status_border}; color:{status_color}; font-family:var(--font-mono); font-size:11px; font-weight:700; letter-spacing:0.6px; padding:5px 11px; border-radius:999px; text-transform:uppercase;">
                    <span style="width:6px; height:6px; border-radius:50%; background:{dot_color}; display:inline-block;"></span>
                    {escape(status_label)}
                </span>
                <div class="hero-avatar">👤</div>
            </div>
        </div>
        <div class="hero-topline">AI-Driven Crime Analytics & Visualization Platform</div>
        <div class="hero-title">Advanced Threat Mitigation System</div>
        <div class="hero-subtitle">
            A modern crime intelligence prototype that transforms fragmented Karnataka crime records into
            interactive dashboards, district drilldowns, police-station maps, trend alerts, anomaly detection,
            predictive risk scoring, and AI-assisted investigation insights.
        </div>
        <div class="hero-footer">
            <span class="badge-soft">District Drilldowns</span>
            <span class="badge-soft">Hotspot Intelligence</span>
            <span class="badge-soft">Anomaly Alerts</span>
            <span class="badge-soft">Risk Scoring</span>
            <span class="badge-soft">Natural Language Search</span>
            <span class="badge-soft">CSV Exports</span>
        </div>
    </div>
    """)


def section_header(title, subtitle=""):
    st.html(f"""
    <div class="section-title">{escape(str(title))}</div>
    <div class="section-subtitle">{escape(str(subtitle))}</div>
    """)


def kpi_card(label, value, description, delta=None, delta_is_bad_if_up=True):
    delta_html = ""
    if delta is not None:
        try:
            up = float(delta) >= 0
        except Exception:
            up = True
        bad = up if delta_is_bad_if_up else (not up)
        cls = "kpi-delta-up" if bad else "kpi-delta-down"
        arrow = "▲" if up else "▼"
        delta_html = f'<span class="{cls}">{arrow} {abs(float(delta)):.1f}%</span>'
    st.html(f"""
    <div class="kpi-card">
        <div class="kpi-label">{escape(str(label))}</div>
        <div class="kpi-value">{escape(str(value))}{delta_html}</div>
        <div class="kpi-desc">{escape(str(description))}</div>
    </div>
    """)


def ai_insight(text):
    st.html(f"""
    <div class="ai-insight">
        <div class="ai-insight-icon">◆</div>
        <div class="ai-insight-body">
            <div class="ai-insight-title">AI Intelligence Insight</div>
            <div class="ai-insight-text">{escape(str(text))}</div>
        </div>
    </div>
    """)


def alert_card(title, text, badge="HIGH RISK"):
    badge_html = f'<div class="alert-badge">{escape(str(badge))}</div>' if badge else ""
    st.html(f"""
    <div class="alert-card">
        {badge_html}
        <div class="alert-title">{escape(str(title))}</div>
        <div class="alert-text">{escape(str(text))}</div>
    </div>
    """)


def info_card(title, text):
    st.html(f"""
    <div class="info-card">
        <div class="alert-title">{escape(str(title))}</div>
        <div class="alert-text">{escape(str(text))}</div>
    </div>
    """)


def risk_panel(level, score, reason):
    level = str(level)
    if level == "High":
        css = "risk-high-panel"
        pill = "risk-pill-high"
    elif level == "Medium":
        css = "risk-medium-panel"
        pill = "risk-pill-medium"
    else:
        css = "risk-low-panel"
        pill = "risk-pill-low"

    st.html(f"""
    <div class="risk-panel {css}">
        <div class="risk-title"><span class="{pill}">{escape(level.upper())} RISK</span> &nbsp; Score: {score:.1f}/100</div>
        <div class="risk-text">{escape(str(reason))}</div>
    </div>
    """)


def risk_pill_html(level):
    level = str(level)
    css = "risk-pill-high" if level == "High" else "risk-pill-medium" if level == "Medium" else "risk-pill-low"
    return f'<span class="{css}">{escape(level.upper())}</span>'


def apply_dark_layout(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#E5E7EB"),
        title_font=dict(size=20, color="#F8FAFC"),
        margin=dict(l=20, r=20, t=60, b=25),
        height=height
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def format_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "0"


def download_button(df, label, filename, key):
    """Reusable CSV export button for any dataframe shown on screen."""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label=f"⬇ {label}",
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=False,
    )


def legend_card(items):
    """items: list of (label, value, color_hex) tuples, matching the prototype's Intelligence Legend."""
    chips = "".join(
        f"""<div class="legend-chip">
                <span class="legend-dot" style="background:{color};"></span>
                <div>
                    <div class="legend-label">{escape(str(label))}</div>
                    <div class="legend-value">{escape(str(value))}</div>
                </div>
            </div>"""
        for label, value, color in items
    )
    st.html(f"""
    <div class="legend-card">
        <div class="legend-heading">Intelligence Legend</div>
        <div class="legend-row">{chips}</div>
    </div>
    """)


def missing_file_notice(path: Path, feature: str):
    alert_card(
        "Data File Missing",
        f"Expected file '{path.name}' was not found in the data folder, so the '{feature}' module cannot load. "
        f"Add the file to the data/ directory and refresh the app."
    )

# -----------------------------
# Login / splash page
# -----------------------------

def _inject_login_css():
    st.html("""
<style>
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }

.login-shell {
    display: flex;
    justify-content: center;
    padding-top: 28px;
}

.login-card {
    width: 420px;
    max-width: 92vw;
    background:
        linear-gradient(180deg, rgba(6,10,20,0.94), rgba(4,7,15,0.98)),
        radial-gradient(circle at 50% 0%, rgba(67,176,248,0.16), transparent 55%);
    border: 1px solid rgba(67,176,248,0.35);
    border-radius: 24px;
    padding: 38px 34px 30px 34px;
    box-shadow:
        0 0 0 1px rgba(67,176,248,0.08),
        0 30px 90px rgba(0,0,0,0.6),
        0 0 60px rgba(67,176,248,0.12);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.login-card::before {
    content: "";
    position: absolute;
    inset: -40%;
    background:
        repeating-linear-gradient(0deg, rgba(67,176,248,0.035) 0px, rgba(67,176,248,0.035) 1px, transparent 1px, transparent 26px),
        repeating-linear-gradient(90deg, rgba(67,176,248,0.035) 0px, rgba(67,176,248,0.035) 1px, transparent 1px, transparent 26px);
    pointer-events: none;
}

.login-shield {
    width: 92px;
    height: 92px;
    margin: 6px auto 18px auto;
    border-radius: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 46px;
    background: radial-gradient(circle at 50% 40%, rgba(67,176,248,0.28), rgba(67,176,248,0.02) 70%);
    border: 1.5px solid rgba(67,176,248,0.55);
    box-shadow: 0 0 40px rgba(67,176,248,0.35), inset 0 0 24px rgba(67,176,248,0.15);
    position: relative;
    z-index: 1;
}

.login-brand-title {
    font-family: var(--font-body);
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 1.5px;
    color: #F3FAFF;
    text-shadow: 0 0 22px rgba(67,176,248,0.55);
    margin-bottom: 4px;
    position: relative;
    z-index: 1;
}

.login-brand-subtitle {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2.6px;
    color: #7FC4F5;
    text-transform: uppercase;
    margin-bottom: 26px;
    position: relative;
    z-index: 1;
}

.login-scan-panel {
    background: rgba(67,176,248,0.06);
    border: 1px solid rgba(67,176,248,0.22);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 22px;
    text-align: left;
    position: relative;
    z-index: 1;
}

.login-scan-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--cw-primary);
    letter-spacing: 0.6px;
    margin-bottom: 8px;
}

.login-progress-track {
    width: 100%;
    height: 6px;
    border-radius: 999px;
    background: rgba(148,163,184,0.15);
    overflow: hidden;
    margin-bottom: 12px;
}

.login-progress-fill {
    height: 100%;
    width: 78%;
    border-radius: 999px;
    background: linear-gradient(90deg, #1E88C7, #43B0F8);
    box-shadow: 0 0 14px rgba(67,176,248,0.7);
    animation: login-scan-pulse 2.2s ease-in-out infinite;
}

@keyframes login-scan-pulse {
    0%   { opacity: 0.75; }
    50%  { opacity: 1; }
    100% { opacity: 0.75; }
}

.login-meta-grid {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 4px;
}

.login-meta-label {
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--cw-neutral);
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 2px;
}

.login-meta-value {
    font-family: var(--font-mono);
    font-size: 11px;
    color: #DCEBFA;
    font-weight: 600;
}

.login-secure-line {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: #6FE3A0;
    letter-spacing: 0.5px;
    text-align: center;
    margin-top: 10px;
    position: relative;
    z-index: 1;
}

.login-secure-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22C55E;
    margin-right: 6px;
    box-shadow: 0 0 8px rgba(34,197,94,0.8);
}

.login-form-label {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--cw-text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
    text-align: left;
    margin: 4px 0 -6px 2px;
    position: relative;
    z-index: 1;
}

.login-card [data-testid="stTextInput"] input,
.login-card [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: rgba(4,8,16,0.9) !important;
    border: 1px solid rgba(67,176,248,0.3) !important;
    color: #F1F8FE !important;
    border-radius: 10px !important;
}

.login-card [data-testid="stTextInput"] input:focus {
    border-color: var(--cw-primary) !important;
    box-shadow: 0 0 0 1px rgba(67,176,248,0.5) !important;
}

.login-card .stButton button {
    width: 100%;
    background: linear-gradient(135deg, #1E88C7, #43B0F8) !important;
    border: 1px solid rgba(67,176,248,0.6) !important;
    color: #04121C !important;
    font-weight: 800 !important;
    letter-spacing: 0.6px;
    border-radius: 11px !important;
    padding: 10px 0 !important;
    margin-top: 10px;
    transition: 0.18s ease-in-out;
}

.login-card .stButton button:hover {
    box-shadow: 0 0 26px rgba(67,176,248,0.65) !important;
    transform: translateY(-1px);
    border-color: #43B0F8 !important;
}

.login-footer-note {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--cw-neutral);
    text-align: center;
    margin-top: 18px;
    letter-spacing: 0.4px;
}
</style>
    """)


def login_page():
    """Renders the CrimeWatch AI splash/login screen and gates access to the dashboard."""
    _inject_login_css()

    left, mid, right = st.columns([1, 1.15, 1])
    with mid:
        st.html("""
<div class="login-shell">
    <div class="login-card">
        <div class="login-shield">🛡️</div>
        <div class="login-brand-title">CRIMEWATCH AI</div>
        <div class="login-brand-subtitle">Advanced Threat Mitigation System</div>
        <div class="login-scan-panel">
            <div class="login-scan-row">
                <span>SCANNING BIOMETRICS...</span>
                <span>78%</span>
            </div>
            <div class="login-progress-track">
                <div class="login-progress-fill"></div>
            </div>
            <div class="login-meta-grid">
                <div>
                    <div class="login-meta-label">Protocol</div>
                    <div class="login-meta-value">AES-256-GCM</div>
                </div>
                <div style="text-align:right;">
                    <div class="login-meta-label">Identity State</div>
                    <div class="login-meta-value">VERIFYING...</div>
                </div>
            </div>
        </div>
    </div>
</div>
        """)

        with st.form("login_form", clear_on_submit=False):
            st.html('<div class="login-form-label">Username</div>')
            username = st.text_input("Username", placeholder="Enter username", label_visibility="collapsed")
            st.html('<div class="login-form-label">Password</div>')
            password = st.text_input("Password", type="password", placeholder="Enter password", label_visibility="collapsed")
            st.html('<div class="login-form-label">Role</div>')
            role = st.selectbox("Role", ["Admin", "Investigator", "Analyst"], label_visibility="collapsed")
            submitted = st.form_submit_button("🔓 Authenticate & Enter")

        if submitted:
            if username == DEMO_USERNAME and password == DEMO_PASSWORD:
                st.session_state["authenticated"] = True
                st.session_state["role"] = role
                st.session_state["username"] = username
                st.rerun()
            else:
                st.html(
                    '<div style="text-align:center; color:#FCA5AD; font-family:var(--font-mono); '
                    'font-size:12.5px; font-weight:700; letter-spacing:0.4px; margin-top:8px;">'
                    '⚠ Invalid credentials</div>'
                )

        st.html("""
<div class="login-secure-line"><span class="login-secure-dot"></span>SECURE CONNECTION ESTABLISHED</div>
<div class="login-footer-note">LATENCY: 12ms &nbsp;|&nbsp; SERVER: US-EAST-01KSC &nbsp;|&nbsp; DEMO CREDENTIALS: admin / admin123</div>
        """)


def logout_button():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="font-family:var(--font-mono); font-size:11px; color:var(--cw-text-dim); '
        f'margin-bottom:6px;">SIGNED IN AS</div>'
        f'<div style="font-size:14px; font-weight:700; color:#F1F5F9; margin-bottom:2px;">{escape(str(st.session_state.get("username") or ""))}</div>'
        f'<div style="font-family:var(--font-mono); font-size:11px; color:var(--cw-primary); margin-bottom:10px;">{escape(str(st.session_state.get("role") or ""))}</div>',
        unsafe_allow_html=True
    )
    if st.sidebar.button("🔒 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["role"] = None
        st.session_state["username"] = None
        st.rerun()


# Data loading and preparation
# -----------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_district_data():
    if not DISTRICT_FILE.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(DISTRICT_FILE)
    df.columns = [c.strip() for c in df.columns]
    df["IPC/BNS Crimes"] = pd.to_numeric(df["IPC/BNS Crimes"], errors="coerce")
    df["SLL Crimes"] = pd.to_numeric(df["SLL Crimes"], errors="coerce")

    region = "Commissionerates"
    regions = []
    for _, row in df.iterrows():
        name = str(row["Districts/Units"]).strip()
        if pd.isna(row["IPC/BNS Crimes"]) and pd.isna(row["SLL Crimes"]):
            region = name
        regions.append(region)
    df["region"] = regions

    df = df.dropna(subset=["IPC/BNS Crimes", "SLL Crimes"], how="all").copy()
    df["district"] = df["Districts/Units"].astype(str).str.strip()
    df["ipc_bns_crimes"] = df["IPC/BNS Crimes"].fillna(0).astype(int)
    df["sll_crimes"] = df["SLL Crimes"].fillna(0).astype(int)
    df["total_crimes"] = df["ipc_bns_crimes"] + df["sll_crimes"]

    state_row = df[df["district"].str.upper() == "STATE"]
    df_units = df[df["district"].str.upper() != "STATE"].copy()

    if df_units.empty:
        return df_units, state_row

    # Explainable risk score: weighted min-max scaling of total volume + IPC/BNS intensity,
    # plus a z-score based outlier flag so extreme districts are easy to explain.
    max_total = df_units["total_crimes"].max() or 1
    max_ipc = df_units["ipc_bns_crimes"].max() or 1
    df_units["risk_score"] = (
        0.75 * (df_units["total_crimes"] / max_total) +
        0.25 * (df_units["ipc_bns_crimes"] / max_ipc)
    ) * 100
    df_units["risk_level"] = pd.cut(
        df_units["risk_score"],
        bins=[-1, 35, 70, 101],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    mean_total = df_units["total_crimes"].mean()
    std_total = df_units["total_crimes"].std(ddof=0) or 1
    df_units["z_score"] = (df_units["total_crimes"] - mean_total) / std_total
    df_units["statistical_outlier"] = df_units["z_score"].abs() >= 1.5
    df_units["rank"] = df_units["total_crimes"].rank(ascending=False, method="min").astype(int)

    return df_units, state_row


@st.cache_data(ttl=3600, show_spinner=False)
def load_crime_review():
    if not CRIME_REVIEW_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(CRIME_REVIEW_FILE)
    df = df.rename(columns={
        "Heads of Crime": "head_group",
        "Major Heads": "major_head",
        "Minor Heads": "minor_head",
        "During the current year upto the end of month under review": "ytd_current_year",
        "During the corresponding month of previous year": "same_month_previous_year",
        "During the previous month": "previous_month",
        "During the current month": "current_month",
    })
    for col in ["ytd_current_year", "same_month_previous_year", "previous_month", "current_month"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0
    for col in ["head_group", "major_head", "minor_head"]:
        df[col] = df[col].fillna("").astype(str).str.replace("\n", " ", regex=False).str.strip()

    df["crime_label"] = np.where(
        df["minor_head"].ne(""),
        df["major_head"] + " - " + df["minor_head"],
        df["major_head"]
    )
    df["mom_change"] = df["current_month"] - df["previous_month"]
    df["yoy_change"] = df["current_month"] - df["same_month_previous_year"]
    df["mom_pct_change"] = np.where(
        df["previous_month"] > 0,
        (df["mom_change"] / df["previous_month"]) * 100,
        np.where(df["current_month"] > 0, 100.0, 0.0)
    )
    df["yoy_pct_change"] = np.where(
        df["same_month_previous_year"] > 0,
        (df["yoy_change"] / df["same_month_previous_year"]) * 100,
        np.where(df["current_month"] > 0, 100.0, 0.0)
    )
    df["alert_level"] = np.select(
        [df["mom_pct_change"] >= 50, df["mom_pct_change"] >= 20, df["mom_pct_change"] > 0],
        ["High", "Medium", "Low"],
        default="No Spike"
    )

    # Statistical anomaly flag using z-score of month-on-month percentage change,
    # complements the simple percentage-threshold alert_level above.
    mean_pct = df["mom_pct_change"].mean()
    std_pct = df["mom_pct_change"].std(ddof=0) or 1
    df["mom_z_score"] = (df["mom_pct_change"] - mean_pct) / std_pct
    df["is_statistical_anomaly"] = df["mom_z_score"] >= 2

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_police_stations():
    if not STATION_FILE.exists():
        return pd.DataFrame(columns=["station_name", "latitude", "longitude", "source_type"])
    df = pd.read_csv(STATION_FILE)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["station_name"] = df["station_name"].fillna("Unknown Station")
    df["source_type"] = df["source_type"].fillna("Police Station")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_special_categories():
    if not WOMEN_CHILD_SCST_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(WOMEN_CHILD_SCST_FILE)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def make_demo_network():
    # Demo-only anonymized linkage data. Real repeat offender tracking needs case-level SCRB/FIR accused IDs.
    rng = np.random.default_rng(7)
    offenders = [f"OFF-{i:03d}" for i in range(1, 26)]
    crimes = ["Theft", "Vehicle Theft", "Cyber Fraud", "Robbery", "Burglary", "Assault"]
    districts = ["Bengaluru City", "Mysuru City", "Mangaluru City", "Belagavi", "Kalaburagi", "Tumakuru"]
    rows = []
    case_id = 1001
    for _ in range(55):
        crime = rng.choice(crimes, p=[0.28, 0.18, 0.18, 0.12, 0.14, 0.10])
        district = rng.choice(districts)
        n_accused = rng.choice([1, 1, 2, 2, 3])
        selected = rng.choice(offenders, size=int(n_accused), replace=False)
        for offender in selected:
            rows.append({
                "case_id": f"CASE-{case_id}",
                "offender_id": offender,
                "crime_type": crime,
                "district": district,
                "year": 2025
            })
        case_id += 1
    return pd.DataFrame(rows)


# -----------------------------
# Analytics helpers
# -----------------------------

def parse_nl_query(query, district_options, crime_review_df):
    """Low-risk natural language parser: maps question only to allowed filters/actions."""
    q = (query or "").lower().strip()
    result = {
        "district": "All",
        "crime_focus": "All",
        "view_hint": "Overview",
        "risk_only": False,
        "alert_only": False,
        "station_search": "",
        "top_n": None,
        "message": "No natural-language filter applied."
    }
    if not q:
        return result

    for d in district_options:
        if d.lower() in q:
            result["district"] = d
            break

    if not crime_review_df.empty:
        crime_terms = sorted(set(crime_review_df["major_head"].dropna().astype(str).tolist()), key=len, reverse=True)
        for term in crime_terms:
            clean = term.lower().strip()
            if len(clean) >= 5 and clean in q:
                result["crime_focus"] = term
                break

    match = re.search(r"top\s+(\d{1,3})", q)
    if match:
        try:
            result["top_n"] = max(1, min(50, int(match.group(1))))
        except Exception:
            result["top_n"] = None

    if any(w in q for w in ["alert", "increase", "spike", "anomaly", "unusual", "rising"]):
        result["view_hint"] = "Trend Alerts"
        result["alert_only"] = True
    if any(w in q for w in ["map", "station", "police station", "location", "geo"]):
        result["view_hint"] = "Geospatial Map"
        result["station_search"] = (
            q.replace("show", "").replace("police stations", "")
            .replace("police station", "").replace("station", "").strip()
        )
    if any(w in q for w in ["risk", "high risk", "danger", "hotspot", "hotspots"]):
        result["view_hint"] = "Risk Ranking"
        result["risk_only"] = True
    if any(w in q for w in ["network", "repeat", "offender", "criminal link", "link analysis"]):
        result["view_hint"] = "Network Demo"
    if any(w in q for w in ["women", "children", "sc", "st", "scheduled caste", "scheduled tribe"]):
        result["view_hint"] = "Special Categories"

    parts = []
    if result["district"] != "All":
        parts.append(f"District filter: {result['district']}")
    if result["crime_focus"] != "All":
        parts.append(f"Crime focus: {result['crime_focus']}")
    if result["top_n"]:
        parts.append(f"Top-N filter: {result['top_n']}")
    parts.append(f"Suggested view: {result['view_hint']}")
    result["message"] = " | ".join(parts)
    return result


def build_network_figure(edges_df):
    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_node(r["offender_id"], node_type="Offender")
        G.add_node(r["case_id"], node_type="Case")
        G.add_edge(r["offender_id"], r["case_id"], crime_type=r["crime_type"])
    pos = nx.spring_layout(G, seed=42, k=0.45)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color="rgba(148,163,184,0.55)"),
        hoverinfo="none",
        mode="lines"
    )

    node_x, node_y, node_text, node_symbol, node_size, node_color = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if str(node).startswith("OFF"):
            node_symbol.append("circle")
            node_size.append(18)
            node_color.append("#38BDF8")
        else:
            node_symbol.append("square")
            node_size.append(11)
            node_color.append("#F97316")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=node_size, symbol=node_symbol, color=node_color, line=dict(width=1, color="#E5E7EB"))
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=540,
        showlegend=False,
        margin=dict(l=10, r=10, t=55, b=10),
        title="Anonymized offender-case link graph",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#E5E7EB"),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )
    return fig


# -----------------------------
# Authentication gate
# -----------------------------

if not st.session_state["authenticated"]:
    login_page()
    st.stop()

# -----------------------------
# Load data
# -----------------------------

with st.spinner("Loading crime datasets..."):
    district_df, state_row = load_district_data()
    review_df = load_crime_review()
    stations_df = load_police_stations()
    special_df = load_special_categories()
    network_df = make_demo_network()

data_ready = not district_df.empty

if data_ready:
    state_total = int(state_row["IPC/BNS Crimes"].fillna(0).sum() + state_row["SLL Crimes"].fillna(0).sum()) if not state_row.empty else int(district_df["total_crimes"].sum())
    total_ipc = int(state_row["IPC/BNS Crimes"].fillna(0).sum()) if not state_row.empty else int(district_df["ipc_bns_crimes"].sum())
    total_sll = int(state_row["SLL Crimes"].fillna(0).sum()) if not state_row.empty else int(district_df["sll_crimes"].sum())
    top_unit = district_df.sort_values("total_crimes", ascending=False).iloc[0]
    high_risk_count = int((district_df["risk_level"] == "High").sum())
else:
    state_total = total_ipc = total_sll = high_risk_count = 0
    top_unit = None

# -----------------------------
# Header + natural-language search
# -----------------------------

if data_ready:
    high_risk_share = high_risk_count / max(len(district_df), 1)
    is_critical_status = high_risk_share >= 0.25
    status_label = "Critical Alert" if is_critical_status else ("Elevated" if high_risk_count > 0 else "Nominal")
else:
    is_critical_status = False
    status_label = "Awaiting Data"

hero_header(status_label=status_label, is_critical=is_critical_status)

if not data_ready:
    missing_file_notice(DISTRICT_FILE, "State Crime Overview / Risk Ranking")

if ui is not None:
    ui.badges(
        badge_list=[
            ("Real public datasets", "default"),
            ("Explainable scoring", "secondary"),
            ("Low-risk NL search", "outline"),
            ("KSP prototype", "destructive"),
        ],
        class_name="flex gap-2",
        key="top_status_badges"
    )

st.html("""
<div class="nl-box">
    <div class="nl-icon">⌕</div>
    <div>
        <div class="nl-title">Ask CrimeWatch AI</div>
        <div class="nl-subtitle">Type a question — the system safely maps it to existing dashboard filters and views. Try "top 5 high risk districts" or "rising crime categories".</div>
    </div>
</div>
""")

nl_query = st.text_input(
    "Ask in plain English",
    placeholder="Examples: Show Bengaluru City crimes | Show top 5 high-risk districts | Show rising crime categories | Show police stations in Bengaluru",
    label_visibility="collapsed"
)
district_choices_for_nl = sorted(district_df["district"].unique()) if data_ready else []
parsed = parse_nl_query(nl_query, district_choices_for_nl, review_df)
ai_insight(parsed["message"])

# Sidebar filters
st.sidebar.markdown("### Manual Filters")
district_options = ["All"] + (sorted(district_df["district"].unique().tolist()) if data_ready else [])
default_district_idx = district_options.index(parsed["district"]) if parsed["district"] in district_options else 0
selected_district = st.sidebar.selectbox("District / Unit", district_options, index=default_district_idx)
risk_options = ["All", "High", "Medium", "Low"]
selected_risk = st.sidebar.selectbox("Risk Level", risk_options)

if not review_df.empty:
    crime_head_options = ["All"] + sorted(review_df["head_group"].replace("", np.nan).dropna().unique().tolist())
else:
    crime_head_options = ["All"]
selected_head = st.sidebar.selectbox("Crime Head Group", crime_head_options)

st.sidebar.markdown("---")
st.sidebar.caption("Filters apply across the Overview, District Drilldown, and Trend Alerts tabs where relevant.")

logout_button()

filtered_district_df = district_df.copy()
if data_ready:
    if selected_district != "All":
        filtered_district_df = filtered_district_df[filtered_district_df["district"] == selected_district]
    if selected_risk != "All":
        filtered_district_df = filtered_district_df[filtered_district_df["risk_level"] == selected_risk]

filtered_review_df = review_df.copy()
if not review_df.empty and selected_head != "All":
    filtered_review_df = filtered_review_df[filtered_review_df["head_group"] == selected_head]

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview",
    "🔎 District Drilldown",
    "📈 Trend Alerts",
    "🗺️ Geospatial Map",
    "🕸️ Network Demo",
    "📋 Build Notes"
])

# -----------------------------
# Tab 1: Overview Dashboard
# -----------------------------
with tab1:
    if not data_ready:
        missing_file_notice(DISTRICT_FILE, "State Crime Overview")
    else:
        section_header(
            "State Crime Overview — 2025",
            "High-level summary of IPC/BNS crimes, SLL crimes, and district-level risk concentration."
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Total Crimes", format_int(state_total), "Combined IPC/BNS + SLL volume")
        with c2:
            kpi_card("IPC/BNS Crimes", format_int(total_ipc), "Core criminal-law category")
        with c3:
            kpi_card("SLL Crimes", format_int(total_sll), "Special and Local Laws category")
        with c4:
            kpi_card("High-Risk Units", format_int(high_risk_count), "Districts/units flagged high risk")

        st.html("<br>")

        top_n_overview = parsed["top_n"] or 12
        left, right = st.columns([1.3, 1])

        with left:
            top_n = district_df.sort_values("total_crimes", ascending=False).head(top_n_overview)
            fig = px.bar(
                top_n,
                x="total_crimes",
                y="district",
                orientation="h",
                title=f"Top {len(top_n)} Districts/Units by Total Crimes",
                hover_data=["ipc_bns_crimes", "sll_crimes", "risk_level"],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Total Crimes", yaxis_title="District / Unit")
            st.plotly_chart(apply_dark_layout(fig, height=510), use_container_width=True)

        with right:
            crime_type_totals = pd.DataFrame({
                "Crime Group": ["IPC/BNS Crimes", "SLL Crimes"],
                "Count": [total_ipc, total_sll]
            })
            fig = px.pie(crime_type_totals, names="Crime Group", values="Count", title="IPC/BNS vs SLL Share", hole=0.45)
            st.plotly_chart(apply_dark_layout(fig, height=510), use_container_width=True)

        section_header("Explainable Risk Ranking", "Risk score uses normalized total crime volume and IPC/BNS intensity. Statistical outliers (|z| ≥ 1.5) are flagged separately.")

        sort_col1, sort_col2 = st.columns([1, 3])
        with sort_col1:
            n_rows = st.number_input("Rows to show", min_value=5, max_value=50, value=min(15, len(district_df)), step=5)

        risk_table = district_df.sort_values("risk_score", ascending=False)[[
            "rank", "district", "region", "ipc_bns_crimes", "sll_crimes", "total_crimes",
            "risk_score", "risk_level", "statistical_outlier"
        ]].head(int(n_rows))
        st.dataframe(risk_table, use_container_width=True, hide_index=True)
        download_button(risk_table, "Download risk ranking (CSV)", "risk_ranking.csv", "dl_risk_overview")

        highest = district_df.sort_values("risk_score", ascending=False).iloc[0]
        outlier_note = " It is also a statistical outlier relative to other districts." if bool(highest["statistical_outlier"]) else ""
        ai_insight(
            f"{highest['district']} is currently ranked highest in the prototype risk model because it has "
            f"{format_int(highest['total_crimes'])} total crimes, including {format_int(highest['ipc_bns_crimes'])} IPC/BNS crimes "
            f"and {format_int(highest['sll_crimes'])} SLL crimes. The model is explainable and based on crime volume plus IPC/BNS intensity."
            f"{outlier_note}"
        )

# -----------------------------
# Tab 2: District Drilldown
# -----------------------------
with tab2:
    if not data_ready:
        missing_file_notice(DISTRICT_FILE, "District Drilldown")
    else:
        section_header("District-Level Drilldown", "Select a district/unit and inspect its crime split, risk contribution, and decision-support explanation.")

        drill_district = selected_district if selected_district != "All" else st.selectbox(
            "Choose a district/unit for drilldown",
            sorted(district_df["district"].unique()),
            key="drill_district"
        )
        row = district_df[district_df["district"] == drill_district].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("District / Unit", row["district"], f"Rank #{int(row['rank'])} statewide by total crimes")
        with c2:
            kpi_card("Total Crimes", format_int(row["total_crimes"]), "Overall reported volume")
        with c3:
            kpi_card("IPC/BNS", format_int(row["ipc_bns_crimes"]), "Core crime category")
        with c4:
            kpi_card("SLL", format_int(row["sll_crimes"]), "Special/local law category")

        reason = (
            f"{row['district']} belongs to {row['region']} and contributes "
            f"{row['total_crimes'] / max(state_total, 1) * 100:.2f}% of the state-level total in this dataset. "
            f"Its crime volume is {row['z_score']:.2f} standard deviations from the statewide district average."
        )
        risk_panel(row["risk_level"], row["risk_score"], reason)

        comp_df = pd.DataFrame({
            "Crime Group": ["IPC/BNS Crimes", "SLL Crimes"],
            "Count": [row["ipc_bns_crimes"], row["sll_crimes"]]
        })
        fig = px.bar(comp_df, x="Crime Group", y="Count", title=f"Crime Split for {row['district']}")
        st.plotly_chart(apply_dark_layout(fig, height=430), use_container_width=True)

        section_header("Compare Against Region", f"How {row['district']} compares with other units in {row['region']}.")
        region_peers = district_df[district_df["region"] == row["region"]].sort_values("total_crimes", ascending=False)
        fig2 = px.bar(
            region_peers, x="district", y="total_crimes",
            title=f"Total Crimes — {row['region']}",
            color=region_peers["district"].eq(row["district"]).map({True: "Selected", False: "Other"}),
            color_discrete_map={"Selected": "#38BDF8", "Other": "#475569"}
        )
        fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Total Crimes")
        st.plotly_chart(apply_dark_layout(fig2, height=380), use_container_width=True)

        ai_insight(
            "This module supports district-level decision-making. In deployment, the same view can connect to police-station-wise records "
            "to show station drilldowns, beat-level hotspots, and patrol recommendations."
        )

# -----------------------------
# Tab 3: Trend Alerts
# -----------------------------
with tab3:
    if review_df.empty:
        missing_file_notice(CRIME_REVIEW_FILE, "Trend Alerts & Anomaly Detection")
    else:
        section_header(
            "Trend Alerts & Anomaly Detection",
            "Uses monthly crime review data to compare current month, previous month, and same month of previous year."
        )

        s1, s2, s3 = st.columns([1.4, 1, 1])
        with s1:
            min_current = st.slider("Minimum current-month cases", 0, int(review_df["current_month"].max()), 5)
        with s2:
            alert_filter = st.selectbox("Alert level", ["All", "High", "Medium", "Low", "No Spike"])
        with s3:
            stat_only = st.checkbox("Statistical outliers only (z ≥ 2)", value=False)

        alerts = filtered_review_df[filtered_review_df["current_month"] >= min_current].copy()
        if alert_filter != "All":
            alerts = alerts[alerts["alert_level"] == alert_filter]
        if stat_only:
            alerts = alerts[alerts["is_statistical_anomaly"]]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Crime Heads Reviewed", format_int(len(review_df)), "Rows analyzed from crime review")
        with c2:
            kpi_card("High Spike Alerts", format_int((review_df["alert_level"] == "High").sum()), "Categories with severe month-on-month rise")
        with c3:
            kpi_card("Statistical Anomalies", format_int(review_df["is_statistical_anomaly"].sum()), "Categories with z-score ≥ 2 on MoM change")
        with c4:
            kpi_card("Current Month Cases", format_int(review_df["current_month"].sum()), "Total current-month cases reviewed")

        st.html("<br>")
        top_n_alerts = parsed["top_n"] or 15
        top_current = filtered_review_df.sort_values("current_month", ascending=False).head(top_n_alerts)
        fig = px.bar(
            top_current,
            x="current_month",
            y="crime_label",
            orientation="h",
            title=f"Top {len(top_current)} Current-Month Crime Categories"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Current Month Cases", yaxis_title="Crime Category")
        st.plotly_chart(apply_dark_layout(fig, height=590), use_container_width=True)

        section_header("Year-on-Year Comparison", "Current month vs the same month last year, for the same top categories.")
        yoy_df = top_current.melt(
            id_vars=["crime_label"],
            value_vars=["current_month", "same_month_previous_year"],
            var_name="Period", value_name="Cases"
        )
        yoy_df["Period"] = yoy_df["Period"].map({"current_month": "Current Month", "same_month_previous_year": "Same Month Last Year"})
        fig_yoy = px.bar(yoy_df, x="crime_label", y="Cases", color="Period", barmode="group", title="Current Month vs Same Month Last Year")
        fig_yoy.update_layout(xaxis_title="", yaxis_title="Cases", xaxis_tickangle=-35)
        st.plotly_chart(apply_dark_layout(fig_yoy, height=430), use_container_width=True)

        section_header("Spike Alert Table", "Filtered anomaly candidates based on month-on-month change and statistical z-score.")
        show_cols = [
            "head_group", "crime_label", "previous_month", "current_month", "mom_change",
            "mom_pct_change", "mom_z_score", "is_statistical_anomaly",
            "same_month_previous_year", "yoy_change", "yoy_pct_change", "alert_level"
        ]
        display_alerts = alerts.sort_values(["alert_level", "mom_pct_change", "current_month"], ascending=[True, False, False])[show_cols].head(50)
        st.dataframe(display_alerts, use_container_width=True, hide_index=True)
        download_button(display_alerts, "Download alert table (CSV)", "spike_alerts.csv", "dl_alerts")

        high_alerts = review_df[(review_df["alert_level"] == "High") & (review_df["current_month"] >= 5)].sort_values("mom_pct_change", ascending=False)
        if not high_alerts.empty:
            for _, a in high_alerts.head(3).iterrows():
                alert_card(
                    f"High Spike: {a['crime_label']}",
                    f"{a['crime_label']} increased from {format_int(a['previous_month'])} cases in the previous month "
                    f"to {format_int(a['current_month'])} cases in the current month.",
                    badge=f"+{a['mom_pct_change']:.0f}% MoM"
                )
            alert_card(
                "Suggested Action",
                "Prioritize monitoring of rising categories, compare with local station-level records, and allocate preventive patrol or investigation resources where the spike is concentrated.",
                badge=None
            )
        else:
            info_card("No High Spikes", "No crime categories currently meet the high-spike threshold under the selected filters.")

# -----------------------------
# Tab 4: Geospatial Map
# -----------------------------
with tab4:
    section_header(
        "Police Station & Outpost Map",
        "Geospatial layer from processed location data. It shows police infrastructure locations across Karnataka/Bengaluru."
    )

    if stations_df.empty:
        missing_file_notice(STATION_FILE, "Geospatial Map")
    else:
        legend_card([
            ("High-Density Hubs", "Critical", "#E63946"),
            ("Low-Activity Zones", "Secure", "#43B0F8"),
            ("Locations Indexed", format_int(len(stations_df)), "#727877"),
        ])

        source_options = ["All"] + sorted(stations_df["source_type"].dropna().unique().tolist())
        c0, c1 = st.columns([1, 2])
        with c0:
            src = st.selectbox("Location type", source_options)
        with c1:
            station_search = st.text_input("🔎 Search station/outpost name", value=parsed.get("station_search", ""))

        map_df = stations_df.copy()
        if src != "All":
            map_df = map_df[map_df["source_type"] == src]
        if station_search:
            map_df = map_df[map_df["station_name"].str.lower().str.contains(station_search.lower(), na=False)]

        c1, c2 = st.columns(2)
        with c1:
            kpi_card("Locations Shown", format_int(len(map_df)), "Filtered map markers")
        with c2:
            kpi_card("Processed Locations", format_int(len(stations_df)), "Total police locations processed")

        st.html("<br>")
        if not map_df.empty:
            st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]], zoom=6)
            display_map_df = map_df[["station_name", "source_type", "latitude", "longitude"]].head(200)
            st.dataframe(display_map_df, use_container_width=True, hide_index=True)
            download_button(display_map_df, "Download filtered locations (CSV)", "police_locations.csv", "dl_map")
        else:
            info_card("No Map Results", "No police station or outpost locations match the current filter.")

        ai_insight(
            "This map is the geospatial base layer. It becomes a true crime hotspot map when station-wise crime counts are connected to these coordinates."
        )

# -----------------------------
# Tab 5: Network Demo
# -----------------------------
with tab5:
    section_header(
        "Network & Repeat Offender Tracking Demo",
        "An anonymized demonstration layer because public aggregated datasets do not contain accused/case linkage records."
    )

    repeat_counts = network_df.groupby("offender_id").agg(
        case_count=("case_id", "nunique"),
        districts=("district", lambda x: ", ".join(sorted(set(x))[:3])),
        crime_types=("crime_type", lambda x: ", ".join(sorted(set(x))[:4]))
    ).reset_index().sort_values("case_count", ascending=False)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Demo Cases", format_int(network_df["case_id"].nunique()), "Anonymized sample cases")
    with c2:
        kpi_card("Demo Offenders", format_int(network_df["offender_id"].nunique()), "Anonymized offender IDs")
    with c3:
        kpi_card("Repeat Offenders", format_int((repeat_counts["case_count"] > 1).sum()), "Appearing in more than one case")

    section_header("Repeat Offender Table", "Offender IDs ranked by number of linked cases.")
    top_offender_rows = st.slider("Offenders to show", 5, len(repeat_counts), min(15, len(repeat_counts)))
    offender_table = repeat_counts.head(top_offender_rows)
    st.dataframe(offender_table, use_container_width=True, hide_index=True)
    download_button(offender_table, "Download offender table (CSV)", "repeat_offenders_demo.csv", "dl_network")

    section_header("Link Analysis Graph", "Offenders are linked to cases to reveal repeated involvement and shared case patterns.")
    top_offenders = repeat_counts.head(8)["offender_id"].tolist()
    graph_df = network_df[network_df["offender_id"].isin(top_offenders)]
    st.plotly_chart(build_network_figure(graph_df), use_container_width=True)

    ai_insight(
        "Deployment requirement: connect this module to official case-level records containing case_id and anonymized accused/offender IDs. "
        "The graph logic is ready; the sensitive source data determines whether repeat offender tracking is real or demo-only."
    )

# -----------------------------
# Tab 6: Build Notes
# -----------------------------
with tab6:
    section_header("Capability Coverage", "How the current prototype maps to Challenge 02 requirements.")

    coverage = pd.DataFrame([
        ["Interactive dashboards", "Covered", "District-wise crime totals and crime category charts"],
        ["Geospatial maps", "Covered", "Police station/outpost data converted to map points"],
        ["Crime hotspot detection", "Partial", "District risk ranking now; station hotspot needs station-wise crime counts"],
        ["District-level drilldowns", "Covered", "District/unit filter, peer comparison, and explainable score"],
        ["Trend alerts", "Covered", "Current month vs previous month vs same month last year"],
        ["Anomaly detection", "Covered", "Percentage-threshold alerts plus z-score statistical outlier flag"],
        ["Network/link analysis", "Demo", "Needs case-level accused linkage records for real deployment"],
        ["Repeat offender tracking", "Demo", "Needs offender_id/accused_id in official records"],
        ["Socio-economic correlation", "Next", "Needs district-level population/literacy/income indicators"],
        ["Predictive risk scoring", "Covered", "Explainable district risk score with outlier flag"],
        ["AI/ML pattern detection", "Covered lightly", "Rule-based + z-score alerts now; clustering/ML can be added later"],
        ["Natural-language search", "Covered", "Maps plain English to safe dashboard filters/actions, including top-N queries"],
        ["Data export", "Covered", "CSV download available on every major table"],
        ["Graceful missing-data handling", "Covered", "Each tab shows a clear notice instead of crashing when a source file is absent"],
    ], columns=["Capability", "Status", "Implementation note"])
    st.dataframe(coverage, use_container_width=True, hide_index=True)
    download_button(coverage, "Download coverage table (CSV)", "capability_coverage.csv", "dl_coverage")

    section_header("Next Data Needed", "Datasets that will make the prototype closer to production quality.")
    ai_insight(
        "To make the remaining modules fully real, look for datasets with: district, police_station, crime_type, month/year, case_count, "
        "case_id, accused_id/offender_id. For socio-economic analysis, add district-level population, literacy rate, unemployment, urbanization, or income indicators."
    )

    if not special_df.empty:
        section_header("Women / Children / SC-ST Crime Data Preview", "Loaded from ka-crimes-women-children-scssts.csv.")
        st.dataframe(special_df.head(50), use_container_width=True, hide_index=True)
        download_button(special_df, "Download special categories data (CSV)", "special_categories.csv", "dl_special")
    else:
        missing_file_notice(WOMEN_CHILD_SCST_FILE, "Special Categories (Women/Children/SC-ST) preview")

# -----------------------------
# Footer
# -----------------------------
st.html(
    '<div class="footer-note">CrimeWatch AI prototype · Built on Streamlit + Plotly · '
    'Risk scores and anomaly flags are explainable heuristics for decision support, not verified predictive claims.</div>'
)