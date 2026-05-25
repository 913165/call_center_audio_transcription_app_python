"""Theme tokens and CSS for the call center Streamlit app."""

from __future__ import annotations

import base64
import hashlib
import html as html_module
import re
from typing import Any, Callable, Dict, Optional

MAX_INLINE_AUDIO_BYTES = 8 * 1024 * 1024

import pandas as pd
import streamlit as st

THEME_TOKENS: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg_start": "#060a10",
        "bg_end": "#0c1219",
        "surface": "rgba(12, 20, 28, 0.92)",
        "surface_alt": "rgba(8, 14, 22, 0.95)",
        "border": "#1a3340",
        "border_soft": "#243848",
        "text": "#e8f2f8",
        "text_muted": "#8fa3b8",
        "text_faint": "#6b7f94",
        "accent": "#2dd4bf",
        "accent_soft": "rgba(45, 212, 191, 0.12)",
        "hero_glow": "rgba(45, 212, 191, 0.15)",
        "track": "#2a3544",
        "score_fill": "linear-gradient(90deg, #14b8a6, #2dd4bf)",
        "shadow": "0 8px 32px rgba(0, 0, 0, 0.35)",
        "chart_bg": "#0a0e14",
        "agent": "#45f4d2",
        "customer": "#f6e08a",
        "sidebar_bg": "rgba(6, 10, 16, 0.98)",
        "metric_bg": "rgba(10, 18, 26, 0.9)",
        "tab_active": "#2dd4bf",
        "success_bg": "#143a2f",
        "success_fg": "#9ef3cf",
        "warn_bg": "#3d3214",
        "warn_fg": "#ffe291",
        "danger_bg": "#421d22",
        "danger_fg": "#ffadad",
        "input_bg": "#0f161e",
        "input_border": "#2a4555",
        "input_text": "#e8f2f8",
        "input_placeholder": "#6b8299",
        "uploader_zone": "#0c1219",
        "uploader_btn_bg": "#0d9488",
        "uploader_btn_text": "#ffffff",
        "uploader_btn_border": "#2dd4bf",
        "uploader_hint": "#9eb0c4",
        "dataframe_bg": "#0f161e",
        "dataframe_header": "#162029",
    },
    "light": {
        "bg_start": "#f0f4f9",
        "bg_end": "#e8eef5",
        "surface": "#ffffff",
        "surface_alt": "#f8fafc",
        "border": "#d4dee8",
        "border_soft": "#e2e8f0",
        "text": "#0f172a",
        "text_muted": "#475569",
        "text_faint": "#64748b",
        "accent": "#0d9488",
        "accent_soft": "rgba(13, 148, 136, 0.1)",
        "hero_glow": "rgba(13, 148, 136, 0.08)",
        "track": "#e2e8f0",
        "score_fill": "linear-gradient(90deg, #0d9488, #14b8a6)",
        "shadow": "0 8px 24px rgba(15, 23, 42, 0.08)",
        "chart_bg": "#f8fafc",
        "agent": "#0d9488",
        "customer": "#b45309",
        "sidebar_bg": "#ffffff",
        "metric_bg": "#f8fafc",
        "tab_active": "#0d9488",
        "success_bg": "#d1fae5",
        "success_fg": "#065f46",
        "warn_bg": "#fef3c7",
        "warn_fg": "#92400e",
        "danger_bg": "#fee2e2",
        "danger_fg": "#991b1b",
        "input_bg": "#ffffff",
        "input_border": "#d4dee8",
        "input_text": "#0f172a",
        "input_placeholder": "#94a3b8",
        "uploader_zone": "#f8fafc",
        "uploader_btn_bg": "#0d9488",
        "uploader_btn_text": "#ffffff",
        "uploader_btn_border": "#0d9488",
        "uploader_hint": "#64748b",
        "dataframe_bg": "#ffffff",
        "dataframe_header": "#f1f5f9",
    },
}


def theme_css(theme: str) -> str:
    """Return CSS wrapped in <style> so Streamlit applies it instead of showing raw text."""
    t = THEME_TOKENS.get(theme, THEME_TOKENS["dark"])
    rules = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

html, body, [class*="css"] {{
  font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
}}

.stApp {{
  background: linear-gradient(165deg, {t['bg_start']} 0%, {t['bg_end']} 55%, {t['bg_start']} 100%);
  color: {t['text']};
}}

/* Streamlit top bar (Deploy menu) — default is white */
header[data-testid="stHeader"],
.stApp > header,
[data-testid="stHeader"] {{
  background: {t['bg_start']} !important;
  background-color: {t['bg_start']} !important;
  border-bottom: 1px solid {t['border_soft']} !important;
}}

/* Colored strip above header on some Streamlit versions */
div[data-testid="stDecoration"] {{
  display: none !important;
}}

[data-testid="stHeader"] button,
[data-testid="stHeader"] a,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] span {{
  color: {t['text_muted']} !important;
}}

[data-testid="stHeader"] button:hover,
[data-testid="stHeader"] a:hover {{
  color: {t['accent']} !important;
}}

/* Use full width of main panel — reduce side gaps */
section.main > div.block-container,
.block-container {{
  padding-top: 1.25rem;
  padding-bottom: 2rem;
  max-width: 100% !important;
  padding-left: 1.25rem !important;
  padding-right: 1.25rem !important;
}}

[data-testid="stMainBlockContainer"] {{
  max-width: 100% !important;
}}

[data-testid="stAppViewContainer"] .main {{
  width: 100%;
}}

[data-testid="stSidebar"] {{
  background: {t['sidebar_bg']} !important;
  border-right: 1px solid {t['border']};
}}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {{
  color: {t['text']} !important;
}}

[data-testid="stMetric"] {{
  background: {t['metric_bg']};
  border: 1px solid {t['border_soft']};
  border-radius: 12px;
  padding: 0.65rem 0.85rem;
  box-shadow: {t['shadow']};
}}

[data-testid="stMetricLabel"] {{
  color: {t['text_muted']} !important;
  font-size: 0.78rem !important;
}}

[data-testid="stMetricValue"] {{
  color: {t['text']} !important;
  font-weight: 700 !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 6px;
  background: {t['surface_alt']};
  border: 1px solid {t['border_soft']};
  border-radius: 14px;
  padding: 6px;
}}

.stTabs [data-baseweb="tab"] {{
  border-radius: 10px;
  color: {t['text_muted']};
  font-weight: 600;
  padding: 0.45rem 0.9rem;
}}

.stTabs [aria-selected="true"] {{
  background: {t['accent_soft']} !important;
  color: {t['accent']} !important;
}}

/* File uploader — dropzone, button, and helper text */
div[data-testid="stFileUploader"] {{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}}

div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
  background: {t['uploader_zone']} !important;
  background-color: {t['uploader_zone']} !important;
  border: 1.5px dashed {t['border']} !important;
  border-radius: 12px !important;
  padding: 1.25rem 1rem !important;
}}

div[data-testid="stFileUploader"] section:hover,
div[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"]:hover {{
  border-color: {t['accent']} !important;
  background: {t['surface_alt']} !important;
  background-color: {t['surface_alt']} !important;
}}

div[data-testid="stFileUploader"] section span,
div[data-testid="stFileUploader"] section small,
div[data-testid="stFileUploader"] section p,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
div[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] span,
div[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] small {{
  color: {t['uploader_hint']} !important;
}}

div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {{
  background: {t['uploader_btn_bg']} !important;
  background-color: {t['uploader_btn_bg']} !important;
  color: {t['uploader_btn_text']} !important;
  border: 1px solid {t['uploader_btn_border']} !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}}

div[data-testid="stFileUploader"] button:hover,
div[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {{
  background: {t['accent']} !important;
  background-color: {t['accent']} !important;
  color: {t['uploader_btn_text']} !important;
  border-color: {t['accent']} !important;
}}

div[data-testid="stFileUploader"] button span,
div[data-testid="stFileUploader"] button p,
div[data-testid="stFileUploader"] button div {{
  color: {t['uploader_btn_text']} !important;
}}

div[data-testid="stFileUploader"] section svg,
div[data-testid="stFileUploader"] button svg {{
  fill: {t['uploader_hint']} !important;
  stroke: {t['uploader_hint']} !important;
}}

div[data-testid="stFileUploader"] button svg {{
  fill: {t['uploader_btn_text']} !important;
  stroke: {t['uploader_btn_text']} !important;
}}

/* Hide white file chips in uploader — files are listed in the table below */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFiles"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] {{
  display: none !important;
}}

/* Text inputs & password fields (main + sidebar) */
.stTextInput input,
.stTextInput textarea,
[data-testid="stSidebar"] .stTextInput input {{
  background-color: {t['input_bg']} !important;
  color: {t['input_text']} !important;
  border: 1px solid {t['input_border']} !important;
  border-radius: 8px !important;
}}

.stTextInput input::placeholder {{
  color: {t['input_placeholder']} !important;
}}

.stTextInput label,
.stSelectbox label,
.stRadio label,
.stTextInput [data-testid="stWidgetLabel"] p,
.stSelectbox [data-testid="stWidgetLabel"] p,
.stRadio [data-testid="stWidgetLabel"] p {{
  color: {t['text_muted']} !important;
}}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
  background-color: {t['input_bg']} !important;
  color: {t['input_text']} !important;
  border-color: {t['input_border']} !important;
  border-radius: 8px !important;
}}

.stSelectbox div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {{
  color: {t['input_text']} !important;
}}

.stSelectbox svg {{
  fill: {t['text_muted']} !important;
}}

/* Radio (theme toggle) */
.stRadio div[role="radiogroup"] label {{
  color: {t['text']} !important;
  background: {t['input_bg']} !important;
  border: 1px solid {t['input_border']} !important;
  border-radius: 8px !important;
  padding: 0.35rem 0.75rem !important;
}}

.stRadio div[role="radiogroup"] label[data-checked="true"],
.stRadio div[role="radiogroup"] label:has(input:checked) {{
  background: {t['accent_soft']} !important;
  border-color: {t['accent']} !important;
  color: {t['accent']} !important;
}}

/* Glide data grid (st.dataframe) — override default white cells */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] div[data-testid="stDataFrameResizable"],
[data-testid="stDataFrame"] .dvn-scroller,
[data-testid="stDataFrame"] .gdg-style {{
  background: {t['dataframe_bg']} !important;
  background-color: {t['dataframe_bg']} !important;
  border: 1px solid {t['border_soft']} !important;
  border-radius: 10px !important;
  --gdg-bg-cell: {t['dataframe_bg']} !important;
  --gdg-bg-cell-medium: {t['surface_alt']} !important;
  --gdg-bg-header: {t['dataframe_header']} !important;
  --gdg-bg-header-has-focus: {t['dataframe_header']} !important;
  --gdg-bg-header-hovered: {t['border_soft']} !important;
  --gdg-text-dark: {t['input_text']} !important;
  --gdg-text-medium: {t['text_muted']} !important;
  --gdg-text-light: {t['text_faint']} !important;
  --gdg-text-header: {t['text']} !important;
  --gdg-text-group-header: {t['text_muted']} !important;
  --gdg-border-color: {t['border_soft']} !important;
  --gdg-accent-color: {t['accent']} !important;
  --gdg-accent-fg: {t['uploader_btn_text']} !important;
  --gdg-accent-light: {t['accent_soft']} !important;
  --gdg-bg-icon-header: {t['text_faint']} !important;
  --gdg-fg-icon-header: {t['text']} !important;
}}

/* Custom HTML tables (upload list, etc.) */
.data-table-wrap {{
  max-height: 420px;
  overflow: auto;
  border: 1px solid {t['border_soft']};
  border-radius: 12px;
  background: {t['dataframe_bg']};
  box-shadow: {t['shadow']};
  margin: 0.75rem 0 1rem 0;
}}

.data-table-wrap.queue-table-scroll {{
  max-height: 280px;
  overflow-y: scroll !important;
  overflow-x: auto !important;
  scrollbar-gutter: stable;
  scrollbar-color: {t['accent']} {t['track']};
}}

.data-table-wrap.queue-table-scroll::-webkit-scrollbar {{
  width: 10px;
  height: 10px;
}}

.data-table-wrap.queue-table-scroll::-webkit-scrollbar-thumb {{
  background: {t['accent']};
  border-radius: 999px;
}}

.data-table-wrap.queue-table-scroll::-webkit-scrollbar-track {{
  background: {t['track']};
  border-radius: 999px;
}}

table.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
}}

table.data-table thead {{
  position: sticky;
  top: 0;
  z-index: 1;
}}

table.data-table th {{
  background: {t['dataframe_header']};
  color: {t['text']};
  font-weight: 700;
  text-align: left;
  padding: 0.65rem 0.85rem;
  border-bottom: 1px solid {t['border']};
  white-space: nowrap;
}}

table.data-table td {{
  color: {t['text_muted']};
  padding: 0.55rem 0.85rem;
  border-bottom: 1px solid {t['border_soft']};
}}

table.data-table tbody tr:hover td {{
  background: {t['surface_alt']};
  color: {t['text']};
}}

table.data-table tbody tr:last-child td {{
  border-bottom: none;
}}

table.data-table tbody tr.row-processing td {{
  background: {t['accent_soft']} !important;
  color: {t['text']} !important;
}}

table.data-table tbody tr.row-done td {{
  color: {t['text_muted']};
}}

.status-text-ready {{
  color: {t['text_faint']};
}}

.status-text-processing {{
  color: {t['accent']};
  font-weight: 700;
}}

.status-text-done {{
  color: #22c55e !important;
  font-weight: 700;
}}

.status-text-error {{
  color: #ef4444 !important;
  font-weight: 700;
}}

/* Progress bar — white track is Streamlit default */
.stProgress,
.stProgress > div,
.stProgress > div > div {{
  background: transparent !important;
  background-color: transparent !important;
}}

.stProgress > div > div > div {{
  background-color: {t['track']} !important;
  border-radius: 999px !important;
  height: 0.55rem !important;
  overflow: hidden !important;
}}

.stProgress > div > div > div > div {{
  background: {t['score_fill']} !important;
  background-color: {t['accent']} !important;
  border-radius: 999px !important;
}}

/* Status / info during batch processing */
[data-testid="stAlert"],
div[data-testid="stNotification"] {{
  background-color: {t['surface_alt']} !important;
  color: {t['text']} !important;
  border: 1px solid {t['border_soft']} !important;
  border-radius: 10px !important;
}}

[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] span {{
  color: {t['text']} !important;
}}

/* Batch progress — fixed height reduces layout shift */
.batch-summary {{
  color: {t['text_muted']};
  font-size: 0.9rem;
  margin: 0.5rem 0 0.75rem 0;
}}

.batch-summary strong {{
  color: {t['text']};
}}

[data-testid="stStatusWidget"] {{
  background: {t['surface']} !important;
  border: 1px solid {t['border_soft']} !important;
  border-radius: 12px !important;
}}

[data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p {{
  color: {t['text']} !important;
}}

/* Spinner container */
.stSpinner > div {{
  border-top-color: {t['accent']} !important;
}}

.stSpinner [data-testid="stMarkdownContainer"] p {{
  color: {t['text_muted']} !important;
}}

/* Alerts / info boxes */
.stAlert {{
  background-color: {t['surface_alt']} !important;
  color: {t['text']} !important;
  border: 1px solid {t['border_soft']} !important;
}}

/* Captions & body text */
.stCaption, .stMarkdown p, .stMarkdown li {{
  color: {t['text_muted']};
}}

h1, h2, h3, .stSubheader {{
  color: {t['text']} !important;
}}

.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {t['accent']}, #14b8a6) !important;
  border: none !important;
  font-weight: 700 !important;
  border-radius: 10px !important;
  box-shadow: 0 4px 14px {t['hero_glow']} !important;
}}

.stButton > button[kind="primary"]:hover {{
  filter: brightness(1.06);
}}

.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {{
  background: {t['input_bg']} !important;
  color: {t['text']} !important;
  border: 1px solid {t['input_border']} !important;
  border-radius: 8px !important;
}}

.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {{
  border-color: {t['accent']} !important;
  color: {t['accent']} !important;
}}

/* App components */
.app-header {{
  margin-bottom: 1.25rem;
}}

.app-title {{
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: {t['text']};
  margin: 0 0 0.25rem 0;
  line-height: 1.2;
}}

.app-title span {{
  color: {t['accent']};
}}

.app-subtitle {{
  color: {t['text_muted']};
  font-size: 0.95rem;
  margin: 0;
}}

.hero {{
  border: 1px solid {t['border']};
  border-radius: 16px;
  padding: 1.25rem 1.35rem;
  background: {t['surface']};
  box-shadow: {t['shadow']};
  position: relative;
  overflow: hidden;
}}

.hero::before {{
  content: '';
  position: absolute;
  top: -40%;
  right: -10%;
  width: 280px;
  height: 280px;
  background: radial-gradient(circle, {t['hero_glow']} 0%, transparent 70%);
  pointer-events: none;
}}

.hero-label {{
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: {t['accent']} !important;
  margin-bottom: 0.5rem;
  opacity: 1 !important;
}}

.badge {{
  display: inline-block;
  padding: 4px 12px;
  border: 1px solid {t['accent']};
  border-radius: 999px;
  color: {t['accent']};
  font-size: 0.78rem;
  font-weight: 600;
  background: {t['accent_soft']};
  margin-top: 0.5rem;
}}

.sidebar-brand {{
  padding: 0.25rem 0 1rem 0;
  border-bottom: 1px solid {t['border_soft']};
  margin-bottom: 1rem;
}}

.sidebar-brand-title {{
  font-size: 1.05rem;
  font-weight: 700;
  color: {t['text']};
  margin: 0;
}}

.sidebar-brand-sub {{
  font-size: 0.78rem;
  color: {t['text_muted']};
  margin: 0.15rem 0 0 0;
}}

.section-label {{
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: {t['text_faint']};
  margin: 0.75rem 0 0.35rem 0;
}}

.kpi-strip {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin: 1rem 0;
}}

@media (max-width: 900px) {{
  .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }}
}}

.kpi-card {{
  border: 1px solid {t['border_soft']};
  border-radius: 12px;
  padding: 0.85rem 1rem;
  background: {t['surface']};
  box-shadow: {t['shadow']};
}}

.kpi-card .kpi-value {{
  font-size: 1.5rem;
  font-weight: 700;
  color: {t['text']};
  line-height: 1.1;
}}

.kpi-card .kpi-label {{
  font-size: 0.75rem;
  color: {t['text_muted']};
  margin-top: 0.2rem;
}}

.turn-agent {{ color: {t['agent']}; }}
.turn-customer {{ color: {t['customer']}; }}

.turn-block {{
  border-left: 3px solid {t['border']};
  padding: 0.5rem 0 0.5rem 0.85rem;
  margin: 0.35rem 0 0.5rem 0;
  background: {t['surface_alt']};
  border-radius: 0 10px 10px 0;
}}

.turn-block.agent {{ border-left-color: {t['agent']}; }}
.turn-block.customer {{ border-left-color: {t['customer']}; }}

.sentiment-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 0.75rem;
  margin: 0.5rem 0 1rem 0;
  width: 100%;
}}

@media (min-width: 1400px) {{
  .sentiment-grid {{
    grid-template-columns: repeat(4, 1fr);
  }}
}}

@media (min-width: 1100px) and (max-width: 1399px) {{
  .sentiment-grid {{
    grid-template-columns: repeat(3, 1fr);
  }}
}}

.sentiment-grid .sentiment-card {{
  margin-bottom: 0;
  height: 100%;
}}

.sentiment-card {{
  background: {t['surface']};
  border: 1px solid {t['border_soft']};
  border-top-width: 3px;
  border-radius: 14px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.85rem;
  box-shadow: {t['shadow']};
}}

.sentiment-file {{
  color: {t['text_faint']};
  font-size: 0.72rem;
  margin-bottom: 0.3rem;
  font-family: ui-monospace, monospace;
}}

.sentiment-label {{
  font-size: 1.15rem;
  font-weight: 700;
  margin-bottom: 0.55rem;
}}

.score-row {{
  display: grid;
  grid-template-columns: 95px 1fr 28px;
  align-items: center;
  gap: 8px;
  margin: 0.2rem 0;
  font-size: 0.76rem;
  color: {t['text_muted']};
}}

.score-track {{
  height: 6px;
  border-radius: 999px;
  background: {t['track']};
  overflow: hidden;
}}

.score-fill {{
  height: 100%;
  border-radius: 999px;
  background: {t['score_fill']};
}}

.score-value {{
  color: {t['text_faint']};
  font-size: 0.72rem;
  text-align: right;
  font-weight: 600;
}}

.topic-row {{
  border: 1px solid {t['border_soft']};
  border-radius: 12px;
  background: {t['surface']};
  padding: 0.7rem 0.9rem;
  margin-bottom: 0.5rem;
  box-shadow: {t['shadow']};
  transition: border-color 0.15s ease;
}}

.topic-row:hover {{
  border-color: {t['border']};
}}

.topic-row-main {{
  display: grid;
  grid-template-columns: minmax(160px, 1.2fr) auto minmax(200px, 2.5fr) auto;
  align-items: center;
  gap: 12px;
}}

@media (max-width: 768px) {{
  .topic-row-main {{
    grid-template-columns: 1fr;
    gap: 6px;
  }}
}}

.topic-file {{
  color: {t['text_muted']};
  font-size: 0.82rem;
  font-family: ui-monospace, monospace;
}}

.topic-pill {{
  padding: 3px 11px;
  border-radius: 8px;
  font-size: 0.76rem;
  border: 1px solid;
  font-weight: 600;
  white-space: nowrap;
}}

.topic-snippet {{
  color: {t['text_faint']};
  font-size: 0.82rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.topic-confidence {{
  color: {t['text_muted']};
  font-size: 0.86rem;
  font-weight: 700;
}}

.breakdown-panel {{
  border: 1px solid {t['border_soft']};
  border-radius: 14px;
  background: {t['surface']};
  padding: 1rem;
  box-shadow: {t['shadow']};
  position: sticky;
  top: 1rem;
}}

.breakdown-title {{
  color: {t['text_faint']};
  letter-spacing: 0.08em;
  font-size: 0.72rem;
  font-weight: 700;
  margin-bottom: 0.85rem;
  text-transform: uppercase;
}}

.break-row {{ margin-bottom: 0.55rem; }}

.break-label {{
  color: {t['text_muted']};
  font-size: 0.86rem;
  margin-bottom: 0.22rem;
  display: flex;
  justify-content: space-between;
}}

.break-track {{
  height: 4px;
  border-radius: 999px;
  background: {t['track']};
  overflow: hidden;
}}

.break-fill {{
  height: 100%;
  border-radius: 999px;
}}

.empty-hint {{
  text-align: center;
  padding: 2.5rem 1.5rem;
  border: 1px dashed {t['border']};
  border-radius: 16px;
  background: {t['surface_alt']};
  color: {t['text_muted']};
}}

.empty-hint strong {{
  color: {t['text']};
  display: block;
  font-size: 1.05rem;
  margin-bottom: 0.35rem;
}}

/* Per-file status dots in upload table */
.file-status {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
}}

.status-dot {{
  display: inline-block;
  width: 11px;
  height: 11px;
  border-radius: 50%;
  flex-shrink: 0;
}}

.status-ready {{
  background: {t['text_faint']};
  opacity: 0.45;
}}

.status-spin {{
  border: 2px solid {t['accent']};
  border-top-color: transparent;
  background: transparent;
  animation: file-status-spin 0.75s linear infinite;
}}

.status-done {{
  background: #22c55e;
  box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.25);
}}

.status-error {{
  background: #ef4444;
  box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.25);
}}

@keyframes file-status-spin {{
  to {{ transform: rotate(360deg); }}
}}

table.data-table th.col-sr,
table.data-table td.col-sr {{
  width: 42px;
  text-align: center;
  padding-left: 0.5rem;
  padding-right: 0.25rem;
  color: {t['text_faint']};
  font-weight: 600;
  font-size: 0.8rem;
}}

table.data-table th.col-indicator,
table.data-table td.col-indicator {{
  width: 36px;
  text-align: center;
  padding-left: 0.25rem;
  padding-right: 0.25rem;
}}

table.data-table th.col-play,
table.data-table td.col-play {{
  width: 40px;
  text-align: center;
  padding-left: 0.35rem;
  padding-right: 0.35rem;
}}

.audio-play-cell {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
}}

.audio-play-cell audio {{
  display: none;
}}

.audio-play-btn {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  padding: 0;
  border: 1px solid {t['border']};
  border-radius: 50%;
  background: {t['accent_soft']};
  color: {t['accent']};
  font-size: 0.72rem;
  line-height: 1;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}}

.audio-play-btn:hover {{
  background: {t['accent']};
  color: {t['uploader_btn_text']};
  border-color: {t['accent']};
}}

.audio-play-btn--disabled {{
  opacity: 0.35;
  cursor: not-allowed;
  pointer-events: none;
}}

.transcript-audio-bar {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
  padding: 0.65rem 0.85rem;
  border: 1px solid {t['border_soft']};
  border-radius: 12px;
  background: {t['surface_alt']};
}}

.transcript-audio-label {{
  color: {t['text_muted']};
  font-size: 0.88rem;
  word-break: break-all;
}}
"""
    return f"<style>{rules}</style>"


def file_status_label_html(status: str) -> str:
    label = {
        "ready": "Waiting",
        "processing": "Processing",
        "done": "Done",
        "error": "Failed",
    }.get(status, "Waiting")
    css_class = {
        "ready": "status-text-ready",
        "processing": "status-text-processing",
        "done": "status-text-done",
        "error": "status-text-error",
    }.get(status, "status-text-ready")
    return f'<span class="{css_class}">{html_module.escape(label)}</span>'


def audio_element_id(prefix: str, filename: str, index: int = 0) -> str:
    digest = hashlib.md5(filename.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", prefix)[:12]
    return f"{safe_prefix}-{index}-{digest}"


_AUDIO_TOGGLE_SCRIPT = """
<script>
function toggleCallAudio(btn) {
  const audioId = btn.getAttribute("data-audio-id");
  const a = document.getElementById(audioId);
  if (!a) return;
  const playIcon = btn.getAttribute("data-play-icon") || "▶";
  const pauseIcon = btn.getAttribute("data-pause-icon") || "⏸";
  if (a.paused || a.ended) {
    document.querySelectorAll("audio").forEach(function (x) {
      if (x === a) return;
      x.pause();
      const otherBtn = document.querySelector('[data-audio-id="' + x.id + '"]');
      if (otherBtn) {
        otherBtn.textContent = playIcon;
        otherBtn.title = "Play recording";
        otherBtn.classList.remove("is-playing");
      }
    });
    a.play().catch(function () {});
    btn.textContent = pauseIcon;
    btn.title = "Pause";
    btn.classList.add("is-playing");
  } else {
    a.pause();
    btn.textContent = playIcon;
    btn.title = "Play recording";
    btn.classList.remove("is-playing");
  }
}
document.addEventListener("ended", function (e) {
  if (!e.target || e.target.tagName !== "AUDIO") return;
  const btn = document.querySelector('[data-audio-id="' + e.target.id + '"]');
  if (!btn) return;
  btn.textContent = btn.getAttribute("data-play-icon") || "▶";
  btn.title = "Play recording";
  btn.classList.remove("is-playing");
}, true);
</script>
"""


def audio_play_button_html(
    element_id: str,
    data: bytes,
    mime: str = "audio/mpeg",
    *,
    max_bytes: int = MAX_INLINE_AUDIO_BYTES,
) -> str:
    """Inline play/pause control for queue table (used inside components.html)."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", element_id)
    audio_dom_id = f"aud-{safe_id}"
    if len(data) > max_bytes:
        return (
            '<span class="audio-play-cell">'
            '<span class="audio-play-btn audio-play-btn--disabled" '
            'title="File too large for inline preview (over 8 MB)">▶</span>'
            "</span>"
        )
    b64 = base64.b64encode(data).decode("ascii")
    safe_mime = html_module.escape(mime or "audio/mpeg")
    return (
        f'<span class="audio-play-cell">'
        f'<audio id="{audio_dom_id}" preload="none" '
        f'src="data:{safe_mime};base64,{b64}"></audio>'
        f'<button type="button" class="audio-play-btn" data-audio-id="{audio_dom_id}" '
        f'data-play-icon="▶" data-pause-icon="⏸" title="Play recording" '
        f'onclick="toggleCallAudio(this)">▶</button>'
        f"</span>"
    )


def file_status_indicator_html(status: str) -> str:
    css_class = {
        "ready": "status-ready",
        "processing": "status-spin",
        "done": "status-done",
        "error": "status-error",
    }.get(status, "status-ready")
    title = {
        "ready": "Waiting",
        "processing": "Processing",
        "done": "Done",
        "error": "Failed",
    }.get(status, "Waiting")
    return (
        f'<span class="file-status"><span class="status-dot {css_class}" '
        f'title="{html_module.escape(title)}"></span></span>'
    )


def _queue_iframe_css(t: Dict[str, str]) -> str:
    return f"""
body {{
  margin: 0;
  font-family: "Source Sans Pro", sans-serif;
  background: {t['dataframe_bg']};
  color: {t['text_muted']};
}}
.data-table-wrap.queue-table-scroll {{
  max-height: inherit;
  overflow-y: auto;
  overflow-x: auto;
  border: 1px solid {t['border_soft']};
  border-radius: 12px;
  background: {t['dataframe_bg']};
}}
table.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
}}
table.data-table thead {{
  position: sticky;
  top: 0;
  z-index: 1;
}}
table.data-table th {{
  background: {t['dataframe_header']};
  color: {t['text']};
  font-weight: 700;
  text-align: left;
  padding: 0.65rem 0.85rem;
  border-bottom: 1px solid {t['border']};
  white-space: nowrap;
}}
table.data-table td {{
  color: {t['text_muted']};
  padding: 0.55rem 0.85rem;
  border-bottom: 1px solid {t['border_soft']};
}}
table.data-table tbody tr:hover td {{
  background: {t['surface_alt']};
  color: {t['text']};
}}
table.data-table tbody tr.row-processing td {{
  background: {t['accent_soft']} !important;
  color: {t['text']} !important;
}}
table.data-table th.col-sr, table.data-table td.col-sr {{
  width: 42px; text-align: center; color: {t['text_faint']}; font-weight: 600;
}}
table.data-table th.col-play, table.data-table td.col-play {{
  width: 40px; text-align: center;
}}
table.data-table th.col-indicator, table.data-table td.col-indicator {{
  width: 36px; text-align: center;
}}
.status-text-ready {{ color: {t['text_faint']}; }}
.status-text-processing {{ color: {t['accent']}; font-weight: 700; }}
.status-text-done {{ color: #22c55e; font-weight: 700; }}
.status-text-error {{ color: #ef4444; font-weight: 700; }}
.status-dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; }}
.status-ready {{ background: {t['text_faint']}; opacity: 0.45; }}
.status-spin {{
  border: 2px solid {t['accent']}; border-top-color: transparent; background: transparent;
  animation: spin 0.75s linear infinite;
}}
.status-done {{ background: #22c55e; }}
.status-error {{ background: #ef4444; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.audio-play-cell {{ display: inline-flex; align-items: center; justify-content: center; }}
.audio-play-cell audio {{ display: none; }}
.audio-play-btn {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 28px; height: 28px; padding: 0; border: 1px solid {t['border']};
  border-radius: 50%; background: {t['accent_soft']}; color: {t['accent']};
  font-size: 0.72rem; cursor: pointer;
}}
.audio-play-btn:hover {{
  background: {t['accent']}; color: {t['uploader_btn_text']}; border-color: {t['accent']};
}}
.audio-play-btn.is-playing {{
  background: {t['accent']};
  color: {t['uploader_btn_text']};
  border-color: {t['accent']};
}}
.audio-play-btn--disabled {{ opacity: 0.35; cursor: not-allowed; pointer-events: none; }}
"""


def render_audio_play_component(
    data: bytes,
    mime: str,
    element_id: str,
    theme: str,
) -> None:
    """Play button that works in Streamlit (runs inside components.html where JS is allowed)."""
    import streamlit.components.v1 as components

    t = THEME_TOKENS.get(theme, THEME_TOKENS["dark"])
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", element_id)
    if len(data) > MAX_INLINE_AUDIO_BYTES:
        components.html(
            f'<p style="margin:0;color:{t["text_muted"]};font-size:0.85rem;">'
            "Recording too large for inline preview (over 8 MB).</p>",
            height=36,
        )
        return

    b64 = base64.b64encode(data).decode("ascii")
    safe_mime = html_module.escape(mime or "audio/mpeg")
    audio_dom_id = f"aud-{safe_id}"
    components.html(
        f"""
        <style>{_queue_iframe_css(t)}</style>
        {_AUDIO_TOGGLE_SCRIPT}
        <span class="audio-play-cell">
          <audio id="{audio_dom_id}" preload="none" src="data:{safe_mime};base64,{b64}"></audio>
          <button type="button" class="audio-play-btn" data-audio-id="{audio_dom_id}"
            data-play-icon="▶" data-pause-icon="⏸" title="Play recording"
            onclick="toggleCallAudio(this)">▶</button>
        </span>
        """,
        height=44,
    )


def render_queue_table_component(
    rows: list[Dict[str, Any]],
    theme: str,
    *,
    max_height_px: int = 280,
) -> None:
    """Queue table with working ▶ play buttons (Streamlit blocks onclick in st.markdown)."""
    import streamlit.components.v1 as components

    t = THEME_TOKENS.get(theme, THEME_TOKENS["dark"])
    body_rows: list[str] = []
    has_processing = False

    for row in rows:
        status = str(row.get("status", "ready"))
        if status == "processing":
            has_processing = True
        tr_attrs = ""
        if status == "processing":
            tr_attrs = ' class="row-processing" id="queue-row-processing"'
        elif status:
            tr_attrs = f' class="row-{html_module.escape(status)}"'

        aid = audio_element_id("queue", str(row["name"]), int(row["sr"]))
        play_html = audio_play_button_html(
            aid, row["data"], str(row.get("mime", "audio/mpeg"))
        )
        body_rows.append(
            f"<tr{tr_attrs}>"
            f'<td class="col-sr">{int(row["sr"])}</td>'
            f'<td class="col-play">{play_html}</td>'
            f'<td class="col-indicator">{file_status_indicator_html(status)}</td>'
            f"<td>{html_module.escape(str(row['name']))}</td>"
            f"<td>{html_module.escape(str(row['duration']))}</td>"
            f"<td>{html_module.escape(str(row['size']))}</td>"
            f"<td>{html_module.escape(str(row['date']))}</td>"
            f"<td>{file_status_label_html(status)}</td>"
            f"</tr>"
        )

    scroll_script = ""
    if has_processing:
        scroll_script = """
<script>
(function () {
  const wrap = document.querySelector(".queue-table-scroll");
  const row = document.getElementById("queue-row-processing");
  if (!wrap || !row) return;
  const offset = row.offsetTop - wrap.offsetTop - (wrap.clientHeight / 2) + (row.clientHeight / 2);
  wrap.scrollTop = Math.max(0, offset);
})();
</script>
"""

    table_html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{_queue_iframe_css(t)}</style></head>
<body>
  <div class="data-table-wrap queue-table-scroll" style="max-height:{max_height_px}px;">
    <table class="data-table">
      <thead><tr>
        <th class="col-sr">Sr.</th>
        <th class="col-play">▶</th>
        <th class="col-indicator"></th>
        <th>Recording file</th>
        <th>Estimated duration</th>
        <th>Size</th>
        <th>Date</th>
        <th>Status</th>
      </tr></thead>
      <tbody>{"".join(body_rows)}</tbody>
    </table>
  </div>
  {_AUDIO_TOGGLE_SCRIPT}
  {scroll_script}
</body>
</html>
"""
    components.html(table_html, height=max_height_px + 12, scrolling=False)


def render_themed_table(
    df: pd.DataFrame,
    theme: str,
    *,
    cell_style: Optional[Dict[str, Callable[[Any], str]]] = None,
    html_columns: Optional[set[str]] = None,
    max_height_px: int = 420,
    wrap_class: str = "",
) -> None:
    """Render a scrollable HTML table that respects light/dark theme (avoids white st.dataframe)."""
    cell_style = cell_style or {}
    html_columns = html_columns or set()
    display_cols = [col for col in df.columns if not str(col).startswith("_")]

    def _col_class(column: str) -> str:
        if column == "Sr.":
            return "col-sr"
        if column == "":
            return "col-indicator"
        if column == "▶":
            return "col-play"
        return ""

    header_cells: list[str] = []
    for col in display_cols:
        css = _col_class(str(col))
        th_class = f' class="{css}"' if css else ""
        if col == "":
            label = ""
        elif col == "▶":
            label = "▶"
        else:
            label = html_module.escape(str(col))
        header_cells.append(f"<th{th_class}>{label}</th>")
    header_html = "".join(header_cells)

    body_rows: list[str] = []
    for _, row in df.iterrows():
        row_status = str(row["_status"]) if "_status" in df.columns else ""
        tr_attrs = ""
        if row_status == "processing":
            tr_attrs = ' class="row-processing" id="queue-row-processing"'
        elif row_status:
            tr_attrs = f' class="row-{html_module.escape(row_status)}"'

        cells: list[str] = []
        for col in display_cols:
            value = row[col]
            extra = ""
            if col in cell_style:
                extra = f' style="{cell_style[col](value)}"'
            css = _col_class(str(col))
            td_class = f' class="{css}"' if css else ""
            if col in html_columns:
                cells.append(f"<td{td_class}{extra}>{value}</td>")
            else:
                cells.append(f"<td{td_class}{extra}>{html_module.escape(str(value))}</td>")
        body_rows.append(f"<tr{tr_attrs}>{''.join(cells)}</tr>")

    wrap_classes = f"data-table-wrap {wrap_class}".strip()
    st.markdown(
        f"""
<div class="{wrap_classes}" style="max-height:{max_height_px}px;">
  <table class="data-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )


def score_cell_style(theme: str, value: float) -> str:
    t = THEME_TOKENS.get(theme, THEME_TOKENS["dark"])
    if value >= 85:
        return f"background-color: {t['success_bg']}; color: {t['success_fg']};"
    if value >= 70:
        return f"background-color: {t['warn_bg']}; color: {t['warn_fg']};"
    return f"background-color: {t['danger_bg']}; color: {t['danger_fg']};"


def chart_background(theme: str) -> str:
    return THEME_TOKENS.get(theme, THEME_TOKENS["dark"])["chart_bg"]


def wordcloud_colormap(theme: str) -> str:
    return "viridis" if theme == "dark" else "GnBu"
