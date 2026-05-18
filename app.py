"""
CAPTCHA Recognition — Streamlit Web Application
Run with: streamlit run app.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT          = Path(__file__).parent
SAVE_DIR      = ROOT / "saved_model"
WEIGHTS_PATH  = SAVE_DIR / "crnn_model.weights.h5"
META_PATH     = SAVE_DIR / "metadata.json"
HISTORY_PATH  = SAVE_DIR / "training_history.json"
PROGRESS_PATH = SAVE_DIR / "progress.json"
LOSS_CURVE    = SAVE_DIR / "loss_curve.png"
SAMPLE_GRID   = SAVE_DIR / "sample_predictions.png"
ZIP_PATH      = ROOT / "Captcha_Recognition_using_CRNN" / "captcha_images_v2.zip"
IMAGE_DIR     = ROOT / "Captcha_Recognition_using_CRNN" / "captcha_images_v2"

st.set_page_config(
    page_title="CAPTCHA Recognition | CRNN",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def _apply_theme():
    if st.session_state.theme == "light":
        st.markdown("""<style>
/* ── Core backgrounds (Streamlit 1.5x / emotion-cache aware) ─────────────── */
html, body,
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main,
[data-testid="stHeader"], .stAppHeader {
    background-color: #f5f4f0 !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] {
    background: #eeede9 !important;
    border-right: 1px solid rgba(200,169,124,0.15) !important;
}
/* ── Fix Streamlit's emotion-cache text-color propagation ─────────────────── */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stMarkdownContainer"],
[data-testid="stElementContainer"],
[data-testid="column"],
[data-testid="stSidebarContent"],
[data-testid="stMainBlockContainer"] {
    color: #1a1a1a !important;
}
/* ── Custom design elements ───────────────────────────────────────────────── */
.hero { background: #f0ede5 !important; border-bottom: 1px solid rgba(200,169,124,0.2) !important; }
.hero::before { background: radial-gradient(circle, rgba(200,169,124,0.1) 0%, transparent 65%) !important; }
.hero-title  { color: #1a1a1a !important; }
.hero-sub    { color: #7a7068 !important; }
.hero-eyebrow { color: #c8a97c !important; }
.marquee-wrap { background: #c8a97c !important; }
.marquee-track span { color: #f5f4f0 !important; }
.surface { background: #eceae3 !important; border-color: rgba(100,90,75,0.2) !important; }
.step-title  { color: #1a1a1a !important; }
.step-body   { color: #7a7068 !important; }
.sec-title   { color: #7a7068 !important; }
.stat-box, .stat-box-full { background: #eceae3 !important; border-color: rgba(100,90,75,0.2) !important; }
.pred-big    { background: #eceae3 !important; border-color: rgba(200,169,124,0.25) !important; }
.logo-name   { color: #1a1a1a !important; }
.logo-sub    { color: #7a7068 !important; }
.empty-state { background: rgba(0,0,0,0.03) !important; border-color: rgba(100,90,75,0.2) !important; }
.empty-txt   { color: #7a7068 !important; }
thead th     { background: rgba(0,0,0,0.04) !important; color: #7a7068 !important; }
tbody td     { color: #3d3530 !important; }
tbody tr:hover td { background: rgba(0,0,0,0.03) !important; color: #1a1a1a !important; }
/* ── Streamlit widgets ────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #eceae3 !important;
    border-color: rgba(100,90,75,0.25) !important;
    color: #1a1a1a !important;
}
[data-testid="stMetricLabel"]  { color: #7a7068 !important; }
[data-testid="stTab"]          { color: #7a7068 !important; }
[data-testid="stExpander"]     { background: #eceae3 !important; border-color: rgba(100,90,75,0.2) !important; }
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] > div {
    background: rgba(200,169,124,0.03) !important;
    border-color: rgba(200,169,124,0.25) !important;
}
::-webkit-scrollbar-track      { background: #e8e5dc !important; }
.sidebar-logo .logo-name       { color: #1a1a1a !important; }
/* ── Fix dropdown/select in light mode ───────────────────────────────────── */
[data-baseweb="select"] > div {
    background-color: #eceae3 !important;
    color: #1a1a1a !important;
    border-color: rgba(100,90,75,0.3) !important;
}
[data-baseweb="select"] svg {
    fill: #3d3530 !important;
}
[data-baseweb="popover"] {
    background-color: #eceae3 !important;
}
[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #eceae3 !important;
    color: #1a1a1a !important;
}
[data-testid="stSelectboxVirtualDropdown"] li,
[data-testid="stSelectboxVirtualDropdown"] [role="option"] {
    background-color: #eceae3 !important;
    color: #1a1a1a !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"] {
    background-color: rgba(200,169,124,0.18) !important;
    color: #1a1a1a !important;
}
/* ── Fix all buttons in light mode ───────────────────────────────────────── */
[data-testid="stBaseButton-secondary"] {
    background: rgba(200,169,124,0.07) !important;
    color: #3d3530 !important;
    border: 1px solid rgba(100,90,75,0.25) !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(200,169,124,0.16) !important;
    color: #1a1a1a !important;
    border-color: rgba(200,169,124,0.45) !important;
    box-shadow: none !important;
}
[data-testid="stBaseButton-secondary"]:focus,
[data-testid="stBaseButton-secondary"]:active {
    background: rgba(200,169,124,0.12) !important;
    color: #1a1a1a !important;
    box-shadow: 0 0 0 2px rgba(200,169,124,0.35) !important;
}
/* ── Fix tooltip popup in light mode ─────────────────────────────────────── */
[data-testid="stTooltipHoverTarget"] + div,
div[role="tooltip"],
.stTooltipContent,
[data-testid="stTooltip"] {
    background-color: #eceae3 !important;
    color: #1a1a1a !important;
    border: 1px solid rgba(100,90,75,0.2) !important;
}
</style>""", unsafe_allow_html=True)

# ─── Design System: lifted from string-tune.fiddle.digital ────────────────────
#  Colors: --c-black #101214 · --c-yellow #c8a97c · --c-red #ff4f36
#          --c-blue #3687ff  · --c-green #4dbf9d  · --c-berry #c8c2cf
#          --c-purple #a399a8 · --c-grey-dark #544d56
#  Easings: --f-cubic cubic-bezier(.35,.35,0,1) · --f-bounce cubic-bezier(.6,.5,0,3)
#  Animations: marquee, blinking, Kerning, opacityGlare, ScrollSmoothly
st.markdown("""
<style>
/* ── Fonts ───────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #101214 !important;
}

/* ── CSS custom properties (mirroring string-tune tokens) ───────────────── */
:root {
    --c-black:      #101214;
    --c-white:      #fff;
    --c-yellow:     #c8a97c;
    --c-red:        #ff4f36;
    --c-blue:       #3687ff;
    --c-green:      #4dbf9d;
    --c-berry:      #c8c2cf;
    --c-purple:     #a399a8;
    --c-grey-dark:  #544d56;
    --c-grey:       #e6dfe4;

    --f-cubic:      cubic-bezier(0.35, 0.35, 0, 1);
    --f-bounce:     cubic-bezier(0.6, 0.5, 0, 3);
    --f-bounce-alt: cubic-bezier(0.6, 0.5, 0, 2);
    --f-fast:       cubic-bezier(0, 0.81, 0.35, 1);
}

/* ── Full page dark background ──────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main {
    background-color: #101214 !important;
}
[data-testid="stSidebar"] {
    background: #0c0e10 !important;
    border-right: 1px solid rgba(200,169,124,0.08) !important;
}

/* ── Marquee ticker ─────────────────────────────────────────────────────── */
.marquee-wrap {
    overflow: hidden;
    background: var(--c-yellow);
    padding: 10px 0;
    position: relative;
    margin-bottom: 0;
}
.marquee-track {
    display: flex;
    width: max-content;
    animation: marquee 80s linear infinite;
}
.marquee-track span {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: #101214;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0 2.5rem;
    white-space: nowrap;
    flex-shrink: 0;
}
.marquee-track span::before {
    content: "·";
    margin-right: 2.5rem;
    color: rgba(16,18,20,0.4);
}
@keyframes marquee {
    from { transform: translate3d(0, 0, 0) rotate(0.01deg); }
    to   { transform: translate3d(-50%, 0, 0) rotate(0.01deg); }
}

/* ── Hero section ───────────────────────────────────────────────────────── */
.hero {
    background: #101214;
    padding: 56px 0 40px 0;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid rgba(200,169,124,0.1);
}
.hero::before {
    content: '';
    position: absolute;
    top: -120px; left: -100px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(200,169,124,0.06) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -60px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,79,54,0.06) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--c-yellow);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px;
    height: 2px;
    background: var(--c-yellow);
}
.hero-title {
    font-size: clamp(2.4rem, 5vw, 4.2rem);
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: var(--c-white);
    margin: 0 0 16px 0;
    animation: Kerning 3.5s var(--f-cubic) both;
}
.hero-title em {
    font-style: normal;
    color: var(--c-yellow);
}
@keyframes Kerning {
    0%   { letter-spacing: -0.1em; opacity: 0; translate: 0 12px; }
    12%  { letter-spacing: 0.02em; }
    100% { letter-spacing: -0.03em; opacity: 1; translate: 0 0; }
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--c-grey-dark);
    max-width: 520px;
    line-height: 1.65;
    margin-bottom: 28px;
}
.hero-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 5px 13px;
    border-radius: 3px;
    border: 1px solid;
}
.tag-y { color: #101214; background: var(--c-yellow); border-color: var(--c-yellow); }
.tag-r { color: var(--c-red);  background: rgba(255,79,54,0.1); border-color: rgba(255,79,54,0.3); }
.tag-b { color: var(--c-blue); background: rgba(54,135,255,0.1); border-color: rgba(54,135,255,0.3); }
.tag-g { color: var(--c-green);background: rgba(77,191,157,0.1); border-color: rgba(77,191,157,0.3); }
.tag-w { color: var(--c-berry);background: rgba(200,194,207,0.08); border-color: rgba(200,194,207,0.2); }

/* ── Hexagon stat card ──────────────────────────────────────────────────── */
.hex-stats {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 20px;
    flex-wrap: wrap;
}
.hex-card {
    position: relative;
    width: 120px; height: 138px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    clip-path: polygon(50% 0, 93% 25%, 93% 75%, 50% 100%, 7% 75%, 7% 25%);
    background: rgba(200,169,124,0.06);
    border: 0;
    transition: all 0.45s cubic-bezier(0.6, 0.5, 0, 2);
    cursor: default;
}
.hex-card:hover {
    background: rgba(200,169,124,0.12);
    transform: translateY(-4px) scale(1.05);
}
.hex-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--c-yellow);
    line-height: 1;
    margin-bottom: 4px;
}
.hex-lbl {
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--c-grey-dark);
    text-align: center;
    padding: 0 8px;
}

/* ── Glare button ──────────────────────────────────────────────────────── */
.glare-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    background: rgba(200,169,124,0.08);
    border: 1px solid rgba(200,169,124,0.25);
    border-radius: 4px;
    padding: 11px 28px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--c-yellow);
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: background 0.6s var(--f-cubic),
                border-color 0.6s var(--f-cubic),
                scale 0.45s cubic-bezier(0.6, 0.5, 0, 2),
                box-shadow 0.6s var(--f-cubic);
    text-decoration: none;
}
.glare-btn::after {
    content: '';
    position: absolute;
    inset: 0;
    background-image: linear-gradient(
        -45deg,
        rgba(200,169,124,0.2) 0%,
        rgba(200,169,124,0.0) 40%,
        rgba(200,169,124,0.0) 60%,
        rgba(200,169,124,0.2) 100%
    );
    border-radius: inherit;
    transition: opacity 0.6s var(--f-cubic);
}
.glare-btn:hover {
    background: rgba(200,169,124,0.14);
    border-color: rgba(200,169,124,0.55);
    box-shadow: 0 0 28px rgba(200,169,124,0.12), 0 0 60px rgba(200,169,124,0.05);
    scale: 1.03;
}

/* ── Section title ─────────────────────────────────────────────────────── */
.sec-title {
    font-size: 0.68rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--c-grey-dark);
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 14px;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(84,77,86,0.5) 0%, transparent 100%);
}

/* ── Metric cards (Streamlit native override) ───────────────────────────── */
[data-testid="metric-container"] {
    background: #131619;
    border: 1px solid rgba(84,77,86,0.4);
    border-radius: 6px;
    padding: 18px 20px;
    transition: border-color 0.45s var(--f-cubic),
                box-shadow 0.45s var(--f-cubic),
                transform 0.45s cubic-bezier(0.6, 0.5, 0, 2);
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--c-yellow);
    opacity: 0;
    transition: opacity 0.3s var(--f-cubic);
}
[data-testid="metric-container"]:hover {
    border-color: rgba(200,169,124,0.25);
    box-shadow: 0 0 30px rgba(200,169,124,0.06), 4px 0 0 var(--c-yellow) inset;
    transform: translateY(-2px);
}
[data-testid="metric-container"]:hover::before { opacity: 1; }
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: var(--c-yellow) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--c-grey-dark) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
/* ── Smaller metrics inside expanders so values fit the narrow columns ── */
[data-testid="stExpander"] [data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
}
[data-testid="stExpander"] [data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    letter-spacing: 0.06em;
}

/* ── Info / status cards ─────────────────────────────────────────────────── */
.status-ready {
    display: flex; align-items: center; gap: 10px;
    background: rgba(77,191,157,0.06);
    border: 1px solid rgba(77,191,157,0.22);
    border-radius: 5px;
    padding: 11px 18px;
}
.status-warn {
    display: flex; align-items: center; gap: 10px;
    background: rgba(255,79,54,0.06);
    border: 1px solid rgba(255,79,54,0.22);
    border-radius: 5px;
    padding: 11px 18px;
}
.dot-live {
    width: 7px; height: 7px;
    background: var(--c-green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--c-green);
    flex-shrink: 0;
    animation: blinking 1.6s ease-in-out infinite;
}
.dot-warn {
    width: 7px; height: 7px;
    background: var(--c-red);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--c-red);
    flex-shrink: 0;
    animation: blinking 1s ease-in-out infinite;
}
@keyframes blinking {
    0%   { opacity: 1; }
    10%  { opacity: 0; }
    20%  { opacity: 1; }
    100% { opacity: 1; }
}

/* ── Surface card ───────────────────────────────────────────────────────── */
.surface {
    background: #131619;
    border: 1px solid rgba(84,77,86,0.3);
    border-radius: 6px;
    padding: 24px 28px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.surface::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle, rgba(200,169,124,0.05) 0%, transparent 70%);
    pointer-events: none;
}

/* ── How-it-works steps ─────────────────────────────────────────────────── */
.step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px solid rgba(84,77,86,0.2);
}
.step:last-child { border-bottom: none; }
.step-num {
    width: 28px; height: 28px;
    flex-shrink: 0;
    background: rgba(200,169,124,0.08);
    border: 1px solid rgba(200,169,124,0.25);
    border-radius: 3px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--c-yellow);
}
.step-title { font-size: 0.9rem; font-weight: 600; color: var(--c-white); margin-bottom: 3px; }
.step-body  { font-size: 0.78rem; color: var(--c-grey-dark); line-height: 1.55; }

/* ── Sidebar overrides ─────────────────────────────────────────────────── */
.sidebar-logo {
    text-align: center;
    padding: 12px 0 20px 0;
}
.logo-hex {
    width: 56px; height: 64px;
    margin: 0 auto 10px;
    clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
    background: rgba(200,169,124,0.1);
    border: none;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    animation: ScrollSmoothly 4s var(--f-cubic) infinite alternate;
}
@keyframes ScrollSmoothly {
    0%   { transform: translateY(-3px); }
    100% { transform: translateY(3px); }
}
.logo-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--c-white);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.logo-sub {
    font-size: 0.68rem;
    color: var(--c-grey-dark);
    letter-spacing: 0.08em;
    margin-top: 3px;
}
.stat-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 14px 0;
}
.stat-box {
    background: #131619;
    border: 1px solid rgba(84,77,86,0.35);
    border-radius: 5px;
    padding: 12px 10px;
    text-align: center;
    transition: border-color 0.3s var(--f-cubic), transform 0.45s cubic-bezier(0.6,0.5,0,2);
}
.stat-box:hover {
    border-color: rgba(200,169,124,0.3);
    transform: scale(1.03);
}
.stat-val { font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; color:var(--c-yellow); }
.stat-lbl { font-size:0.6rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--c-grey-dark); margin-top:3px; }
.stat-box-full {
    background: #131619;
    border: 1px solid rgba(84,77,86,0.35);
    border-radius: 5px;
    padding: 10px 14px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
}
.stat-box-full .stat-val { font-size:0.95rem; }
.stat-box-full .stat-lbl { font-size:0.6rem; text-align:left; }

/* ── Progress bar ───────────────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--c-yellow), var(--c-red)) !important;
    border-radius: 2px !important;
}

/* ── Predict result card ─────────────────────────────────────────────────── */
.pred-big {
    background: #131619;
    border: 1px solid rgba(200,169,124,0.18);
    border-radius: 6px;
    padding: 36px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-big::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--c-yellow), transparent);
    animation: borderScan 3s var(--f-cubic) infinite;
}
@keyframes borderScan {
    0%   { opacity: 0; transform: scaleX(0); transform-origin: left; }
    50%  { opacity: 1; transform: scaleX(1); transform-origin: left; }
    51%  { transform-origin: right; }
    100% { opacity: 0; transform: scaleX(0); transform-origin: right; }
}
.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--c-grey-dark);
    margin-bottom: 12px;
}
.pred-chars {
    font-family: 'Space Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    color: var(--c-yellow);
    text-shadow: 0 0 40px rgba(200,169,124,0.25), 0 0 80px rgba(200,169,124,0.08);
    margin: 0;
    line-height: 1;
}
.conf-bar-wrap {
    margin-top: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: center;
}
.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* ── Sample grid labels ─────────────────────────────────────────────────── */
.sample-ok  { font-family:'Space Mono',monospace; font-size:0.8rem; color:var(--c-green); font-weight:700; }
.sample-err { font-family:'Space Mono',monospace; font-size:0.8rem; color:var(--c-red);   font-weight:700; }
.sample-sub { font-size:0.72rem; color:var(--c-grey-dark); margin-top:1px; }

/* ── Char vocab chip ────────────────────────────────────────────────────── */
.vocab-wrap { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }
.vocab-chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: var(--c-berry);
    background: rgba(200,194,207,0.07);
    border: 1px solid rgba(200,194,207,0.15);
    border-radius: 3px;
    padding: 4px 10px;
    transition: background 0.3s var(--f-cubic), border-color 0.3s, color 0.3s, transform 0.45s cubic-bezier(0.6,0.5,0,2);
}
.vocab-chip:hover {
    background: rgba(200,169,124,0.08);
    border-color: rgba(200,169,124,0.3);
    color: var(--c-yellow);
    transform: translateY(-2px) scale(1.08);
}

/* ── Architecture table ─────────────────────────────────────────────────── */
table { width:100%; border-collapse:collapse; }
thead th {
    font-family:'Space Mono',monospace;
    font-size:0.65rem;
    text-transform:uppercase;
    letter-spacing:0.12em;
    color:var(--c-grey-dark);
    padding:10px 14px;
    border-bottom:1px solid rgba(84,77,86,0.4);
    text-align:left;
    background:rgba(16,18,20,0.5);
}
tbody td {
    font-size:0.83rem;
    color:var(--c-berry);
    padding:9px 14px;
    border-bottom:1px solid rgba(84,77,86,0.15);
}
tbody tr:hover td {
    background: rgba(200,169,124,0.025);
    color: var(--c-grey);
}

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0e1012 !important;
    border: 1px solid rgba(84,77,86,0.3) !important;
    border-radius: 5px !important;
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(200,169,124,0.2) !important;
    border-radius: 6px !important;
    background: rgba(200,169,124,0.02) !important;
    transition: border-color 0.3s var(--f-cubic) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(200,169,124,0.4) !important;
    background: rgba(200,169,124,0.04) !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
[data-testid="stTab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--c-grey-dark) !important;
}
[data-testid="stTab"][aria-selected="true"] {
    color: var(--c-yellow) !important;
    border-bottom-color: var(--c-yellow) !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: #0c0e10; }
::-webkit-scrollbar-thumb { background: #544d56; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-yellow); }

/* ── Empty state ─────────────────────────────────────────────────────────── */
.empty-state {
    border: 1px dashed rgba(84,77,86,0.35);
    border-radius: 6px;
    padding: 52px 24px;
    text-align: center;
    background: rgba(16,18,20,0.4);
}
.empty-icon { font-size:2.2rem; margin-bottom:12px; }
.empty-txt  { font-size:0.85rem; color:var(--c-grey-dark); }

</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
for key, val in [("model",None),("metadata",None),("history",None),("training",False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model_cached(weights_path: str, meta_path: str, mtime: float = 0):
    from data_utils import load_metadata
    from model import build_crnn_model, ctc_loss_fn
    meta = load_metadata(meta_path)
    model = build_crnn_model(
        image_height=meta["image_height"], image_width=meta["image_width"],
        num_classes=len(meta["all_chars"]), use_attention=meta.get("use_attention", True),
    )
    model.compile(optimizer="adam", loss=ctc_loss_fn)
    model.load_weights(weights_path)
    return model, meta

def _weights_mtime() -> float:
    return WEIGHTS_PATH.stat().st_mtime if WEIGHTS_PATH.exists() else 0.0

def _meta_mtime() -> float:
    return META_PATH.stat().st_mtime if META_PATH.exists() else 0.0

def try_load_model():
    if WEIGHTS_PATH.exists() and META_PATH.exists():
        return load_model_cached(str(WEIGHTS_PATH), str(META_PATH), _weights_mtime())
    return None, None

def model_is_ready() -> bool:
    return WEIGHTS_PATH.exists() and META_PATH.exists()

@st.cache_data
def _load_meta_dict(path: str, mtime: float = 0) -> dict:
    return json.loads(Path(path).read_text())

@st.cache_data
def _load_history_dict(path: str, mtime: float = 0) -> dict:
    return json.loads(Path(path).read_text())

def predict_image(pil_img: Image.Image, model, meta: dict) -> dict:
    from data_utils import decode_predictions
    h, w = meta["image_height"], meta["image_width"]
    img  = pil_img.convert("L").resize((w, h))
    arr  = np.array(img, dtype=np.float32)[..., np.newaxis]
    arr  = np.expand_dims(arr, 0)
    raw  = model.predict(arr, verbose=0)
    pred = decode_predictions(raw, meta["int_to_char"])[0]
    probs = raw[0]
    conf  = float(probs.max(axis=-1).mean())
    return {"text": pred, "confidence": conf, "probs": probs}

def dark_fig(w=10, h=4, ncols=1):
    light = st.session_state.get("theme") == "light"
    BG    = "#f5f4f0" if light else "#0d0f11"
    AX    = "#eceae3" if light else "#111418"
    TC    = "#7a7068" if light else "#544d56"
    SC    = "#d4cfcb" if light else "#1c2026"
    TITLE = "#3d3530" if light else "#c8c2cf"
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    axs = [axes] if ncols == 1 else list(axes)
    for ax in axs:
        ax.set_facecolor(AX)
        ax.tick_params(colors=TC, labelsize=8.5)
        ax.xaxis.label.set_color(TC); ax.yaxis.label.set_color(TC)
        ax.title.set_color(TITLE)
        for sp in ax.spines.values(): sp.set_edgecolor(SC)
    return fig, (axs if ncols > 1 else axs[0])


# ─── Marquee ticker (always visible) ─────────────────────────────────────────
_items = (
    "CRNN Architecture" + "   " +
    "CTC Loss Decoding" + "   " +
    "BiLSTM × 2" + "   " +
    "Multi-Head Attention" + "   " +
    "98.1% Word Accuracy" + "   " +
    "VGG Double Conv Blocks" + "   " +
    "Data Augmentation" + "   " +
    "TensorFlow 2.21" + "   "
) * 3
st.markdown(f"""
<div class="marquee-wrap">
  <div class="marquee-track">
    {"".join(f'<span>{item.strip()}</span>' for item in _items.split("   ") if item.strip())}
    {"".join(f'<span>{item.strip()}</span>' for item in _items.split("   ") if item.strip())}
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-hex">🔐</div>
        <div class="logo-name">CAPTCHA AI</div>
        <div class="logo-sub">CRNN · CTC · Attention</div>
    </div>
    """, unsafe_allow_html=True)
    # Theme toggle
    is_light = st.session_state.theme == "light"
    icon  = "☀️" if is_light else "🌙"
    label = "Light Mode" if is_light else "Dark Mode"
    col_tgl, col_lbl = st.columns([1, 3])
    with col_tgl:
        if st.button(icon, key="theme_toggle", help="Toggle light / dark theme",
                     use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    with col_lbl:
        st.markdown(f'<div style="font-size:0.75rem;color:#544d56;padding-top:10px;font-weight:500;">{label}</div>',
                    unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid rgba(84,77,86,0.3);margin:8px 0 16px 0;">', unsafe_allow_html=True)

    if st.button("↺  Reload Model", use_container_width=True,
                 help="Force-reload weights from disk (clears cache)"):
        load_model_cached.clear()
        _load_meta_dict.clear()
        _load_history_dict.clear()
        st.rerun()

    if model_is_ready():
        meta_raw = _load_meta_dict(str(META_PATH), _meta_mtime())
        m = meta_raw.get("metrics", {})
        wa  = m.get("word_accuracy", 0)
        ca  = m.get("char_accuracy", 0)
        cer = m.get("cer", 0)
        correct = m.get("correct", "?")
        total   = m.get("total", "?")

        st.markdown(f"""
        <div class="status-ready">
            <span class="dot-live"></span>
            <span style="font-size:0.82rem;font-weight:600;color:#4dbf9d;">Model Ready</span>
        </div>
        <div class="stat-row" style="margin-top:14px;">
            <div class="stat-box">
                <div class="stat-val">{wa*100:.1f}%</div>
                <div class="stat-lbl">Word Acc</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{ca*100:.1f}%</div>
                <div class="stat-lbl">Char Acc</div>
            </div>
        </div>
        <div class="stat-box-full">
            <div>
                <div class="stat-lbl">Char Error Rate</div>
                <div class="stat-val" style="color:#ff4f36;">{cer*100:.2f}%</div>
            </div>
            <div>
                <div class="stat-lbl">Test Correct</div>
                <div class="stat-val" style="font-size:0.9rem;">{correct}/{total}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-warn">
            <span class="dot-warn"></span>
            <span style="font-size:0.82rem;font-weight:600;color:#ff4f36;">No model found</span>
        </div>
        <div style="font-size:0.76rem;color:var(--c-grey-dark);margin-top:10px;padding:0 2px;">
            Go to the <b style="color:#c8c2cf;">Retrain</b> tab to train a model.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title" style="margin-top:20px;">Retrain Config</div>', unsafe_allow_html=True)
    cfg_epochs  = st.slider("Epochs", 10, 60, 30, step=5)
    cfg_batch   = st.selectbox("Batch Size", [8, 16, 32], index=1)
    cfg_lr      = st.select_slider("Learning Rate", options=[1e-4,5e-4,1e-3,3e-3], value=1e-3,
                                   format_func=lambda x: f"{x:.0e}")
    cfg_augment = st.checkbox("Data Augmentation", value=True)
    cfg_attn    = st.checkbox("Self-Attention", value=True)

    st.markdown('<hr style="border:none;border-top:1px solid rgba(84,77,86,0.2);margin:16px 0 8px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.62rem;text-align:center;color:rgba(84,77,86,0.6);letter-spacing:0.08em;">TensorFlow · Streamlit · CRNN</div>', unsafe_allow_html=True)


# Apply theme class to document body
_apply_theme()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_predict, tab_dashboard, tab_train = st.tabs(["Predict", "Dashboard", "Retrain"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB — RETRAIN (optional, model ships pre-trained)
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    # ── Hero ────────────────────────────────────────────────────────────────
    hero_col, stats_col = st.columns([3, 1], gap="large")

    with hero_col:
        if model_is_ready():
            _hero_meta = _load_meta_dict(str(META_PATH), _meta_mtime())
            _wa = _hero_meta.get("metrics", {}).get("word_accuracy", 0)
            _acc_badge = f"{_wa*100:.1f}% Accuracy"
        else:
            _acc_badge = "98.1% Accuracy"
        st.markdown(f"""
        <div class="hero">
            <div class="hero-eyebrow">Deep Learning · OCR</div>
            <h1 class="hero-title">Retrain the <em>CRNN</em><br>your way</h1>
            <p class="hero-sub">
                Customise epochs, batch size, learning rate, augmentation and attention,
                then launch a fresh training run. The pre-trained model still works
                while you experiment.
            </p>
            <div class="hero-badges">
                <span class="tag tag-y">{_acc_badge}</span>
                <span class="tag tag-r">CTC Loss</span>
                <span class="tag tag-b">BiLSTM × 2</span>
                <span class="tag tag-g">Attention</span>
                <span class="tag tag-w">1040 Images</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with stats_col:
        if model_is_ready():
            meta_raw = _load_meta_dict(str(META_PATH), _meta_mtime())
            m   = meta_raw.get("metrics", {})
            wa  = m.get("word_accuracy", 0)
            ca  = m.get("char_accuracy", 0)
            cer = m.get("cer", 0)
            st.markdown(f"""
            <div class="hex-stats" style="flex-direction:column;align-items:center;padding-top:48px;">
                <div class="hex-card">
                    <div class="hex-val">{wa*100:.1f}%</div>
                    <div class="hex-lbl">Word Acc</div>
                </div>
                <div class="hex-card">
                    <div class="hex-val">{ca*100:.1f}%</div>
                    <div class="hex-lbl">Char Acc</div>
                </div>
                <div class="hex-card" style="background:rgba(255,79,54,0.06);">
                    <div class="hex-val" style="color:#ff4f36;">{cer*100:.2f}%</div>
                    <div class="hex-lbl">CER</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="sec-title">Dataset & Architecture</div>', unsafe_allow_html=True)

        with st.expander("Dataset Overview", expanded=not model_is_ready()):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Images", "1,040")
            c2.metric("Image Size", "200 × 50")
            c3.metric("Label Chars", "5")
            st.markdown("""
            <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap;">
                <span class="tag tag-g">Train 70%  ·  728</span>
                <span class="tag tag-b">Val 20%  ·  208</span>
                <span class="tag tag-w">Test 10%  ·  104</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Model Architecture", expanded=False):
            st.markdown("""
| Layer | Details |
|---|---|
| Input | (50, 200, 1) grayscale |
| Rescaling | → [0, 1] |
| Conv Block 1 | 64 × Conv2D × 2, BN, Pool (2,2) |
| Conv Block 2 | 128 × Conv2D × 2, BN, Pool (2,2) |
| Conv Block 3 | 256 × Conv2D × 2, BN, Pool (2,1) |
| Reshape | (25, 3072) |
| Dense Proj | 256 units, ReLU |
| BiLSTM 1 | 128 units bidirectional |
| BiLSTM 2 | 64 units bidirectional |
| Attention | 4 heads, key_dim = 32 |
| Output | (25, #classes + 1) softmax |
| Loss | CTC |
            """)

    with col_right:
        st.markdown('<div class="sec-title">How it works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="surface">
            <div class="step">
                <div class="step-num">01</div>
                <div>
                    <div class="step-title">CNN Feature Extraction</div>
                    <div class="step-body">VGG-style double conv blocks extract spatial features across 3 scales (64→128→256 filters).</div>
                </div>
            </div>
            <div class="step">
                <div class="step-num">02</div>
                <div>
                    <div class="step-title">BiLSTM Sequence Modelling</div>
                    <div class="step-body">Two bidirectional LSTM layers learn character order and long-range dependencies in both directions.</div>
                </div>
            </div>
            <div class="step">
                <div class="step-num">03</div>
                <div>
                    <div class="step-title">Attention + CTC Decode</div>
                    <div class="step-body">Multi-head attention weighs global context. CTC decodes the output sequence without explicit alignment.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Retraining</div>', unsafe_allow_html=True)

    col_btn, col_stat = st.columns([1, 3])
    with col_btn:
        train_clicked = st.button("▶  Start Retraining", type="primary",
                                   disabled=st.session_state.training,
                                   use_container_width=True)
    with col_stat:
        if model_is_ready():
            st.markdown("""
            <div class="status-ready">
                <span class="dot-live"></span>
                <span style="font-size:0.84rem;font-weight:600;color:#4dbf9d;">Pre-trained model active — retraining will replace it</span>
            </div>""", unsafe_allow_html=True)

    if train_clicked:
        st.session_state.training = True
        load_model_cached.clear()
        _load_meta_dict.clear()
        _load_history_dict.clear()

        progress_bar = st.progress(0.0, text="Starting …")
        metric_slot  = st.empty()
        chart_slot   = st.empty()
        epoch_data   = {"loss": [], "val_loss": []}

        def on_epoch(epoch: int, logs: dict):
            total = cfg_epochs
            loss  = logs.get("loss", 0.0)
            val   = logs.get("val_loss", 0.0)
            epoch_data["loss"].append(loss)
            epoch_data["val_loss"].append(val)
            progress_bar.progress((epoch+1)/total,
                text=f"Epoch {epoch+1}/{total}  ·  loss {loss:.4f}  ·  val {val:.4f}")
            with metric_slot.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Epoch",      f"{epoch+1}/{total}")
                c2.metric("Train Loss", f"{loss:.4f}")
                c3.metric("Val Loss",   f"{val:.4f}")
                _cur_lr = logs.get("learning_rate", logs.get("lr", cfg_lr))
                c4.metric("LR",         f"{_cur_lr:.0e}")
            if len(epoch_data["loss"]) > 1:
                fig, ax = dark_fig(9, 3)
                xs = range(1, len(epoch_data["loss"])+1)
                ax.plot(xs, epoch_data["loss"],     color="#c8a97c", lw=2, label="Train")
                ax.plot(xs, epoch_data["val_loss"], color="#ff4f36", lw=2, ls="--", label="Val")
                ax.fill_between(xs, epoch_data["loss"],     alpha=0.06, color="#c8a97c")
                ax.fill_between(xs, epoch_data["val_loss"], alpha=0.06, color="#ff4f36")
                ax.set_xlabel("Epoch"); ax.set_ylabel("CTC Loss")
                ax.set_title("Live Training Loss")
                ax.legend(labelcolor="#7a7068" if st.session_state.get("theme")=="light" else "#a399a8",
                           facecolor="#eceae3" if st.session_state.get("theme")=="light" else "#0d0f11",
                           edgecolor="#d4cfcb" if st.session_state.get("theme")=="light" else "#1c2026")
                ax.grid(True, alpha=0.15, color="#d4cfcb" if st.session_state.get("theme")=="light" else "#1c2026")
                plt.tight_layout(); chart_slot.pyplot(fig); plt.close()

        from train import train_model
        with st.spinner("Training in progress …"):
            results = train_model(epochs=cfg_epochs, batch_size=cfg_batch,
                                  learning_rate=cfg_lr, use_attention=cfg_attn,
                                  augment=cfg_augment, on_epoch_end_fn=on_epoch)

        st.session_state.training = False
        progress_bar.progress(1.0, text="Training complete!")
        r = results["metrics"]
        st.balloons()
        st.markdown("""
        <div class="status-ready" style="margin:12px 0;">
            <span class="dot-live"></span>
            <span style="font-size:0.9rem;font-weight:700;color:#4dbf9d;">Training complete — model saved to <code style="color:#c8a97c;">saved_model/</code></span>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Word Accuracy", f"{r['word_accuracy']*100:.2f}%")
        c2.metric("Char Accuracy", f"{r['char_accuracy']*100:.2f}%")
        c3.metric("CER",           f"{r['cer']*100:.2f}%")
        c4.metric("Correct/Total", f"{r['correct']}/{r['total']}")
        if SAMPLE_GRID.exists():
            st.image(str(SAMPLE_GRID), caption="Sample test predictions")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB — PREDICT (default / landing tab)
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    _is_light = st.session_state.get("theme") == "light"
    _h_col   = "#1a1a1a" if _is_light else "#fff"
    _p_col   = "#7a7068" if _is_light else "#544d56"
    st.markdown(f"""
    <div class="hero-eyebrow" style="padding:28px 0 0 0;">Live Inference</div>
    <h2 style="font-size:1.9rem;font-weight:700;letter-spacing:-0.02em;color:{_h_col};margin:6px 0 4px 0;">
        Decode a <span style="color:#c8a97c;">CAPTCHA</span>
    </h2>
    <p style="color:{_p_col};font-size:0.88rem;margin:0 0 24px 0;">
        Upload any CAPTCHA image and the CRNN model will decode it instantly.
    </p>
    """, unsafe_allow_html=True)

    if not model_is_ready():
        st.markdown("""
        <div class="status-warn">
            <span class="dot-warn"></span>
            <span style="font-size:0.85rem;font-weight:600;color:#ff4f36;">
                No trained model — go to the <b>Train</b> tab first.
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        _model, _meta = try_load_model()
        if _model is None:
            st.error("Failed to load model weights. Please retrain.")
        else:
            col_up, col_res = st.columns([1, 1], gap="large")

            with col_up:
                st.markdown('<div class="sec-title">Upload Image</div>', unsafe_allow_html=True)
                uploaded = st.file_uploader(
                    "CAPTCHA image", type=["png","jpg","jpeg"],
                    label_visibility="collapsed")
                if uploaded:
                    pil_img = Image.open(uploaded)
                    st.image(pil_img, caption="Input", use_container_width=True)

            with col_res:
                st.markdown('<div class="sec-title">Result</div>', unsafe_allow_html=True)
                if uploaded:
                    with st.spinner("Running inference …"):
                        result = predict_image(pil_img, _model, _meta)

                    pred  = result["text"]
                    conf  = result["confidence"]
                    cpct  = conf * 100
                    cclr  = "#c8a97c" if cpct >= 80 else "#ff4f36" if cpct < 60 else "#4dbf9d"

                    st.markdown(f"""
                    <div class="pred-big">
                        <div class="pred-label">Decoded Text</div>
                        <div class="pred-chars">{pred}</div>
                        <div class="conf-bar-wrap">
                            <span class="conf-pct" style="color:{cclr};">{cpct:.1f}%</span>
                            <span style="font-size:0.7rem;color:#544d56;letter-spacing:0.08em;text-transform:uppercase;">confidence</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("")
                    st.progress(conf, text=f"Confidence: {cpct:.1f}%")

                    # Vocab activation chart
                    st.markdown('<div class="sec-title" style="margin-top:20px;">Vocabulary Activation</div>', unsafe_allow_html=True)
                    probs = result["probs"]
                    top_k = probs[:, :-1].max(0)
                    chars = _meta["all_chars"]
                    colors = ["#c8a97c" if c in pred else "#1c2026" for c in chars]
                    ec     = ["#c8a97c" if c in pred else "#544d56" for c in chars]
                    fig, ax = dark_fig(7, 2.5)
                    ax.bar(range(len(chars)), top_k, color=colors, edgecolor=ec, linewidth=0.8, width=0.72)
                    ax.set_xticks(range(len(chars)))
                    ax.set_xticklabels(chars, fontsize=8.5, fontfamily="monospace")
                    ax.set_ylim(0, 1.05)
                    ax.set_ylabel("Max CTC prob")
                    ax.set_title("Active chars in yellow")
                    ax.grid(True, axis="y", alpha=0.15, color="#d4cfcb" if st.session_state.get("theme")=="light" else "#1c2026")
                    plt.tight_layout(); st.pyplot(fig); plt.close()
                else:
                    st.markdown("""
                    <div class="empty-state">
                        <div class="empty-icon">🔍</div>
                        <div class="empty-txt">Upload an image to see predictions</div>
                    </div>""", unsafe_allow_html=True)

            # Random samples
            st.markdown('<div class="sec-title" style="margin-top:8px;">Random Test Samples</div>', unsafe_allow_html=True)
            col_n, col_b = st.columns([3, 1])
            with col_n:
                num_samples = st.slider("Number of samples", 3, 12, 6,
                                        label_visibility="collapsed")
            with col_b:
                load_clicked = st.button("Roll Samples", use_container_width=True)

            if load_clicked:
                if not IMAGE_DIR.exists():
                    from data_utils import extract_dataset
                    extract_dataset(str(ZIP_PATH), str(IMAGE_DIR))
                import random
                all_imgs = sorted(IMAGE_DIR.glob("*.png"))
                chosen   = random.sample(all_imgs, min(num_samples, len(all_imgs)))
                cols = st.columns(3, gap="small")
                for i, img_path in enumerate(chosen):
                    true_lbl = img_path.stem
                    pil_img  = Image.open(img_path)
                    res  = predict_image(pil_img, _model, _meta)
                    pred = res["text"]
                    conf = res["confidence"]
                    ok   = pred == true_lbl
                    with cols[i % 3]:
                        st.image(pil_img, use_container_width=True)
                        css  = "sample-ok" if ok else "sample-err"
                        badge = "✓" if ok else "✗"
                        st.markdown(
                            f'<div class="{css}">{badge} {pred}</div>'
                            f'<div class="sample-sub">true: {true_lbl}  ·  {conf*100:.0f}%</div>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    _is_light = st.session_state.get("theme") == "light"
    _h_col    = "#1a1a1a" if _is_light else "#fff"
    _p_col    = "#7a7068" if _is_light else "#544d56"
    st.markdown(f"""
    <div class="hero-eyebrow" style="padding:28px 0 0 0;">Analytics</div>
    <h2 style="font-size:1.9rem;font-weight:700;letter-spacing:-0.02em;color:{_h_col};margin:6px 0 4px 0;">
        Model <span style="color:#c8a97c;">Dashboard</span>
    </h2>
    <p style="color:{_p_col};font-size:0.88rem;margin:0 0 24px 0;">
        Training history, evaluation metrics, and model specification.
    </p>
    """, unsafe_allow_html=True)

    if not model_is_ready():
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-txt">Train the model first to see dashboard metrics.</div>
        </div>""", unsafe_allow_html=True)
    else:
        meta_raw = _load_meta_dict(str(META_PATH), _meta_mtime())
        metrics  = meta_raw.get("metrics", {})

        st.markdown('<div class="sec-title">Test-Set Performance</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Word Accuracy",   f"{metrics.get('word_accuracy',0)*100:.2f}%",
                  help="All 5 chars correct")
        c2.metric("Char Accuracy",   f"{metrics.get('char_accuracy',0)*100:.2f}%",
                  help="Per-character correct rate")
        c3.metric("CER",             f"{metrics.get('cer',0)*100:.2f}%",
                  help="Character Error Rate — lower is better")
        c4.metric("Correct / Total", f"{metrics.get('correct','?')}/{metrics.get('total','?')}",
                  help="Exact-match samples on test set")

        # Training curves
        if HISTORY_PATH.exists():
            st.markdown('<div class="sec-title" style="margin-top:8px;">Training Curves</div>', unsafe_allow_html=True)
            hist    = _load_history_dict(str(HISTORY_PATH),
                                         HISTORY_PATH.stat().st_mtime if HISTORY_PATH.exists() else 0)
            best_ep = int(np.argmin(hist["val_loss"]))
            xs      = range(1, len(hist["loss"]) + 1)

            fig, (ax1, ax2) = dark_fig(12, 4, ncols=2)
            for ax, data_y, log in [(ax1, hist, False), (ax2, hist, True)]:
                fn = ax.semilogy if log else ax.plot
                fn(xs, data_y["loss"],     color="#c8a97c", lw=2, label="Train")
                fn(xs, data_y["val_loss"], color="#ff4f36", lw=2, ls="--", label="Val")
                ax.fill_between(xs, data_y["loss"],     alpha=0.05, color="#c8a97c")
                ax.fill_between(xs, data_y["val_loss"], alpha=0.05, color="#ff4f36")
                ax.axvline(best_ep+1, color="#4dbf9d", ls=":", alpha=0.7,
                           label=f"Best: ep {best_ep+1}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("CTC Loss" + (" (log)" if log else ""))
                ax.set_title("Log Scale" if log else "Linear Scale")
                ax.legend(labelcolor="#7a7068" if st.session_state.get("theme")=="light" else "#a399a8",
                           facecolor="#eceae3" if st.session_state.get("theme")=="light" else "#0d0f11",
                           edgecolor="#d4cfcb" if st.session_state.get("theme")=="light" else "#1c2026")
                ax.grid(True, alpha=0.15, color="#d4cfcb" if st.session_state.get("theme")=="light" else "#1c2026", which="both" if log else "major")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            with st.expander("Epoch-level table"):
                import pandas as pd
                df = pd.DataFrame({
                    "Epoch": list(range(1, len(hist["loss"])+1)),
                    "Train Loss": [f"{v:.4f}" for v in hist["loss"]],
                    "Val Loss":   [f"{v:.4f}" for v in hist["val_loss"]],
                    "Delta":      ["—"] + [f"{hist['val_loss'][i]-hist['val_loss'][i-1]:+.4f}"
                                   for i in range(1, len(hist["val_loss"]))],
                })
                st.dataframe(df, use_container_width=True, height=280)

        # Sample grid
        st.markdown('<div class="sec-title">Sample Test Predictions</div>', unsafe_allow_html=True)
        if SAMPLE_GRID.exists():
            st.image(str(SAMPLE_GRID), use_container_width=True,
                     caption="Predictions on held-out test samples")
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🖼️</div>
                <div class="empty-txt">Sample grid will appear after training.</div>
            </div>""", unsafe_allow_html=True)

        # Model & dataset detail
        st.markdown('<div class="sec-title">Specification</div>', unsafe_allow_html=True)
        col_m, col_d = st.columns(2, gap="large")

        n_chars = len(meta_raw.get("all_chars", []))
        img_h   = meta_raw.get("image_height", 50)
        img_w   = meta_raw.get("image_width", 200)
        max_len = meta_raw.get("max_label_len", 5)

        with col_m:
            st.markdown("""
            <div class="surface">
                <div style="font-size:0.62rem;font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:0.15em;color:#544d56;margin-bottom:16px;">Architecture</div>
                <table><tbody>
                    <tr><td style="color:#544d56;width:44%;font-size:0.78rem;">Model</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">Improved CRNN</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Conv</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">3 × double-conv (64→128→256)</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Recurrent</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">BiLSTM(128) + BiLSTM(64)</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Attention</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">4 heads, key_dim=32</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Loss</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">CTC</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Optimiser</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">Adam + ReduceLROnPlateau</td></tr>
                </tbody></table>
            </div>
            """, unsafe_allow_html=True)

        with col_d:
            st.markdown(f"""
            <div class="surface">
                <div style="font-size:0.62rem;font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:0.15em;color:#544d56;margin-bottom:16px;">Dataset</div>
                <table><tbody>
                    <tr><td style="color:#544d56;width:44%;font-size:0.78rem;">Name</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">captcha_images_v2</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Total</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">1,040 images</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Image size</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">{img_w} × {img_h} px</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Vocabulary</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">{n_chars} chars</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Label len</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">{max_len} characters</td></tr>
                    <tr><td style="color:#544d56;font-size:0.78rem;">Augmentation</td>
                        <td style="color:#c8c2cf;font-size:0.83rem;font-weight:600;">noise · brightness · contrast</td></tr>
                </tbody></table>
            </div>
            """, unsafe_allow_html=True)

        # Vocabulary
        st.markdown('<div class="sec-title">Character Vocabulary</div>', unsafe_allow_html=True)
        all_chars = meta_raw.get("all_chars", [])
        chips = " ".join(f'<span class="vocab-chip">{ch}</span>' for ch in all_chars)
        st.markdown(f'<div class="vocab-wrap">{chips}</div>', unsafe_allow_html=True)
