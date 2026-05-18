"""
CAPTCHA Recognition — Streamlit Web Application
Run: streamlit run app.py
"""

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from data_utils import decode_predictions, extract_dataset, load_metadata
from model import build_crnn_model, ctc_loss_fn

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
ASSETS_DIR   = ROOT / "assets"
SAVE_DIR     = ROOT / "saved_model"
WEIGHTS_PATH = SAVE_DIR / "crnn_model.weights.h5"
META_PATH    = SAVE_DIR / "metadata.json"
HISTORY_PATH = SAVE_DIR / "training_history.json"
SAMPLE_GRID  = SAVE_DIR / "sample_predictions.png"
ZIP_PATH     = ROOT / "Captcha_Recognition_using_CRNN" / "captcha_images_v2.zip"
IMAGE_DIR    = ROOT / "Captcha_Recognition_using_CRNN" / "captcha_images_v2"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAPTCHA Recognition | CRNN",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ── CSS ────────────────────────────────────────────────────────────────────────
@st.cache_data
def _load_css(filename: str) -> str:
    return (ASSETS_DIR / filename).read_text(encoding="utf-8")

st.markdown(f"<style>{_load_css('dark.css')}</style>", unsafe_allow_html=True)

def _apply_theme():
    if st.session_state.theme == "light":
        st.markdown(f"<style>{_load_css('light.css')}</style>", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in [("model", None), ("metadata", None), ("history", None), ("training", False)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Model helpers ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model_cached(weights_path: str, meta_path: str, mtime: float = 0):
    meta  = load_metadata(meta_path)
    model = build_crnn_model(
        image_height=meta["image_height"],
        image_width=meta["image_width"],
        num_classes=len(meta["all_chars"]),
        use_attention=meta.get("use_attention", True),
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

# ── JSON cache ─────────────────────────────────────────────────────────────────
@st.cache_data
def _load_json(path: str, mtime: float = 0) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

# ── Inference ──────────────────────────────────────────────────────────────────
def predict_image(pil_img: Image.Image, model, meta: dict) -> dict:
    h, w = meta["image_height"], meta["image_width"]
    img  = pil_img.convert("L").resize((w, h))
    arr  = np.array(img, dtype=np.float32)[..., np.newaxis]
    arr  = np.expand_dims(arr, 0)
    raw  = model.predict(arr, verbose=0)
    pred = decode_predictions(raw, meta["int_to_char"])[0]
    conf = float(raw[0].max(axis=-1).mean())
    return {"text": pred, "confidence": conf, "probs": raw[0]}

# ── Plot helpers ───────────────────────────────────────────────────────────────
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
        ax.xaxis.label.set_color(TC)
        ax.yaxis.label.set_color(TC)
        ax.title.set_color(TITLE)
        for sp in ax.spines.values():
            sp.set_edgecolor(SC)
    return fig, (axs if ncols > 1 else axs[0])


# ── Marquee ticker ─────────────────────────────────────────────────────────────
_TICKER = (
    "CRNN Architecture   CTC Loss Decoding   BiLSTM × 2   "
    "Multi-Head Attention   98.1% Word Accuracy   VGG Double Conv Blocks   "
    "Data Augmentation   TensorFlow 2.x   "
) * 3
st.markdown(f"""
<div class="marquee-wrap">
  <div class="marquee-track">
    {"".join(f'<span>{t.strip()}</span>' for t in _TICKER.split("   ") if t.strip()) * 2}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-hex">🔐</div>
        <div class="logo-name">CAPTCHA AI</div>
        <div class="logo-sub">CRNN · CTC · Attention</div>
    </div>
    """, unsafe_allow_html=True)

    is_light = st.session_state.theme == "light"
    col_tgl, col_lbl = st.columns([1, 3])
    with col_tgl:
        if st.button("☀️" if is_light else "🌙", key="theme_toggle",
                     help="Toggle light / dark theme", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    with col_lbl:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#544d56;padding-top:10px;font-weight:500;">'
            f'{"Light Mode" if is_light else "Dark Mode"}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border:none;border-top:1px solid rgba(84,77,86,0.3);margin:8px 0 16px 0;">', unsafe_allow_html=True)

    if st.button("↺  Reload Model", use_container_width=True,
                 help="Force-reload weights from disk (clears cache)"):
        load_model_cached.clear()
        _load_json.clear()
        st.rerun()

    if model_is_ready():
        meta_raw = _load_json(str(META_PATH), _meta_mtime())
        m   = meta_raw.get("metrics", {})
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
    cfg_lr      = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 3e-3],
                                   value=1e-3, format_func=lambda x: f"{x:.0e}")
    cfg_augment = st.checkbox("Data Augmentation", value=True)
    cfg_attn    = st.checkbox("Self-Attention", value=True)

    st.markdown('<hr style="border:none;border-top:1px solid rgba(84,77,86,0.2);margin:16px 0 8px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.62rem;text-align:center;color:rgba(84,77,86,0.6);letter-spacing:0.08em;">TensorFlow · Streamlit · CRNN</div>', unsafe_allow_html=True)

_apply_theme()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_predict, tab_dashboard, tab_train = st.tabs(["Predict", "Dashboard", "Retrain"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    _is_light = st.session_state.get("theme") == "light"
    _h_col    = "#1a1a1a" if _is_light else "#fff"
    _p_col    = "#7a7068" if _is_light else "#544d56"
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
                No trained model — go to the <b>Retrain</b> tab first.
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
                    "CAPTCHA image", type=["png", "jpg", "jpeg"],
                    label_visibility="collapsed")
                if uploaded:
                    pil_img = Image.open(uploaded)
                    st.image(pil_img, caption="Input", use_container_width=True)

            with col_res:
                st.markdown('<div class="sec-title">Result</div>', unsafe_allow_html=True)
                if uploaded:
                    with st.spinner("Running inference …"):
                        result = predict_image(pil_img, _model, _meta)

                    pred = result["text"]
                    conf = result["confidence"]
                    cpct = conf * 100
                    cclr = "#c8a97c" if cpct >= 80 else "#ff4f36" if cpct < 60 else "#4dbf9d"

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

                    st.markdown('<div class="sec-title" style="margin-top:20px;">Vocabulary Activation</div>', unsafe_allow_html=True)
                    probs  = result["probs"]
                    top_k  = probs[:, :-1].max(0)
                    chars  = _meta["all_chars"]
                    colors = ["#c8a97c" if c in pred else "#1c2026" for c in chars]
                    ec     = ["#c8a97c" if c in pred else "#544d56" for c in chars]
                    fig, ax = dark_fig(7, 2.5)
                    ax.bar(range(len(chars)), top_k, color=colors, edgecolor=ec,
                           linewidth=0.8, width=0.72)
                    ax.set_xticks(range(len(chars)))
                    ax.set_xticklabels(chars, fontsize=8.5, fontfamily="monospace")
                    ax.set_ylim(0, 1.05)
                    ax.set_ylabel("Max CTC prob")
                    ax.set_title("Active chars in yellow")
                    ax.grid(True, axis="y", alpha=0.15,
                            color="#d4cfcb" if _is_light else "#1c2026")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
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
                    extract_dataset(str(ZIP_PATH), str(IMAGE_DIR))
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
                        css   = "sample-ok" if ok else "sample-err"
                        badge = "✓" if ok else "✗"
                        st.markdown(
                            f'<div class="{css}">{badge} {pred}</div>'
                            f'<div class="sample-sub">true: {true_lbl}  ·  {conf*100:.0f}%</div>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB — DASHBOARD
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
        meta_raw = _load_json(str(META_PATH), _meta_mtime())
        metrics  = meta_raw.get("metrics", {})

        st.markdown('<div class="sec-title">Test-Set Performance</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Word Accuracy",   f"{metrics.get('word_accuracy', 0)*100:.2f}%",
                  help="All 5 chars correct")
        c2.metric("Char Accuracy",   f"{metrics.get('char_accuracy', 0)*100:.2f}%",
                  help="Per-character correct rate")
        c3.metric("CER",             f"{metrics.get('cer', 0)*100:.2f}%",
                  help="Character Error Rate — lower is better")
        c4.metric("Correct / Total", f"{metrics.get('correct','?')}/{metrics.get('total','?')}",
                  help="Exact-match samples on test set")

        if HISTORY_PATH.exists():
            st.markdown('<div class="sec-title" style="margin-top:8px;">Training Curves</div>', unsafe_allow_html=True)
            hist    = _load_json(str(HISTORY_PATH),
                                 HISTORY_PATH.stat().st_mtime if HISTORY_PATH.exists() else 0)
            best_ep = int(np.argmin(hist["val_loss"]))
            xs      = range(1, len(hist["loss"]) + 1)
            gc      = "#d4cfcb" if _is_light else "#1c2026"
            lc      = "#7a7068" if _is_light else "#a399a8"
            fc      = "#eceae3" if _is_light else "#0d0f11"
            ec      = "#d4cfcb" if _is_light else "#1c2026"

            fig, (ax1, ax2) = dark_fig(12, 4, ncols=2)
            for ax, log in [(ax1, False), (ax2, True)]:
                fn = ax.semilogy if log else ax.plot
                fn(xs, hist["loss"],     color="#c8a97c", lw=2, label="Train")
                fn(xs, hist["val_loss"], color="#ff4f36", lw=2, ls="--", label="Val")
                ax.fill_between(xs, hist["loss"],     alpha=0.05, color="#c8a97c")
                ax.fill_between(xs, hist["val_loss"], alpha=0.05, color="#ff4f36")
                ax.axvline(best_ep + 1, color="#4dbf9d", ls=":", alpha=0.7,
                           label=f"Best: ep {best_ep+1}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("CTC Loss" + (" (log)" if log else ""))
                ax.set_title("Log Scale" if log else "Linear Scale")
                ax.legend(labelcolor=lc, facecolor=fc, edgecolor=ec)
                ax.grid(True, alpha=0.15, color=gc, which="both" if log else "major")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            with st.expander("Epoch-level table"):
                import pandas as pd
                df = pd.DataFrame({
                    "Epoch":      list(range(1, len(hist["loss"]) + 1)),
                    "Train Loss": [f"{v:.4f}" for v in hist["loss"]],
                    "Val Loss":   [f"{v:.4f}" for v in hist["val_loss"]],
                    "Delta":      ["—"] + [
                        f"{hist['val_loss'][i] - hist['val_loss'][i-1]:+.4f}"
                        for i in range(1, len(hist["val_loss"]))
                    ],
                })
                st.dataframe(df, use_container_width=True, height=280)

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

        st.markdown('<div class="sec-title">Character Vocabulary</div>', unsafe_allow_html=True)
        all_chars = meta_raw.get("all_chars", [])
        chips = " ".join(f'<span class="vocab-chip">{ch}</span>' for ch in all_chars)
        st.markdown(f'<div class="vocab-wrap">{chips}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB — RETRAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    hero_col, stats_col = st.columns([3, 1], gap="large")

    with hero_col:
        if model_is_ready():
            _hero_meta = _load_json(str(META_PATH), _meta_mtime())
            _wa        = _hero_meta.get("metrics", {}).get("word_accuracy", 0)
            _acc_badge = f"{_wa*100:.1f}% Accuracy"
        else:
            _acc_badge = "98.1% Accuracy"
        st.markdown(f"""
        <div class="hero">
            <div class="hero-eyebrow">Deep Learning · OCR</div>
            <h1 class="hero-title">Retrain the <em>CRNN</em><br>your way</h1>
            <p class="hero-sub">
                Customise epochs, batch size, learning rate, augmentation and attention,
                then launch a fresh training run. The pre-trained model stays active
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
            meta_raw = _load_json(str(META_PATH), _meta_mtime())
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
        _load_json.clear()

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
            progress_bar.progress((epoch + 1) / total,
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
                xs = range(1, len(epoch_data["loss"]) + 1)
                ax.plot(xs, epoch_data["loss"],     color="#c8a97c", lw=2, label="Train")
                ax.plot(xs, epoch_data["val_loss"], color="#ff4f36", lw=2, ls="--", label="Val")
                ax.fill_between(xs, epoch_data["loss"],     alpha=0.06, color="#c8a97c")
                ax.fill_between(xs, epoch_data["val_loss"], alpha=0.06, color="#ff4f36")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("CTC Loss")
                ax.set_title("Live Training Loss")
                _lc = "#7a7068" if st.session_state.get("theme") == "light" else "#a399a8"
                _fc = "#eceae3" if st.session_state.get("theme") == "light" else "#0d0f11"
                _ec = "#d4cfcb" if st.session_state.get("theme") == "light" else "#1c2026"
                ax.legend(labelcolor=_lc, facecolor=_fc, edgecolor=_ec)
                ax.grid(True, alpha=0.15, color=_ec)
                plt.tight_layout()
                chart_slot.pyplot(fig)
                plt.close()

        from train import train_model
        with st.spinner("Training in progress …"):
            results = train_model(
                epochs=cfg_epochs, batch_size=cfg_batch,
                learning_rate=cfg_lr, use_attention=cfg_attn,
                augment=cfg_augment, on_epoch_end_fn=on_epoch,
            )

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
