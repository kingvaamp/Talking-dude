import streamlit as st
import streamlit.components.v1 as components
import html as _html
import json
import os
import re
import queue
import threading
import time
import difflib
import array as _array_mod  # top-level: avoids sys.modules lookup in hot audio loop
import math    # top-level: used in _audio_waveform_html called every 300ms rerun
import socket
import qrcode
import io
import pyaudio
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)


st.set_page_config(page_title="Talking Dude — Live Interpreter", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

# audioop was removed in Python 3.13 — provide a pure-Python fallback
try:
    import audioop
    _HAS_AUDIOOP = True
except ImportError:
    _HAS_AUDIOOP = False

def _to_mono(raw: bytes) -> bytes:
    """Stereo int16 → mono int16 (pure Python fallback)."""
    if _HAS_AUDIOOP:
        return audioop.tomono(raw, 2, 0.5, 0.5)
    samps = _array_mod.array('h', raw)
    # BUG FIX: an O(n) python generator `(samps[i]+... for i in range)` executing
    # 96,000 times per second severely throttled the GIL. Slice `[::2]` takes the
    # left channel using C-optimized native evaluation almost instantly.
    return samps[::2].tobytes()

def _ratecv(raw: bytes, in_rate: int, out_rate: int, state):
    """Sample-rate conversion int16 mono (pure Python fallback). Returns (bytes, state)."""
    if _HAS_AUDIOOP:
        return audioop.ratecv(raw, 2, 1, in_rate, out_rate, state)
    samps  = _array_mod.array('h', raw)
    n_in   = len(samps)
    n_out  = int(n_in * out_rate / in_rate)
    # BUG-5 FIX: pre-allocate to avoid repeated realloc on every append.
    result = _array_mod.array('h', [0] * n_out)
    for i in range(n_out):
        pos  = i * in_rate / out_rate
        idx  = int(pos)
        frac = pos - idx
        a    = samps[idx]         if idx     < n_in else 0
        b    = samps[idx + 1]     if idx + 1 < n_in else a
        result[i] = max(-32768, min(32767, int(a + frac * (b - a))))
    return result.tobytes(), None  # stateless fallback

try:
    from deep_translator import GoogleTranslator
except ImportError:
    st.error("❌ `deep-translator` manquant. Lance : `pip install deep-translator`")
    st.stop()

try:
    from openai import OpenAI as GroqOpenAI
except ImportError:
    st.error("❌ `openai` manquant. Lance : `pip install openai`")
    st.stop()

# ── Settings persistence ──────────────────────────────────────────────────────
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(data):
    try:
        existing = load_settings()
        existing.update(data)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

@st.cache_data
def generate_qr_bytes(url: str) -> bytes:
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=3)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a2e", back_color="#f0f0f5")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return bytes(buf.getvalue())

_persisted = load_settings()

# ── CSS ───────────────────────────────────────────────────────────────────────
# ── Session state init ─────────────────────────────────────────────────────────
def _init_ss(key, factory):
    if key not in st.session_state:
        st.session_state[key] = factory()

# STABILITY FIX: AUDIO_QUEUE maxsize raised from 60 → 400 to handle pauses without dropping;
# UI_UPDATE_QUEUE maxsize raised from 200 → 500 to handle rapid transcription bursts.
_init_ss("AUDIO_QUEUE",       lambda: queue.Queue())
_init_ss("TRANSCRIPT_QUEUE",  lambda: queue.Queue())
_init_ss("TRANSLATION_QUEUE", lambda: queue.Queue(maxsize=15))
_init_ss("UI_UPDATE_QUEUE",   lambda: queue.Queue(maxsize=500))
_init_ss("STATUS_QUEUE",      lambda: queue.Queue(maxsize=50))
_init_ss("SUMMARY_QUEUE",               lambda: queue.Queue())
_init_ss("HIGHLIGHT_SUMMARY_QUEUE",     lambda: queue.Queue())
_init_ss("STOP_EVENT",                  threading.Event)
_init_ss("SUMMARY_LOCK",                threading.Lock)
_init_ss("interim_text",                lambda: "")
_init_ss("history",                     list)
_init_ss("history_map",                 dict)
_init_ss("status_dict",                 lambda: {"running": False, "msg": "Prêt — appuie sur Start."})
_init_ss("current_sentence",            lambda: {"original": "", "translation": "", "id": 0.0})
_init_ss("audio_level",                 lambda: 0.0)
_init_ss("summary",                     lambda: "")
_init_ss("summary_loading",             lambda: False)
_init_ss("summary_in_progress",         lambda: False)
_init_ss("summary_start_time",          lambda: 0.0)
_init_ss("current_page",                lambda: "main")  # "main" ou "summary"
_init_ss("sidebar_hidden",              lambda: False)
_init_ss("highlighted_cards",           set)
_init_ss("highlighted_summaries",       dict)  # {item_id: summary_text}
_init_ss("highlighted_summaries_loading", lambda: False)
_init_ss("highlight_summary_start_time", lambda: 0.0)

# Audio device name locked at session start — never re-enumerated mid-session
_init_ss("_locked_device_name", lambda: None)


# ── Theme config ──────────────────────────────────────────────────────────────
_init_ss("theme", lambda: _persisted.get("theme", "dark"))

def toggle_theme():
    new_theme = "light" if st.session_state.theme == "dark" else "dark"
    st.session_state.theme = new_theme
    save_settings({"theme": new_theme})

if st.session_state.theme == "dark":
    theme_vars = """
        --bg-main: #0A0E17;
        --text-main: #CBD5E1;
        --header-bg: rgba(10, 14, 23, 0.92);
        --live-bg-1: rgba(15,20,35,0.92);
        --live-bg-2: rgba(20,28,50,0.82);
        --live-border: rgba(99,102,241,0.18);
        --live-title: #818CF8;
        --live-glow: 0 0 12px #EF4444, 0 0 24px rgba(249,115,22,0.35);
        --live-text: #F1F5F9;
        --ghost-text: 0.32;
        --trans-text: #A5B4FC;
        --hist-bg: rgba(255,255,255,0.03);
        --hist-hover: rgba(255,255,255,0.055);
        --hist-border: rgba(99,102,241,0.30);
        --hist-orig: #E2E8F0;
        --hist-trans: #A5B4FC;
        --hist-hl-bg: rgba(167,139,250,0.12);
        --hist-hl-border: #A78BFA;
        --hist-hl-glow: 0 0 0 2px rgba(167,139,250,0.20), 0 4px 20px rgba(139,92,246,0.18);
        --btn-bg: rgba(255,255,255,0.04);
        --btn-border: rgba(255,255,255,0.08);
        --btn-hover: rgba(99,102,241,0.12);
        --btn-hover-border: rgba(99,102,241,0.45);
        --btn-shadow: none;
        --btn-primary-bg: linear-gradient(135deg, #10B981, #34D399);
        --btn-primary-border: #10B981;
        --btn-primary-hover: linear-gradient(135deg, #059669, #10B981);
        --btn-primary-text: #ffffff;
        --status-bg: rgba(255,255,255,0.035);
        --sb-bg: rgba(10,14,23,0.97);
        --sb-border: rgba(255,255,255,0.06);
        --subtitle: #64748B;
        --wv-bg-grad: linear-gradient(180deg,rgba(99,102,241,0.05) 0%,rgba(49,46,129,0.12) 100%);
        --input-bg: rgba(255,255,255,0.05);
        --input-border: rgba(255,255,255,0.10);
        --input-text: #F1F5F9;
        --label-text: #CBD5E1;
        --toggle-icon: #34D399;
        --accent: #6366F1;
        --accent-light: #818CF8;
        --highlight: #A78BFA;
        --highlight-light: #C4B5FD;
    """
else:
    theme_vars = """
        --bg-main: #F8FAFC;
        --text-main: #1E293B;
        --header-bg: rgba(248, 250, 252, 0.92);
        --live-bg-1: rgba(255,255,255,0.95);
        --live-bg-2: rgba(241,245,249,0.88);
        --live-border: rgba(99,102,241,0.15);
        --live-title: #4F46E5;
        --live-glow: 0 0 10px #EF4444, 0 0 20px rgba(239,68,68,0.35);
        --live-text: #0F172A;
        --ghost-text: 0.40;
        --trans-text: #4338CA;
        --hist-bg: rgba(255,255,255,0.95);
        --hist-hover: rgba(241,245,249,1);
        --hist-border: rgba(99,102,241,0.30);
        --hist-orig: #1E293B;
        --hist-trans: #4338CA;
        --hist-hl-bg: rgba(139,92,246,0.08);
        --hist-hl-border: #7C3AED;
        --hist-hl-glow: 0 0 0 2px rgba(139,92,246,0.18), 0 4px 18px rgba(124,58,237,0.14);
        --btn-bg: #ffffff;
        --btn-border: #CBD5E1;
        --btn-hover: rgba(99,102,241,0.08);
        --btn-hover-border: rgba(99,102,241,0.50);
        --btn-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
        --btn-primary-bg: linear-gradient(135deg, #10B981, #34D399);
        --btn-primary-border: #10B981;
        --btn-primary-hover: linear-gradient(135deg, #059669, #10B981);
        --btn-primary-text: #ffffff;
        --status-bg: rgba(241,245,249,1);
        --sb-bg: rgba(248,250,252,0.97);
        --sb-border: rgba(0,0,0,0.06);
        --subtitle: #64748B;
        --wv-bg-grad: linear-gradient(180deg,rgba(99,102,241,0.04) 0%,rgba(99,102,241,0.10) 100%);
        --input-bg: #ffffff;
        --input-border: #CBD5E1;
        --input-text: #0F172A;
        --label-text: #334155;
        --toggle-icon: #0F172A;
        --accent: #4F46E5;
        --accent-light: #6366F1;
        --highlight: #7C3AED;
        --highlight-light: #8B5CF6;
    """

# ── CSS ───────────────────────────────────────────────────────────────────────
_rec_css = ""
if st.session_state.status_dict.get("running"):
    _rec_css = """
@keyframes td-rec-blink{0%,100%{opacity:1}50%{opacity:0.25}}
.td-rec-btn .stButton > button {
    border-color: rgba(239,68,68,0.35) !important;
}
.td-rec-btn .stButton > button::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #EF4444;
    border-radius: 50%;
    box-shadow: 0 0 6px #EF4444, 0 0 14px rgba(239,68,68,0.35);
    animation: td-rec-blink 1.2s ease-in-out infinite;
    margin-right: 8px;
    vertical-align: middle;
    position: relative;
    top: -1px;
}"""

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        {theme_vars}
        --ease-out-strong: cubic-bezier(0.23, 1, 0.32, 1);
        --ease-in-out-strong: cubic-bezier(0.77, 0, 0.175, 1);
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: var(--bg-main);
        color: var(--text-main);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    [data-testid="stMainBlockContainer"] {{
        padding-top: 3.5rem !important;
    }}
    [id="stHeader"] {{
        background-color: transparent !important;
    }}

    .stAppHeader {{
        background-color: var(--header-bg) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid var(--live-border);
    }}

    /* ── Live Box ── */
    .live-box {{
        background: linear-gradient(145deg, var(--live-bg-1) 0%, var(--live-bg-2) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 28px 32px;
        border: 1px solid var(--live-border);
        box-shadow:
            0 8px 32px rgba(0,0,0,0.12),
            0 0 0 1px rgba(255,255,255,0.03),
            inset 0 1px 0 rgba(255,255,255,0.04);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }}
    .live-box::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--live-title), transparent);
        opacity: 0.5;
    }}

    .live-title {{
        color: var(--live-title);
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.68rem;
        letter-spacing: 3.5px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .live-dot {{
        width: 7px;
        height: 7px;
        background: #EF4444;
        border-radius: 50%;
        box-shadow: var(--live-glow);
        animation: td-pulse 2s ease-in-out infinite;
        display: inline-block;
    }}
    @keyframes td-pulse {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.35); opacity: 0.45; }}
    }}

    .live-text {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        line-height: 1.6;
        color: var(--live-text);
        min-height: 2.5rem;
        letter-spacing: -0.01em;
    }}
    .ghost-text {{
        opacity: var(--ghost-text);
        font-style: italic;
        font-weight: 400;
    }}

    .translation-text {{
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 500;
        color: var(--trans-text);
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid var(--sb-border);
        font-style: italic;
        line-height: 1.6;
        letter-spacing: 0.01em;
    }}

    /* ── History Cards ── */
    .history-card {{
        background: var(--hist-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 3px solid var(--hist-border);
        transition: transform 200ms var(--ease-out-strong),
                    background 200ms ease,
                    border-color 200ms ease,
                    box-shadow 200ms ease;
        position: relative;
        cursor: pointer;
        user-select: none;
        opacity: 0;
        animation: td-card-in 350ms var(--ease-out-strong) forwards;
        animation-delay: var(--stagger, 0ms);
    }}
    @keyframes td-card-in {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .history-card:hover {{
        background: var(--hist-hover);
        border-left-color: var(--live-title);
        transform: translateX(3px);
    }}
    .history-card.highlighted {{
        background: var(--hist-hl-bg) !important;
        border-left: 3px solid var(--hist-hl-border) !important;
        box-shadow: var(--hist-hl-glow);
        transform: translateX(3px);
    }}
    .history-card.highlighted .history-original {{
        color: var(--hist-hl-border);
        font-weight: 600;
    }}
    .history-card.highlighted::after {{
        content: "";
        position: absolute;
        right: 14px;
        top: 50%;
        transform: translateY(-50%);
        width: 8px; height: 8px;
        background: var(--hist-hl-border);
        border-radius: 2px;
        transform: translateY(-50%) rotate(45deg);
        opacity: 0.7;
    }}
    .history-original {{
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: var(--hist-orig);
        margin-bottom: 6px;
        line-height: 1.55;
        transition: color 200ms ease;
    }}
    .history-translation {{
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        color: var(--hist-trans);
        font-style: italic;
        line-height: 1.55;
        opacity: 0.92;
    }}

    /* ── Buttons ── */
    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 10px;
        border: 1px solid var(--btn-border);
        background: var(--btn-bg) !important;
        color: var(--text-main) !important;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.01em;
        transition: transform 160ms var(--ease-out-strong),
                    background 200ms ease,
                    border-color 200ms ease,
                    box-shadow 200ms ease;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: var(--btn-shadow);
    }}
    .stButton > button:active,
    .stDownloadButton > button:active {{
        transform: scale(0.97) !important;
    }}
    .stButton > button:disabled,
    .stDownloadButton > button:disabled {{
        background: var(--btn-bg) !important;
        border-color: var(--btn-border) !important;
        color: var(--text-main) !important;
        opacity: 0.4;
        cursor: not-allowed;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        background: var(--btn-hover) !important;
        border-color: var(--btn-hover-border) !important;
        color: var(--live-title) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-1px);
        outline: none !important;
    }}
    .stButton > button:focus,
    .stDownloadButton > button:focus {{
        outline: 2px solid var(--live-title) !important;
        outline-offset: 2px;
    }}

    /* ── Primary Button (Résumé) ── */
    .stButton button[kind="primary"], .stButton button[data-testid="baseButton-primary"] {{
        background: var(--btn-primary-bg) !important;
        border: 1px solid var(--btn-primary-border) !important;
        color: var(--btn-primary-text) !important;
        box-shadow: 0 0 15px var(--live-border);
    }}
    .stButton button[kind="primary"]:hover, .stButton button[data-testid="baseButton-primary"]:hover {{
        background: var(--btn-primary-hover) !important;
        box-shadow: 0 4px 20px rgba(16,185,129,0.25);
        color: var(--btn-primary-text) !important;
        transform: translateY(-1px);
    }}
    .stButton button[kind="primary"]:active, .stButton button[data-testid="baseButton-primary"]:active {{
        transform: scale(0.97) !important;
    }}
    .stButton button[kind="primary"]:disabled, .stButton button[data-testid="baseButton-primary"]:disabled {{
        background: var(--btn-primary-bg) !important;
        border-color: var(--btn-primary-border) !important;
        color: var(--btn-primary-text) !important;
        opacity: 0.35;
        box-shadow: none;
    }}

    /* ── Status ── */
    .stInfo, .stSuccess, .stError, .stWarning {{
        background: var(--status-bg) !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem !important;
    }}
    .stSuccess {{ border-color: rgba(16,185,129,0.30) !important; }}
    .stError   {{ border-color: rgba(239,68,68,0.30) !important; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: var(--sb-bg) !important;
        border-right: 1px solid var(--sb-border);
    }}

    /* ── Native Sidebar Arrow Colors ── */
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {{
        fill: var(--toggle-icon) !important;
        color: var(--toggle-icon) !important;
        opacity: 1 !important;
    }}

    /* ── Header ── */
    .app-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }}
    .app-header-icon {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px; height: 36px;
        background: linear-gradient(135deg, var(--accent, #6366F1), var(--highlight, #A78BFA));
        border-radius: 10px;
        flex-shrink: 0;
    }}
    .app-header-icon svg {{
        width: 20px; height: 20px;
        stroke: white; fill: none;
        stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
    }}
    .app-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--live-text);
        letter-spacing: -0.03em;
    }}
    .app-subtitle {{
        font-size: 0.82rem;
        color: var(--subtitle);
        font-family: 'JetBrains Mono', monospace;
    }}

    
    /* ── Streamlit Widgets Overrides ── */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
        color: var(--input-text) !important;
        border-radius: 10px !important;
    }}
    
    [data-baseweb="select"] span,
    [data-baseweb="base-input"] {{
        background-color: transparent !important;
        color: var(--input-text) !important;
    }}

    [data-baseweb="input"] input {{
        background-color: transparent !important;
        color: var(--input-text) !important;
        -webkit-text-fill-color: var(--input-text) !important;
    }}
    
    .stSelectbox label p,
    .stTextInput label p {{
        color: var(--label-text) !important;
        font-family: 'Inter', sans-serif;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: var(--text-main) !important;
        font-family: 'Outfit', sans-serif !important;
    }}

    div[data-testid="stSubheader"] {{
        color: var(--subtitle) !important;
        font-family: 'Outfit', sans-serif;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-weight: 600;
        margin-top: 8px;
    }}

    /* ── Metric cards ── */
    [data-testid="stMetric"] {{
        background: var(--hist-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--sb-border);
        border-radius: 12px;
        padding: 16px !important;
    }}
    [data-testid="stMetric"] label {{
        font-family: 'Inter', sans-serif !important;
        color: var(--subtitle) !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-family: 'Outfit', sans-serif !important;
        color: var(--live-text) !important;
    }}

    /* ── prefers-reduced-motion ── */
    @media (prefers-reduced-motion: reduce) {{
        .live-dot {{ animation: none; opacity: 1; }}
        .history-card {{ animation: none; opacity: 1; }}
        .history-card:hover {{ transform: none; }}
        .stButton > button:hover {{ transform: none; }}
        .stButton > button:active {{ transform: none !important; }}
    }}

/* Waveform Visualizer */
.wv-wrap{{margin:10px 0 4px;padding:0;}}
.wv-lbl{{font-size:0.60rem;color:var(--subtitle);text-transform:uppercase;
         letter-spacing:2.5px;margin-bottom:7px;font-family:'Inter',sans-serif;font-weight:500;}}
.wv{{display:flex;align-items:flex-end;gap:2px;height:46px;
     background:var(--wv-bg-grad);
     border-radius:10px;padding:5px 6px;
     border:1px solid rgba(99,102,241,0.10);box-sizing:border-box;}}
.wb{{flex:1;min-width:1px;border-radius:2px 2px 1px 1px;
     background:linear-gradient(180deg,var(--c,#818CF8) 0%,rgba(79,70,229,0.5) 100%);
     box-shadow:0 0 4px rgba(99,102,241,0.18);
     transform-origin:bottom center;
     transition: height 0.08s ease;}}

{_rec_css}
</style>
""", unsafe_allow_html=True)

# STABILITY FIX: PA_LOCK is only used during audio device enumeration (UI side),
# NOT inside the producer_worker which runs its own private PyAudio instance.
# This prevents the producer thread from starving/deadlocking during Streamlit reruns.
if "PA_LOCK" not in st.session_state:
    st.session_state.PA_LOCK = threading.Lock()
PA_LOCK = st.session_state.PA_LOCK

AUDIO_QUEUE       = st.session_state.AUDIO_QUEUE

TRANSCRIPT_QUEUE  = st.session_state.TRANSCRIPT_QUEUE
TRANSLATION_QUEUE = st.session_state.TRANSLATION_QUEUE
UI_UPDATE_QUEUE   = st.session_state.UI_UPDATE_QUEUE
STATUS_QUEUE      = st.session_state.STATUS_QUEUE
STOP_EVENT        = st.session_state.STOP_EVENT

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-icon">
        <svg viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
    </div>
    <span class="app-title">Talking Dude</span>
</div>
""", unsafe_allow_html=True)

# ── Audio detection ───────────────────────────────────────────────────────────
def get_audio_devices(force_refresh: bool = False):
    """Returns a dictionary of {name: index} for available BLACKHOLE-ONLY input devices.

    BLACKHOLE LOCK: Only devices whose name contains 'blackhole' are returned.
    This makes it impossible for the sidebar selectbox to ever expose a mic
    or any other input device — even if macOS renumbers/renames devices.
    """
    if not force_refresh and "AUDIO_DEVICES_CACHE" in st.session_state:
        return st.session_state["AUDIO_DEVICES_CACHE"]

    st.session_state.pop("AUDIO_DEVICES_CACHE", None)

    with PA_LOCK:
        p = pyaudio.PyAudio()
        devices = {}
        try:
            for i in range(p.get_device_count()):
                try:
                    info = p.get_device_info_by_index(i)
                    # BLACKHOLE LOCK: only list devices that contain "blackhole" in the name
                    if info["maxInputChannels"] > 0 and "blackhole" in info["name"].lower():
                        devices[info["name"]] = i
                except Exception:
                    continue
            st.session_state["AUDIO_DEVICES_CACHE"] = devices
            return devices
        finally:
            p.terminate()



def find_blackhole_index():
    """Find the BlackHole audio device index (UI-side, cached)."""
    devices = get_audio_devices()
    for name, idx in devices.items():
        if "blackhole" in name.lower():
            return idx
    return None


def _resolve_blackhole_live(p) -> tuple:
    """Scan a live PyAudio instance and return (name, index) of the first
    BlackHole input device found.

    Called inside producer_worker on EVERY stream open so we survive macOS
    renumbering devices when a new peripheral is detected.
    Returns (None, None) if BlackHole is not currently present.
    NEVER returns a non-BlackHole device — caller must check for None.
    """
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and "blackhole" in info["name"].lower():
                return info["name"], i
        except Exception:
            continue
    return None, None


# ── Audio waveform visualiser (defined early — used inside sidebar) ─────────────
def _audio_waveform_html(level: float) -> str:
    """Return an HTML/CSS animated waveform bar visualizer for the sidebar."""
    # math is imported at top-level (no per-call import overhead)
    boosted_level = min(1.0, (level * 4.5) ** 0.65)
    
    num_bars = 30
    max_h = 40   # px
    min_h = 3    # px
    bars = []
    
    for i in range(num_bars):
        # Bell-curve shape: tall in centre, shorter at edges
        edge_factor = 1.0 - 0.45 * abs((i - num_bars / 2) / (num_bars / 2)) ** 1.4
        
        sine_factor = 0.60 + 0.40 * abs(math.sin(i * 0.73 + boosted_level * 5.0))
        
        h = max(min_h, int(min_h + (max_h - min_h) * boosted_level * edge_factor * sine_factor))
        
        bright = int(160 + 95 * edge_factor)
        colour = f"rgb({int(bright*0.5)},{int(bright*0.55)},{bright})"
        
        bars.append(
            f'<span class="wb" style="height:{h}px;--c:{colour}"></span>'
        )
        
    bars_html = "".join(bars)
    
    return f"""
<div class="wv-wrap">
  <div class="wv-lbl">Niveau audio</div>
  <div class="wv">{bars_html}</div>
</div>
"""

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<br>", unsafe_allow_html=True)
col_thm = st.sidebar.columns([1])[0]
theme_icon = "☀️ Mode Clair" if st.session_state.theme == "dark" else "🌙 Mode Sombre"
col_thm.button(theme_icon, on_click=toggle_theme, use_container_width=True)
st.sidebar.markdown("---")

st.sidebar.header("Langues")

source_lang_map = {"en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "it": "it-IT", "pt": "pt-BR", "ar": "ar"}
target_lang_map = {"fr": "fr", "en": "en", "es": "es", "de": "de", "it": "it", "pt": "pt", "ar": "ar"}

_src_keys  = list(source_lang_map.keys())
_tgt_keys  = list(target_lang_map.keys())
_saved_src = _persisted.get("source_lang", "en")
_saved_tgt = _persisted.get("target_lang", "fr")
_src_idx   = _src_keys.index(_saved_src) if _saved_src in _src_keys else 0
_tgt_idx   = _tgt_keys.index(_saved_tgt) if _saved_tgt in _tgt_keys else 0

source_display   = st.sidebar.selectbox("Langue source (audio)", _src_keys, index=_src_idx)
target_display   = st.sidebar.selectbox("Langue cible (traduction)", _tgt_keys, index=_tgt_idx)
source_lang_code = source_lang_map[source_display]
target_lang_code = target_lang_map[target_display]

if source_display != _saved_src or target_display != _saved_tgt:
    save_settings({"source_lang": source_display, "target_lang": target_display})


st.sidebar.markdown("---")
st.sidebar.header("Audio")

# BLACKHOLE LOCK — THREE WALLS:
#   1. get_audio_devices() only returns BlackHole devices (mic never listed)
#   2. Sidebar shows error + disables Start if BlackHole is absent
#   3. start_translating() hard-blocks any non-BlackHole device name
devices_dict = get_audio_devices()
device_names  = list(devices_dict.keys())

if not device_names:
    # BlackHole not installed / not detectable
    st.sidebar.error(
        "⛔ BlackHole introuvable.  \n"
        "Installe [BlackHole 2ch](https://existential.audio/blackhole/) "
        "puis redémarre l'app."
    )
    selected_device_name  = None
    SELECTED_DEVICE_INDEX = None
else:
    _saved_device = _persisted.get("audio_device", device_names[0])
    _dev_idx      = device_names.index(_saved_device) if _saved_device in device_names else 0

    selected_device_name = st.sidebar.selectbox(
        "Entrée audio (BlackHole uniquement)",
        device_names,
        index=_dev_idx,
        help="Seuls les périphériques BlackHole sont listés. Aucun microphone ne peut être sélectionné."
    )
    SELECTED_DEVICE_INDEX = devices_dict[selected_device_name]

    if selected_device_name != _saved_device:
        save_settings({"audio_device": selected_device_name})

    st.sidebar.success(f"✅ {selected_device_name}")

if st.sidebar.button("🔄 Rafraîchir les périphériques"):
    if "AUDIO_DEVICES_CACHE" in st.session_state:
        del st.session_state["AUDIO_DEVICES_CACHE"]
    st.rerun()

if st.session_state.status_dict["running"]:
    st.sidebar.markdown(
        _audio_waveform_html(st.session_state.get("audio_level", 0.0)),
        unsafe_allow_html=True
    )


st.sidebar.markdown("---")
st.sidebar.header("Résumé IA")

if "_groq_key" not in st.session_state:
    st.session_state._groq_key = _persisted.get("groq_key", "")

groq_key_input = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=st.session_state._groq_key,
    placeholder="gsk_...",
    help="Gratuit sur console.groq.com",
)
if groq_key_input != st.session_state._groq_key:
    save_settings({"groq_key": groq_key_input})
    st.session_state._groq_key = groq_key_input

GROQ_API_KEY = st.session_state._groq_key

if GROQ_API_KEY:
    st.sidebar.success("✅ Groq configuré.")
else:
    st.sidebar.warning("⚠️ Clé Groq requise pour les résumés.")

# Langue du résumé
_sum_lang_options = {"Français": "français", "English": "english", "Español": "español"}
_saved_sum_lang = _persisted.get("summary_lang", "Français")
_sum_lang_idx = list(_sum_lang_options.keys()).index(_saved_sum_lang) if _saved_sum_lang in _sum_lang_options else 0
selected_sum_lang_label = st.sidebar.selectbox("Langue du résumé", list(_sum_lang_options.keys()), index=_sum_lang_idx)
SUMMARY_LANG = _sum_lang_options[selected_sum_lang_label]
if selected_sum_lang_label != _saved_sum_lang:
    save_settings({"summary_lang": selected_sum_lang_label})

# Bouton Summarize — defined here, rendered in the main page
def go_to_summary():
    if not st.session_state.history:
        try:
            STATUS_QUEUE.put_nowait("⚠️ Aucune transcription à résumer.")
        except queue.Full:
            pass
        return
    if not GROQ_API_KEY:
        try:
            STATUS_QUEUE.put_nowait("⚠️ Entre ta clé Groq dans la barre latérale.")
        except queue.Full:
            pass
        return
    st.session_state.current_page = "summary"
    st.session_state.summary = ""
    st.session_state.summary_loading = True
    st.session_state.summary_in_progress = False
    # Reset highlighted summaries for fresh generation
    st.session_state.highlighted_summaries = {}
    st.session_state.highlighted_summaries_loading = False
    generate_summary_groq()
    # Also generate individual summaries for highlighted phrases (parallel)
    if st.session_state.highlighted_cards:
        generate_highlight_summaries_groq()


st.sidebar.markdown("---")
st.sidebar.header("APIs")

if "_dg_key" not in st.session_state:
    saved_key = _persisted.get("dg_key", "")
    st.session_state._dg_key = saved_key

dg_key_input = st.sidebar.text_input(
    "Deepgram API Key",
    type="password",
    value=st.session_state._dg_key,
    placeholder="Colle ta clé ici...",
    help="Crée un compte gratuit sur console.deepgram.com — $200 de crédits offerts",
)
if dg_key_input != st.session_state._dg_key:
    save_settings({"dg_key": dg_key_input})
    st.session_state._dg_key = dg_key_input

DG_API_KEY = st.session_state._dg_key

if not DG_API_KEY:
    st.sidebar.warning("⚠️ Clé Deepgram requise.")
else:
    st.sidebar.success("✅ Deepgram configuré.")

# Model selector
DG_MODEL_OPTIONS = {
    "nova-2 (Général — recommandé)": "nova-2",
    "nova-2-medical (Médical)": "nova-2-medical",
    "nova-2-meeting (Réunions)": "nova-2-meeting",
    "enhanced (Précis)": "enhanced",
    "base (Rapide)": "base",
}
_saved_model_label = _persisted.get("dg_model_label", list(DG_MODEL_OPTIONS.keys())[0])
_model_idx = list(DG_MODEL_OPTIONS.keys()).index(_saved_model_label) if _saved_model_label in DG_MODEL_OPTIONS else 0
selected_model_label = st.sidebar.selectbox("Modèle Deepgram", list(DG_MODEL_OPTIONS.keys()), index=_model_idx)
DG_MODEL = DG_MODEL_OPTIONS[selected_model_label]
if selected_model_label != _saved_model_label:
    save_settings({"dg_model_label": selected_model_label})

# ── iPhone Access ─────────────────────────────────────────────────────
st.sidebar.markdown("---")

if "show_qr" not in st.session_state:
    st.session_state.show_qr = False

if st.sidebar.button("📱 iPhone Access"):
    st.session_state.show_qr = not st.session_state.show_qr

if st.session_state.show_qr:
    ip = get_local_ip()
    url = f"http://{ip}:8501"
    if st.session_state.get("_last_qr_ip") != ip:
        generate_qr_bytes.clear()
        st.session_state["_last_qr_ip"] = ip
    st.sidebar.image(generate_qr_bytes(url), width=180, caption=url)

# Glossary logic removed per user request
glossary_list       = []
glossary_trans_list = []


# ── Helpers ────────────────────────────────────────────────────────────────────


def apply_glossary(text, glossary_array):
    if not text or not glossary_array:
        return text
    refined = text
    for term in glossary_array:
        target = term.split("(")[0].strip()
        if not target:
            continue
        pattern = re.compile(re.escape(target), re.IGNORECASE)
        refined = pattern.sub(term, refined)
    return refined

_CMP_STRIP = str.maketrans("", "", ".,!?;:\"'()-[]")

def _cmp(word):
    return word.lower().translate(_CMP_STRIP)

def find_new_words(committed_text, new_transcript):
    committed_text = committed_text.strip()
    new_transcript = new_transcript.strip()
    if not committed_text:
        return new_transcript
    if not new_transcript:
        return ""
    c_words = committed_text.split()
    n_words = new_transcript.split()
    max_check = min(len(c_words), len(n_words), 40)
    for overlap in range(max_check, 1, -1):
        c_tail = " ".join(_cmp(w) for w in c_words[-overlap:])
        n_head = " ".join(_cmp(w) for w in n_words[:overlap])
        if difflib.SequenceMatcher(None, c_tail, n_head).ratio() >= 0.70:
            return " ".join(n_words[overlap:])
    return new_transcript

def _is_sentence_end(text):
    t = text.rstrip()
    return bool(t) and t[-1] in ".?!"

# ── Deepgram streaming worker ──────────────────────────────────────────────────
# STABILITY FIXES applied:
#  1. KeepAlive interval reduced from 8s → 4s (Deepgram disconnects at 10s idle)
#  2. Use SDK's keep_alive() method instead of raw JSON send — guaranteed text frame
#  3. Skip sending empty audio packets (can trigger server-side close)
#  4. Auto-reconnect loop is still present with 2s backoff
def deepgram_stream_worker(api_key, model, lang_code, audio_q, transcript_q, STATUS_Q, stop_event):
    """Robust Deepgram streaming worker with auto-reconnect and KeepAlive for long sessions."""

    live_options = LiveOptions(
        model=model,
        language=lang_code[:2],
        smart_format=True,
        interim_results=True,
        encoding="linear16",
        channels=1,
        sample_rate=16000,
        endpointing=800,          # attend 800ms de silence avant de finaliser (défaut ~10ms)
        utterance_end_ms="2000",  # pause de 2s = fin d'une phrase
        vad_events=True,          # active les événements Voice Activity Detection
    )

    while not stop_event.is_set():
        connection_closed = threading.Event()

        try:
            config = DeepgramClientOptions(options={"keepalive": "true"})
            client = DeepgramClient(api_key, config)
            dg_connection = client.listen.websocket.v("1")

            def on_message(self, result, **kwargs):
                try:
                    transcript = result.channel.alternatives[0].transcript
                    if transcript:
                        # BUG-1 FIX: put_nowait — never block the Deepgram WebSocket
                        # callback thread. A blocking put() here stalls the SDK's
                        # internal event loop and kills the connection on long sessions.
                        try:
                            transcript_q.put_nowait((transcript, result.is_final))
                        except queue.Full:
                            # Queue full: drop oldest, insert newest (keep audio fresh)
                            try:
                                transcript_q.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                transcript_q.put_nowait((transcript, result.is_final))
                            except queue.Full:
                                pass
                except Exception as e:
                    # BUG-15 FIX: route errors to STATUS_Q, not stdout (lost in prod)
                    try:
                        STATUS_Q.put_nowait(f"⚠️ Erreur callback Deepgram: {e}")
                    except queue.Full:
                        pass

            def on_metadata(self, metadata, **kwargs):
                try:
                    STATUS_Q.put_nowait("✅ Deepgram connecté & prêt")
                except queue.Full:
                    pass

            def on_speech_started(self, speech_started, **kwargs):
                try:
                    STATUS_Q.put_nowait("🎙️ Voix détectée...")
                except queue.Full:
                    pass

            def on_error(self, error, **kwargs):
                try:
                    STATUS_Q.put_nowait(f"❌ Deepgram erreur: {error}")
                except queue.Full:
                    pass
                connection_closed.set()

            def on_close(self, close, **kwargs):
                try:
                    STATUS_Q.put_nowait("🔌 Connexion Deepgram fermée — reconnexion...")
                except queue.Full:
                    pass
                connection_closed.set()

            def on_utterance_end(self, utterance_end, **kwargs):
                """Deepgram signal: silence long — force commit of pending segment.
                BUG-2 FIX: was transcript_q.put() — blocking on the WebSocket thread.
                Now uses try/put_nowait so the callback always returns immediately.
                """
                try:
                    transcript_q.put_nowait(("__UTTERANCE_END__", True))
                except queue.Full:
                    pass
                except Exception:
                    pass

            dg_connection.on(LiveTranscriptionEvents.Transcript,    on_message)
            dg_connection.on(LiveTranscriptionEvents.Metadata,      on_metadata)
            dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
            dg_connection.on(LiveTranscriptionEvents.UtteranceEnd,  on_utterance_end)
            dg_connection.on(LiveTranscriptionEvents.Error,         on_error)
            dg_connection.on(LiveTranscriptionEvents.Close,         on_close)

            try:
                STATUS_Q.put_nowait("⚡ Connexion à Deepgram...")
            except queue.Full:
                pass

            if not dg_connection.start(live_options):
                try:
                    STATUS_Q.put_nowait("❌ Connexion Deepgram échouée. Nouvelle tentative dans 3s...")
                except queue.Full:
                    pass
                time.sleep(3)
                continue

            try:
                STATUS_Q.put_nowait("⚡ Streaming Deepgram actif")
            except queue.Full:
                pass
            # BUG-M FIX: TCP socket block isolation. Deepgram's dg_connection.send() is synchronous 
            # and indefinitely halts if the OS write buffer fills due to network hiccups. 
            # We offload it to a single dedicated thread. If it hangs, the queue fills, and we cleanly 
            # detect the timeout here to force a reconnection!
            # BUG FIX: Unbounded queue (maxsize=0). A bounded limit caused frame drops 
            # (which permanently deleted audio and caused skipped words) during brief WiFi delays.
            dg_send_q = queue.Queue()
            def _dg_sender():
                while not stop_event.is_set() and not connection_closed.is_set():
                    try:
                        packet = dg_send_q.get(timeout=0.1)
                        if packet is None: break
                        try:
                            dg_connection.send(packet)
                        except Exception as e:
                            connection_closed.set()
                            break
                    except queue.Empty:
                        pass
                        
            sender_t = threading.Thread(target=_dg_sender, daemon=True)
            sender_t.start()

            last_keepalive_t = time.time()
            while not stop_event.is_set() and not connection_closed.is_set():
                try:
                    data_list = [audio_q.get(timeout=0.1)]
                    while not audio_q.empty():
                        try:
                            data_list.append(audio_q.get_nowait())
                        except queue.Empty:
                            break
                    
                    combined_data = b"".join(data_list)
                    if combined_data:
                        try:
                            dg_send_q.put_nowait(combined_data)
                        except queue.Full:
                            pass
                        
                        # BUG-G FIX: Do NOT reset last_keepalive_t here.
                        # Since we stream audio bytes continuously (even during silence),
                        # resetting the timer here prevented the KeepAlive frame from EVER
                        # being sent. Deepgram disconnects if 10-15s passes without speech
                        # UNLESS a KeepAlive text frame is actively sent.
                except queue.Empty:
                    pass
                except Exception as e:
                    try:
                        STATUS_Q.put_nowait(f"❌ Erreur envoi Deepgram: {e}")
                    except queue.Full:
                        pass
                    connection_closed.set()
                    break

                now = time.time()
                if now - last_keepalive_t >= 3.0:
                    try:
                        dg_connection.keep_alive()
                        last_keepalive_t = now
                    except Exception as e:
                        # BUG-I FIX: Never fallback to dg_connection.send(json) because send() 
                        # forces binary frames designed for raw PCM audio. Sending text strings 
                        # here corrupted the AI decoder stream, causing it to "freeze" permanently.
                        pass

            # BUG-J FIX: If the WebSocket dropped ungracefully (e.g. WiFi hiccup),
            # finish() will block the thread forever, stalling the entire worker 
            # and permanently preventing it from looping to start a new connection!
            def _async_finish(conn):
                try:
                    conn.finish()
                except Exception:
                    pass
            threading.Thread(target=_async_finish, args=(dg_connection,), daemon=True).start()

            if stop_event.is_set():
                try:
                    STATUS_Q.put_nowait("🛑 Arrêt du flux Deepgram")
                except queue.Full:
                    pass
                break

            # Connection dropped — wait briefly then reconnect
            try:
                STATUS_Q.put_nowait("🔄 Reconnexion Deepgram dans 2s...")
            except queue.Full:
                pass
            time.sleep(2)

        except Exception as e:
            try:
                STATUS_Q.put_nowait(f"❌ Échec critique Deepgram: {e} — Reconnexion dans 3s...")
            except queue.Full:
                pass
            time.sleep(3)




# ── Audio producer (BlackHole → bytes 16kHz mono) ─────────────────────────────
TARGET_RATE = 16000

def _get_peak(raw_bytes):
    if not raw_bytes:
        return 0
    try:
        samps = _array_mod.array('h', raw_bytes)  # BUG-8 FIX: use top-level import
        if not samps: return 0
        return max(map(abs, samps)) / 32768.0
    except Exception:
        return 0

# BLACKHOLE LOCK: producer_worker always resolves the device by the keyword
# "blackhole" — never by index or cached name. This survives macOS renumbering
# devices when AirPods / Bluetooth / USB peripherals are detected mid-session.
# If BlackHole is absent at open time, the worker waits and retries instead of
# silently falling through to the system microphone.
def producer_worker(input_device_name, STATUS_Q, AUDIO_Q, UI_Q, stop_event):
    """Audio capture worker — EXCLUSIVELY reads from BlackHole.

    Uses _resolve_blackhole_live() on every (re-)open to scan current device
    list by keyword, not by index.  If BlackHole is not found the worker sleeps
    and retries; it NEVER opens any other input device.
    """
    CONFIGS = [(2, 48000, 2400), (1, 48000, 2400), (2, 44100, 2205), (1, 16000, 800)]
    MAX_CONSECUTIVE_ERRORS = 10
    # BUG-A FIX: per-second dedup guards for status messages.
    # Without these, `int(time.time()) % N == 0` fires for every audio chunk
    # during that second (~100 chunks @ 48 kHz/2400 chunk), spamming STATUS_Q
    # with ~100 identical messages and evicting legitimate status messages.
    _last_warn_s = -1   # second when "no signal" was last emitted
    _last_cap_s  = -1   # second when "capture en cours" was last emitted

    while not stop_event.is_set():
        stream    = None
        p         = None
        CHANNELS = RATE = CHUNK = None

        try:
            # ── Open PyAudio stream — BLACKHOLE ONLY ──────────────────
            with PA_LOCK:
                p = pyaudio.PyAudio()

                # Resolve BlackHole by keyword, ignoring index (macOS renumbers).
                bh_name, target_index = _resolve_blackhole_live(p)

                if target_index is None:
                    # BlackHole is not in the device list right now.
                    # HARD STOP — do NOT open any fallback device.
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    p = None
                    try:
                        STATUS_Q.put_nowait(
                            "⛔ BlackHole introuvable — attente reconnnexion... "
                            "(vérifie que BlackHole est installé)"
                        )
                    except queue.Full:
                        pass
                    time.sleep(3)
                    continue  # retry outer while loop

            # BUG-N FIX: Complete PyAudio callback re-architecture. 
            # Blocking `stream.read()` combined with `stream.abort()` can cause
            # mutual deadlocks deep inside PortAudio's C extensions. By moving
            # audio ingestion to a strict, non-blocking C-callback, we ensure zero
            # frame drops and we can safely detect Apple CoreAudio suspensions.

            RAW_Q = queue.Queue(maxsize=300)
            last_read_t = time.time()

            def audio_callback(in_data, frame_count, time_info, status):
                nonlocal last_read_t
                if in_data:
                    last_read_t = time.time()
                    try:
                        RAW_Q.put_nowait(in_data)
                    except queue.Full:
                        pass
                # paContinue tells PortAudio to keep firing this callback
                return (None, pyaudio.paContinue)

            opened_ok = False
            for ch, rate, chunk in CONFIGS:
                try:
                    stream = p.open(
                        format=pyaudio.paInt16, channels=ch, rate=rate,
                        input=True, input_device_index=target_index,
                        frames_per_buffer=chunk,
                        stream_callback=audio_callback
                    )
                    CHANNELS, RATE, CHUNK = ch, rate, chunk
                    opened_ok = True
                    break
                except Exception:
                    continue

            if not opened_ok or not stream:
                raise RuntimeError("Aucune config BlackHole compatible.")

            try:
                STATUS_Q.put_nowait(f"🎤 BlackHole ({bh_name}): {CHANNELS}ch @ {RATE}Hz actif")
            except queue.Full:
                pass
                
            stream.start_stream()
            ratecv_state     = None
            consecutive_errs = 0

            # ── Main processing loop (NO blocking stream.read) ──────────
            while not stop_event.is_set():
                try:
                    raw = RAW_Q.get(timeout=0.2)
                except queue.Empty:
                    # If CoreAudio sleeps for 5s, the callback stops ticking.
                    if time.time() - last_read_t > 5.0:
                        raise RuntimeError("Audio stream frozen (CoreAudio 5-minute sleep). Redémarrage forcé...")
                    continue
                
                try:
                    if CHANNELS == 2:
                        raw = _to_mono(raw)
                    if RATE != TARGET_RATE:
                        raw, ratecv_state = _ratecv(raw, RATE, TARGET_RATE, ratecv_state)

                    # STABILITY FIX: only send non-empty chunks to Deepgram queue
                    if not raw or len(raw) == 0:
                        continue

                    peak = _get_peak(raw)
                    try:
                        UI_Q.put_nowait(("level", peak))
                    except queue.Full:
                        pass

                    # STABILITY FIX: if AUDIO_Q is full, drop the OLDEST frame (not
                    # the newest) so Deepgram always gets fresh audio.
                    if AUDIO_Q.full():
                        try:
                            AUDIO_Q.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        AUDIO_Q.put_nowait(raw)
                    except queue.Full:
                        pass  # extremely rare since we just drained above

                    _now_s = int(time.time())
                    if peak < 0.0001:
                        # Emit at most once per 15-second window — not once per chunk
                        if _now_s % 15 == 0 and _now_s != _last_warn_s:
                            _last_warn_s = _now_s
                            try:
                                STATUS_Q.put_nowait("⚠️ Pas de signal audio (vérifie BlackHole)")
                            except queue.Full:
                                pass
                    else:
                        # Emit at most once per 60-second window
                        if _now_s % 60 == 0 and _now_s != _last_cap_s:
                            _last_cap_s = _now_s
                            try:
                                STATUS_Q.put_nowait("🎤 Capture en cours...")
                            except queue.Full:
                                pass

                except Exception as e:
                    consecutive_errs += 1
                    try:
                        STATUS_Q.put_nowait(f"⚠️ Erreur audio ({consecutive_errs}/{MAX_CONSECUTIVE_ERRORS}): {e}")
                    except queue.Full:
                        pass
                    if consecutive_errs >= MAX_CONSECUTIVE_ERRORS:
                        try:
                            STATUS_Q.put_nowait("🔄 Trop d'erreurs — réouverture du flux audio...")
                        except queue.Full:
                            pass
                        break  # exit inner loop → reopen stream
                    time.sleep(0.05)

        except Exception as e:
            try:
                STATUS_Q.put_nowait(f"❌ Audio critique: {e}")
            except queue.Full:
                pass
            time.sleep(3)

        finally:
            # ── Always clean up before retrying (brief lock) ──────────
            with PA_LOCK:
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                if p:
                    try:
                        p.terminate()
                    except Exception:
                        pass

        if stop_event.is_set():
            break

        try:
            STATUS_Q.put_nowait("🔄 Réouverture du flux audio dans 2s...")
        except queue.Full:
            pass
        time.sleep(2)

    try:
        STATUS_Q.put_nowait("🛑 Flux audio fermé.")
    except queue.Full:
        pass


# ── Consumer (transcripts → UI queue) ─────────────────────────────────────────
def consumer_worker(source_lang, target_lang, history_list, status_dict,
                    active_glossary, active_glossary_trans,
                    STATUS_Q, TRANSCRIPT_Q, TRANS_Q, UI_Q, stop_event):

    last_speech_t = time.time()
    local_orig = ""
    current_sid = time.time()

    def _commit_sentence():
        nonlocal local_orig, current_sid
        if not local_orig.strip():
            # BUG-H FIX: if we commit on silence but have no final text,
            # force a wipe of any ghost interim text from the screen!
            try:
                UI_Q.put_nowait(("clear_interim", None))
            except queue.Full:
                pass
            return
        try:
            UI_Q.put_nowait(("commit", current_sid))
        except queue.Full:
            pass
        local_orig = ""
        current_sid = time.time()

    while not stop_event.is_set():
        try:
            text, is_final = TRANSCRIPT_Q.get(timeout=0.1)

            # ── STABILITY FIX: utterance_end sentinel → force immediate commit ──────
            # on_utterance_end now sends ("__UTTERANCE_END__", True) instead of ("", True).
            # This avoids resetting last_speech_t every 2s during silence (which was
            # permanently blocking the 5s fallback commit). We commit whatever we have.
            if text == "__UTTERANCE_END__":
                _commit_sentence()
                continue

            if is_final:
                clean_text = text.strip()
                if not clean_text and not local_orig:
                    continue  # rien à committer, on ignore silencieusement

                # STABILITY FIX: if clean_text is empty but local_orig has content,
                # commit immediately instead of resetting last_speech_t (old bug).
                if not clean_text:
                    _commit_sentence()
                    continue

                sep = " " if local_orig else ""
                local_orig += sep + clean_text
                last_speech_t = time.time()  # only reset when we have real spoken text
                try:
                    UI_Q.put_nowait(("final", (local_orig, current_sid)))
                except queue.Full:
                    pass

                # Envoyer pour traduction — vider l'ancienne requête puis envoyer la nouvelle
                try:
                    TRANS_Q.get_nowait()
                except Exception:
                    pass
                # BUG-4 FIX: use put_nowait since we just freed a slot with get_nowait above.
                # Old timeout=0.1 blocked the consumer thread 100ms per sentence.
                try:
                    TRANS_Q.put_nowait((local_orig, source_lang, target_lang, current_sid))
                except queue.Full:
                    pass  # still full after drain — skip this translation, keep transcribing

                # Commit on sentence end, or every 30 words (was 50 — more responsive commits)
                if _is_sentence_end(clean_text) or len(local_orig.split()) >= 30:
                    _commit_sentence()
            else:
                last_speech_t = time.time()  # Track interim speech too!
                try:
                    UI_Q.put_nowait(("interim", text.strip()))
                except queue.Full:
                    pass

        except queue.Empty:
            # Fallback commit: if speech stopped > 5s ago
            if time.time() - last_speech_t > 5.0:
                if local_orig:
                    _commit_sentence()
                else:
                    # Wipe ghost interim text if it's stuck on screen
                    try:
                        UI_Q.put_nowait(("clear_interim", None))
                    except queue.Full:
                        pass
                # Reset to prevent spamming the queue
                last_speech_t = time.time()
            continue

# ── Translation worker ─────────────────────────────────────────────────────────
# STABILITY FIX: exponential backoff on translation failure to avoid hammering
# Google Translate API during network hiccups which could freeze the thread.
def translation_worker(STATUS_Q, TRANS_Q, UI_Q, stop_event):
    _backoff = 0.0
    _cached_translator = None
    _last_src = None
    _last_dest = None

    while not stop_event.is_set():
        try:
            payload = TRANS_Q.get(timeout=0.5)
        except queue.Empty:
            continue
        text, src, dest, sid = payload
        try:
            _s = src[:2]
            if _cached_translator is None or _last_src != _s or _last_dest != dest:
                _cached_translator = GoogleTranslator(source=_s, target=dest)
                _last_src = _s
                _last_dest = dest

            translated = _cached_translator.translate(text)
            if translated:
                try:
                    UI_Q.put_nowait(("translation", (translated, sid)))
                except queue.Full:
                    pass
            _backoff = 0.0  # reset on success
        except Exception as e:
            _backoff = min(_backoff + 1.0, 8.0)
            try:
                UI_Q.put_nowait(("translation", (f"[Erreur traduction: {e}]", sid)))
            except queue.Full:
                pass
            # FREEZE FIX: sleep in small increments so stop_event is checked frequently.
            # Old code: time.sleep(_backoff) — blocked the thread up to 8s, ignoring stop.
            if _backoff > 0:
                deadline = time.time() + _backoff
                while time.time() < deadline and not stop_event.is_set():
                    time.sleep(0.1)

# ── Start / Stop / Clear ───────────────────────────────────────────────────────
def start_translating(input_name, dg_model, src_lang, tgt_lang, g_list, g_trans_list):
    dg_key = st.session_state.get("_dg_key", "")
    if not dg_key:
        # BUG-3 FIX: was blocking put() — use put_nowait to never stall the main thread.
        try:
            STATUS_QUEUE.put_nowait("❌ Entre ta clé Deepgram dans la barre latérale.")
        except queue.Full:
            pass
        return

    # Always do a fresh device scan right before starting — never trust a stale cache
    devices = get_audio_devices(force_refresh=True)
    if input_name not in devices:
        try:
            STATUS_QUEUE.put_nowait(f"❌ Appareil '{input_name}' introuvable ou déconnecté.")
        except queue.Full:
            pass
        return

    # BLACKHOLE LOCK: hard-block if the resolved device is not BlackHole.
    # This is a second safety net — the sidebar should already only offer BlackHole
    # devices, but we enforce it here too so no code path can bypass it.
    if not input_name or "blackhole" not in input_name.lower():
        try:
            STATUS_QUEUE.put_nowait(
                "⛔ Démarrage refusé : seul BlackHole est autorisé comme entrée audio. "
                "Vérifie que BlackHole est installé et sélectionné."
            )
        except queue.Full:
            pass
        return

    # Signal stop to any existing threads
    if "STOP_EVENT" in st.session_state:
        st.session_state.STOP_EVENT.set()
        # BUG-9 FIX: give old worker threads a moment to notice the stop event before
        # we replace the queues. Without this, an old consumer_worker that's mid-commit
        # writes to the stale queue object and the first sentences of the new session are lost.
        time.sleep(0.35)
    st.session_state.STOP_EVENT = threading.Event()

    # STABILITY FIX: reinitialise queues with larger buffers for long sessions
    st.session_state.AUDIO_QUEUE       = queue.Queue()
    st.session_state.TRANSCRIPT_QUEUE  = queue.Queue()
    st.session_state.TRANSLATION_QUEUE = queue.Queue(maxsize=15)
    # BUG-F FIX: UI_UPDATE_QUEUE must be unbounded (maxsize=0).
    # If the Streamlit UI ever hangs for >15s (e.g. browser throttling, tab switch),
    # the 30Hz stream of 'level' and 'interim' events fills a 500-item queue.
    # When full, put_nowait() raised queue.Full and SILENTLY DROPPED critical
    # 'commit' and 'final' messages. To the user, transcription "froze forever".
    st.session_state.UI_UPDATE_QUEUE   = queue.Queue()

    stop_ev = st.session_state.STOP_EVENT
    audio_q = st.session_state.AUDIO_QUEUE
    dg_q    = st.session_state.TRANSCRIPT_QUEUE
    trans_q = st.session_state.TRANSLATION_QUEUE
    ui_q    = st.session_state.UI_UPDATE_QUEUE

    # STABILITY FIX: lock the audio device name for the entire session.
    # Even if the sidebar selectbox changes during a session, workers keep using the
    # device they were started with — preventing mid-session audio routing changes.
    st.session_state._locked_device_name = input_name

    st.session_state.status_dict["running"] = True
    st.session_state.status_dict["msg"]     = "⚡ Démarrage..."

    # BUG-C FIX: pass ui_q (captured after the queue was freshly created above).
    # All four workers should write to the same queue object via the local ref,
    # not by re-reading from session_state which could diverge in a future refactor.
    threading.Thread(
        target=producer_worker,
        args=(input_name, STATUS_QUEUE, audio_q, ui_q, stop_ev),
        daemon=True
    ).start()

    threading.Thread(
        target=deepgram_stream_worker,
        args=(dg_key, dg_model, src_lang, audio_q, dg_q, STATUS_QUEUE, stop_ev),
        daemon=True
    ).start()


    threading.Thread(
        target=consumer_worker,
        args=(src_lang, tgt_lang,
              st.session_state.history, st.session_state.status_dict,
              g_list, g_trans_list,
              STATUS_QUEUE, dg_q, trans_q, ui_q, stop_ev),
        daemon=True
    ).start()

    threading.Thread(
        target=translation_worker,
        args=(STATUS_QUEUE, trans_q, ui_q, stop_ev),
        daemon=True
    ).start()

# MAX highlights per session — prevents runaway API usage & freezes on long sessions
_HL_MAX = 10

def highlight_summary_worker(api_key, highlighted_items, full_history, lang, result_queue):
    """Background worker: per-highlight Groq analysis.

    BUG-FIX (freeze prevention):
    - Emits each result individually via put_nowait right after it arrives
      (no accumulation → UI shows progress phrase by phrase, no end-of-batch freeze)
    - 25s timeout per API call (prevents single hung call from blocking all others)
    - Sends __HL_DONE__ sentinel at the end so drain loop can clear loading flag
      reliably even if some calls failed
    - Capped at _HL_MAX phrases (enforced by generate_highlight_summaries_groq)
    """
    if not api_key or not highlighted_items:
        try:
            result_queue.put_nowait({"__HL_DONE__": True})
        except Exception:
            pass
        return

    # BUG-8 FIX: cap context at 80 items (was 300). On a 200-item history each item
    # averages ~15 tokens → 300 items ≈ 4 500 tokens context PER highlight call.
    # With 10 highlights that's 45 000 tokens just in context, risking Groq rate-limit
    # errors and very long wait times that make the UI appear frozen.
    # 80 items keeps the context meaningful while staying well under token limits.
    context_items = list(reversed(full_history))[-80:]
    full_context = "\n".join([
        f"[Original] {item.get('original', '')}\n[Traduction] {item.get('translation', '')}"
        for item in context_items
    ])

    try:
        client = GroqOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0,   # BUG-FIX: global client timeout — prevents hung TCP from
                            # blocking the thread indefinitely on network issues
        )
        for item in highlighted_items:
            item_id  = item.get("id")
            original = item.get("original", "")
            transl   = item.get("translation", "")
            if not original.strip():
                continue
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""Tu es un assistant expert en analyse de conversations.

L'utilisateur a marqué comme IMPORTANTE une phrase spécifique d'une conversation transcrite.
Ta tâche est d'analyser cette phrase dans son contexte complet et de répondre en {lang}.

Structure ta réponse ainsi :

**❓ Question / Enjeu soulevé**
(En 1-2 phrases : quelle est la question, le problème ou le point important que cette phrase soulève ?)

**💬 Contexte & Réponses dans la conversation**
(Quels éléments de la conversation répondent à cette question ou l'éclairent ? Sois précis.)

**⚡ Implications & Points d'action**
(Quelles décisions, actions ou suites découlent de ce point ? Ou notez "Aucune action directe identifiée" si applicable.)

Sois concis mais précis. Maximum 250 mots. Réponds uniquement sur cette phrase."""
                        },
                        {
                            "role": "user",
                            "content": f"""Transcription complète (contexte) :

{full_context}

---
Phrase marquée :
[Original] {original}
[Traduction] {transl}

Analyse cette phrase."""
                        }
                    ],
                    temperature=0.3,
                    max_tokens=400,
                )
                analysis = resp.choices[0].message.content
            except Exception as e:
                analysis = f"❌ Erreur d'analyse : {e}"

            # BUG-FIX: emit result immediately after each phrase
            # (old code accumulated all in a dict and sent once at the end —
            #  if the worker had 10 phrases and took 50s, the UI showed nothing
            #  until all were done, causing perceived freeze)
            try:
                result_queue.put_nowait({item_id: analysis})
            except Exception:
                pass

    except Exception as e:
        # Client init failure — emit error for all pending items
        for item in highlighted_items:
            item_id = item.get("id")
            if item_id:
                try:
                    result_queue.put_nowait({item_id: f"❌ Erreur Groq : {e}"})
                except Exception:
                    pass

    # Always send the done sentinel so the drain loop can clear loading state
    try:
        result_queue.put_nowait({"__HL_DONE__": True})
    except Exception:
        pass


def generate_highlight_summaries_groq():
    """Trigger async generation of per-highlight summaries (does nothing if no highlights).

    BUG-FIX: caps at _HL_MAX phrases to prevent runaway API usage.
    On long 60-min sessions a user could highlight 60 phrases; without a cap
    the worker would make 60 sequential API calls and the UI would appear frozen
    for the entire remaining timeout window.
    """
    api_key = st.session_state.get("_groq_key", "")
    highlighted_ids = st.session_state.get("highlighted_cards", set())
    history = st.session_state.history

    if not api_key or not highlighted_ids or not history:
        return

    # Build list of highlighted items (preserve conversation order — newest first)
    highlighted_items = [
        item for item in history
        if item.get("id") in highlighted_ids
    ]
    if not highlighted_items:
        return

    # CAP: never process more than _HL_MAX phrases to bound API calls & wait time
    if len(highlighted_items) > _HL_MAX:
        highlighted_items = highlighted_items[:_HL_MAX]

    st.session_state.highlighted_summaries = {}
    st.session_state.highlighted_summaries_loading = True
    st.session_state.highlight_summary_start_time = time.time()

    # BUG-B FIX: use pass (not break) so one get_nowait() exception (Empty race)
    # doesn't abort the drain early and leave a ghost __HL_DONE__ sentinel in the
    # queue — which would flip highlighted_summaries_loading=False the instant the
    # newly-launched worker checks in for the first time.
    while not st.session_state.HIGHLIGHT_SUMMARY_QUEUE.empty():
        try:
            st.session_state.HIGHLIGHT_SUMMARY_QUEUE.get_nowait()
        except Exception:
            pass

    threading.Thread(
        target=highlight_summary_worker,
        args=(api_key, highlighted_items, list(history),
              SUMMARY_LANG, st.session_state.HIGHLIGHT_SUMMARY_QUEUE),
        daemon=True
    ).start()


def summary_worker(api_key, history, lang, result_queue):
    """Background worker to generate summary using Groq without blocking the UI."""
    if not api_key or not history:
        # FREEZE FIX: was result_queue.put() — blocking with no timeout on an unbounded queue.
        # Use put_nowait; if it somehow fails, the summary_loading timeout (45s) will recover.
        try:
            result_queue.put_nowait("❌ Clé Groq manquante ou aucune transcription.")
        except Exception:
            pass
        return

    # Construire le texte complet
    full_text = "\n".join([
        f"[Original] {item.get('original', '')}\n[Traduction] {item.get('translation', '')}"
        for item in reversed(history)
    ])

    try:
        # BUG-9 FIX: add timeout=30.0 to match highlight_summary_worker — prevents
        # hung TCP from blocking the summary thread indefinitely on network issues.
        client = GroqOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0,
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"""Tu es un assistant expert en analyse et synthèse de conversations professionnelles.
Ta tâche est de traiter une transcription (phrases originales et traductions) et de générer un résumé exhaustif, hautement détaillé et ultra-précis en {lang}.
Tu dois respecter scrupuleusement la vérité de la transcription et n'inventer aucune information. Prends soin de restituer tous les détails importants.

Génère le résumé avec exactement cette structure structurée et complète :

## 📋 Résumé Exécutif Détaillé
(Fournis un résumé substantiel de 4 à 6 phrases décrivant le contexte complet, les enjeux et la conclusion de la conversation, de manière parfaitement fidèle)

## 🔑 Thèmes Principaux et Points Clés
(Liste de manière exhaustive TOUS les sujets abordés. Ne te limite pas à 5 points. Utilise autant de points que nécessaire pour capturer le contenu entier)
- **Point majeur 1** : Description détaillée de ce point.
  - Détail ou fait marquant
  - Éléments de discussion affiliés
- **Point majeur 2** : Description détaillée.
  - Détail ou argument avancé
- ... (ajoute autant de points que le contenu l'exige)

## 💡 Décisions, Engagements et Actions
(Capture de façon granulaire tout ce qui a été décidé, les actions urgentes ou les points de désaccord/résolution)
- Décision / Action 1
- Décision / Action 2
- ... (ajoute tout élément applicable, ou indique "Aucune action spécifique formulée" si c'est le cas)

## 📊 Statistiques de session
- Nombre de phrases traitées : {len(history)}
- Analyse linguistique : Source → Cible

Livre un compte-rendu très complet avec beaucoup de bullet points. Valorise la quantité de détails justes plutôt que la simple brièveté."""
                },
                {
                    "role": "user",
                    "content": f"Voici la transcription de la session :\n\n{full_text}"
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )
        summary_text = response.choices[0].message.content
        # BUG-1 FIX: was result_queue.put(summary_text) — blocking call on an unbounded
        # queue that already held the previous result. If the UI hasn't drained yet, this
        # blocks the summary_worker thread indefinitely, freezing the background thread.
        try:
            result_queue.put_nowait(summary_text)
        except Exception:
            pass  # UI will recover via the 45s timeout safety net

    except Exception as e:
        try:
            result_queue.put_nowait(f"❌ Erreur lors de la génération du résumé : {e}")
        except Exception:
            pass  # FREEZE FIX: was queue.Full only — catch all so thread always exits cleanly

def generate_summary_groq():
    """Déclenche la génération de résumé en arrière-plan (asynchrone)."""
    api_key = st.session_state.get("_groq_key", "")
    history = st.session_state.history
    lang = SUMMARY_LANG
    
    if not api_key or not history:
        st.session_state.summary_loading = False
        st.session_state.summary = "❌ Clé Groq manquante ou aucune transcription."
        return

    # Si déjà en cours, ne pas relancer - atomic check-and-set
    with st.session_state.SUMMARY_LOCK:
        if st.session_state.get("summary_in_progress"):
            return
        st.session_state.summary_in_progress = True

    # Lancer le thread de génération
    st.session_state.summary_loading = True
    st.session_state.summary = ""
    st.session_state.summary_start_time = time.time()
    
    threading.Thread(
        target=summary_worker,
        args=(api_key, list(history), lang, st.session_state.SUMMARY_QUEUE),
        daemon=True
    ).start()

def stop_translating():
    st.session_state.STOP_EVENT.set()
    st.session_state.status_dict["running"] = False
    st.session_state.status_dict["msg"]     = "🛑 Arrêté."
    try:
        STATUS_QUEUE.put_nowait("🛑 Arrêté.")
    except queue.Full:
        pass
    # Clear the locked device name on stop
    st.session_state._locked_device_name = None
    # Marquer que le résumé doit être généré au prochain cycle Streamlit
    if st.session_state.get("_groq_key") and st.session_state.history:
        generate_summary_groq()

def clear_conversation():
    st.session_state.history                         = []
    st.session_state.history_map                     = {}
    st.session_state.current_sentence["original"]    = ""
    st.session_state.current_sentence["translation"] = ""
    st.session_state.interim_text                    = ""
    st.session_state.summary                         = ""
    st.session_state.summary_loading                 = False
    st.session_state.summary_in_progress             = False
    st.session_state.current_page                    = "main"
    st.session_state.highlighted_cards               = set()
    st.session_state.highlighted_summaries           = {}
    st.session_state.highlighted_summaries_loading   = False
    # BUG-D FIX: drain stale results from both background queues.
    # A summary/highlight worker still running at clear-time will eventually
    # post its result (or the __HL_DONE__ sentinel) into its queue. Without
    # this drain, the next rerun picks it up and either overwrites summary=""
    # with old text, or flips highlighted_summaries_loading=False prematurely
    # on a fresh generation triggered right after the clear.
    for _stale_q in (st.session_state.SUMMARY_QUEUE,
                     st.session_state.HIGHLIGHT_SUMMARY_QUEUE):
        while not _stale_q.empty():
            try:
                _stale_q.get_nowait()
            except Exception:
                break
    try:
        STATUS_QUEUE.put_nowait("🗑️ Conversation effacée.")
    except queue.Full:
        pass

# ── Drain queues → session state ───────────────────────────────────────────────
# Drain all pending status messages — keep only the LATEST one for display
# BUG-10 FIX: cap drain at 50 items/rerun. If a bug floods STATUS_QUEUE with 500 items
# the old unbounded while loop would spin through all 500 before rendering, blocking
# the main thread for a visible freeze (each get_nowait() has OS overhead).
_last_status_msg = None
_status_drain_limit = 50
_status_drain_count = 0
while _status_drain_count < _status_drain_limit and not STATUS_QUEUE.empty():
    try:
        msg = STATUS_QUEUE.get_nowait()
        if isinstance(msg, str):
            _last_status_msg = msg
    except Exception:
        pass  # FREEZE FIX: was break — one bad item must not abort the whole drain
    _status_drain_count += 1
if _last_status_msg is not None:
    st.session_state.status_dict["msg"] = _last_status_msg

# BUG-11 FIX: cap UI drain at 100 items/rerun (was unbounded). On burst transcription
# all 500 queued items would drain synchronously before the first render, causing a
# perceptible freeze. 100 items/cycle keeps processing lean; the next 300ms rerun
# picks up the remainder. This is the single most important freeze-prevention cap.
_ui_drain_limit = 100
_ui_drain_count = 0
while _ui_drain_count < _ui_drain_limit and not st.session_state.UI_UPDATE_QUEUE.empty():
    try:
        topic, val = st.session_state.UI_UPDATE_QUEUE.get_nowait()
        if topic == "interim":
            st.session_state.interim_text = val
        elif topic == "clear_interim":
            st.session_state.interim_text = ""
        elif topic == "final":
            st.session_state.interim_text = ""
            text, sid = val
            st.session_state.current_sentence["original"] = text
            st.session_state.current_sentence["id"]       = sid
        elif topic == "translation":
            translated, sid = val
            # Update current if matches
            if sid == st.session_state.current_sentence.get("id"):
                st.session_state.current_sentence["translation"] = translated
            # Retroactively update history
            if sid in st.session_state.history_map:
                st.session_state.history_map[sid]["translation"] = translated
        elif topic == "level":
            st.session_state.audio_level = val
        elif topic == "commit":
            sid = val
            orig = st.session_state.current_sentence["original"]
            trad = st.session_state.current_sentence["translation"]
            if orig.strip():
                new_item = {"original": orig, "translation": trad, "id": sid}
                st.session_state.history.insert(0, new_item)
                st.session_state.history_map[sid] = new_item
                # BUG-7 FIX: trim history to 200 items so Streamlit session memory stays
                # bounded over a 40-minute session. Without this, memory grows unboundedly
                # and reruns slow down until the UI queue starves.
                if len(st.session_state.history) > 200:
                    st.session_state.history = st.session_state.history[:200]
                    # BUG-5 FIX: prune stale highlight IDs that were trimmed out of history.
                    # Without this, highlighted_cards holds IDs that no longer exist in history.
                    # They silently waste _HL_MAX quota in generate_highlight_summaries_groq()
                    # and cause highlight_summary_worker to analyse phantom items.
                    _live_ids = {it["id"] for it in st.session_state.history}
                    st.session_state.history_map = {k: v for k, v in st.session_state.history_map.items() if k in _live_ids}
                    st.session_state.highlighted_cards = {
                        hid for hid in st.session_state.highlighted_cards if hid in _live_ids
                    }
            st.session_state.current_sentence["original"]    = ""
            st.session_state.current_sentence["translation"] = ""
            st.session_state.current_sentence["id"]          = 0.0
            st.session_state.interim_text                    = "" # BUG-H FIX: absolutely ensure interim text wipes on commit
    except Exception:
        pass  # BUG-6 FIX: was break — one malformed message must not abort the drain
    _ui_drain_count += 1

# BUG-E FIX: cap SUMMARY_QUEUE drain at 5 items and use pass (not continue).
# summary_worker posts exactly 1 item, but after a timeout→regen→clear cycle a
# previous worker's result can still arrive. The old `continue` inside the while
# loop created an infinite spin if an exception was raised on every iteration.
_sum_drain = 0
while _sum_drain < 5 and not st.session_state.SUMMARY_QUEUE.empty():
    try:
        res = st.session_state.SUMMARY_QUEUE.get_nowait()
        st.session_state.summary = res
        st.session_state.summary_loading = False
        st.session_state.summary_in_progress = False
    except Exception:
        pass  # keep draining — one bad item must not abort
    _sum_drain += 1

# Drain highlighted phrase summaries queue
# BUG-FIX: drain ALL pending items per rerun cycle (old code only handled one item
# and then left the rest for the next cycle — caused 500ms+/phrase display lag).
# The __HL_DONE__ sentinel is the authoritative signal to clear loading state;
# individual items {item_id: text} are applied progressively to highlighted_summaries.
_hl_drain_limit = 50   # safety cap — never spin more than 50 items per rerun
_hl_drain_count = 0
while _hl_drain_count < _hl_drain_limit and not st.session_state.HIGHLIGHT_SUMMARY_QUEUE.empty():
    try:
        res = st.session_state.HIGHLIGHT_SUMMARY_QUEUE.get_nowait()
        if isinstance(res, dict):
            if "__HL_DONE__" in res:
                # Sentinel received — worker finished all phrases
                st.session_state.highlighted_summaries_loading = False
            else:
                # Partial result for one phrase — apply immediately
                st.session_state.highlighted_summaries.update(res)
    except Exception:
        pass
    # BUG-6 FIX: always increment OUTSIDE the try/except so non-dict or exception
    # items don't bypass the counter, preventing an infinite spin on a corrupt queue.
    _hl_drain_count += 1

# Lancer la génération si en attente
if st.session_state.summary_loading and not st.session_state.summary:
    generate_summary_groq()

# ══════════════════════════════════════════════════════════════
# PAGE PRINCIPALE
# ══════════════════════════════════════════════════════════════
if st.session_state.current_page == "main":

    # ── Control buttons ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    with col1:
        if not st.session_state.status_dict["running"]:
            _start_disabled = not DG_API_KEY or selected_device_name is None
            _start_help = None if not _start_disabled else (
                "Clé Deepgram requise" if not DG_API_KEY else "BlackHole introuvable — installe BlackHole 2ch"
            )
            st.button("▶ Démarrer", on_click=start_translating,
                      args=(selected_device_name, DG_MODEL, source_lang_code, target_lang_code, glossary_list, glossary_trans_list),
                      use_container_width=True, disabled=_start_disabled, help=_start_help)
        else:
            st.markdown('<div class="td-rec-btn">', unsafe_allow_html=True)
            st.button("Arrêter", on_click=stop_translating, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.button("Effacer", on_click=clear_conversation, use_container_width=True)
    with col3:
        summarize_disabled = not st.session_state.history or not GROQ_API_KEY
        st.button(
            "Résumé",
            type="primary",
            on_click=go_to_summary,
            use_container_width=True,
            disabled=summarize_disabled,
            help="Disponible après une session de transcription avec une clé Groq"
        )
    with col4:
        if st.session_state.status_dict["running"]:
            st.markdown("<p style='color:#34D399;font-size:0.8rem;padding-top:8px;font-family:JetBrains Mono,monospace;font-weight:600;letter-spacing:1px'>● LIVE</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:var(--subtitle);font-size:0.8rem;padding-top:8px;font-family:JetBrains Mono,monospace;letter-spacing:1px'>● Off</p>", unsafe_allow_html=True)

    if not DG_API_KEY:
        st.warning("👈 Entre ta clé Deepgram dans la barre latérale pour commencer.")

    # ── Status bar ────────────────────────────────────────────
    msg = st.session_state.status_dict["msg"]
    if "❌" in msg:
        st.error(msg)
    elif "✅" in msg or "⚡" in msg or "🎤" in msg:
        st.success(msg)
    else:
        st.info(msg)

    # ── Notification résumé prêt ──────────────────────────────
    if st.session_state.summary and not st.session_state.status_dict["running"]:
        st.success("✅ Résumé IA prêt ! Clique sur **📋 Voir le résumé** dans la barre latérale.")

    # ── Live transcription box ────────────────────────────────
    _orig_raw  = st.session_state.current_sentence["original"]
    _interim   = st.session_state.interim_text
    _trans_raw = st.session_state.current_sentence["translation"]

    display_text = _html.escape(_orig_raw) if _orig_raw else ""
    if _interim:
        ghost = _html.escape(_interim)
        display_text += (" " if _orig_raw else "") + f'<span class="ghost-text">{ghost}…</span>'

    orig  = display_text if display_text else "<span style='opacity:0.3;font-weight:300'>En attente de l'audio…</span>"
    trans = _html.escape(_trans_raw) if _trans_raw else "<span style='opacity:0.3'>La traduction apparaîtra ici…</span>"

    st.markdown(f"""
    <div class="live-box">
        <div class="live-title"><span class="live-dot"></span>TRANSCRIPTION EN DIRECT</div>
        <div class="live-text">{orig}</div>
        <div class="translation-text">{trans}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── History ───────────────────────────────────────────────
    st.subheader("Phrases précédentes")
    if not st.session_state.history:
        st.caption("Les phrases complètes apparaissent ici après chaque pause ou fin de phrase.")
    else:
        _is_dark = st.session_state.theme == "dark"

        # Theme-aware highlight colors — emerald green palette (inline styles — works on all devices)
        if _is_dark:
            HL_CARD  = "background:rgba(16,185,129,0.12)!important;border-left:3px solid #34D399!important;box-shadow:0 0 0 2px rgba(16,185,129,0.22),0 4px 22px rgba(16,185,129,0.18);transform:translateX(3px);"
            HL_ORIG  = "color:#6EE7B7;font-weight:600;"
            HL_COLOR = "#34D399"
        else:
            HL_CARD  = "background:rgba(16,185,129,0.10)!important;border-left:3px solid #059669!important;box-shadow:0 0 0 2px rgba(5,150,105,0.22),0 4px 22px rgba(16,185,129,0.16);transform:translateX(3px);"
            HL_ORIG  = "color:#065F46;font-weight:600;"
            HL_COLOR = "#059669"

        for idx, item in enumerate(st.session_state.history[:60]):
            # BUG FIX: use the sentence's stable unique id for highlight tracking,
            # NOT the list index — history is prepended so indices shift on every new sentence.
            item_id = item.get("id", idx)
            is_hl   = item_id in st.session_state.highlighted_cards
            s_orig  = _html.escape(item.get("original", ""))
            s_trans = _html.escape(item.get("translation", ""))

            card_style = HL_CARD if is_hl else ""
            orig_style = HL_ORIG if is_hl else ""
            sparkle    = f'<span style="position:absolute;right:14px;top:50%;transform:translateY(-50%) rotate(45deg);width:10px;height:10px;background:{HL_COLOR};border-radius:2px;opacity:0.7"></span>' if is_hl else ""

            # Stagger delay: 40ms per card, capped at 400ms (10 cards visible stagger)
            _stagger_ms = min(idx * 40, 400)

            col_card, col_btn = st.columns([11, 1])
            with col_card:
                st.markdown(f"""
                <div class="history-card" style="--stagger:{_stagger_ms}ms;{card_style}position:relative;">
                    <div class="history-original" style="{orig_style}">{s_orig}</div>
                    <div class="history-translation">{s_trans}</div>
                    {sparkle}
                </div>""", unsafe_allow_html=True)
            with col_btn:
                btn_icon = "✦" if is_hl else "◇"
                if st.button(btn_icon, key=f"hl_{item_id}", help="Toggle highlight"):
                    if item_id in st.session_state.highlighted_cards:
                        st.session_state.highlighted_cards.discard(item_id)
                    else:
                        st.session_state.highlighted_cards.add(item_id)
                    st.rerun()

    # BUG-5 FIX: Only rerun when data actually arrives! By halting execution inside a
    # sleep loop here, Streamlit stays passive and avoids needlessly re-rendering all
    # 60 complex history cards every 300ms, which was thrashing the client browser.
    _needs_rerun = st.session_state.status_dict["running"] or st.session_state.summary_loading
    if _needs_rerun:
        _ui_changed = False
        while not st.session_state.STOP_EVENT.is_set():
            _sleep_ms = 0.15 if st.session_state.status_dict["running"] else 0.5
            time.sleep(_sleep_ms)
            
            # Watch queues passively
            if not st.session_state.UI_UPDATE_QUEUE.empty() or \
               not st.session_state.STATUS_QUEUE.empty() or \
               not st.session_state.SUMMARY_QUEUE.empty() or \
               st.session_state.highlighted_summaries_loading:
                _ui_changed = True
                break
                
        if _ui_changed:
            try:
                st.rerun()
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════
# PAGE RÉSUMÉ
# ══════════════════════════════════════════════════════════════
elif st.session_state.current_page == "summary":

    _is_dark_sum = st.session_state.theme == "dark"

    # ── Header page résumé ────────────────────────────────────
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Retour", use_container_width=True):
            st.session_state.current_page = "main"
            st.rerun()
    with col_title:
        st.markdown("""
        <div style='padding-top:6px'>
            <span style='font-family:Outfit,sans-serif;font-size:1.35rem;font-weight:700;color:var(--live-text)'>Résumé de session</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:var(--sb-border);margin:0.8rem 0'>", unsafe_allow_html=True)

    # ── Boutons actions ───────────────────────────────────────
    # Build combined download content (highlights + global)
    _hl_ids   = st.session_state.get("highlighted_cards", set())
    _hl_sums  = st.session_state.get("highlighted_summaries", {})
    _hl_items = [it for it in st.session_state.history if it.get("id") in _hl_ids]

    _download_parts = []
    if _hl_items and _hl_sums:
        _download_parts.append("=== ✦ ANALYSE DES PHRASES CLÉS ===")
        for _it in _hl_items:
            _iid = _it.get("id")
            _download_parts.append(f"\n— Phrase : {_it.get('original','')}")
            _download_parts.append(f"  Traduction : {_it.get('translation','')}")
            _download_parts.append(_hl_sums.get(_iid, "(en cours...)"))
        _download_parts.append("="*50)
    if st.session_state.summary:
        _download_parts.append("\n=== 📋 RÉSUMÉ GLOBAL ===")
        _download_parts.append(st.session_state.summary)

    _combined_download = "\n".join(_download_parts)

    col_r1, col_r2, col_r3 = st.columns([2, 2, 4])
    with col_r1:
        if st.button("Régénérer", use_container_width=True):
            st.session_state.summary = ""
            st.session_state.summary_loading = True
            st.session_state.summary_in_progress = False
            st.session_state.highlighted_summaries = {}
            st.session_state.highlighted_summaries_loading = False
            generate_summary_groq()
            if st.session_state.highlighted_cards:
                generate_highlight_summaries_groq()
    with col_r2:
        if _combined_download and "❌" not in _combined_download:
            st.download_button(
                "Télécharger",
                data=_combined_download,
                file_name="resume_session.txt",
                mime="text/plain",
                use_container_width=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 1 — RÉSUMÉ GLOBAL (en haut, prominent)
    # ══════════════════════════════════════════════════════════
    if _hl_ids:
        st.markdown("""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:16px'>
            <span style='font-family:Outfit,sans-serif;font-size:0.9rem;font-weight:700;color:var(--live-title);
                         text-transform:uppercase;letter-spacing:2.5px'>Résumé Global</span>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.summary_loading and not st.session_state.summary:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.18);
                    border-radius:16px;padding:36px;text-align:center;margin-bottom:24px;
                    backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)'>
            <div style='width:36px;height:36px;border:2px solid var(--live-title);border-top-color:transparent;
                        border-radius:50%;animation:td-spin 0.8s linear infinite;margin:0 auto 14px'></div>
            <div style='color:var(--live-title);font-family:Outfit,sans-serif;font-size:1rem;font-weight:600'>Génération du résumé en cours...</div>
            <div style='color:var(--subtitle);font-size:0.82rem;margin-top:8px'>Groq analyse la session complète</div>
        </div>
        <style>@keyframes td-spin { to { transform: rotate(360deg); } }</style>
        """, unsafe_allow_html=True)
        timeout_limit = 45.0
        elapsed = time.time() - st.session_state.get("summary_start_time", 0)
        if elapsed > timeout_limit:
            st.session_state.summary_loading = False
            st.session_state.summary_in_progress = False
            st.session_state.summary = "❌ Délai dépassé (Timeout). Groq ne répond pas, réessaie plus tard."
            st.rerun()
        time.sleep(0.5)
        st.rerun()

    elif st.session_state.summary:
        # Decorative top bar for the global summary card
        st.markdown("""
        <div style='background:linear-gradient(145deg,rgba(15,20,35,0.95),rgba(20,28,50,0.88));
                    border:1px solid rgba(99,102,241,0.20);border-radius:16px;
                    padding:6px 36px 4px;margin-bottom:0;
                    backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
                    box-shadow:0 8px 32px rgba(0,0,0,0.3);position:relative;overflow:hidden'>
            <div style='position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,transparent,#818CF8,#6366F1,transparent);
                        opacity:0.5'></div>
        </div>
        """, unsafe_allow_html=True)
        # Native markdown renders the full structured summary (bold headers, bullets, etc.)
        st.markdown(st.session_state.summary)

        # Session stats bar
        st.markdown("<hr style='border-color:rgba(99,102,241,0.10);margin:1.2rem 0'>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Phrases transcrites", len(st.session_state.history))
        with col_s2:
            total_words = sum(len(item.get("original", "").split()) for item in st.session_state.history)
            st.metric("Mots transcrits", total_words)
        with col_s3:
            _n_hl = len(st.session_state.get("highlighted_cards", set()))
            st.metric("Phrases clés", _n_hl)

    else:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);
                    border-radius:16px;padding:40px;text-align:center;color:var(--subtitle);
                    margin-bottom:24px;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)'>
            <div style='width:40px;height:40px;border-radius:10px;background:rgba(99,102,241,0.10);
                        display:flex;align-items:center;justify-content:center;margin:0 auto 14px'>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--subtitle)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>
            </div>
            <div style='font-family:Outfit,sans-serif;font-size:1rem;font-weight:600'>Aucun résumé disponible</div>
            <div style='font-size:0.85rem;margin-top:8px'>Lance une session de transcription puis reviens ici</div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 2 — ✦ ANALYSE DES PHRASES CLÉS (en bas)
    # ══════════════════════════════════════════════════════════
    if _hl_ids:
        st.markdown("<br>", unsafe_allow_html=True)

        if _is_dark_sum:
            _hl_section_bg   = "linear-gradient(145deg,rgba(139,92,246,0.06),rgba(88,28,135,0.08))"
            _hl_section_bdr  = "rgba(139,92,246,0.22)"
            _hl_card_bg      = "rgba(139,92,246,0.06)"
            _hl_card_bdr     = "#A78BFA"
            _hl_card_glow    = "0 0 0 1px rgba(139,92,246,0.15), 0 4px 18px rgba(139,92,246,0.12)"
            _hl_phrase_color = "#C4B5FD"
            _hl_trans_color  = "#A78BFA"
            _hl_tag_bg       = "rgba(139,92,246,0.15)"
            _hl_tag_color    = "#A78BFA"
        else:
            _hl_section_bg   = "linear-gradient(145deg,rgba(124,58,237,0.05),rgba(88,28,135,0.07))"
            _hl_section_bdr  = "rgba(124,58,237,0.22)"
            _hl_card_bg      = "rgba(124,58,237,0.06)"
            _hl_card_bdr     = "#7C3AED"
            _hl_card_glow    = "0 0 0 1px rgba(124,58,237,0.18), 0 4px 18px rgba(124,58,237,0.12)"
            _hl_phrase_color = "#5B21B6"
            _hl_trans_color  = "#7C3AED"
            _hl_tag_bg       = "rgba(124,58,237,0.10)"
            _hl_tag_color    = "#7C3AED"

        # ── Section banner ──────────────────────────────────────
        _hl_loading = st.session_state.get("highlighted_summaries_loading", False)
        _done_count = sum(1 for it in _hl_items if it.get("id") in _hl_sums)
        _loading_label = (
            f"⏳ {_done_count}/{len(_hl_items)} analysées..."
            if _hl_loading else
            f"{len(_hl_items)} phrase{'s' if len(_hl_items)>1 else ''}"
        )

        st.markdown(f"""
        <div style='background:{_hl_section_bg};border:1px solid {_hl_section_bdr};
                    border-radius:16px;padding:20px 24px 16px;margin-bottom:22px;
                    position:relative;overflow:hidden'>
            <div style='position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,transparent,{_hl_card_bdr},transparent);opacity:0.6'></div>
            <div style='display:flex;align-items:center;gap:10px'>
                <span style='width:8px;height:8px;background:{_hl_card_bdr};border-radius:2px;transform:rotate(45deg);flex-shrink:0'></span>
                <span style='font-family:Outfit,sans-serif;font-size:0.82rem;font-weight:700;color:{_hl_phrase_color};
                             text-transform:uppercase;letter-spacing:2.5px'>Analyse des Phrases Clés</span>
                <span style='background:{_hl_tag_bg};color:{_hl_tag_color};font-size:0.68rem;
                             font-weight:600;padding:3px 10px;border-radius:20px;
                             letter-spacing:0.5px;font-family:Inter,sans-serif'>{_loading_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-phrase cards ────────────────────────────────────
        for _rank, _item in enumerate(_hl_items, 1):
            _iid      = _item.get("id")
            _orig     = _html.escape(_item.get("original", ""))
            _trad     = _html.escape(_item.get("translation", ""))
            _analysis = _hl_sums.get(_iid)

            # Phrase header card
            st.markdown(f"""
            <div style='background:{_hl_card_bg};border:1px solid {_hl_card_bdr};
                        border-radius:12px;padding:18px 20px 14px;margin-bottom:6px;
                        box-shadow:{_hl_card_glow};position:relative'>
                <div style='position:absolute;top:12px;right:14px;background:{_hl_tag_bg};
                            color:{_hl_tag_color};font-size:0.62rem;font-weight:700;
                            padding:2px 8px;border-radius:10px;letter-spacing:0.5px;
                            font-family:JetBrains Mono,monospace'># {_rank}</div>
                <div style='font-family:Outfit,sans-serif;font-size:0.95rem;font-weight:600;color:{_hl_phrase_color};
                            margin-bottom:5px;line-height:1.55;padding-right:44px'>{_orig}</div>
                <div style='font-family:Inter,sans-serif;font-size:0.85rem;color:{_hl_trans_color};font-style:italic;
                            line-height:1.55;opacity:0.85'>{_trad}</div>
            </div>
            """, unsafe_allow_html=True)

            # Analysis content (or loading / fallback state)
            if _analysis:
                st.markdown(_analysis)
            elif _hl_loading:
                st.markdown("""
                <div style='padding:10px 4px 20px;color:var(--subtitle);font-size:0.82rem;
                            font-style:italic;font-family:Inter,sans-serif'>Analyse en cours...</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='padding:8px 4px 18px;color:var(--subtitle);font-size:0.82rem;
                            font-style:italic;font-family:Inter,sans-serif'>Aucune analyse disponible.</div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)

        # Timeout safety: force-clear loading flag if stuck for >120s
        # BUG-7 FIX: also trigger st.rerun() after clearing — without it the UI
        # stays showing "loading" until the next natural 500ms cycle fires.
        if _hl_loading:
            _hl_elapsed = time.time() - st.session_state.get("highlight_summary_start_time", 0)
            if _hl_elapsed > 120.0:
                st.session_state.highlighted_summaries_loading = False
                try:
                    st.rerun()
                except Exception:
                    pass

    # ── Rerun while any generation is in progress ─────────────
    if st.session_state.get("highlighted_summaries_loading") or \
       (st.session_state.summary_loading and not st.session_state.summary):
        time.sleep(0.5)
        try:
            st.rerun()
        except Exception:
            pass
