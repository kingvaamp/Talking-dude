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
    mono  = _array_mod.array('h', ((samps[i] + samps[i + 1]) // 2 for i in range(0, len(samps), 2)))
    return mono.tobytes()

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
_init_ss("AUDIO_QUEUE",       lambda: queue.Queue(maxsize=400))
_init_ss("TRANSCRIPT_QUEUE",  lambda: queue.Queue())
_init_ss("TRANSLATION_QUEUE", lambda: queue.Queue(maxsize=15))
_init_ss("UI_UPDATE_QUEUE",   lambda: queue.Queue(maxsize=500))
_init_ss("STATUS_QUEUE",      lambda: queue.Queue(maxsize=50))
_init_ss("SUMMARY_QUEUE",     lambda: queue.Queue())
_init_ss("STOP_EVENT",        threading.Event)
_init_ss("interim_text",      lambda: "")
_init_ss("history",           list)
_init_ss("status_dict",       lambda: {"running": False, "msg": "Prêt — appuie sur Start."})
_init_ss("current_sentence",  lambda: {"original": "", "translation": "", "id": 0.0})
_init_ss("audio_level",       lambda: 0.0)
_init_ss("summary",           lambda: "")
_init_ss("summary_loading",   lambda: False)
_init_ss("summary_in_progress", lambda: False)
_init_ss("summary_start_time", lambda: 0.0)
_init_ss("current_page",      lambda: "main")  # "main" ou "summary"
_init_ss("sidebar_hidden", lambda: False)
_init_ss("highlighted_cards", set)

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
        --bg-main: #080b10;
        --text-main: #c8d0e0;
        --header-bg: rgba(8, 11, 16, 0.95);
        --live-bg-1: rgba(10,16,28,0.95);
        --live-bg-2: rgba(16,22,38,0.85);
        --live-border: rgba(0, 200, 255, 0.15);
        --live-title: #00c8ff;
        --live-glow: 0 0 10px #ff2255, 0 0 20px rgba(255,34,85,0.4);
        --live-text: #f0f4ff;
        --ghost-text: 0.35;
        --trans-text: #00e5ff;
        --hist-bg: rgba(255,255,255,0.025);
        --hist-hover: rgba(255,255,255,0.04);
        --hist-border: rgba(0, 200, 255, 0.35);
        --hist-orig: #d0d8ee;
        --hist-trans: #00b8d4;
        --hist-hl-bg: rgba(0, 150, 255, 0.14);
        --hist-hl-border: #00c8ff;
        --hist-hl-glow: 0 0 0 2px rgba(0,200,255,0.18), 0 4px 20px rgba(0,150,255,0.18);
        --btn-bg: rgba(255,255,255,0.04);
        --btn-border: rgba(255,255,255,0.08);
        --btn-hover: rgba(0, 200, 255, 0.12);
        --btn-hover-border: rgba(0, 200, 255, 0.4);
        --btn-shadow: none;
        --btn-primary-bg: linear-gradient(135deg, #00b359, #00d26a);
        --btn-primary-border: #00d26a;
        --btn-primary-hover: linear-gradient(135deg, #00994d, #00b359);
        --btn-primary-text: #ffffff;
        --status-bg: rgba(255,255,255,0.04);
        --sb-bg: rgba(8,11,16,0.98);
        --sb-border: rgba(255,255,255,0.05);
        --subtitle: #4a5570;
        --wv-bg-grad: linear-gradient(180deg,rgba(0,200,255,0.04) 0%,rgba(0,40,80,0.12) 100%);
        --input-bg: rgba(255,255,255,0.05);
        --input-border: rgba(255,255,255,0.1);
        --input-text: #f0f4ff;
        --label-text: #c8d0e0;
        --toggle-icon: #00d26a;
    """
else:
    theme_vars = """
        --bg-main: #f8fafc;
        --text-main: #334155;
        --header-bg: rgba(248, 250, 252, 0.95);
        --live-bg-1: rgba(255,255,255,0.95);
        --live-bg-2: rgba(241,245,249,0.85);
        --live-border: rgba(2, 132, 199, 0.15);
        --live-title: #0284c7;
        --live-glow: 0 0 10px #ef4444, 0 0 20px rgba(239,68,68,0.4);
        --live-text: #0f172a;
        --ghost-text: 0.45;
        --trans-text: #007fb1;
        --hist-bg: rgba(255,255,255,1);
        --hist-hover: rgba(241,245,249,1);
        --hist-border: rgba(2, 132, 199, 0.35);
        --hist-orig: #1e293b;
        --hist-trans: #006080;
        --hist-hl-bg: rgba(2, 132, 199, 0.10);
        --hist-hl-border: #0284c7;
        --hist-hl-glow: 0 0 0 2px rgba(2,132,199,0.15), 0 4px 18px rgba(2,132,199,0.12);
        --btn-bg: #ffffff;
        --btn-border: #94a3b8;
        --btn-hover: rgba(2, 132, 199, 0.08);
        --btn-hover-border: rgba(2, 132, 199, 0.6);
        --btn-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --btn-primary-bg: linear-gradient(135deg, #10b981, #34d399);
        --btn-primary-border: #10b981;
        --btn-primary-hover: linear-gradient(135deg, #059669, #10b981);
        --btn-primary-text: #ffffff;
        --status-bg: rgba(241,245,249,1);
        --sb-bg: rgba(248,250,252,0.98);
        --sb-border: rgba(0,0,0,0.05);
        --subtitle: #64748b;
        --wv-bg-grad: linear-gradient(180deg,rgba(2,132,199,0.04) 0%,rgba(2,132,199,0.12) 100%);
        --input-bg: #ffffff;
        --input-border: #cbd5e1;
        --input-text: #0f172a;
        --label-text: #334155;
        --toggle-icon: #000000;
    """

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {{
        {theme_vars}
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: var(--bg-main);
        color: var(--text-main);
        font-family: 'Space Grotesk', sans-serif;
    }}

    [data-testid="stMainBlockContainer"] {{
        padding-top: 3.5rem !important;
    }}
    [id="stHeader"] {{
        background-color: transparent !important;
    }}

    .stAppHeader {{
        background-color: var(--header-bg) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--live-border);
    }}

    /* ── Live Box ── */
    .live-box {{
        background: linear-gradient(145deg, var(--live-bg-1) 0%, var(--live-bg-2) 100%);
        backdrop-filter: blur(24px);
        border-radius: 18px;
        padding: 28px 32px;
        border: 1px solid var(--live-border);
        box-shadow:
            0 8px 32px rgba(0,0,0,0.1),
            0 0 0 1px rgba(255,255,255,0.03),
            inset 0 1px 0 rgba(255,255,255,0.05);
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
        opacity: 0.6;
    }}

    .live-title {{
        color: var(--live-title);
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 3px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .live-dot {{
        width: 7px;
        height: 7px;
        background: #ff2255;
        border-radius: 50%;
        box-shadow: var(--live-glow);
        animation: pulse 1.8s ease-in-out infinite;
        display: inline-block;
    }}
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.4); opacity: 0.4; }}
    }}

    .live-text {{
        font-size: 1.55rem;
        font-weight: 600;
        line-height: 1.5;
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
        font-size: 1.5rem;
        font-weight: 500;
        color: var(--trans-text);
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid var(--sb-border);
        font-style: italic;
        line-height: 1.5;
        letter-spacing: 0.02em;
    }}

    /* ── History Cards ── */
    .history-card {{
        background: var(--hist-bg);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border-left: 3px solid var(--hist-border);
        transition: all 0.22s cubic-bezier(0.4,0,0.2,1);
        position: relative;
        cursor: pointer;
        user-select: none;
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
        content: "✦";
        position: absolute;
        right: 14px;
        top: 50%;
        transform: translateY(-50%);
        color: var(--hist-hl-border);
        font-size: 0.75rem;
        opacity: 0.7;
    }}
    .history-original {{
        font-size: 1rem;
        color: var(--hist-orig);
        margin-bottom: 6px;
        line-height: 1.5;
        transition: color 0.22s ease;
    }}
    .history-translation {{
        font-size: 1.25rem;
        color: var(--hist-trans);
        font-style: italic;
        line-height: 1.5;
        opacity: 0.95;
    }}

    /* ── Buttons ── */
    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 10px;
        border: 1px solid var(--btn-border);
        background: var(--btn-bg) !important;
        color: var(--text-main) !important;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.02em;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(8px);
        box-shadow: var(--btn-shadow);
    }}
    .stButton > button:disabled,
    .stDownloadButton > button:disabled {{
        background: var(--btn-bg) !important;
        border-color: var(--btn-border) !important;
        color: var(--text-main) !important;
        opacity: 0.5;
        cursor: not-allowed;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stButton > button:active,
    .stDownloadButton > button:active,
    .stButton > button:focus,
    .stDownloadButton > button:focus {{
        background: var(--btn-hover) !important;
        border-color: var(--btn-hover-border) !important;
        color: var(--live-title) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
        outline: none !important;
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
        box-shadow: 0 0 20px var(--btn-primary-border);
        color: var(--btn-primary-text) !important;
        transform: translateY(-2px);
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
        font-size: 0.85rem !important;
    }}
    .stSuccess {{ border-color: rgba(0,255,127,0.25) !important; }}
    .stError   {{ border-color: rgba(255,34,85,0.25) !important; }}

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
        align-items: baseline;
        gap: 14px;
        margin-bottom: 6px;
    }}
    .app-title {{
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--live-text);
        letter-spacing: -0.03em;
    }}
    .app-subtitle {{
        font-size: 0.85rem;
        color: var(--subtitle);
        font-family: 'JetBrains Mono', monospace;
    }}

    
    /* ── Streamlit Widgets Overrides ── */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
        color: var(--input-text) !important;
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
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: var(--text-main) !important;
    }}

    div[data-testid="stSubheader"] {{
        color: var(--subtitle) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        margin-top: 8px;
    }}
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
    <span class="app-title">🎙️ Talking Dude</span>
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
        
        bright = int(180 + 75 * edge_factor)
        colour = f"rgb(0,{bright},255)"
        
        bars.append(
            f'<span class="wb" style="height:{h}px;--c:{colour}"></span>'
        )
        
    bars_html = "".join(bars)
    
    return f"""
<style>
.wv-wrap{{margin:10px 0 4px;padding:0;}}
.wv-lbl{{font-size:0.62rem;color:#3d4d66;text-transform:uppercase;
         letter-spacing:2px;margin-bottom:7px;font-family:'Space Grotesk',sans-serif;}}
.wv{{display:flex;align-items:flex-end;gap:2px;height:46px;
     background:var(--wv-bg-grad);
     border-radius:8px;padding:5px 6px;
     border:1px solid rgba(0,200,255,0.10);box-sizing:border-box;}}
.wb{{flex:1;min-width:1px;border-radius:2px 2px 1px 1px;
     background:linear-gradient(180deg,var(--c,#00c8ff) 0%,rgba(0,80,200,0.6) 100%);
     box-shadow:0 0 5px rgba(0,200,255,0.20);
     transform-origin:bottom center;
     transition: height 0.08s ease;}}
</style>
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

st.sidebar.header("🌍 Langues")

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
st.sidebar.header("🎙️ Audio")

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
st.sidebar.header("🤖 Résumé IA")

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
    generate_summary_groq()


st.sidebar.markdown("---")
st.sidebar.header("🔑 APIs")

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
            last_keepalive_t = time.time()

            while not stop_event.is_set() and not connection_closed.is_set():
                try:
                    data = audio_q.get(timeout=0.1)
                    # STABILITY FIX: Never send empty packets — triggers server-side close
                    if data and len(data) > 0:
                        dg_connection.send(data)
                        last_keepalive_t = time.time()
                except queue.Empty:
                    # STABILITY FIX: KeepAlive every 3s (was 8s) — Deepgram timeout is 10s.
                    # Use SDK keep_alive() which sends the proper text WebSocket frame.
                    now = time.time()
                    if now - last_keepalive_t >= 3.0:
                        try:
                            dg_connection.keep_alive()
                            last_keepalive_t = now
                        except Exception:
                            # Fallback: try raw JSON if SDK method unavailable
                            try:
                                dg_connection.send(json.dumps({"type": "KeepAlive"}))
                                last_keepalive_t = now
                            except Exception:
                                pass
                    continue
                except Exception as e:
                    try:
                        STATUS_Q.put_nowait(f"❌ Erreur envoi Deepgram: {e}")
                    except queue.Full:
                        pass
                    connection_closed.set()
                    break

            try:
                dg_connection.finish()
            except Exception:
                pass

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
        return max(abs(s) for s in samps) / 32768.0
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

                opened_ok = False
                for ch, rate, chunk in CONFIGS:
                    try:
                        stream = p.open(
                            format=pyaudio.paInt16, channels=ch, rate=rate,
                            input=True, input_device_index=target_index,
                            frames_per_buffer=chunk
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
            ratecv_state     = None
            consecutive_errs = 0

            # ── Read loop (NO lock held here) ──────────────────────────
            while not stop_event.is_set():
                try:
                    raw = stream.read(CHUNK, exception_on_overflow=False)
                    consecutive_errs = 0  # reset on success

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

                    if peak < 0.0001:
                        if int(time.time()) % 15 == 0:
                            try:
                                STATUS_Q.put_nowait("⚠️ Pas de signal audio (vérifie BlackHole)")
                            except queue.Full:
                                pass
                    else:
                        if int(time.time()) % 60 == 0:
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
                try:
                    UI_Q.put_nowait(("interim", text.strip()))
                except queue.Full:
                    pass

        except queue.Empty:
            # Fallback commit: if speech stopped > 5s ago and we still have pending text
            if local_orig and time.time() - last_speech_t > 5.0:
                _commit_sentence()
            continue

# ── Translation worker ─────────────────────────────────────────────────────────
# STABILITY FIX: exponential backoff on translation failure to avoid hammering
# Google Translate API during network hiccups which could freeze the thread.
def translation_worker(STATUS_Q, TRANS_Q, UI_Q, stop_event):
    _backoff = 0.0
    while not stop_event.is_set():
        try:
            payload = TRANS_Q.get(timeout=0.5)
        except queue.Empty:
            continue
        text, src, dest, sid = payload
        try:
            translated = GoogleTranslator(source=src[:2], target=dest).translate(text)
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
    st.session_state.AUDIO_QUEUE       = queue.Queue(maxsize=400)
    st.session_state.TRANSCRIPT_QUEUE  = queue.Queue()
    st.session_state.TRANSLATION_QUEUE = queue.Queue(maxsize=15)
    st.session_state.UI_UPDATE_QUEUE   = queue.Queue(maxsize=500)

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

    threading.Thread(
        target=producer_worker,
        args=(input_name, STATUS_QUEUE, audio_q, st.session_state.UI_UPDATE_QUEUE, stop_ev),
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
        client = GroqOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
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
        result_queue.put(summary_text)

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

    # Si déjà en cours, ne pas relancer
    if st.session_state.get("summary_in_progress"):
        return

    # Lancer le thread de génération
    st.session_state.summary_loading = True
    st.session_state.summary = ""
    st.session_state.summary_in_progress = True
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
        st.session_state.summary = ""
        st.session_state.summary_loading = True
        st.session_state.summary_in_progress = False
        generate_summary_groq()

def clear_conversation():
    st.session_state.history                         = []
    st.session_state.current_sentence["original"]    = ""
    st.session_state.current_sentence["translation"] = ""
    st.session_state.interim_text                    = ""
    st.session_state.summary                         = ""
    st.session_state.summary_loading                 = False
    st.session_state.summary_in_progress             = False
    st.session_state.current_page                    = "main"
    st.session_state.highlighted_cards               = set()  # BUG FIX: reset highlights so stale indices don't persist
    try:
        STATUS_QUEUE.put_nowait("🗑️ Conversation effacée.")
    except queue.Full:
        pass

# ── Drain queues → session state ───────────────────────────────────────────────
# Drain all pending status messages — keep only the LATEST one for display
_last_status_msg = None
while not STATUS_QUEUE.empty():
    try:
        msg = STATUS_QUEUE.get_nowait()
        if isinstance(msg, str):
            _last_status_msg = msg
    except Exception:
        continue  # FREEZE FIX: was break — one bad item must not abort the whole drain
if _last_status_msg is not None:
    st.session_state.status_dict["msg"] = _last_status_msg

while not st.session_state.UI_UPDATE_QUEUE.empty():
    try:
        topic, val = st.session_state.UI_UPDATE_QUEUE.get_nowait()
        if topic == "interim":
            st.session_state.interim_text = val
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
            for item in st.session_state.history:
                if item.get("id") == sid:
                    item["translation"] = translated
                    break
        elif topic == "level":
            st.session_state.audio_level = val
        elif topic == "commit":
            sid = val
            orig = st.session_state.current_sentence["original"]
            trad = st.session_state.current_sentence["translation"]
            if orig.strip():
                st.session_state.history.insert(0, {"original": orig, "translation": trad, "id": sid})
                # BUG-7 FIX: trim history to 200 items so Streamlit session memory stays
                # bounded over a 40-minute session. Without this, memory grows unboundedly
                # and reruns slow down until the UI queue starves.
                if len(st.session_state.history) > 200:
                    st.session_state.history = st.session_state.history[:200]
            st.session_state.current_sentence["original"]    = ""
            st.session_state.current_sentence["translation"] = ""
            st.session_state.current_sentence["id"]          = 0.0
    except Exception:
        continue  # BUG-6 FIX: was break — one malformed message must not abort the drain

while not st.session_state.SUMMARY_QUEUE.empty():
    try:
        res = st.session_state.SUMMARY_QUEUE.get_nowait()
        st.session_state.summary = res
        st.session_state.summary_loading = False
        st.session_state.summary_in_progress = False
    except Exception:
        continue  # BUG-12 FIX: was break — keep draining even if one item fails

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
            st.button("🟢 Démarrer", on_click=start_translating,
                      args=(selected_device_name, DG_MODEL, source_lang_code, target_lang_code, glossary_list, glossary_trans_list),
                      use_container_width=True, disabled=_start_disabled, help=_start_help)
        else:
            st.button("🔴 Arrêter", on_click=stop_translating, use_container_width=True)
    with col2:
        st.button("🗑️ Effacer", on_click=clear_conversation, use_container_width=True)
    with col3:
        summarize_disabled = not st.session_state.history or not GROQ_API_KEY
        st.button(
            "📋 Résumé",
            type="primary",
            on_click=go_to_summary,
            use_container_width=True,
            disabled=summarize_disabled,
            help="Disponible après une session de transcription avec une clé Groq"
        )
    with col4:
        if st.session_state.status_dict["running"]:
            st.markdown("<p style='color:#00ff7f;font-size:0.85rem;padding-top:8px'>● LIVE</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#4a5570;font-size:0.85rem;padding-top:8px'>● Off</p>", unsafe_allow_html=True)

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

        # Theme-aware highlight colors (inline styles — works on all devices)
        if _is_dark:
            HL_CARD  = "background:rgba(0,210,100,0.13)!important;border-left:3px solid #00d26a!important;box-shadow:0 0 0 2px rgba(0,210,100,0.22),0 4px 22px rgba(0,210,100,0.18);transform:translateX(3px);"
            HL_ORIG  = "color:#00d26a;font-weight:600;"
            HL_COLOR = "#00d26a"
        else:
            HL_CARD  = "background:rgba(16,185,129,0.22)!important;border-left:3px solid #047857!important;box-shadow:0 0 0 3px rgba(4,120,87,0.40),0 4px 22px rgba(16,185,129,0.30);transform:translateX(3px);"
            HL_ORIG  = "color:#065f46;font-weight:600;"
            HL_COLOR = "#047857"

        for idx, item in enumerate(st.session_state.history[:60]):
            # BUG FIX: use the sentence's stable unique id for highlight tracking,
            # NOT the list index — history is prepended so indices shift on every new sentence.
            item_id = item.get("id", idx)
            is_hl   = item_id in st.session_state.highlighted_cards
            s_orig  = _html.escape(item.get("original", ""))
            s_trans = _html.escape(item.get("translation", ""))

            card_style = HL_CARD if is_hl else ""
            orig_style = HL_ORIG if is_hl else ""
            sparkle    = f'<span style="position:absolute;right:14px;top:50%;transform:translateY(-50%);font-size:1.4rem;color:{HL_COLOR};line-height:1">✦</span>' if is_hl else ""

            col_card, col_btn = st.columns([11, 1])
            with col_card:
                st.markdown(f"""
                <div class="history-card" style="{card_style}position:relative;">
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

    # ── Refresh loop ──────────────────────────────────────────
    if st.session_state.status_dict["running"]:
        time.sleep(0.3)
        try:
            st.rerun()
        except Exception:
            pass

    # Rerun si résumé en cours de génération
    if st.session_state.summary_loading:
        time.sleep(0.5)
        try:
            st.rerun()
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════
# PAGE RÉSUMÉ
# ══════════════════════════════════════════════════════════════
elif st.session_state.current_page == "summary":

    # ── Header page résumé ────────────────────────────────────
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Retour", use_container_width=True):
            st.session_state.current_page = "main"
            st.rerun()
    with col_title:
        st.markdown("""
        <div style='padding-top:6px'>
            <span style='font-size:1.4rem;font-weight:700;color:var(--live-text)'>📋 Résumé de session</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:var(--sb-border);margin:0.8rem 0'>", unsafe_allow_html=True)

    # ── Boutons actions ───────────────────────────────────────
    col_r1, col_r2, col_r3 = st.columns([2, 2, 4])
    with col_r1:
        if st.button("🔄 Régénérer", use_container_width=True):
            st.session_state.summary = ""
            st.session_state.summary_loading = True
            st.session_state.summary_in_progress = False
            generate_summary_groq()  # BUG FIX: call directly — relying on cycle was unreliable
    with col_r2:
        if st.session_state.summary and "❌" not in st.session_state.summary:
            st.download_button(
                "💾 Télécharger",
                data=st.session_state.summary,
                file_name="resume_session.txt",
                mime="text/plain",
                use_container_width=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Affichage du résumé ───────────────────────────────────
    if st.session_state.summary_loading and not st.session_state.summary:
        st.markdown("""
        <div style='background:rgba(0,200,255,0.05);border:1px solid rgba(0,200,255,0.15);
                    border-radius:16px;padding:40px;text-align:center'>
            <div style='font-size:2rem;margin-bottom:16px'>⏳</div>
            <div style='color:#00c8ff;font-size:1rem;font-weight:600'>Génération du résumé en cours...</div>
            <div style='color:#4a5570;font-size:0.85rem;margin-top:8px'>Groq analyse ta session</div>
        </div>
        """, unsafe_allow_html=True)
        # La génération est déjà déclenchée par le cycle principal
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
        # Afficher le résumé formaté
        summary_escaped = st.session_state.summary
        st.markdown(f"""
        <div style='background:linear-gradient(145deg,rgba(10,16,28,0.98),rgba(16,22,38,0.9));
                    border:1px solid rgba(0,200,255,0.2);border-radius:18px;
                    padding:32px 36px;line-height:1.9;color:#c8d0e0;font-size:0.95rem;
                    box-shadow:0 8px 32px rgba(0,0,0,0.5);position:relative;overflow:hidden'>
            <div style='position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,transparent,#00c8ff,#0066ff,transparent);
                        opacity:0.5'></div>
        </div>
        """, unsafe_allow_html=True)

        # Utiliser st.markdown natif pour le rendu Markdown du résumé
        st.markdown(summary_escaped)

        # Stats de session
        st.markdown("<hr style='border-color:#1a2030;margin:1.5rem 0'>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Phrases transcrites", len(st.session_state.history))
        with col_s2:
            total_words = sum(len(item.get("original", "").split()) for item in st.session_state.history)
            st.metric("Mots transcrits", total_words)
        with col_s3:
            st.metric("Modèle IA", "Groq Llama 3.3")

    else:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);
                    border-radius:16px;padding:40px;text-align:center;color:#4a5570'>
            <div style='font-size:2rem;margin-bottom:12px'>🤖</div>
            <div style='font-size:1rem'>Aucun résumé disponible</div>
            <div style='font-size:0.85rem;margin-top:8px'>Lance une session de transcription puis reviens ici</div>
        </div>
        """, unsafe_allow_html=True)
