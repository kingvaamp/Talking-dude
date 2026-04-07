import streamlit as st
import html as _html
import json
import os
import re
import queue
import threading
import time
import difflib
import asyncio
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
        return audioop.tomono(raw, 2, 1, 1)
    import array as _arr
    samps = _arr.array('h', raw)
    mono  = _arr.array('h', ((samps[i] + samps[i + 1]) // 2 for i in range(0, len(samps), 2)))
    return mono.tobytes()

def _ratecv(raw: bytes, in_rate: int, out_rate: int, state):
    """Sample-rate conversion int16 mono (pure Python fallback). Returns (bytes, state)."""
    if _HAS_AUDIOOP:
        return audioop.ratecv(raw, 2, 1, in_rate, out_rate, state)
    import array as _arr
    samps  = _arr.array('h', raw)
    n_in   = len(samps)
    n_out  = int(n_in * out_rate / in_rate)
    result = _arr.array('h')
    for i in range(n_out):
        pos  = i * in_rate / out_rate
        idx  = int(pos)
        frac = pos - idx
        a    = samps[idx]         if idx     < n_in else 0
        b    = samps[idx + 1]     if idx + 1 < n_in else a
        result.append(max(-32768, min(32767, int(a + frac * (b - a)))))
    return result.tobytes(), None  # stateless fallback

try:
    from deep_translator import GoogleTranslator
except ImportError:
    st.error("❌ `deep-translator` manquant. Lance : `pip install deep-translator`")
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

_persisted = load_settings()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #080b10;
        color: #c8d0e0;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Move everything up slightly but keep it visible */
    [data-testid="stMainBlockContainer"] {
        padding-top: 3.5rem !important;
    }
    [id="stHeader"] {
        background-color: transparent !important;
    }

    .stAppHeader {
        background-color: rgba(8, 11, 16, 0.95) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(0, 200, 255, 0.08);
    }

    /* ── Live Box ── */
    .live-box {
        background: linear-gradient(145deg, rgba(10,16,28,0.95) 0%, rgba(16,22,38,0.85) 100%);
        backdrop-filter: blur(24px);
        border-radius: 18px;
        padding: 28px 32px;
        border: 1px solid rgba(0, 200, 255, 0.15);
        box-shadow:
            0 8px 32px rgba(0,0,0,0.6),
            0 0 0 1px rgba(255,255,255,0.03),
            inset 0 1px 0 rgba(255,255,255,0.05);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .live-box::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00c8ff, #0066ff, transparent);
        opacity: 0.6;
    }

    .live-title {
        color: #00c8ff;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 3px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .live-dot {
        width: 7px;
        height: 7px;
        background: #ff2255;
        border-radius: 50%;
        box-shadow: 0 0 10px #ff2255, 0 0 20px rgba(255,34,85,0.4);
        animation: pulse 1.8s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.4); opacity: 0.4; }
    }

    .live-text {
        font-size: 1.55rem;
        font-weight: 600;
        line-height: 1.5;
        color: #f0f4ff;
        min-height: 2.5rem;
        letter-spacing: -0.01em;
    }
    .ghost-text {
        opacity: 0.35;
        font-style: italic;
        font-weight: 400;
    }

    .translation-text {
        font-size: 1.2rem;
        font-weight: 400;
        color: #5bb8ff;
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid rgba(255,255,255,0.06);
        font-style: italic;
        line-height: 1.5;
        letter-spacing: 0.01em;
    }

    /* ── History Cards ── */
    .history-card {
        background: rgba(255,255,255,0.025);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border-left: 3px solid rgba(0, 200, 255, 0.35);
        transition: all 0.2s ease;
        position: relative;
    }
    .history-card:hover {
        background: rgba(255,255,255,0.04);
        border-left-color: rgba(0, 200, 255, 0.7);
        transform: translateX(3px);
    }
    .history-original {
        font-size: 1rem;
        color: #d0d8ee;
        margin-bottom: 6px;
        line-height: 1.5;
    }
    .history-translation {
        font-size: 0.95rem;
        color: #4a9fd4;
        font-style: italic;
        line-height: 1.5;
        opacity: 0.85;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.04);
        color: #c8d0e0;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
        backdrop-filter: blur(8px);
    }
    .stButton > button:hover {
        background: rgba(0, 200, 255, 0.12) !important;
        border-color: rgba(0, 200, 255, 0.4) !important;
        color: #00c8ff !important;
        box-shadow: 0 0 20px rgba(0, 200, 255, 0.15);
    }

    /* ── Status ── */
    .stInfo, .stSuccess, .stError, .stWarning {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem !important;
    }
    .stSuccess { border-color: rgba(0,255,127,0.25) !important; }
    .stError   { border-color: rgba(255,34,85,0.25) !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(8,11,16,0.98) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* ── Header ── */
    .app-header {
        display: flex;
        align-items: baseline;
        gap: 14px;
        margin-bottom: 6px;
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f4ff;
        letter-spacing: -0.03em;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #4a5570;
        font-family: 'JetBrains Mono', monospace;
    }

    div[data-testid="stSubheader"] {
        color: #4a5570 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
def _init_ss(key, factory):
    if key not in st.session_state:
        st.session_state[key] = factory()

_init_ss("AUDIO_QUEUE",       lambda: queue.Queue())
_init_ss("TRANSCRIPT_QUEUE",  lambda: queue.Queue())
_init_ss("TRANSLATION_QUEUE", lambda: queue.Queue(maxsize=1))
_init_ss("UI_UPDATE_QUEUE",   lambda: queue.Queue())
_init_ss("STATUS_QUEUE",      lambda: queue.Queue())
_init_ss("STOP_EVENT",        threading.Event)
_init_ss("interim_text",      lambda: "")
_init_ss("history",           list)
_init_ss("status_dict",       lambda: {"running": False, "msg": "Prêt — appuie sur Start."})
_init_ss("current_sentence",  lambda: {"original": "", "translation": ""})
_init_ss("audio_level",       lambda: 0.0)
st.session_state.sidebar_hidden = False

# ── Global Lock for PyAudio Stability ──────────────────────────────────────────
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
@st.cache_resource(show_spinner=False)
def get_audio_devices_cached():
    """Returns a dictionary of {name: index} for available input devices (cached)."""
    with PA_LOCK:
        p = pyaudio.PyAudio()
        devices = {}
        try:
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    devices[info["name"]] = i
            return devices
        finally:
            p.terminate()

def get_audio_devices():
    return get_audio_devices_cached()

def find_blackhole_index():
    """Find the BlackHole audio device index."""
    devices = get_audio_devices()
    for name, idx in devices.items():
        if "blackhole" in name.lower():
            return idx
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("🔑 Deepgram API")

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
st.sidebar.header("🎤 Audio")

devices_dict = get_audio_devices()
device_names = list(devices_dict.keys())

# Default to BlackHole if present, else first available
bh_idx = find_blackhole_index()
default_dev_name = None
if bh_idx is not None:
    for name, idx in devices_dict.items():
        if idx == bh_idx:
            default_dev_name = name
            break
if not default_dev_name and device_names:
    default_dev_name = device_names[0]

_saved_device = _persisted.get("audio_device", default_dev_name)
_dev_idx = device_names.index(_saved_device) if _saved_device in device_names else 0

selected_device_name = st.sidebar.selectbox("Entrée audio", device_names, index=_dev_idx)
SELECTED_DEVICE_INDEX = devices_dict[selected_device_name]

if selected_device_name != _saved_device:
    save_settings({"audio_device": selected_device_name})

st.sidebar.info(f"Appareil ID: {SELECTED_DEVICE_INDEX}")

if st.session_state.status_dict["running"]:
    st.sidebar.write("Niveau audio")
    st.sidebar.progress(min(1.0, st.session_state.get("audio_level", 0.0)))

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
# ── Deepgram streaming worker ──────────────────────────────────────────────────
def deepgram_stream_worker(api_key, model, lang_code, audio_q, transcript_q, STATUS_Q, stop_event):
    """Sync-style connection for stability in multi-threaded environment."""
    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        client = DeepgramClient(api_key, config)
        dg_connection = client.listen.live.v("1")

        packets_received = 0

        def on_message(self, result, **kwargs):
            nonlocal packets_received
            try:
                # v3 structure parsing
                transcript = result.channel.alternatives[0].transcript
                if transcript:
                    packets_received += 1
                    transcript_q.put((transcript, result.is_final))
                    # Periodically report activity to UI
                    if packets_received % 5 == 0:
                        STATUS_Q.put(f"⚡ Streaming Deepgram : {packets_received} packets")
            except Exception:
                pass

        def on_error(self, error, **kwargs):
            STATUS_Q.put(f"❌ Deepgram: {error}")

        def on_close(self, close, **kwargs):
            STATUS_Q.put("🔌 Connexion Deepgram fermée.")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)

        options = LiveOptions(
            model=model,
            language=lang_code[:2],
            smart_format=True,
            interim_results=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
        )

        if not dg_connection.start(options):
            STATUS_Q.put("❌ Connexion Deepgram échouée.")
            return

        STATUS_Q.put("⚡ Streaming Deepgram actif")

        # Stream audio from queue to Deepgram
        while not stop_event.is_set():
            try:
                data = audio_q.get(timeout=0.1)
                dg_connection.send(data)
            except queue.Empty:
                continue
            except Exception as e:
                STATUS_Q.put(f"❌ Erreur envoi: {e}")
                break

        dg_connection.finish()

    except Exception as e:
        STATUS_Q.put(f"❌ Deepgram critique: {e}")

# No longer need run_dg_async_bridge as worker is sync-style for robust threading


# ── Audio producer (BlackHole → bytes 16kHz mono) ─────────────────────────────
TARGET_RATE = 16000

def _get_peak(raw_bytes):
    import array as _arr
    if not raw_bytes: return 0
    samps = _arr.array('h', raw_bytes)
    if not samps: return 0
    return max(abs(s) for s in samps) / 32768.0

def producer_worker(bh_index, STATUS_Q, AUDIO_Q, UI_Q, stop_event):
    CONFIGS = [(2, 48000, 2400), (1, 48000, 2400), (2, 44100, 2205), (1, 16000, 800)]
    stream = None
    p      = None
    CHANNELS = RATE = CHUNK = None

    try:
        with PA_LOCK:
            p = pyaudio.PyAudio()
            for ch, rate, chunk in CONFIGS:
                try:
                    stream = p.open(
                        format=pyaudio.paInt16, channels=ch, rate=rate,
                        input=True, input_device_index=bh_index,
                        frames_per_buffer=chunk
                    )
                    CHANNELS, RATE, CHUNK = ch, rate, chunk
                    STATUS_Q.put(f"🎤 Audio: {ch}ch @ {rate}Hz (Lancement...)")
                    break
                except Exception:
                    continue

        if not stream:
            raise RuntimeError("Impossible d'ouvrir l'entrée audio — vérifie tes réglages.")

        STATUS_Q.put(f"🎤 Audio: {CHANNELS}ch @ {RATE}Hz actif")
        ratecv_state = None
        while not stop_event.is_set():
            try:
                raw = stream.read(CHUNK, exception_on_overflow=False)
                if CHANNELS == 2:
                    raw = _to_mono(raw)
                if RATE != TARGET_RATE:
                    raw, ratecv_state = _ratecv(raw, RATE, TARGET_RATE, ratecv_state)

                peak = _get_peak(raw)
                UI_Q.put(("level", peak))
                AUDIO_Q.put(raw)
            except Exception:
                continue

    except Exception as e:
        STATUS_Q.put(f"❌ Producteur audio: {e}")
    finally:
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
        STATUS_Q.put("🛑 Flux audio fermé.")


# ── Consumer (transcripts → UI queue) ─────────────────────────────────────────
def consumer_worker(source_lang, target_lang, history_list, status_dict,
                    active_glossary, active_glossary_trans,
                    STATUS_Q, TRANSCRIPT_Q, TRANS_Q, UI_Q, stop_event):

    last_speech_t = time.time()
    local_orig = ""

    def _commit_sentence():
        nonlocal local_orig
        if not local_orig.strip():
            return
        UI_Q.put(("commit", None))
        local_orig = ""

    while not stop_event.is_set():
        try:
            text, is_final = TRANSCRIPT_Q.get(timeout=0.1)
            if is_final:
                clean_text = text.strip()
                sep = " " if local_orig else ""

                local_orig += sep + clean_text
                last_speech_t = time.time()
                UI_Q.put(("final", local_orig))

                # Envoyer pour traduction
                try:
                    TRANS_Q.get_nowait()
                except Exception:
                    pass
                TRANS_Q.put((local_orig, source_lang, target_lang))

                if _is_sentence_end(clean_text) or len(local_orig.split()) >= 50:
                    _commit_sentence()
            else:
                UI_Q.put(("interim", text.strip()))

        except queue.Empty:
            if local_orig and time.time() - last_speech_t > 5.0:
                _commit_sentence()
            continue

# ── Translation worker ─────────────────────────────────────────────────────────
def translation_worker(STATUS_Q, TRANS_Q, UI_Q, stop_event):
    while not stop_event.is_set():
        try:
            payload = TRANS_Q.get(timeout=0.5)
        except queue.Empty:
            continue
        text, src, dest = payload
        try:
            translated = GoogleTranslator(source=src[:2], target=dest).translate(text)
            if translated:
                UI_Q.put(("translation", translated))
        except Exception as e:
            UI_Q.put(("translation", f"[Erreur traduction: {e}]"))

# ── Start / Stop / Clear ───────────────────────────────────────────────────────
def start_translating():
    dg_key = st.session_state.get("_dg_key", "")
    if not dg_key:
        STATUS_QUEUE.put("❌ Entre ta clé Deepgram dans la barre latérale.")
        return

    input_index = SELECTED_DEVICE_INDEX

    # Signal stop to any existing threads
    st.session_state.STOP_EVENT        = threading.Event()
    st.session_state.AUDIO_QUEUE       = queue.Queue()
    st.session_state.TRANSCRIPT_QUEUE  = queue.Queue()
    st.session_state.TRANSLATION_QUEUE = queue.Queue(maxsize=1)
    st.session_state.UI_UPDATE_QUEUE   = queue.Queue()

    stop_ev = st.session_state.STOP_EVENT
    audio_q = st.session_state.AUDIO_QUEUE
    dg_q    = st.session_state.TRANSCRIPT_QUEUE
    trans_q = st.session_state.TRANSLATION_QUEUE
    ui_q    = st.session_state.UI_UPDATE_QUEUE

    st.session_state.status_dict["running"] = True
    st.session_state.status_dict["msg"]     = "⚡ Démarrage..."

    threading.Thread(
        target=producer_worker,
        args=(input_index, STATUS_QUEUE, audio_q, st.session_state.UI_UPDATE_QUEUE, stop_ev),
        daemon=True
    ).start()

    threading.Thread(
        target=deepgram_stream_worker,
        args=(dg_key, DG_MODEL, source_lang_code, audio_q, dg_q, STATUS_QUEUE, stop_ev),
        daemon=True
    ).start()


    threading.Thread(
        target=consumer_worker,
        args=(source_lang_code, target_lang_code,
              st.session_state.history, st.session_state.status_dict,
              glossary_list, glossary_trans_list,
              STATUS_QUEUE, dg_q, trans_q, ui_q, stop_ev),
        daemon=True
    ).start()

    threading.Thread(
        target=translation_worker,
        args=(STATUS_QUEUE, trans_q, ui_q, stop_ev),
        daemon=True
    ).start()

def stop_translating():
    st.session_state.STOP_EVENT.set()
    st.session_state.status_dict["running"] = False
    st.session_state.status_dict["msg"]     = "🛑 Arrêté."
    STATUS_QUEUE.put("🛑 Arrêté.")

def clear_conversation():
    st.session_state.history                         = []
    st.session_state.current_sentence["original"]    = ""
    st.session_state.current_sentence["translation"] = ""
    st.session_state.interim_text                    = ""
    STATUS_QUEUE.put("🗑️ Conversation effacée.")

# ── Drain queues → session state ───────────────────────────────────────────────
while not STATUS_QUEUE.empty():
    try:
        msg = STATUS_QUEUE.get_nowait()
        if isinstance(msg, str):
            st.session_state.status_dict["msg"] = msg
    except Exception:
        break

while not UI_UPDATE_QUEUE.empty():
    try:
        topic, val = UI_UPDATE_QUEUE.get_nowait()
        if topic == "interim":
            st.session_state.interim_text = val
        elif topic == "final":
            st.session_state.interim_text = ""
            st.session_state.current_sentence["original"] = val
        elif topic == "translation":
            st.session_state.current_sentence["translation"] = val
        elif topic == "level":
            st.session_state.audio_level = val
        elif topic == "commit":
            orig = st.session_state.current_sentence["original"]
            trad = st.session_state.current_sentence["translation"]
            if orig.strip():
                st.session_state.history.insert(0, {"original": orig, "translation": trad})
            st.session_state.current_sentence["original"]    = ""
            st.session_state.current_sentence["translation"] = ""
    except Exception:
        break

# ── Control buttons ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 3, 2])
with col1:
    if not st.session_state.status_dict["running"]:
        st.button("🟢 Démarrer", on_click=start_translating,
                  use_container_width=True, disabled=not DG_API_KEY)
    else:
        st.button("🔴 Arrêter", on_click=stop_translating, use_container_width=True)
with col2:
    st.button("🗑️ Effacer la conversation", on_click=clear_conversation, use_container_width=True)
with col3:
    # Indicateur de statut compact
    if st.session_state.status_dict["running"]:
        st.markdown("<p style='color:#00ff7f;font-size:0.85rem;padding-top:8px'>● EN DIRECT</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#4a5570;font-size:0.85rem;padding-top:8px'>● Inactif</p>", unsafe_allow_html=True)

if not DG_API_KEY:
    st.warning("👈 Entre ta clé Deepgram dans la barre latérale pour commencer.")

# Sidebar hide CSS logic removed

# ── Status bar ─────────────────────────────────────────────────────────────────
msg = st.session_state.status_dict["msg"]
if "❌" in msg:
    st.error(msg)
elif "✅" in msg or "⚡" in msg or "🎤" in msg:
    st.success(msg)
else:
    st.info(msg)

# ── Live transcription box ─────────────────────────────────────────────────────
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

# ── History ────────────────────────────────────────────────────────────────────
st.subheader("Phrases précédentes")
if not st.session_state.history:
    st.caption("Les phrases complètes apparaissent ici après chaque pause ou fin de phrase.")
else:
    for item in st.session_state.history:
        s_orig  = _html.escape(item.get("original", ""))
        s_trans = _html.escape(item.get("translation", ""))
        st.markdown(f"""
        <div class="history-card">
            <div class="history-original">{s_orig}</div>
            <div class="history-translation">{s_trans}</div>
        </div>""", unsafe_allow_html=True)

# ── Refresh loop ───────────────────────────────────────────────────────────────
if st.session_state.status_dict["running"]:
    time.sleep(0.08)
    try:
        st.rerun()
    except Exception:
        pass
