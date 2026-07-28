import re

with open("Talking_Dude_NiceGUI.py", "r", encoding="utf-8") as f:
    code = f.read()

start_marker_str = "# ══════════════════════════════════════════════════════════════════════════════\n# CSS THEME\n# ══════════════════════════════════════════════════════════════════════════════\n"
end_marker_str = "# ══════════════════════════════════════════════════════════════════════════════\n# UI — NiceGUI\n# ══════════════════════════════════════════════════════════════════════════════\n"

head = code.split(start_marker_str)[0]
tail = code.split(end_marker_str)[1]

# Apple HIG UI Replacement string
new_ui = """DARK_VARS = '''
    --bg: #000000;
    --bg2: #1c1c1e;
    --surface: #1c1c1e;
    --surface2: #2c2c2e;
    --border: rgba(255,255,255,0.08);
    --border2: rgba(255,255,255,0.04);
    --accent: #0a84ff;
    --accent-glow: rgba(10,132,255,0.15);
    --accent-hover: #007aff;
    --accent2: #30d158;
    --accent2-glow: rgba(48,209,88,0.1);
    --red: #ff453a;
    --text: #f5f5f7;
    --text-muted: #86868b;
    --text-dim: #98989d;
    --text-trans: #30d158;

    --bg-main: var(--bg);
    --text-main: var(--text);
    --header-bg: rgba(28,28,30,0.65);

    --live-bg-1: #1c1c1e;
    --live-bg-2: #2c2c2e;
    --live-border: rgba(255,255,255,0.05);
    --live-title: var(--text-dim);
    --live-text: #ffffff;
    --ghost-text: 0.4;
    --trans-text: var(--text-trans);

    --hist-bg: #151516;
    --hist-hover: #1c1c1e;
    --hist-border: rgba(255,255,255,0.05);
    --hist-orig: var(--text);
    --hist-trans: var(--text-trans);

    --btn-bg: #1c1c1e;
    --btn-border: rgba(255,255,255,0.08); /* Minimal border for depth */
    --btn-hover: #3a3a3c;
    --btn-hover-border: rgba(255,255,255,0.12);

    --btn-primary-bg: #0a84ff;
    --btn-primary-border: transparent;
    --btn-primary-hover: #007aff;
    --btn-primary-text: #ffffff;

    --status-bg: rgba(28,28,30,0.6);
    --sb-bg: transparent;
    --sb-border: rgba(255,255,255,0.05);
    --subtitle: var(--text-muted);

    --input-bg: rgba(118,118,128,0.24);
    --input-border: transparent;
    --input-text: #ffffff;
    --label-text: var(--text-dim);

    --hl-card-bg: rgba(10,132,255,0.08);
    --hl-card-border: rgba(10,132,255,0.3);
    --hl-orig-color: var(--accent);
    --hl-diamond-color: var(--accent);

    --glass-blur: blur(40px) saturate(200%);
    --noise: none;
'''

LIGHT_VARS = '''
    --bg: #f5f5f7;
    --bg2: #e5e5ea;
    --surface: #ffffff;
    --surface2: #f2f2f7;
    --border: rgba(0,0,0,0.06);
    --border2: rgba(0,0,0,0.03);
    --accent: #007aff;
    --accent-glow: rgba(0,122,255,0.1);
    --accent-hover: #006ce4;
    --accent2: #34c759;
    --accent2-glow: rgba(52,199,89,0.1);
    --red: #ff3b30;
    --text: #1d1d1f;
    --text-muted: #86868b;
    --text-dim: #8e8e93;
    --text-trans: #34c759;

    --bg-main: var(--bg);
    --text-main: var(--text);
    --header-bg: rgba(255,255,255,0.65);

    --live-bg-1: #ffffff;
    --live-bg-2: #f2f2f7;
    --live-border: rgba(0,0,0,0.04);
    --live-title: var(--text-muted);
    --live-text: #1d1d1f;
    --ghost-text: 0.45;
    --trans-text: var(--text-trans);

    --hist-bg: #ffffff;
    --hist-hover: #f2f2f7;
    --hist-border: rgba(0,0,0,0.05);
    --hist-orig: var(--text);
    --hist-trans: var(--text-trans);

    --btn-bg: #e5e5ea;
    --btn-border: transparent;
    --btn-hover: #d1d1d6;
    --btn-hover-border: transparent;

    --btn-primary-bg: #007aff;
    --btn-primary-border: transparent;
    --btn-primary-hover: #006ce4;
    --btn-primary-text: #ffffff;

    --status-bg: rgba(255,255,255,0.6);
    --sb-bg: transparent;
    --sb-border: rgba(0,0,0,0.05);
    --subtitle: var(--text-muted);

    --input-bg: rgba(118,118,128,0.12);
    --input-border: transparent;
    --input-text: #1d1d1f;
    --label-text: var(--text-dim);

    --hl-card-bg: rgba(0,122,255,0.06);
    --hl-card-border: rgba(0,122,255,0.25);
    --hl-orig-color: var(--accent);
    --hl-diamond-color: var(--accent);

    --glass-blur: blur(40px) saturate(200%);
    --noise: none;
'''

def _get_theme_vars():
    return DARK_VARS

GLOBAL_CSS = f'''
body.dark {{ {DARK_VARS} }}
body.light {{ {LIGHT_VARS} }}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    transition: background 0.4s ease, color 0.4s ease;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.011em;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 8px; border: 2px solid var(--bg); }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text-dim); }}

/* ── Quasar overrides ── */
.nicegui-content {{ padding: 0 !important; }}
.q-page {{ background: var(--bg) !important; }}
.q-drawer {{
    background: var(--bg) !important;
    border-right: 1px solid var(--border) !important;
}}
.q-header {{
    background: var(--header-bg) !important;
    backdrop-filter: var(--glass-blur) !important;
    -webkit-backdrop-filter: var(--glass-blur) !important;
    border-bottom: 1px solid var(--border) !important;
}}
.q-separator {{ background: var(--border) !important; opacity: 1; }}
.q-field__control {{ background: var(--input-bg) !important; }}
.q-field__native, .q-field__input {{
    color: var(--input-text) !important; 
    font-family: -apple-system, BlinkMacSystemFont, sans-serif !important; 
}}
.q-field__label {{ 
    color: var(--label-text) !important; 
    font-family: -apple-system, BlinkMacSystemFont, sans-serif !important; 
    font-size: 0.75rem !important; 
    letter-spacing: 0; 
}}
.q-field--outlined .q-field__control {{ border-color: transparent !important; }}
.q-field--outlined:hover .q-field__control {{ background: var(--input-bg) !important; }}
.q-item {{ color: var(--text) !important; }}
.q-menu {{ 
    background: rgba(28,28,30,0.85) !important; 
    backdrop-filter: blur(20px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
    border: 1px solid var(--border) !important; 
    border-radius: 12px !important; 
}}
body.light .q-menu {{
    background: rgba(255,255,255,0.85) !important;
}}

/* ── App Header ── */
.app-header {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.app-logo {{
    width: 32px; height: 32px;
    border-radius: 8px; /* HIG squircle-ish */
    background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}}
.app-logo svg {{ width: 16px; height: 16px; stroke: white; }}
.app-title {{
    font-size: 1.05rem; /* HIG Navigation Bar Title Size */
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.015em;
}}

/* ── Live Indicator Badge ── */
.live-indicator {{
    font-family: -apple-system, BlinkMacSystemFont, monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 4px 10px;
    border-radius: 12px;
    border: none;
    color: var(--text-muted);
    background: var(--input-bg);
    transition: all 0.3s ease;
}}
.live-indicator.active {{
    color: var(--accent);
    background: var(--hl-card-bg);
    animation: badge-breathe 2.5s ease-in-out infinite;
}}
body.light .live-indicator.active {{
    color: var(--accent);
    background: var(--hl-card-bg);
}}
@keyframes badge-breathe {{
    0%, 100% {{ opacity: 0.8; }}
    50% {{ opacity: 1; transform: scale(1.02); }}
}}

/* ── Sidebar Section Title ── */
.sidebar-section-title {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 22px 0 8px;
    display: flex;
    align-items: center;
    padding-left: 8px;
}}
.sidebar-label {{ font-size: 0.75rem; font-weight: 500; color: var(--label-text); margin-bottom: 4px; padding-left: 8px; }}
.sidebar-success {{
    color: var(--accent2);
    font-size: 0.75rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 6px;
    padding-left: 8px;
}}
.sidebar-warning {{
    color: #ff9f0a; /* Apple Orange */
    font-size: 0.75rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 6px;
    padding-left: 8px;
}}
.sidebar-error {{
    color: var(--red);
    font-size: 0.75rem;
    font-weight: 500;
    line-height: 1.4;
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(255,69,58,0.1);
    margin: 0 8px;
}}

/* ── Control Buttons ── */
.td-btn {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    border-radius: 12px;
    border: 1px solid var(--btn-border);
    background: var(--btn-bg);
    color: var(--text);
    padding: 10px 18px;
    width: 100%;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    transition: all 0.2s ease;
    white-space: nowrap;
}}
.td-btn:hover:not(:disabled) {{
    background: var(--btn-hover);
    border-color: var(--btn-hover-border);
    transform: scale(0.98);
}}
.td-btn:disabled {{
    opacity: 0.4;
    cursor: not-allowed;
}}
.td-btn.btn-primary {{
    background: var(--btn-primary-bg) !important;
    border-color: var(--btn-primary-border) !important;
    color: var(--btn-primary-text) !important;
}}
.td-btn.btn-primary:hover:not(:disabled) {{
    background: var(--btn-primary-hover) !important;
    transform: scale(0.98);
}}
.td-btn.recording {{
    color: #ffffff !important;
    border-color: transparent !important;
    background: var(--red) !important;
}}
.td-btn.recording:hover {{
    background: #e03225 !important;
}}

/* ── Status Bar ── */
.status-bar {{
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-muted);
    padding: 8px 12px;
    background: var(--status-bg);
    border: none;
    border-radius: 10px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
body.light .status-bar {{
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}}

/* ── Live Box (Glassmorphism look) ── */
.live-box {{
    background: var(--live-bg-1);
    border: 1px solid var(--live-border);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}}
body.dark .live-box {{
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}}
body.light .live-box {{
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}}
.live-title {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.live-dot {{
    width: 8px; height: 8px;
    background: var(--red); /* iOS orange/red active indicator */
    border-radius: 50%;
    flex-shrink: 0;
}}
.live-text {{
    font-size: 1.35rem;
    font-weight: 400;
    color: var(--live-text);
    line-height: 1.5;
    min-height: 2rem;
    letter-spacing: -0.015em;
}}
.ghost-text {{
    opacity: var(--ghost-text);
    color: var(--text-dim);
}}
.translation-text {{
    font-size: 1.1rem;
    font-weight: 400;
    color: var(--text-trans);
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    line-height: 1.5;
}}

/* ── Section Header ── */
.section-subheader {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin-bottom: 12px;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* ── History Cards ── */
.history-card {{
    background: var(--hist-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
    cursor: default;
}}
body.light .history-card {{
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}}
.history-card:hover {{
    background: var(--hist-hover);
}}
.history-card.highlighted {{
    border-color: var(--hl-card-border) !important;
    background: var(--hl-card-bg) !important;
}}
.history-card.highlighted .history-original {{
    color: var(--accent) !important;
}}
.history-original {{
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--hist-orig);
    margin-bottom: 4px;
    line-height: 1.4;
}}
.history-translation {{
    font-size: 0.85rem;
    font-weight: 400;
    color: var(--text-muted);
    line-height: 1.4;
}}

/* ── Waveform ── */
.wv-wrap {{ margin: 16px 0 6px; }}
.wv-lbl {{
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}}
.wv {{
    display: flex;
    align-items: center;
    gap: 3px;
    height: 48px;
    background: transparent;
    padding: 0;
}}
.wb {{
    flex: 1;
    min-width: 3px;
    border-radius: 10px; /* Fully rounded pills for Apple style voice memos */
    background: var(--red);
    opacity: 0.9;
    transition: height 0.1s cubic-bezier(0.23,1,0.32,1);
}}

/* ── Summary & Metrics ── */
.summary-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 32px 36px;
    margin-bottom: 24px;
}}
body.dark .summary-card {{
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}}
body.light .summary-card {{
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}}
.summary-content {{
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text);
}}
.summary-content h2 {{
    font-weight: 700;
    font-size: 1.25rem;
    letter-spacing: -0.015em;
    color: var(--text);
    margin: 24px 0 12px;
}}
.summary-content p {{ margin-bottom: 12px; }}
.summary-content ul {{ padding-left: 20px; }}
.summary-content li {{ margin-bottom: 8px; }}

.metric-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    flex: 1;
}}
body.light .metric-card {{ box-shadow: 0 1px 4px rgba(0,0,0,0.02); }}
.metric-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 6px;
}}
.metric-value {{
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--accent);
    line-height: 1;
}}

/* ── Spinner ── */
.td-spinner {{
    width: 28px; height: 28px;
    border: 2px solid var(--border);
    border-top-color: var(--text-muted);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 20px;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

/* ── Focus ring — keyboard accessibility ── */
*:focus-visible {{
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 6px;
}}

/* ── Quasar native button overrides ── */
.q-btn {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}}
.q-btn--flat {{
    color: var(--accent) !important;
}}
.q-btn--flat:hover {{
    background: var(--input-bg) !important;
}}
.q-btn .q-ripple {{ opacity: 0.04 !important; }}

/* ── Quasar select/input polish ── */
.q-field--outlined .q-field__control {{
    border-radius: 10px !important;
    background: var(--input-bg) !important;
}}
.q-field--outlined .q-field__control::before {{
    border-color: var(--input-border) !important;
}}
.q-field--outlined:hover .q-field__control::before {{
    border-color: var(--input-border) !important;
}}
.q-field--outlined.q-field--focused .q-field__control::before {{
    border-color: var(--accent) !important;
    border-width: 2px !important;
}}
.q-field__native, .q-field__input, .q-field__prefix, .q-field__suffix {{
    color: var(--input-text) !important;
    font-size: 0.85rem !important;
}}
.q-field__label {{
    color: var(--label-text) !important;
    font-size: 0.75rem !important;
    text-transform: none !important;
}}
.q-select__dropdown-icon {{
    color: var(--text-dim) !important;
}}
.q-item {{ font-weight: 500 !important; padding: 10px 16px !important; }}
.q-item:hover {{ background: rgba(0,122,255,0.1) !important; color: var(--accent) !important; }}

/* ── Header menu button ── */
.q-header .q-btn {{
    color: var(--accent) !important;
}}

/* ── Separator ── */
.q-separator {{
    background: var(--border) !important;
    margin: 8px 0 !important;
}}

/* ── QR code image ── */
.q-img {{ border-radius: 12px; overflow: hidden; }}

/* ── Page transition ── */
.q-page-container {{ transition: padding 0.3s ease; }}

/* ── Highlighted card button ── */
.q-btn[aria-label*="◇"],
.q-btn[aria-label*="✦"] {{
    transition: transform 0.15s ease, color 0.15s ease !important;
}}

/* ── Responsive ── */
@media (max-width: 768px) {{
    .live-text {{ font-size: 1.15rem !important; }}
    .translation-text {{ font-size: 1rem !important; }}
    .live-box {{ padding: 20px 22px !important; }}
    .metric-value {{ font-size: 1.8rem !important; }}
    .summary-card {{ padding: 24px !important; }}
}}
'''

def _waveform_html(level: float) -> str:
    boosted = min(1.0, (level * 5.0) ** 0.65)
    num_bars = 28
    max_h = 36; min_h = 4
    bars = []
    for i in range(num_bars):
        center = num_bars / 2
        edge = 1.0 - 0.5 * abs((i - center) / center) ** 1.8
        wave = 0.55 + 0.45 * abs(math.sin(i * 0.7 + boosted * 4.5))
        h = max(min_h, int(min_h + (max_h - min_h) * boosted * edge * wave))
        # Apple Voice Memos styling uses exact rounded bars
        bars.append(f'<span class="wb" style="height:{h}px"></span>')
    return f'<div class="wv-wrap"><div class="wv-lbl">Voice Level</div><div class="wv">{{ "".join(bars) }}</div></div>'
"""

new_code = head + start_marker_str + new_ui + "\n\n" + end_marker_str + tail

with open("Talking_Dude_NiceGUI.py", "w", encoding="utf-8") as f:
    f.write(new_code)
print("UI patched")
