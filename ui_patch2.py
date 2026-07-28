import re

with open("Talking_Dude_NiceGUI.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Strip out old specific font references in inline styles
code = code.replace("font-family:Syne,sans-serif", "font-family:-apple-system,BlinkMacSystemFont,sans-serif")
code = code.replace("font-family:DM Mono,monospace", "font-family:-apple-system,BlinkMacSystemFont,monospace")

# 2. Add Apple-style frosted glass to the sidebar drawer
code = code.replace(
    ".q-drawer {\n    background: var(--bg) !important;\n    border-right: 1px solid var(--border) !important;\n}",
    ".q-drawer {\n    background: var(--header-bg) !important;\n    backdrop-filter: blur(40px) saturate(200%) !important;\n    -webkit-backdrop-filter: blur(40px) saturate(200%) !important;\n    border-right: 1px solid var(--border) !important;\n}"
)

# 3. Modify button classes to have the pill shape and no ripple
code = code.replace(
    ".q-btn {\n    font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;\n    font-weight: 500 !important;\n    letter-spacing: 0 !important;\n    border-radius: 10px !important;\n    transition: all 0.2s ease !important;\n}",
    ".q-btn {\n    font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;\n    font-weight: 600 !important;\n    letter-spacing: -0.01em !important;\n    border-radius: 24px !important;\n    transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1) !important;\n    box-shadow: none !important;\n    text-transform: none !important;\n}"
)
code = code.replace(".q-btn .q-ripple { opacity: 0.04 !important; }", ".q-btn .q-ripple { display: none !important; }")

# Also modify the custom `.td-btn` to use pill shape
code = code.replace(
    "border-radius: 12px;\n    border: 1px solid var(--btn-border);\n    background: var(--btn-bg);",
    "border-radius: 24px;\n    border: 1px solid var(--btn-border);\n    background: var(--btn-bg);"
)

with open("Talking_Dude_NiceGUI.py", "w", encoding="utf-8") as f:
    f.write(code)

print("sidebar and buttons styled to HIG.")
