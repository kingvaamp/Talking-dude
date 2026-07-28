import sys

with open("Talking_Dude_NiceGUI.py", "r", encoding="utf-8") as f:
    code = f.read()

start_marker = "# ══════════════════════════════════════════════════════════════════════════════\n# CSS THEME\n# ══════════════════════════════════════════════════════════════════════════════"
end_marker = "# ══════════════════════════════════════════════════════════════════════════════\n# UI — NiceGUI\n# ══════════════════════════════════════════════════════════════════════════════"

if start_marker not in code:
    print("Start marker not found!")
    sys.exit(1)
if end_marker not in code:
    print("End marker not found!")
    sys.exit(1)

head = code.split(start_marker)[0]
tail = code.split(end_marker)[1]

from ui_patch import new_ui

new_code = head + start_marker + "\n" + new_ui + "\n" + end_marker + tail

with open("Talking_Dude_NiceGUI.py", "w", encoding="utf-8") as f:
    f.write(new_code)
print("UI successfully patched.")
