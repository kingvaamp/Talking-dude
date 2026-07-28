import re

with open("Talking_Dude_NiceGUI.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace any q-mb-sm or w-full on selects to explicit margins
replaces = [
    (".classes('w-full q-mb-sm')", ".classes('w-full').style('margin-bottom: 16px;')"),
    (".classes('w-full')", ".classes('w-full')") # fallback
]

for src, tgt in replaces:
    code = code.replace(src, tgt)

# Increase bottom margin on section titles
code = code.replace("margin: 22px 0 8px;", "margin: 24px 0 12px;")

with open("Talking_Dude_NiceGUI.py", "w", encoding="utf-8") as f:
    f.write(code)

print("done")
