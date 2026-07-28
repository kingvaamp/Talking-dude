with open("Talking_Dude_NiceGUI.py", "r", encoding="utf-8") as f:
    code = f.read()

replaces = [
    ('label="Langue source (audio)"\n        ).classes(\'w-full\')', 'label="Langue source (audio)"\n        ).classes(\'w-full q-mb-sm\')'),
    ('label="Langue cible (traduction)"\n        ).classes(\'w-full\')', 'label="Langue cible (traduction)"\n        ).classes(\'w-full q-mb-sm\')'),
    ('label="Entrée audio (BlackHole)"\n            ).classes(\'w-full\')', 'label="Entrée audio (BlackHole)"\n            ).classes(\'w-full q-mb-sm\')'),
    ('value=S._groq_key, placeholder="gsk_..."\n        ).classes(\'w-full\')', 'value=S._groq_key, placeholder="gsk_..."\n        ).classes(\'w-full q-mb-sm\')'),
    ('label="Langue du résumé"\n        ).classes(\'w-full\')', 'label="Langue du résumé"\n        ).classes(\'w-full q-mb-sm\')'),
    ('value=S._dg_key, placeholder="Paste your key here..."\n        ).classes(\'w-full\')', 'value=S._dg_key, placeholder="Paste your key here..."\n        ).classes(\'w-full q-mb-sm\')'),
    ('label="Modèle"\n        ).classes(\'w-full\')', 'label="Modèle"\n        ).classes(\'w-full q-mb-sm\')'),
]

for src, tgt in replaces:
    code = code.replace(src, tgt)

with open("Talking_Dude_NiceGUI.py", "w", encoding="utf-8") as f:
    f.write(code)

print("done")
