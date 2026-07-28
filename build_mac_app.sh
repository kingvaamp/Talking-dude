#!/bin/bash

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Clean up previous builds
rm -rf build dist

# The icon path
ICON_PATH="/Users/admin/Downloads/Image 3427.icns"

# Build the PyInstaller command
PYINSTALLER_CMD=(
    /Library/Frameworks/Python.framework/Versions/3.11/bin/pyinstaller
    --noconfirm
    --windowed
    --name "Talking Dude"
    --icon "$ICON_PATH"
    --add-data "Talking_Dude.py:."
    --add-data "settings.json:."
    --collect-all streamlit
    --collect-all pywebview
    --collect-all deepgram
    --collect-all deep_translator
    --collect-all openai
    --collect-all qrcode
    --hidden-import="streamlit.web.cli"
    --hidden-import="streamlit"
    --hidden-import="webview"
    --hidden-import="pyaudio"
    --hidden-import="deepgram"
    --hidden-import="deep_translator"
    --hidden-import="openai"
    --hidden-import="qrcode"
    "launch.py"
)

echo "🚀 Building Talking Dude Mac App..."
"${PYINSTALLER_CMD[@]}"

echo "✅ Build Complete! You can find the app in the 'dist' folder."
