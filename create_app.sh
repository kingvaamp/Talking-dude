#!/bin/bash

# Configuration
APP_NAME="Talking Dude"
ICON_PATH="/Users/admin/Downloads/Image 3427.icns"
PROJECT_DIR="/Users/admin/Desktop/New shit/TALKING DUDE BETA  001"
LAUNCH_SCRIPT="launch.py"
OUTPUT_PATH="/Applications/${APP_NAME}.app"

# 1. Create a temporary AppleScript to run the command
TEMP_APPLESCRIPT="/tmp/talking_dude_launcher.applescript"
echo "do shell script \"cd '$PROJECT_DIR' && /usr/local/bin/python3 '$PROJECT_DIR/$LAUNCH_SCRIPT' > /dev/null 2>&1 &\"" > "$TEMP_APPLESCRIPT"

# 2. Compile to .app bundle
echo "🚀 Creating App bundle in /Applications..."
osacompile -o "$OUTPUT_PATH" "$TEMP_APPLESCRIPT"

# 3. Apply the icon
if [ -f "$ICON_PATH" ]; then
    echo "🎨 Applying icon..."
    cp "$ICON_PATH" "$OUTPUT_PATH/Contents/Resources/applet.icns"
    # Update the timestamp of the bundle to force macOS to refresh the icon
    touch "$OUTPUT_PATH"
else
    echo "⚠️ Icon not found at $ICON_PATH. Skipping icon application."
fi

# Clean up
rm "$TEMP_APPLESCRIPT"

echo "✅ Done! You can now find '$APP_NAME' in your Applications folder and Launchpad."
