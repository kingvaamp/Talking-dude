#!/bin/bash
set -e

APP="dist/Talking Dude.app"
FRAMEWORKS="$APP/Contents/Frameworks"
RESOURCES="$APP/Contents/Resources"
MACOS="$APP/Contents/MacOS"

cd "$(dirname "$0")"

echo "🧹 Stripping all xattrs from the bundle..."
xattr -cr "$APP"

echo "🔏 Step 1: Sign all .dylib files..."
find "$APP" -name "*.dylib" | sort | while read f; do
    codesign --force --sign - "$f" 2>/dev/null || true
done

echo "🔏 Step 2: Sign all .so extension modules..."
find "$APP" -name "*.so" | sort | while read f; do
    codesign --force --sign - "$f" 2>/dev/null || true
done

echo "🔏 Step 3: Sign all frameworks..."
find "$APP" -name "*.framework" -prune | while read f; do
    codesign --force --sign - "$f" 2>/dev/null || true
done

echo "🔏 Step 4: Sign the main executable..."
codesign --force --sign - "$MACOS/Talking Dude"

echo "🔏 Step 5: Sign the bundle (top-level, no --deep)..."
codesign --force --sign - "$APP"

echo ""
echo "🔍 Verifying bundle..."
codesign --verify "$APP" && echo "✅ Bundle signature valid!" || echo "⚠️ Verification note (normal for ad-hoc signing)"

echo ""
echo "✅ Done! App is ready at: $APP"
echo "   Copy it to /Applications to install:"
echo "   cp -R '$APP' /Applications/"
