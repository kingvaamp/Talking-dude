import os
import json
import asyncio
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

# Load key from settings.json
SETTINGS_FILE = "settings.json"

def load_key():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f).get("dg_key", "")
    return ""

async def test_deepgram():
    api_key = load_key()
    if not api_key:
        print("❌ No API key found in settings.json")
        return

    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        client = DeepgramClient(api_key, config)
        
        # Test both patterns
        print("Testing websocket.v('1')...")
        try:
            dg_conn = client.listen.websocket.v("1")
            print("✅ websocket.v('1') initialized")
        except Exception as e:
            print(f"❌ websocket.v('1') failed: {e}")

        print("Testing live.v('1')...")
        try:
            dg_conn = client.listen.live.v("1")
            print("✅ live.v('1') initialized")
        except Exception as e:
            print(f"❌ live.v('1') failed: {e}")

    except Exception as e:
        print(f"❌ Generic failure: {e}")

if __name__ == "__main__":
    asyncio.run(test_deepgram())
