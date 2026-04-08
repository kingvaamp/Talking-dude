import pyaudio
import array
try:
    import audioop
    _HAS_AUDIOOP = True
except:
    _HAS_AUDIOOP = False

def _to_mono(raw):
    if _HAS_AUDIOOP: return audioop.tomono(raw, 2, 0.5, 0.5)
    return raw

def _ratecv(raw, in_rate, out_rate, state):
    if _HAS_AUDIOOP: return audioop.ratecv(raw, 2, 1, in_rate, out_rate, state)
    return raw, None

def test():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=2, rate=48000, 
                    input=True, input_device_index=3, frames_per_buffer=2400)
    ratecv_state = None
    import time
    for _ in range(15):
        raw = stream.read(2400, exception_on_overflow=False)
        orig_peak = max(abs(s) for s in array.array('h', raw)) / 32768.0 if raw else 0
        mono = _to_mono(raw)
        res, ratecv_state = _ratecv(mono, 48000, 16000, ratecv_state)
        final_peak = max(abs(s) for s in array.array('h', res)) / 32768.0 if res else 0
        print(f"Orig: {orig_peak:.6f}, Final: {final_peak:.6f}")
        time.sleep(0.1)
    p.terminate()

test()
