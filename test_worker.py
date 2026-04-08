import pyaudio
import time
import queue
import array

try:
    import audioop
    _HAS_AUDIOOP = True
except:
    _HAS_AUDIOOP = False

def _to_mono(raw: bytes) -> bytes:
    if _HAS_AUDIOOP:
        return audioop.tomono(raw, 2, 0.5, 0.5)
    return raw

def _ratecv(raw: bytes, in_rate: int, out_rate: int, state):
    if _HAS_AUDIOOP:
        return audioop.ratecv(raw, 2, 1, in_rate, out_rate, state)
    return raw, None

def _get_peak(raw_bytes):
    if not raw_bytes: 
        return 0
    try:
        samps = array.array('h', raw_bytes)
        if not samps: return 0
        return max(abs(s) for s in samps) / 32768.0
    except Exception:
        return 0

def run_test():
    p = pyaudio.PyAudio()
    bh_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if 'blackhole' in info['name'].lower() and info['maxInputChannels'] > 0:
            bh_index = i
            break
            
    stream = p.open(
        format=pyaudio.paInt16, channels=2, rate=48000,
        input=True, input_device_index=bh_index,
        frames_per_buffer=2400
    )
    
    ratecv_state = None
    TARGET_RATE = 16000
    RATE = 48000
    CHANNELS = 2
    CHUNK = 2400
    
    print("Testing Stream...")
    for i in range(20):
        raw = stream.read(CHUNK, exception_on_overflow=False)
        orig_peak = _get_peak(raw)
        
        if CHANNELS == 2:
            raw = _to_mono(raw)
        if RATE != TARGET_RATE:
            raw, ratecv_state = _ratecv(raw, RATE, TARGET_RATE, ratecv_state)

        peak = _get_peak(raw)
        
        print(f"Orig peak: {orig_peak:.5f} | Final peak: {peak:.5f}")
        time.sleep(0.05)
        
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    run_test()
