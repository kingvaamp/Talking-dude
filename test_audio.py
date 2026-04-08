import pyaudio
import sys

def test_mic():
    p = pyaudio.PyAudio()
    bh_idx = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if 'blackhole' in info['name'].lower():
            bh_idx = i
            break
    
    if bh_idx is None:
        print("No BlackHole found.")
        sys.exit(1)
        
    print(f"Using device {bh_idx}...")
    try:
        stream = p.open(format=pyaudio.paInt16, channels=2, rate=48000, 
                        input=True, input_device_index=bh_idx, frames_per_buffer=1024)
        for _ in range(20):
            data = stream.read(1024, exception_on_overflow=False)
            peak = max(abs(s) for s in __import__('array').array('h', data))
            print(f"Peak: {peak}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()

test_mic()
