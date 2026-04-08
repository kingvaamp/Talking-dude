import pyaudio
import threading
import array
import time

def worker(bh_idx):
    print("Worker starting on device:", bh_idx)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=2, rate=48000, 
                    input=True, input_device_index=bh_idx, frames_per_buffer=2400)
    for _ in range(10):
        raw = stream.read(2400, exception_on_overflow=False)
        samps = array.array('h', raw)
        peak = max(abs(s) for s in samps) / 32768.0 if samps else 0
        print(f"[Thread] Peak: {peak:.6f}")
        time.sleep(0.1)
    stream.stop_stream()
    stream.close()
    p.terminate()

def run_test():
    p = pyaudio.PyAudio()
    bh_idx = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if 'blackhole' in info['name'].lower() and info['maxInputChannels'] > 0:
            bh_idx = i
            break
    p.terminate()

    if bh_idx is None:
        print("No BlackHole")
        return

    print("Found device:", bh_idx)
    
    # 1. Main thread read
    print("--- MAIN THREAD TEST ---")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=2, rate=48000, 
                    input=True, input_device_index=bh_idx, frames_per_buffer=2400)
    raw = stream.read(2400, exception_on_overflow=False)
    samps = array.array('h', raw)
    peak = max(abs(s) for s in samps) / 32768.0 if samps else 0
    print(f"[Main] Peak: {peak:.6f}")
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("--- WORKER THREAD TEST ---")
    t = threading.Thread(target=worker, args=(bh_idx,))
    t.start()
    t.join()

if __name__ == '__main__':
    run_test()
