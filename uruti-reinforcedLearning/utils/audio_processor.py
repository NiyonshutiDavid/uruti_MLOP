import pyaudio
import wave
import threading
from datetime import datetime

class AudioRecorder:
    def __init__(self, rate=44100, channels=1, chunk=1024):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.audio_format = pyaudio.paInt16
        self.frames = []
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording_thread = None
    
    def start_recording(self):
        """Start audio recording in a separate thread"""
        self.frames = []
        self.recording = True
        
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        print("Audio recording started...")
    
    def _record(self):
        """Internal recording method"""
        while self.recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
    
    def stop_recording(self):
        """Stop recording and save to file"""
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Save to file
        filename = f"pitch_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        print(f"Audio saved as: {filename}")
        return filename
    
    def __del__(self):
        if self.audio:
            self.audio.terminate()