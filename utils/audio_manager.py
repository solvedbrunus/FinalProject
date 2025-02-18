import pygame
import io
import threading
import time

class AudioManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            pygame.mixer.init()
            self._initialized = True
            self.current_audio = None
            self._lock = threading.Lock()
    
    def play_audio(self, audio_bytes):
        if not audio_bytes:
            return
            
        with self._lock:
            try:
                # Stop any currently playing audio
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                
                # Load and play new audio
                audio_file = io.BytesIO(audio_bytes)
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # Wait for audio to finish in a non-blocking way
                def wait_for_audio():
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                
                threading.Thread(target=wait_for_audio, daemon=True).start()
                
            except Exception as e:
                print(f"Error playing audio: {str(e)}")
