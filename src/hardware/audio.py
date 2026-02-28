# src/hardware/audio.py
"""
Audio manager for microphone recording and speaker playback
"""

import pyaudio
import wave
import subprocess
import threading
import os
import logging
from typing import Optional

logger = logging.getLogger("hardware.audio")

class AudioManager:
    def __init__(self, config: dict = None):
        """
        Initialize audio manager
        
        Args:
            config: Audio configuration dict
        """
        self.config = config or {}
        
        # Audio settings
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.channels = self.config.get("channels", 1)
        self.chunk_size = self.config.get("chunk_size", 1024)
        self.format = pyaudio.paInt16
        
        # PyAudio instance
        self.audio: Optional[pyaudio.PyAudio] = None
        try:
            self.audio = pyaudio.PyAudio()
            logger.info("Audio manager initialized")
        except Exception as e:
            logger.error("PyAudio initialization failed: %s", e)
            
        # Recording state
        self.is_recording = False
        self.recording_stream: Optional[pyaudio.Stream] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.recording_frames = []
        self.output_file = ""
        
        # Playback
        self.playback_process: Optional[subprocess.Popen] = None
        
    def start_recording(self, output_file: str):
        """
        Start recording audio (non-blocking)
        
        Args:
            output_file: Path to save recording
        """
        if not self.audio or self.is_recording:
            logger.warning("Already recording or audio not available")
            return
            
        logger.info("Recording to %s", output_file)
        
        self.recording_frames = []
        self.output_file = output_file
        self.is_recording = True
        
        try:
            self.recording_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=None
            )
            
            # Start recording in background thread
            self.recording_thread = threading.Thread(
                target=self._record_loop,
                daemon=True
            )
            self.recording_thread.start()
            
        except Exception as e:
            logger.error("Recording start error: %s", e)
            self.is_recording = False
            
    def _record_loop(self):
        """Background recording loop"""
        try:
            while self.is_recording and self.recording_stream:
                try:
                    data = self.recording_stream.read(
                        self.chunk_size,
                        exception_on_overflow=False
                    )
                    self.recording_frames.append(data)
                except Exception as e:
                    logger.error("Recording read error: %s", e)
                    break
        except Exception as e:
            logger.error("Recording loop error: %s", e)
            
    def stop_recording(self):
        """Stop recording and save to file"""
        if not self.is_recording:
            return
            
        logger.info("Stopping recording")
        self.is_recording = False
        
        # Wait for recording thread
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
            
        # Stop and close stream
        if self.recording_stream:
            try:
                self.recording_stream.stop_stream()
                self.recording_stream.close()
            except:
                pass
                
        # Save to WAV file
        if self.recording_frames and self.audio:
            try:
                with wave.open(self.output_file, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.format))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(self.recording_frames))
                logger.info("Recording saved to %s", self.output_file)
            except Exception as e:
                logger.error("Error saving recording: %s", e)
        else:
            logger.warning("No audio data to save")
            
    def play(self, audio_file: str):
        """
        Play audio file (blocking)
        
        Args:
            audio_file: Path to audio file
        """
        if not os.path.exists(audio_file):
            logger.error("Audio file not found: %s", audio_file)
            return
            
        logger.info("Playing %s", audio_file)
        
        try:
            # Use aplay (ALSA) for reliable playback on Pi
            result = subprocess.run(
                ["aplay", "-q", audio_file],
                check=True,
                timeout=30
            )
            logger.info("Playback finished")
        except subprocess.TimeoutExpired:
            logger.warning("Playback timeout")
        except subprocess.CalledProcessError as e:
            logger.error("Playback error: %s", e)
        except FileNotFoundError:
            logger.error("aplay not found - install alsa-utils")
            
    def stop_playback(self):
        """Stop any ongoing playback"""
        try:
            subprocess.run(
                ["killall", "aplay"],
                stderr=subprocess.DEVNULL,
                timeout=1
            )
            logger.info("Playback stopped")
        except:
            pass
            
    def cleanup(self):
        """Cleanup audio resources"""
        if self.is_recording:
            self.stop_recording()
        self.stop_playback()
        if self.audio:
            self.audio.terminate()
        logger.info("Audio manager cleaned up")
