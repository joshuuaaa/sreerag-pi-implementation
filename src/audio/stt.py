# src/audio/stt.py
"""
Whisper.cpp Speech-to-Text wrapper
"""

import subprocess
import os
from typing import Optional

class WhisperSTT:
    def __init__(self, config: dict):
        """
        Initialize Whisper STT
        
        Args:
            config: Whisper configuration dict
        """
        self.model_path = os.path.expanduser(config.get("model_path", ""))
        self.whisper_bin = os.path.expanduser(config.get("binary_path", ""))
        self.language = config.get("language", "en")
        
        # Validate paths
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Whisper model not found: {self.model_path}")
        if not os.path.exists(self.whisper_bin):
            raise FileNotFoundError(f"Whisper binary not found: {self.whisper_bin}")
            
        print(f"✅ Whisper STT initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Binary: {self.whisper_bin}")
        
    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_file: Path to WAV file
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return ""
            
        print(f"📝 Transcribing: {audio_file}")
        
        try:
            # Run whisper.cpp
            result = subprocess.run(
                [
                    self.whisper_bin,
                    "-m", self.model_path,
                    "-f", audio_file,
                    "-l", self.language,
                    "-nt",  # No timestamps
                    "-np"   # No print special tokens
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse output
            text = result.stdout.strip()
            
            # Clean up output
            text = text.replace("[BLANK_AUDIO]", "")
            text = text.replace("  ", " ").strip()
            
            # Remove common whisper.cpp artifacts
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # Take last non-empty line (usually the transcription)
            if lines:
                text = lines[-1]
            
            print(f"✅ Transcribed: '{text}'")
            return text
            
        except subprocess.TimeoutExpired:
            print("❌ Whisper timeout")
            return ""
        except Exception as e:
            print(f"❌ Whisper error: {e}")
            return ""
