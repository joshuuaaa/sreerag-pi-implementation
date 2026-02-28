# src/audio/tts.py
"""
Piper Text-to-Speech wrapper
"""

import subprocess
import os
import logging
import tempfile
from typing import Optional

logger = logging.getLogger("audio.tts")

class PiperTTS:
    def __init__(self, config: dict):
        """
        Initialize Piper TTS
        
        Args:
            config: Piper configuration dict
        """
        self.model_path = os.path.expanduser(config.get("model_path", ""))
        self.piper_bin = os.path.expanduser(config.get("binary_path", ""))
        
        # Validate paths
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Piper model not found: {self.model_path}")
        if not os.path.exists(self.piper_bin):
            raise FileNotFoundError(f"Piper binary not found: {self.piper_bin}")
            
        logger.info("Piper TTS initialized")
        logger.debug("Piper model: %s", self.model_path)
        logger.debug("Piper binary: %s", self.piper_bin)
        
    def synthesize(self, text: str) -> Optional[str]:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to generated WAV file, or None on error
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text, skipping TTS")
            return None
            
        # Create temp file
        output_file = tempfile.mktemp(suffix=".wav", prefix="tts_")
        
        logger.info("Synthesizing speech")
        
        try:
            # Run Piper
            process = subprocess.Popen(
                [
                    self.piper_bin,
                    "--model", self.model_path,
                    "--output_file", output_file
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Send text and wait
            stdout, stderr = process.communicate(input=text.encode('utf-8'), timeout=30)
            
            # Check if file was created
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info("TTS generated audio: %s", output_file)
                return output_file
            else:
                logger.error("TTS failed to generate audio")
                if stderr:
                    logger.error("Piper stderr: %s", stderr.decode(errors="ignore"))
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Piper timeout")
            process.kill()
            return None
        except Exception as e:
            logger.error("Piper error: %s", e)
            return None
