#!/usr/bin/env python3
"""
Crisis Assistant - Main Entry Point
Offline, voice-activated emergency guidance for Raspberry Pi 5

Pipeline: Button → Record → STT → Analyze → Decide → RAG → LLM → TTS → Play
"""

import os
import sys
import time
import signal
import logging
import tempfile
import traceback
from pathlib import Path

from src.utils import load_config, ensure_dir

# ── logging ──────────────────────────────────────────────────────────────────
def _setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    level   = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
    logfile = log_cfg.get("file", "logs/crisis_assistant.log")
    ensure_dir(os.path.dirname(logfile))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )

logger = logging.getLogger("main")


# ── graceful-shutdown sentinel ────────────────────────────────────────────────
_running = True

def _sigterm_handler(sig, frame):
    global _running
    logger.info("Received shutdown signal")
    _running = False

signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT,  _sigterm_handler)


# ── CrisisAssistant ───────────────────────────────────────────────────────────
class CrisisAssistant:
    """
    Top-level application controller.

    Manages hardware lifecycle, routes button events into the AI pipeline,
    and handles error recovery so the device stays responsive even when
    individual components fail.
    """

    def __init__(self, config: dict):
        self.config   = config
        self.lcd      = None
        self.audio    = None
        self.button   = None
        self.stt      = None
        self.tts      = None
        self.orch     = None
        self._tmp_wav = None   # path of last recording

        # Lazy state flags
        self._recording  = False
        self._processing = False

    # ── initialisation ────────────────────────────────────────────────────────

    def init_hardware(self):
        """Boot display, audio, and button – fail gracefully on non-Pi."""
        # LCD
        try:
            from src.hardware.lcd import ConversationLCD
            hw = self.config.get("hardware", {})
            self.lcd = ConversationLCD(
                i2c_address=hw.get("lcd_address", 0x27),
                cols=hw.get("lcd_cols", 16),
                rows=hw.get("lcd_rows", 2),
            )
        except Exception as e:
            logger.warning("LCD unavailable: %s", e)

        # Audio
        try:
            from src.hardware.audio import AudioManager
            self.audio = AudioManager(self.config.get("audio", {}))
        except Exception as e:
            logger.warning("AudioManager unavailable: %s", e)

        # Button (GPIO – only on Pi)
        try:
            from src.hardware.button import SmartButton
            hw = self.config.get("hardware", {})
            self.button = SmartButton(pin=hw.get("button_pin", 17))
            self.button.on_press_start  = self._on_press_start
            self.button.on_press_end    = self._on_press_end
            self.button.on_triple_press = self._on_triple_press
            self.button.on_long_hold    = self._on_long_hold
        except Exception as e:
            logger.warning("GPIO button unavailable: %s – running in demo mode", e)

    def init_ai(self):
        """Load STT, TTS, and the full AI orchestrator."""
        # STT
        try:
            from src.audio.stt import WhisperSTT
            self.stt = WhisperSTT(self.config.get("whisper", {}))
        except Exception as e:
            logger.warning("Whisper STT unavailable: %s", e)

        # TTS
        try:
            from src.audio.tts import PiperTTS
            self.tts = PiperTTS(self.config.get("piper", {}))
        except Exception as e:
            logger.warning("Piper TTS unavailable: %s", e)

        # Orchestrator (loads LLM, RAG, decision engine)
        from src.orchestrator import IntelligentOrchestrator
        self.orch = IntelligentOrchestrator(self.config)

    def start(self):
        """Start a fresh session and greet the user."""
        greeting = self.orch.start_session()
        self._speak(greeting)
        self._display("idle")
        logger.info("Session started")

    # ── button callbacks ──────────────────────────────────────────────────────

    def _on_press_start(self):
        """Button held down – begin recording."""
        if self._processing:
            logger.debug("Still processing previous request, ignoring press")
            return

        self._recording = True
        self._display("listening")
        self._stop_playback()

        tmp = tempfile.mktemp(suffix=".wav", prefix="rec_")
        self._tmp_wav = tmp
        if self.audio:
            self.audio.start_recording(tmp)
        logger.info("Recording started → %s", tmp)

    def _on_press_end(self, hold_duration: float):
        """Button released – stop recording and run the pipeline."""
        if not self._recording:
            return

        self._recording = False
        if self.audio:
            self.audio.stop_recording()

        logger.info("Recording stopped (%.2fs)", hold_duration)

        if hold_duration < 0.4:
            logger.info("Press too short, ignoring")
            self._display("idle")
            return

        self._run_pipeline(self._tmp_wav)

    def _on_triple_press(self):
        """Triple press – reset conversation."""
        logger.info("Triple press: resetting conversation")
        if self._recording and self.audio:
            self.audio.stop_recording()
            self._recording = False

        self.orch.reset_session()
        greeting = self.orch.start_session()
        self._speak(greeting)
        self._display("idle")

    def _on_long_hold(self):
        """5-second long hold – safe shutdown."""
        logger.info("Long hold: initiating shutdown")
        if self.lcd:
            self.lcd.show("Shutting down", "Goodbye...")
        time.sleep(1)
        self.shutdown()
        os.system("sudo shutdown -h now")

    # ── core pipeline ─────────────────────────────────────────────────────────

    def _run_pipeline(self, wav_file: str):
        """Full voice-to-voice pipeline (blocking, runs in caller's thread)."""
        self._processing = True
        try:
            # 1. STT ──────────────────────────────────────────────────────────
            self._display("processing")
            user_text = ""
            if self.stt and wav_file and os.path.exists(wav_file):
                user_text = self.stt.transcribe(wav_file)
            else:
                logger.warning("STT skipped – no audio file or STT unavailable")

            if not user_text.strip():
                self._speak("I didn't catch that. Please try again.")
                self._display("idle")
                return

            logger.info("STT result: '%s'", user_text)

            # 2. Orchestrate ──────────────────────────────────────────────────
            result = self.orch.process_message(user_text)
            response_text = result.get("response", "")
            lcd_text      = result.get("lcd_display", "")
            state         = result.get("state", "responding")
            analysis      = result.get("analysis", {})

            logger.info("Response (%s): '%s'", state, response_text[:80])

            # 3. Update display ───────────────────────────────────────────────
            if state == "critical":
                self._display("critical")
            elif lcd_text and self.lcd:
                self.lcd.show("Crisis Assistant", lcd_text[:16])
            else:
                self._display("responding")

            # 4. TTS + playback ───────────────────────────────────────────────
            self._speak(response_text)

            # 5. Post-response idle state ─────────────────────────────────────
            if state == "critical" and self.lcd:
                self.lcd.show_state("critical")
            else:
                self._display("idle")

        except Exception as e:
            logger.error("Pipeline error: %s\n%s", e, traceback.format_exc())
            self._speak("Something went wrong. Please try again.")
            self._display("error")
        finally:
            self._processing = False
            # Clean up temp wav
            if wav_file and os.path.exists(wav_file):
                try:
                    os.remove(wav_file)
                except OSError:
                    pass

    # ── helpers ───────────────────────────────────────────────────────────────

    def _speak(self, text: str):
        """Synthesize and play text, with console fallback."""
        if not text:
            return
        print(f"\n🤖 ASSISTANT: {text}\n")
        if self.tts:
            wav = self.tts.synthesize(text)
            if wav and self.audio:
                self.audio.play(wav)
                try:
                    os.remove(wav)
                except OSError:
                    pass

    def _display(self, state: str):
        """Send predefined state to LCD (no-op if LCD unavailable)."""
        if self.lcd:
            self.lcd.show_state(state)

    def _stop_playback(self):
        """Interrupt any ongoing TTS playback."""
        if self.audio and hasattr(self.audio, "stop_playback"):
            self.audio.stop_playback()

    # ── demo mode (no GPIO) ───────────────────────────────────────────────────

    def run_demo_loop(self):
        """Interactive text loop for development / testing without hardware."""
        print("\n" + "="*60)
        print("  CRISIS ASSISTANT – DEMO MODE (type to interact)")
        print("  Commands: 'reset' | 'quit' | any message")
        print("="*60 + "\n")

        while _running:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            if user_input.lower() == "reset":
                self.orch.reset_session()
                greeting = self.orch.start_session()
                print(f"🔄 Session reset\n🤖 ASSISTANT: {greeting}\n")
                continue

            result = self.orch.process_message(user_input)
            response  = result.get("response", "")
            state     = result.get("state", "")
            analysis  = result.get("analysis", {})

            print(f"🤖 ASSISTANT: {response}")
            if analysis.get("conditions"):
                conds = [f"{c['type']}({c['severity']})" for c in analysis["conditions"]]
                print(f"   [Analysis] phase={analysis['phase']} | conditions={conds}")
            print()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def run(self):
        """
        Main run loop.

        - If GPIO button is available: wait for button events.
        - Otherwise: fall back to interactive text demo.
        """
        if self.button:
            logger.info("Hardware mode: waiting for button events")
            self._display("idle")
            try:
                while _running:
                    time.sleep(0.1)
            finally:
                self.shutdown()
        else:
            self.run_demo_loop()
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down Crisis Assistant")
        if self.button:
            try:
                self.button.cleanup()
            except Exception:
                pass
        if self.lcd:
            try:
                self.lcd.clear()
            except Exception:
                pass
        if self.audio:
            try:
                if hasattr(self.audio, "cleanup"):
                    self.audio.cleanup()
            except Exception:
                pass
        logger.info("Shutdown complete")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    config_path = os.environ.get("CRISIS_CONFIG", "configs/pi_production.yaml")
    config = load_config(config_path)
    if not config:
        print("❌ Failed to load configuration. Exiting.")
        sys.exit(1)

    _setup_logging(config)
    logger.info("━━━ Crisis Assistant v%s starting ━━━", config.get("app", {}).get("version", "?"))

    # Ensure required directories exist
    ensure_dir("logs")
    ensure_dir("data/index")

    app = CrisisAssistant(config)

    print("\n🏥 Initializing hardware...")
    app.init_hardware()

    print("🧠 Loading AI components (this may take 1-2 minutes)...")
    app.init_ai()

    print("✅ All systems ready\n")
    app.start()
    app.run()


if __name__ == "__main__":
    main()
