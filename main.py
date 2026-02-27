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
import threading
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


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y"}:
            return True
        if normalized in {"0", "false", "no", "off", "n"}:
            return False
    return default


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
        env_text_only = os.environ.get("CRISIS_TEXT_ONLY")
        cfg_text_only = config.get("app", {}).get("text_only", False)
        self.text_only = _as_bool(env_text_only, _as_bool(cfg_text_only, False))
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
        self._record_start_time = 0.0
        self._record_timeout_timer = None

        app_cfg = self.config.get("app", {})
        self._min_press_seconds = float(app_cfg.get("min_press_seconds", 0.2))
        self._max_record_seconds = float(app_cfg.get("max_record_seconds", 20.0))

    # ── initialisation ────────────────────────────────────────────────────────

    def init_hardware(self):
        """Boot display, audio, and button – fail gracefully on non-Pi."""
        if self.text_only:
            logger.info("Text-only mode enabled: skipping hardware init")
            return

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
            self.button.on_long_hold    = None
        except Exception as e:
            logger.warning("GPIO button unavailable: %s – running in demo mode", e)

    def init_ai(self):
        """Load STT, TTS, and the full AI orchestrator."""
        if self.text_only:
            logger.info("Text-only mode enabled: skipping STT/TTS init")
        else:
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

        if self._recording:
            logger.debug("Already recording, ignoring duplicate press")
            return

        self._recording = True
        self._record_start_time = time.time()
        self._display("listening")
        self._stop_playback()

        tmp = tempfile.mktemp(suffix=".wav", prefix="rec_")
        self._tmp_wav = tmp
        if self.audio:
            self.audio.start_recording(tmp)
        logger.info("Recording started → %s", tmp)
        self._schedule_record_timeout()

    def _on_press_end(self, hold_duration: float):
        """Button released – stop recording and run the pipeline."""
        if not self._recording:
            return

        self._finalize_recording(hold_duration=hold_duration, source="button_release")

    def _schedule_record_timeout(self):
        self._cancel_record_timeout()
        if self._max_record_seconds <= 0:
            return

        self._record_timeout_timer = threading.Timer(
            self._max_record_seconds, self._on_record_timeout
        )
        self._record_timeout_timer.daemon = True
        self._record_timeout_timer.start()

    def _cancel_record_timeout(self):
        if self._record_timeout_timer and self._record_timeout_timer.is_alive():
            self._record_timeout_timer.cancel()
        self._record_timeout_timer = None

    def _on_record_timeout(self):
        if not self._recording or self._processing:
            return

        elapsed = time.time() - self._record_start_time if self._record_start_time else self._max_record_seconds
        logger.warning("Recording timed out after %.2fs; auto-submitting", elapsed)
        self._finalize_recording(hold_duration=elapsed, source="timeout")

    def _finalize_recording(self, hold_duration: float, source: str):
        self._cancel_record_timeout()

        if not self._recording:
            return

        self._recording = False
        if self.audio:
            self.audio.stop_recording()

        elapsed = time.time() - self._record_start_time if self._record_start_time else hold_duration
        measured_hold = max(hold_duration, elapsed)
        self._record_start_time = 0.0

        logger.info("Recording stopped (%.2fs, source=%s)", measured_hold, source)

        if measured_hold < self._min_press_seconds:
            logger.info(
                "Press too short (%.2fs < %.2fs), ignoring",
                measured_hold,
                self._min_press_seconds,
            )
            self._display("idle")
            return

        self._run_pipeline(self._tmp_wav)

    def _on_triple_press(self):
        """Triple press – reset conversation."""
        logger.info("Triple press: resetting conversation")
        self._cancel_record_timeout()
        if self._recording and self.audio:
            self.audio.stop_recording()
            self._recording = False
            self._record_start_time = 0.0

        self.orch.reset_session()
        greeting = self.orch.start_session()
        self._speak(greeting)
        self._display("idle")

    def _on_long_hold(self):
        """Long hold handler (disabled)."""
        logger.info("Long hold detected (shutdown disabled)")
        return False

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

    # ── hold-to-talk text mode (button present, no mic) ─────────────────────

    def run_button_text_loop(self):
        """
        True hold-to-talk text loop.

        - HOLD button  → input prompt appears, characters are accepted.
        - RELEASE      → input stops immediately, message auto-submits.
        - Nothing typed on release → discarded silently.
        - Triple press → reset session.
        - Long hold → no power action.
        """
        import threading
        import termios
        import tty
        import select

        _pressed = threading.Event()
        _released = threading.Event()

        def _on_press():
            if not self._processing:
                self._recording = True
                _pressed.set()
                _released.clear()

        def _on_release(hold_duration: float):
            self._recording = False
            _released.set()

        def _on_triple():
            self.orch.reset_session()
            greeting = self.orch.start_session()
            print(f"\n🔄 Session reset\n🤖 ASSISTANT: {greeting}\n")

        self.button.on_press_start  = _on_press
        self.button.on_press_end    = _on_release
        self.button.on_triple_press = _on_triple

        print("\n" + "="*60)
        print("  CRISIS ASSISTANT – HOLD-TO-TALK TEXT MODE")
        print("  Hold button + type → release to send")
        print("  Triple press = reset  |  Long hold = no action  |  Ctrl-C = quit")
        print("="*60 + "\n")

        self._display("idle")
        logger.info("Hold-to-talk text loop started")

        while _running:
            _pressed.clear()
            _released.clear()
            print("⏳  Hold button and type your message...")

            # ── wait for button press ─────────────────────────────────────
            while _running and not _pressed.wait(timeout=0.5):
                pass
            if not _running:
                break

            self._display("listening")
            print("\n🎙️  HOLD + TYPE  (release to send)\n  > ", end="", flush=True)

            # ── collect chars while button held, raw mode ─────────────────
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            chars = []
            try:
                tty.setraw(fd)
                while not _released.is_set() and _running:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if ready:
                        ch = sys.stdin.read(1)
                        if ch in ("\x03", "\x04"):   # Ctrl-C / Ctrl-D
                            raise KeyboardInterrupt
                        elif ch in ("\x7f", "\x08"):  # backspace
                            if chars:
                                chars.pop()
                                sys.stdout.write("\b \b")
                                sys.stdout.flush()
                        elif ch.isprintable():
                            chars.append(ch)
                            sys.stdout.write(ch)
                            sys.stdout.flush()
            except KeyboardInterrupt:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print()
                break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            user_input = "".join(chars).strip()
            print(f"\n  [sent: {user_input!r}]\n" if user_input else "\n  (nothing typed)\n")

            if not user_input:
                self._display("idle")
                continue

            if user_input.lower() == "quit":
                break

            # ── process ───────────────────────────────────────────────────
            self._display("processing")
            self._processing = True
            try:
                result   = self.orch.process_message(user_input)
                response = result.get("response", "")
                state    = result.get("state", "")
                analysis = result.get("analysis", {})

                logger.info("Response (%s): '%s'", state, response[:80])
                self._speak(response)

                if analysis.get("conditions"):
                    conds = [
                        f"{c['type']}({c['severity']})"
                        for c in analysis["conditions"]
                    ]
                    print(f"   [Analysis] phase={analysis['phase']} | conditions={conds}")
                print()
            except Exception as e:
                logger.error("Pipeline error: %s", e)
                print(f"❌ Error: {e}\n")
            finally:
                self._processing = False
                self._display("idle")

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
        if self.text_only:
            logger.info("Text-only mode: interactive console loop")
            self.run_demo_loop()
            self.shutdown()
            return

        if self.button and self.audio:
            # Full hardware mode: button triggers mic recording
            logger.info("Hardware mode: waiting for button events (mic active)")
            self._display("idle")
            try:
                while _running:
                    time.sleep(0.1)
            finally:
                self.shutdown()
        elif self.button and not self.audio:
            # Button present but no mic: gated text input
            logger.info("Button-gated text mode: mic unavailable, using keyboard")
            self.run_button_text_loop()
            self.shutdown()
        else:
            # No button: free-form text demo
            self.run_demo_loop()
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down Crisis Assistant")
        self._cancel_record_timeout()
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
    if _as_bool(os.environ.get("CRISIS_TEXT_ONLY"), _as_bool(config.get("app", {}).get("text_only", False))):
        logger.info("Text-only mode is active")

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
