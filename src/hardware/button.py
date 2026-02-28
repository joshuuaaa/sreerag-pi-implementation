# src/hardware/button.py
"""
Smart button handler for hold-to-talk interface.
Uses gpiozero (Pi 5 compatible via rpi-lgpio backend).

Gestures:
- Press & hold  → start recording
- Release       → process input
- Triple press  → reset conversation
"""

import time
import logging
from collections import deque
from typing import Callable, Optional

from gpiozero import Button as _GpioButton

logger = logging.getLogger("hardware.button")


class SmartButton:
    def __init__(
        self,
        pin: int = 17,
        long_hold_threshold: float = 5.0,
        triple_press_window: float = 1.0,
    ):
        """
        Initialize smart button on a BCM GPIO pin.

        Args:
            pin:                  BCM GPIO pin number.
            long_hold_threshold:  Seconds held to trigger long-hold callback.
            triple_press_window:  Max seconds for three presses to count as triple.
        """
        self.pin                 = pin
        self.long_hold_threshold = long_hold_threshold
        self.triple_press_window = triple_press_window

        # State tracking
        self._is_pressed    = False
        self._press_start   = 0.0
        self._press_times:  deque = deque(maxlen=3)

        # Public callbacks – set these after construction
        self.on_press_start:  Optional[Callable]             = None
        self.on_press_end:    Optional[Callable[[float], None]] = None
        self.on_triple_press: Optional[Callable]             = None
        self.on_long_hold:    Optional[Callable]             = None

        # gpiozero Button (pull_up=True → active-low, same as before)
        self._btn = _GpioButton(pin, pull_up=True, bounce_time=0.05)
        self._btn.when_pressed  = self._on_pressed
        self._btn.when_released = self._on_released

        logger.info("Button initialized on GPIO %s (gpiozero/lgpio)", pin)

    # ── internal handlers ─────────────────────────────────────────────────────

    def _on_pressed(self):
        """Called by gpiozero on falling edge (button down)."""
        now = time.time()
        self._is_pressed  = True
        self._press_start = now
        self._press_times.append(now)

        # Triple-press detection
        if len(self._press_times) == 3:
            span = self._press_times[-1] - self._press_times[0]
            if span < self.triple_press_window:
                logger.info("Triple press detected")
                self._press_times.clear()
                self._is_pressed = False
                if self.on_triple_press:
                    self.on_triple_press()
                return

        logger.debug("Button pressed")
        if self.on_press_start:
            self.on_press_start()

    def _on_released(self):
        """Called by gpiozero on rising edge (button up)."""
        if not self._is_pressed:
            return

        hold = time.time() - self._press_start
        self._is_pressed = False

        logger.debug("Button released (held %.2fs)", hold)

        if self.on_press_end:
            self.on_press_end(hold)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def cleanup(self):
        """Release GPIO resources."""
        self._btn.close()
        logger.info("Button GPIO cleaned up")

