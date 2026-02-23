# src/hardware/button.py
"""
Smart button handler for hold-to-talk interface
- Press & hold: Start recording
- Release: Process input
- Triple press: Reset conversation
- Long hold (5s): Shutdown
"""

import time
import RPi.GPIO as GPIO
from collections import deque
from typing import Callable, Optional

class SmartButton:
    def __init__(self, pin: int = 17, long_hold_threshold: float = 5.0, triple_press_window: float = 1.0):
        """
        Initialize smart button
        
        Args:
            pin: GPIO pin number (BCM mode)
            long_hold_threshold: Seconds to trigger long hold
            triple_press_window: Time window for triple press detection
        """
        self.pin = pin
        self.long_hold_threshold = long_hold_threshold
        self.triple_press_window = triple_press_window
        
        # State tracking
        self.is_pressed = False
        self.press_start_time = None
        self.press_times = deque(maxlen=3)
        
        # Callbacks
        self.on_press_start: Optional[Callable] = None
        self.on_press_end: Optional[Callable[[float], None]] = None
        self.on_triple_press: Optional[Callable] = None
        self.on_long_hold: Optional[Callable] = None
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Add event detection
        GPIO.add_event_detect(
            self.pin,
            GPIO.BOTH,
            callback=self._handle_event,
            bouncetime=50
        )
        
        print(f"✅ Button initialized on GPIO {pin}")
        
    def _handle_event(self, channel):
        """Handle button state changes"""
        current_state = GPIO.input(self.pin)
        current_time = time.time()
        
        # Button PRESSED (LOW with pull-up resistor)
        if current_state == GPIO.LOW and not self.is_pressed:
            self.is_pressed = True
            self.press_start_time = current_time
            self.press_times.append(current_time)
            
            # Check for triple press
            if len(self.press_times) == 3:
                time_span = self.press_times[-1] - self.press_times[0]
                if time_span < self.triple_press_window:
                    print("🔄 TRIPLE PRESS DETECTED")
                    if self.on_triple_press:
                        self.on_triple_press()
                    self.press_times.clear()
                    self.is_pressed = False
                    return
            
            # Normal press start
            print("🔽 Button PRESSED")
            if self.on_press_start:
                self.on_press_start()
                
        # Button RELEASED (HIGH with pull-up resistor)
        elif current_state == GPIO.HIGH and self.is_pressed:
            hold_duration = current_time - self.press_start_time
            self.is_pressed = False
            
            print(f"🔼 Button RELEASED (held {hold_duration:.2f}s)")
            
            # Check for long hold
            if hold_duration >= self.long_hold_threshold:
                print("⚠️ LONG HOLD DETECTED")
                if self.on_long_hold:
                    self.on_long_hold()
            else:
                # Normal release
                if self.on_press_end:
                    self.on_press_end(hold_duration)
                    
    def cleanup(self):
        """Cleanup GPIO"""
        GPIO.cleanup()
        print("🧹 Button GPIO cleaned up")

