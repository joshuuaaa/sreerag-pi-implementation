# src/hardware/lcd.py
"""
LCD display manager for conversation states
Supports I2C 16x2 character displays
"""

from RPLCD.i2c import CharLCD
import time
from typing import Optional, Dict

class ConversationLCD:
    def __init__(self, i2c_address: int = 0x27, cols: int = 16, rows: int = 2):
        """
        Initialize LCD display
        
        Args:
            i2c_address: I2C address (0x27 or 0x3f typically)
            cols: Number of columns (16 or 20)
            rows: Number of rows (2 or 4)
        """
        self.cols = cols
        self.rows = rows
        self.lcd: Optional[CharLCD] = None
        
        try:
            self.lcd = CharLCD(
                i2c_expander='PCF8574',
                address=i2c_address,
                port=1,
                cols=cols,
                rows=rows,
                dotsize=8,
                charmap='A00',
                auto_linebreaks=True
            )
            self.lcd.clear()
            print(f"✅ LCD initialized at 0x{i2c_address:02x} ({cols}x{rows})")
        except Exception as e:
            print(f"❌ LCD initialization failed: {e}")
            print("   Continuing without LCD...")
            
    def show_state(self, state: str):
        """
        Display predefined conversation state
        
        Args:
            state: One of: idle, listening, processing, responding, critical, ready
        """
        states: Dict[str, tuple] = {
            "idle": ("Crisis Assistant", "Press to talk"),
            "listening": ("LISTENING", "Speak now..."),
            "processing": ("ANALYZING", "Please wait..."),
            "responding": ("SPEAKING", "Listen..."),
            "critical": ("!!! CRITICAL", "CALL 911 NOW"),
            "ready": ("Ready", "Press again"),
            "error": ("Error", "Try again")
        }
        
        lines = states.get(state, (state, ""))
        self.show(lines[0], lines[1])
        
    def show(self, line1: str, line2: str = ""):
        """
        Display custom text
        
        Args:
            line1: First line text
            line2: Second line text (optional)
        """
        if not self.lcd:
            # Fallback to console
            print(f"LCD: {line1} | {line2}")
            return
            
        try:
            self.lcd.clear()
            self.lcd.write_string(line1[:self.cols])
            if line2 and self.rows >= 2:
                self.lcd.crlf()
                self.lcd.write_string(line2[:self.cols])
        except Exception as e:
            print(f"❌ LCD error: {e}")
            
    def show_analysis(self, analysis: dict):
        """
        Display situation analysis info
        
        Args:
            analysis: Analysis dict with conditions and turn_count
        """
        if not analysis or not analysis.get("conditions"):
            self.show("Assessing...")
            return
            
        primary = analysis["conditions"][0]
        condition = primary["type"][:10]  # Truncate to fit
        severity = primary.get("severity", "?")[:4].upper()
        turn = analysis.get("turn_count", 0)
        
        line1 = f"{condition} {severity}"
        line2 = f"Turn {turn}"
        
        self.show(line1, line2)
        
    def clear(self):
        """Clear display"""
        if self.lcd:
            try:
                self.lcd.clear()
            except:
                pass
                
    def scroll_text(self, text: str, delay: float = 0.3):
        """
        Scroll long text horizontally (blocking)
        
        Args:
            text: Text to scroll
            delay: Delay between scroll steps
        """
        if not self.lcd or len(text) <= self.cols:
            self.show(text)
            return
            
        try:
            padded = text + "    "
            for i in range(len(padded) - self.cols + 1):
                self.lcd.clear()
                self.lcd.write_string(padded[i:i+self.cols])
                time.sleep(delay)
        except Exception as e:
            print(f"❌ Scroll error: {e}")
