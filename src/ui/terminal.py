import os
import re
import shutil
import textwrap
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

# ── ANSI helpers ──────────────────────────────────────────────────────────────
R   = "\x1b[0m"       # reset
B   = "\x1b[1m"       # bold
DIM = "\x1b[2m"

FG_RED     = "\x1b[31m"
FG_GREEN   = "\x1b[32m"
FG_YELLOW  = "\x1b[33m"
FG_BLUE    = "\x1b[34m"
FG_CYAN    = "\x1b[36m"
FG_WHITE   = "\x1b[37m"
FG_BR_RED  = "\x1b[91m"
FG_BR_GRN  = "\x1b[92m"
FG_BR_YEL  = "\x1b[93m"
FG_BR_CYN  = "\x1b[96m"
FG_BR_WHT  = "\x1b[97m"

_STATE_COLOR = {
    "Idle":                   FG_BR_GRN,
    "Ready":                  FG_BR_GRN,
    "Listening":              FG_BR_CYN,
    "Processing":             FG_BR_YEL,
    "Responding":             FG_BLUE,
    "Critical":               FG_BR_RED,
    "Error":                  FG_RED,
    "Demo mode":              FG_CYAN,
    "Session reset":          FG_BR_GRN,
    "Shutting down":          DIM,
    "Initializing hardware":  FG_YELLOW,
    "Loading AI components":  FG_YELLOW,
    "Starting":               FG_YELLOW,
}

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_ACTIVE_STATES = {
    "Listening", "Processing", "Responding",
    "Initializing hardware", "Loading AI components", "Starting",
}

# Pipeline stage text — full width (≥ 64 cols) and compact (< 60 cols)
_PIPELINE_FULL = {
    "Listening":  "[ ● REC  ·  STT  ·  AI  ·  TTS ]",
    "Processing": "[   REC  · ● STT  · ● AI  ·  TTS ]",
    "Responding": "[   REC  ·  STT  ·  AI  · ● TTS ]",
}
_PIPELINE_COMPACT = {
    "Listening":  "●REC · STT · AI · TTS",
    "Processing": "REC · ●STT · ●AI · TTS",
    "Responding": "REC · STT · AI · ●TTS",
}

# Width threshold below which compact layout is used
_COMPACT_THRESHOLD = 60


def _visible_len(s: str) -> int:
    """Length of string with ANSI escape codes stripped."""
    return len(re.sub(r"\x1b\[[0-9;]*m", "", s))


def _pad_ansi(s: str, width: int) -> str:
    """Left-justify s to `width` visible characters (handles ANSI codes)."""
    pad = width - _visible_len(s)
    return s + " " * max(pad, 0)


class TerminalUI:
    """
    Full-screen ANSI terminal UI.

    Automatically switches between full layout (≥ 60 cols) and a compact
    layout optimised for 3.5″ displays (~53 × 30 chars) when the terminal
    width is below the threshold.

    Full layout  : title + clock header, pipeline bar, YOU, ASSISTANT,
                   activity log (6 lines), hint bar.
    Compact layout: title + spinner+status on one row (no clock), compact
                    pipeline bar, YOU, ASSISTANT, activity log (3 lines),
                    hint bar — everything wrapped to the narrow column count.
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        title: str = "Crisis Assistant",
        enabled: bool = True,
        max_cols: Optional[int] = None,
        max_rows: Optional[int] = None,
    ):
        self.title    = title
        self.enabled  = enabled
        self._max_cols = max_cols   # hard cap; None = use terminal size
        self._max_rows = max_rows   # reserved for future scroll-capping
        self._lock    = threading.Lock()

        self._status    = "Starting"
        self._user_text = ""
        self._asst_text = ""
        self._hint      = ""
        self._pipeline  = ""
        self._log: deque = deque(maxlen=6)

        self._spinner_idx   = 0
        self._spinner_timer: threading.Timer | None = None
        self._last_render   = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def set_status(self, status: str):
        self._status = status or ""
        # Pick pipeline text based on current width
        compact = self._is_compact()
        table   = _PIPELINE_COMPACT if compact else _PIPELINE_FULL
        self._pipeline = table.get(self._status, "")
        if self._status in _ACTIVE_STATES:
            self._start_spinner()
        else:
            self._stop_spinner()
        self.render()

    def set_user_text(self, text: str):
        self._user_text = (text or "").strip()
        self.render()

    def set_assistant_text(self, text: str):
        self._asst_text = (text or "").strip()
        self.render()

    def set_hint(self, hint: str):
        self._hint = hint or ""
        self.render()

    def add_log(self, message: str):
        """Append a timestamped entry to the activity log."""
        ts = datetime.now().strftime("%H:%M")
        self._log.append(f"{DIM}{ts}{R} {message}")
        self.render()

    def set_pipeline_stage(self, stage: str):
        """Manually override the pipeline stage label."""
        self._pipeline = stage
        self.render()

    # ── rendering ─────────────────────────────────────────────────────────────

    def _is_compact(self) -> bool:
        """Return True when the effective terminal width is below threshold."""
        cols = self._effective_cols()
        return cols < _COMPACT_THRESHOLD

    def _effective_cols(self) -> int:
        detected = shutil.get_terminal_size((80, 24)).columns
        if self._max_cols:
            return min(detected, self._max_cols)
        return detected

    def render(self):
        if not self.enabled:
            return
        with self._lock:
            self._render_locked()

    def _render_locked(self):
        cols    = self._effective_cols()
        compact = cols < _COMPACT_THRESHOLD

        if compact:
            # 3.5″ display: clamp to 53, minimum 36
            width = max(36, min(cols - 1, 53))
        else:
            width = max(64, min(cols - 1, 120))

        inner = width - 4   # usable text width inside ║ … ║

        # Recalculate pipeline text for current width in case set_status ran
        # before we knew the width.
        pipeline_table = _PIPELINE_COMPACT if compact else _PIPELINE_FULL
        pipeline = pipeline_table.get(self._status, self._pipeline)

        color = _STATE_COLOR.get(self._status, FG_WHITE)
        spin  = f"{_SPINNER[self._spinner_idx]}" if self._status in _ACTIVE_STATES else " "

        lines: list[str] = []

        if compact:
            # ── COMPACT layout (3.5″ display) ─────────────────────────────
            # Row 1: ╔══…══╗ + "Crisis Asst │ ● Processing"
            short_title = self.title.replace("Crisis Assistant", "Crisis Asst")
            title_str   = f"{B}{FG_BR_WHT}{short_title}{R}"
            status_str  = f"{color}{B}{self._status}{R}"
            header_body = f"{title_str} {DIM}│{R} {color}{spin}{R} {status_str}"
            lines.append(f"╔{'═' * (width - 2)}╗")
            lines.append(self._row(header_body, inner))

            # Row: pipeline (compact one-liner)
            if pipeline:
                lines.append(f"╠{'═' * (width - 2)}╣")
                pipe_str = f"{color}{B}{pipeline}{R}"
                lines.append(self._row(pipe_str, inner))

            # YOU section
            if self._user_text:
                lines.append(f"╠{'═' * (width - 2)}╣")
                lines.append(self._row(f"{FG_BR_CYN}{B} YOU{R}", inner))
                lines.append(f"╟{'─' * (width - 2)}╢")
                for chunk in textwrap.wrap(self._user_text, width=inner - 1) or [""]:
                    lines.append(self._row(f" {chunk}", inner))

            # ASSISTANT section
            if self._asst_text:
                lines.append(f"╠{'═' * (width - 2)}╣")
                lines.append(self._row(f"{FG_BR_GRN}{B} ASST{R}", inner))
                lines.append(f"╟{'─' * (width - 2)}╢")
                for chunk in textwrap.wrap(self._asst_text, width=inner - 1) or [""]:
                    lines.append(self._row(f" {chunk}", inner))

            # Activity log — last 3 entries only, no timestamps to save space
            log_entries = list(self._log)[-3:]
            if log_entries:
                lines.append(f"╠{'═' * (width - 2)}╣")
                lines.append(self._row(f"{FG_YELLOW}{B} LOG{R}", inner))
                lines.append(f"╟{'─' * (width - 2)}╢")
                for entry in log_entries:
                    # Strip timestamp prefix (HH:MM + space) for compact display
                    plain = re.sub(r"\x1b\[[0-9;]*m", "", entry)
                    body  = plain[6:].strip() if len(plain) > 6 else plain
                    for chunk in textwrap.wrap(body, width=inner - 1) or [""]:
                        lines.append(self._row(f" {chunk}", inner))

            # Hint bar
            lines.append(f"╠{'═' * (width - 2)}╣")
            hint_text = self._hint or "Waiting…"
            # Truncate hint to fit on one line
            if len(hint_text) > inner - 1:
                hint_text = hint_text[:inner - 2] + "…"
            lines.append(self._row(f" {DIM}{hint_text}{R}", inner, raw=True))
            lines.append(f"╚{'═' * (width - 2)}╝")

        else:
            # ── FULL layout (desktop / large terminal) ────────────────────
            title_str = f"{B}{FG_BR_WHT}{self.title}{R}"
            ts_str    = f"{DIM}{datetime.now().strftime('%H:%M:%S')}{R}"
            spin_pad  = f" {spin} "

            lines.append(f"╔{'═' * (width - 2)}╗")
            lines.append(self._row(
                f"{title_str}  {DIM}│{R}  {color}{spin_pad}{self._status}{R}",
                inner, right=ts_str,
            ))

            if pipeline:
                lines.append(f"╠{'═' * (width - 2)}╣")
                pipe_display = f"{DIM}PIPELINE{R}  {color}{B}{pipeline}{R}"
                lines.append(self._row(pipe_display, inner))

            if self._user_text:
                lines.append(f"╠{'═' * (width - 2)}╣")
                lines.append(self._row(f"{FG_BR_CYN}{B}  YOU{R}", inner))
                lines.append(f"╟{'─' * (width - 2)}╢")
                for chunk in textwrap.wrap(self._user_text, width=inner) or [""]:
                    lines.append(self._row(f"  {chunk}", inner))

            if self._asst_text:
                lines.append(f"╠{'═' * (width - 2)}╣")
                lines.append(self._row(f"{FG_BR_GRN}{B}  ASSISTANT{R}", inner))
                lines.append(f"╟{'─' * (width - 2)}╢")
                for chunk in textwrap.wrap(self._asst_text, width=inner) or [""]:
                    lines.append(self._row(f"  {chunk}", inner))

            if self._log:
                lines.append(f"╠{'═' * (width - 2)}╣")
                lines.append(self._row(f"{FG_YELLOW}{B}  ACTIVITY LOG{R}", inner))
                lines.append(f"╟{'─' * (width - 2)}╢")
                for entry in self._log:
                    for chunk in textwrap.wrap(entry, width=inner + 20) or [""]:
                        vis = _visible_len(chunk)
                        if vis <= inner:
                            lines.append(self._row(f"  {chunk}", inner, raw=True))

            lines.append(f"╠{'═' * (width - 2)}╣")
            hint_text = self._hint or "Waiting…"
            lines.append(self._row(f"  {DIM}{hint_text}{R}", inner, raw=True))
            lines.append(f"╚{'═' * (width - 2)}╝")

        screen = "\x1b[2J\x1b[H" + "\n".join(lines) + "\n"
        os.write(1, screen.encode("utf-8", errors="ignore"))
        self._last_render = time.time()

    def shutdown(self):
        if not self.enabled:
            return
        self._stop_spinner()
        os.write(1, b"\x1b[2J\x1b[H")

    # ── spinner ───────────────────────────────────────────────────────────────

    def _start_spinner(self):
        if self._spinner_timer and self._spinner_timer.is_alive():
            return
        self._tick_spinner()

    def _tick_spinner(self):
        if not self.enabled:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER)
        with self._lock:
            self._render_locked()
        if self._status in _ACTIVE_STATES:
            self._spinner_timer = threading.Timer(0.1, self._tick_spinner)
            self._spinner_timer.daemon = True
            self._spinner_timer.start()

    def _stop_spinner(self):
        if self._spinner_timer:
            self._spinner_timer.cancel()
            self._spinner_timer = None

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row(content: str, inner: int, right: str = "", raw: bool = False) -> str:
        """Render a box row: ║ content … right ║"""
        vis_content = _visible_len(content)
        vis_right   = _visible_len(right)
        gap = inner - vis_content - vis_right
        if gap < 0:
            gap = 0
        return f"║ {content}{' ' * gap}{right} ║"
