"""Modern CLI help formatter for the Affine CLI.

Renders fully-enclosed boxes (┌─ Title ─┐ / │ content │ / └───────┘) with
colored command/option names on wide terminals (>= WIDE_THRESHOLD columns).
Falls back to standard Click formatting on narrow terminals.
"""

import re
import shutil
import sys
from contextlib import contextmanager

import click
from click.formatting import iter_rows, measure_table, wrap_text


# ------------------------------------------------------------------
# Custom Context + Group that wires in AffineHelpFormatter.
# Click 8.3.1 does not accept formatter_class as a context_setting,
# so we hook through context_class / make_formatter instead.
# ------------------------------------------------------------------

class AffineContext(click.Context):
    """Context subclass that produces an AffineHelpFormatter."""

    def make_formatter(self) -> "AffineHelpFormatter":
        return AffineHelpFormatter(
            width=self.terminal_width,
            max_width=self.max_content_width,
        )


class AffineGroup(click.Group):
    """Group subclass that uses AffineContext (and thus AffineHelpFormatter)."""
    context_class = AffineContext


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

WIDE_THRESHOLD = 100   # columns — below this, fall back to plain Click

# ANSI escape codes
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_CYAN    = "\033[36m"
_YELLOW  = "\033[33m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _term_width() -> int:
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def _has_color() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _visible_len(s: str) -> int:
    """Return the visible (display) length of *s*, ignoring ANSI codes."""
    return len(_ANSI_RE.sub("", s))


# ------------------------------------------------------------------
# Formatter
# ------------------------------------------------------------------

class AffineHelpFormatter(click.HelpFormatter):
    """Help formatter with full-box sections for wide terminals."""

    def __init__(self, **kwargs):
        self._tw = _term_width()
        self._modern = self._tw >= WIDE_THRESHOLD
        self._color = _has_color()
        if self._modern:
            if not kwargs.get("width"):
                kwargs["width"] = min(self._tw, 80)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ansi(self, text: str, *codes: str) -> str:
        """Wrap *text* in ANSI codes; no-op when color is disabled."""
        if not self._color:
            return text
        return "".join(codes) + text + _RESET

    # ------------------------------------------------------------------
    # Usage line — colored "Usage:" prefix + bold program name
    # ------------------------------------------------------------------

    def write_usage(self, prog: str, args: str = "", prefix=None) -> None:
        if not self._modern:
            super().write_usage(prog, args, prefix)
            return

        if prefix is None:
            prefix = "Usage: "

        indent = " " * self.current_indent
        colored_prefix = self._ansi(prefix, _BOLD, _YELLOW)  # bold yellow "Usage: "
        colored_prog   = self._ansi(prog,   _BOLD, _YELLOW)  # bold yellow program name
        dim_args       = self._ansi(args,   _DIM)             # dimmed args
        self.write(f"{indent}{colored_prefix}{colored_prog} {dim_args}\n")

    # ------------------------------------------------------------------
    # Section — buffers content, then wraps in a full enclosed box
    # ------------------------------------------------------------------

    @contextmanager
    def section(self, name: str):
        if not self._modern:
            # Standard Click behavior
            self.write_paragraph()
            self.write_heading(name)
            self.indent()
            try:
                yield
            finally:
                self.dedent()
            return

        # Capture everything written inside the section
        saved = self.buffer
        self.buffer = []
        self.indent()
        try:
            yield
        finally:
            self.dedent()
            content = "".join(self.buffer)
            self.buffer = saved
            self._emit_box(name, content)

    def _emit_box(self, title: str, content: str) -> None:
        w = self.width or 80

        # Top border:  ┌─ Title ──────────────────────────────┐
        label = f" {title} "
        right = max(w - 3 - len(label), 2)   # 3 = ┌─ + ┐
        top = f"┌─{label}{'─' * right}┐"

        # Bottom border: └──────────────────────────────────────┘
        bot = f"└{'─' * (w - 2)}┘"

        # Blank separator — only when there is already preceding content
        if self.buffer:
            self.write("\n")

        self.write(f"{self._ansi(top, _DIM)}\n")

        # Content lines: │<line padded to inner width>│
        inner_w = w - 2   # chars between the two │ borders
        for line in content.splitlines():
            pad = " " * max(inner_w - _visible_len(line), 0)
            self.write(
                f"{self._ansi('│', _DIM)}{line}{pad}{self._ansi('│', _DIM)}\n"
            )

        self.write(f"{self._ansi(bot, _DIM)}\n")

    # ------------------------------------------------------------------
    # Definition list — colors the first column (command / option names)
    # ------------------------------------------------------------------

    def write_dl(self, rows, col_max: int = 30, col_spacing: int = 2) -> None:
        if not self._modern:
            super().write_dl(rows, col_max, col_spacing)
            return

        # Coerce None values to "" so measure_table / wrap_text don't crash
        rows = [
            (r[0] or "", r[1] or "") if len(r) >= 2 else (r[0] or "",)
            for r in rows
        ]
        if not rows:
            return

        widths = measure_table(rows)
        if len(widths) != 2:
            super().write_dl(rows, col_max, col_spacing)
            return

        first_col  = min(widths[0], col_max)
        indent_str = " " * self.current_indent
        text_width = max(
            (self.width or 80) - self.current_indent - first_col - col_spacing,
            10,
        )

        for first, second in iter_rows(rows, len(widths)):
            colored = self._ansi(first, _CYAN)

            if not second:
                self.write(f"{indent_str}{colored}\n")
                continue

            if len(first) <= first_col:
                pad    = " " * (first_col - len(first) + col_spacing)
                prefix = f"{indent_str}{colored}{pad}"
                cont   = " " * (self.current_indent + first_col + col_spacing)
            else:
                # Name overflows — put description on next line
                self.write(f"{indent_str}{colored}\n")
                cont   = " " * (self.current_indent + first_col + col_spacing)
                prefix = cont

            wrapped = wrap_text(second, text_width, preserve_paragraphs=True).splitlines()
            if not wrapped:
                self.write(f"{prefix}\n")
                continue

            self.write(f"{prefix}{wrapped[0]}\n")
            for line in wrapped[1:]:
                self.write(f"{cont}{line}\n")
