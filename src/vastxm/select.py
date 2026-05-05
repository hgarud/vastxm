from __future__ import annotations

import sys
import termios
import tty

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

DEFAULT_LIMIT = 20


def _fmt_int(v) -> str:
    if v is None:
        return "?"
    try:
        return f"{int(v)}"
    except (TypeError, ValueError):
        return "?"


def _fmt_float(v, digits: int = 1) -> str:
    if v is None:
        return "?"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "?"


def _fmt_price(v) -> str:
    if v is None:
        return "?"
    try:
        return f"${float(v):.3f}"
    except (TypeError, ValueError):
        return "?"


def _fmt_ram_gb(mib) -> str:
    if mib is None:
        return "?"
    try:
        return f"{float(mib) / 1024:.0f}"
    except (TypeError, ValueError):
        return "?"


def _fmt_reliability(r) -> str:
    if r is None:
        return "?"
    try:
        return f"{float(r) * 100:.1f}%"
    except (TypeError, ValueError):
        return "?"


def _build_table(offers: list[dict], selected_idx: int) -> Table:
    table = Table(
        title=f"Available offers (showing {len(offers)}, cheapest first)",
        title_style="bold",
        show_lines=False,
        header_style="bold cyan",
        expand=False,
    )
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("GPU", no_wrap=True)
    table.add_column("$/hr", justify="right", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("RAM", justify="right", no_wrap=True)
    table.add_column("Disk", justify="right", no_wrap=True)
    table.add_column("Net ↓/↑", justify="right", no_wrap=True)
    table.add_column("Rel.", justify="right", no_wrap=True)
    table.add_column("Loc.", no_wrap=True)
    table.add_column("CUDA", no_wrap=True)

    for i, o in enumerate(offers):
        gpu_ram = _fmt_int(o.get("gpu_ram"))
        gpu_ram_part = f" ({gpu_ram}GB)" if gpu_ram != "?" else ""
        gpu_cell = f"{_fmt_int(o.get('num_gpus'))}× {o.get('gpu_name') or '?'}{gpu_ram_part}"
        net = f"{_fmt_float(o.get('inet_down'), 0)}/{_fmt_float(o.get('inet_up'), 0)}"
        marker = "▶" if i == selected_idx else " "
        row = [
            f"{marker} {i + 1}",
            gpu_cell,
            _fmt_price(o.get("dph_total")),
            _fmt_int(o.get("cpu_cores_effective") or o.get("cpu_cores")),
            f"{_fmt_ram_gb(o.get('cpu_ram'))}G",
            f"{_fmt_int(o.get('disk_space'))}G",
            net,
            _fmt_reliability(o.get("reliability2") or o.get("reliability")),
            (o.get("geolocation") or "?"),
            str(o.get("cuda_max_good") or "?"),
        ]
        style = "reverse bold" if i == selected_idx else ""
        table.add_row(*row, style=style)

    return table


_HELP = Text.from_markup(
    "[dim]↑/↓ or j/k to move · Enter to select · q to abort[/dim]"
)


def _read_key(fd: int) -> str:
    """Read one logical key. Returns 'up', 'down', 'enter', 'q', or '' for ignored."""
    ch = sys.stdin.read(1)
    if ch == "\x1b":  # ESC — possibly arrow sequence
        # Read next two bytes (CSI sequence). If nothing follows, treat as abort.
        seq = sys.stdin.read(2)
        if seq == "[A":
            return "up"
        if seq == "[B":
            return "down"
        return "abort"
    if ch in ("\r", "\n"):
        return "enter"
    if ch in ("q", "Q", "\x03", "\x04"):  # q, Ctrl-C, Ctrl-D
        return "abort"
    if ch in ("k", "K"):
        return "up"
    if ch in ("j", "J"):
        return "down"
    return ""


def choose_offer(offers: list[dict], *, limit: int = DEFAULT_LIMIT) -> dict:
    """Show offers in a table and let the user pick one with arrow keys. Requires a TTY."""
    if not offers:
        raise RuntimeError("choose_offer called with no offers")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise RuntimeError(
            "Interactive offer selection requires a TTY (stdin/stdout is not a terminal). "
            "Re-run vastxm in an interactive shell."
        )

    shown = offers[:limit]
    selected = 0
    console = Console(stderr=True)
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        with Live(
            Group(_build_table(shown, selected), _HELP),
            console=console,
            auto_refresh=False,
            transient=False,
        ) as live:
            while True:
                key = _read_key(fd)
                if key == "up":
                    selected = (selected - 1) % len(shown)
                elif key == "down":
                    selected = (selected + 1) % len(shown)
                elif key == "enter":
                    return shown[selected]
                elif key == "abort":
                    raise RuntimeError("aborted by user")
                else:
                    continue
                live.update(Group(_build_table(shown, selected), _HELP), refresh=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
