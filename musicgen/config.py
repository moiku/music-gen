"""Paths and presets. Stdlib only (imported by both the server and the ACE worker)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("MUSICGEN_LIBRARY", ROOT / "output/library"))
JOBS = LIBRARY / "_jobs"
ACE_PYTHON = ROOT / "vendor/ACE-Step-1.5/.venv/bin/python"
ROOT_PYTHON = ROOT / ".venv/bin/python"

SR = 48000
TARGET_LUFS = -14.0
# ACE-Step tends to stop ~2.5 s before the requested length (survey.md §6),
# so we generate longer and cut at the target.
GEN_MARGIN_S = 2.5
IDLE_UNLOAD_S = 10 * 60  # worker frees the model after this much idle time

# How a take ends -- the only thing that changes the prompt (worker.STYLE).
ENDINGS = {
    "loop": "ループ",            # steady groove, end flows back into the start (Stories / Reels)
    "button": "きっちり終わる",  # builds up and resolves to a final chord (CM)
    "auto": "おまかせ",
}
# Length presets for the form (survey.md §7). Requests made before 2026-11-07 stored "purpose".
LENGTH_PRESETS = [
    {"label": "ストーリーズ 15秒", "seconds": 15, "ending": "loop"},
    {"label": "リール 7秒", "seconds": 7, "ending": "loop"},
    {"label": "リール 15秒", "seconds": 15, "ending": "loop"},
    {"label": "リール 30秒", "seconds": 30, "ending": "loop"},
    {"label": "CM 15秒", "seconds": 15, "ending": "button"},
    {"label": "CM 30秒", "seconds": 30, "ending": "button"},
    {"label": "60秒", "seconds": 60, "ending": "auto"},
]
_PURPOSE_TO_ENDING = {"stories": "loop", "reels": "loop", "cm": "button", "free": "auto"}


def ending_of(req: dict) -> str:
    return req.get("ending") or _PURPOSE_TO_ENDING.get(req.get("purpose", ""), "auto")


def bar_seconds(bpm: float, beats_per_bar: int = 4) -> float:
    return 60.0 / bpm * beats_per_bar


def snap_to_bars(seconds: float, bpm: float, beats_per_bar: int = 4) -> tuple[int, float]:
    """Round a length to whole bars so loops and bar-aligned edits work. Returns (bars, seconds)."""
    bar = bar_seconds(bpm, beats_per_bar)
    bars = max(1, round(seconds / bar))
    return bars, round(bars * bar, 3)
