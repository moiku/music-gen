"""Measure a take against its order and cut it to length (root venv).

    python -m musicgen.analysis <item_dir>     # raw.wav -> take.wav + meta["analysis"]

Lessons from the CM prototype (survey.md §6):
- librosa's tempo estimate and beat regression were unstable; the reliable check is the offset of
  a fixed grid (bpm, starting at t=0) in 3 windows -> constant offset = shift, growing = drift.
- ACE sometimes renders ~1% fast/slow: fix with a pitch-preserving stretch (Rubber Band).
- Beats commonly land 20-35 ms late: trim the head. Half-beat offsets (~bar/8) are rejected.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SR
from .library import read_json, update_json

HOP = 64


def load(p: Path) -> np.ndarray:
    x, sr = sf.read(p, always_2d=True)
    if sr != SR:
        import librosa
        x = librosa.resample(x.T, orig_sr=sr, target_sr=SR).T
    return x.astype(np.float64)


def rms_db(y: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))


def _onset_env(mono: np.ndarray):
    import librosa
    env = librosa.onset.onset_strength(y=mono.astype(np.float32), sr=SR, hop_length=HOP)
    return env, librosa.times_like(env, sr=SR, hop_length=HOP)


def grid_offsets(x: np.ndarray, bpm: float, seconds: float, n_windows: int = 3) -> list[float]:
    """Best-fitting offset (ms, in [-beat/2, beat/2)) of a fixed beat grid, per window."""
    beat = 60.0 / bpm
    mono = x[: int(seconds * SR)].mean(axis=1)
    env, t = _onset_env(mono)
    edges = np.linspace(0, seconds, n_windows + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        best, val = 0.0, -1.0
        for o in np.arange(-beat / 2, beat / 2, 0.002):
            ts = np.arange(0, seconds, beat) + o
            ts = ts[(ts >= a) & (ts < b)]
            if len(ts) == 0:
                continue
            idx = np.clip(np.searchsorted(t, ts), 0, len(env) - 1)
            v = float(np.mean([env[max(i - 2, 0):i + 3].max() for i in idx]))
            if v > val:
                best, val = o, v
        out.append(round(best * 1000, 1))
    return out


def end_time(mono: np.ndarray, thresh_db: float = -30.0) -> float:
    whole = rms_db(mono)
    w = int(0.05 * SR)
    loud = [i for i in range(0, len(mono) - w, w) if rms_db(mono[i:i + w]) - whole > thresh_db]
    return round((loud[-1] + w) / SR, 2) if loud else 0.0


def accent_ratio(x: np.ndarray, bpm: float, seconds: float, beat_index: int = 2) -> float:
    """Onset strength on beat `beat_index` (0-based) of each bar vs the median beat."""
    beat = 60.0 / bpm
    env, t = _onset_env(x[: int(seconds * SR)].mean(axis=1))
    ts = np.arange(0, seconds - 0.05, beat)
    idx = np.clip(np.searchsorted(t, ts), 0, len(env) - 1)
    per = np.array([env[max(i - 2, 0):i + 3].max() for i in idx])
    sel = per[beat_index::4]
    return round(float(sel.mean() / (np.median(per) + 1e-9)), 2) if len(sel) else 0.0


def measure(x: np.ndarray, bpm: float, seconds: float) -> dict:
    """Numbers shown in the gallery. `x` is already cut to `seconds`."""
    import pyloudnorm as pyln
    mono = x.mean(axis=1)
    offs = grid_offsets(x, bpm, seconds)
    m = {
        "grid_ms": offs,
        "drift_ms": round(offs[-1] - offs[0], 1),
        "start_db": round(rms_db(mono[: int(0.05 * SR)]) - rms_db(mono), 1),
        "end_s": end_time(mono),
        "tail_db": round(rms_db(mono[-int(0.3 * SR):]) - rms_db(mono), 1),
        "accent3": accent_ratio(x, bpm, seconds),
        "lufs": round(float(pyln.Meter(SR).integrated_loudness(x)), 1),
    }
    if x.shape[1] >= 2:
        m["lr_corr"] = round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 2)
    return m


def verdict(m: dict, seconds: float) -> dict:
    """Pass/fail flags against the order. Thresholds from the CM prototype."""
    flags = {
        "on_grid": all(abs(o) <= 40 for o in m["grid_ms"]),
        "steady": abs(m["drift_ms"]) <= 40,
        "starts_at_0": m["start_db"] > -20,
        "fills_length": m["end_s"] >= seconds - 1.5,
        "mono_safe": m.get("lr_corr", 1.0) > 0.2,
    }
    flags = {k: bool(v) for k, v in flags.items()}
    flags["ok"] = all(flags.values())
    return flags


def align(x: np.ndarray, bpm: float, seconds: float) -> tuple[np.ndarray, dict]:
    """Stretch to exact tempo if drifting, then trim the head so beats sit on the grid."""
    info = {"stretch": 1.0, "head_trim_ms": 0.0}
    offs = grid_offsets(x, bpm, seconds)
    drift = offs[-1] - offs[0]
    # windows are seconds/3 apart: drift over (2/3)*seconds -> tempo ratio
    span = seconds * 2 / 3
    if abs(drift) > 25 and abs(drift) < 200:
        ratio = 1 + (drift / 1000) / span  # >1: beats arrive later and later -> too slow
        from pedalboard import time_stretch
        x = time_stretch(x.T.astype(np.float32), SR, stretch_factor=ratio,
                         high_quality=True).T.astype(np.float64)
        info["stretch"] = round(float(ratio), 4)
        offs = grid_offsets(x, bpm, seconds)
    head = offs[0]
    if 0 < head <= 80:
        x = x[int(head / 1000 * SR):]
        info["head_trim_ms"] = head
    return x, info


def fit(x: np.ndarray, seconds: float) -> np.ndarray:
    n = int(round(seconds * SR))
    return x[:n] if len(x) >= n else np.vstack([x, np.zeros((n - len(x), x.shape[1]))])


def declick_tail(x: np.ndarray, ms: float = 8) -> np.ndarray:
    n = int(ms / 1000 * SR)
    x = x.copy()
    x[-n:] *= np.cos(np.linspace(0, np.pi / 2, n))[:, None]
    return x


def analyze_item(d: Path) -> dict:
    meta = read_json(d / "meta.json")
    bpm, seconds = meta["bpm"], meta["seconds"]
    raw = load(d / "raw.wav")
    x, info = align(raw, bpm, seconds)
    take = declick_tail(fit(x, seconds))
    sf.write(d / "take.wav", take.astype(np.float32), SR, subtype="FLOAT")
    m = measure(take, bpm, seconds)
    return update_json(d / "meta.json", analysis={**m, **info}, verdict=verdict(m, seconds),
                       status="ready")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(analyze_item(Path(p)).get("verdict"))
