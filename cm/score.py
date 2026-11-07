"""Score CM candidates against the brief's measurable requirements.

    uv run python cm/score.py [output/cm/candidates]

Checks (96 BPM, 4/4, grid starting at t=0, 15.0 s):
  tempo_bpm     librosa tempo estimate (96 expected; 48/192 count as octave errors)
  grid_offset   phase (ms) of the beat grid that best fits the onsets; 0 = beats land on k*0.625 s
  grid_ratio    onset strength on the t=0 grid / mean onset strength (higher = clearer beat on grid)
  start_db      level of the first 50 ms relative to the whole clip (should not be silent)
  hits          onset strength on beat 3 of bars 1-3 relative to the median beat (brief: "決め")
  tail_db       level of the last 300 ms relative to the whole clip (very low = faded/ended early,
                near 0 = cut off while still playing)
  dur_s         duration
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np

BPM, BEAT = 96.0, 60.0 / 96.0
HOP = 256


def rms_db(y: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)


def end_time(y: np.ndarray, sr: int, whole_db: float, thresh: float = -30.0) -> float:
    """Last time (s) where the 50 ms level is within `thresh` dB of the whole-clip level."""
    w = int(0.05 * sr)
    lv = [rms_db(y[i:i + w]) - whole_db for i in range(0, len(y) - w, w)]
    loud = [i for i, v in enumerate(lv) if v > thresh]
    return (loud[-1] + 1) * w / sr if loud else 0.0


def score(path: Path) -> dict:
    y, sr = librosa.load(path, sr=48000, mono=True)
    dur = len(y) / sr
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    t_env = librosa.times_like(env, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=HOP,
                                                      start_bpm=BPM))[0])

    def grid_strength(offset: float) -> float:
        ts = np.arange(offset, dur - 0.05, BEAT)
        idx = np.clip(np.searchsorted(t_env, ts), 0, len(env) - 1)
        # allow +-1 frame (~5 ms) jitter
        return float(np.mean([env[max(i - 1, 0):i + 2].max() for i in idx]))

    offsets = np.arange(0, BEAT, 0.005)
    strengths = np.array([grid_strength(o) for o in offsets])
    best = offsets[strengths.argmax()]
    best_ms = (best if best < BEAT / 2 else best - BEAT) * 1000

    beat_vals = [grid_strength(0)]  # placeholder for median below
    beat_ts = np.arange(0, dur - 0.05, BEAT)
    bidx = np.clip(np.searchsorted(t_env, beat_ts), 0, len(env) - 1)
    per_beat = np.array([env[max(i - 1, 0):i + 2].max() for i in bidx])
    hit_beats = [b for b in (2, 6, 10) if b < len(per_beat)]  # beat 3 of bars 1-3
    hits = float(np.mean(per_beat[hit_beats]) / (np.median(per_beat) + 1e-9))

    whole = rms_db(y)
    return {
        "file": path.name,
        "dur_s": round(dur, 3),
        "tempo_bpm": round(tempo, 1),
        "grid_offset_ms": round(best_ms, 0),
        "grid_ratio": round(grid_strength(0) / (env.mean() + 1e-9), 2),
        "start_db": round(rms_db(y[: int(0.05 * sr)]) - whole, 1),
        "hits": round(hits, 2),
        "tail_db": round(rms_db(y[-int(0.3 * sr):]) - whole, 1),
        "end_s": round(end_time(y, sr, whole), 2),
    }


def total(s: dict) -> float:
    """Single number for ranking (heuristic; ears decide in the end)."""
    t = 0.0
    t -= abs(s["tempo_bpm"] - BPM) / 2          # tempo error
    t -= abs(s["grid_offset_ms"]) / 25           # grid misalignment
    t += min(s["grid_ratio"], 3)                 # clear beat on the grid
    t -= max(0, -12 - s["start_db"]) / 3         # silent / slow start
    t += min(s["hits"], 2) / 2                   # accents on beat 3
    t -= max(0, -25 - s["tail_db"]) / 5          # fades away too early
    t -= max(0, s["tail_db"] + 3) / 2            # still loud at the cut = chopped
    return round(t, 2)


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "output/cm/candidates")
    rows = [score(p) for p in sorted(d.glob("*.wav"))]
    for r in rows:
        r["total"] = total(r)
    rows.sort(key=lambda r: -r["total"])
    cols = ["file", "total", "tempo_bpm", "grid_offset_ms", "grid_ratio", "start_db", "hits",
            "tail_db", "end_s"]
    print("  ".join(f"{c:>14s}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>14s}" for c in cols))
    (d / "scores.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
