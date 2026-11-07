"""Turn one chosen candidate into the deliverables of the CM brief.

    uv run python cm/finish.py output/cm/candidates/<file>.wav [--loop-src output/cm/loop/<file>.wav]

Writes to output/cm/final/<stem>/:
  full_15s.wav     15.000 s, 48 kHz, -14 LUFS (peak <= -1 dBFS), 8 ms de-click at the very end
  short_6s.wav     bar 1 + bar 6 (2 x 2.5 s = 5.0 s), 10 ms equal-power join on the bar line
  loop_15s.wav     (if --loop-src) 15.000 s whose end flows into its start
  stems/           drums / bass / other (Demucs htdemucs), same gain as full_15s
  report.json      measured tempo / grid / loudness of each file
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sheet import limit  # noqa: E402
from score import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR, BAR, LEN = 48000, 2.5, 15.0
TARGET = -14.0


def load48(p: Path) -> np.ndarray:
    x, sr = sf.read(p, always_2d=True)
    if sr != SR:
        import librosa
        x = librosa.resample(x.T, orig_sr=sr, target_sr=SR).T
    return x


def fit(x: np.ndarray, seconds: float) -> np.ndarray:
    n = int(round(seconds * SR))
    return x[:n] if len(x) >= n else np.vstack([x, np.zeros((n - len(x), x.shape[1]))])


def declick(x: np.ndarray, ms: float = 8) -> np.ndarray:
    n = int(ms / 1000 * SR)
    x = x.copy()
    x[-n:] *= np.cos(np.linspace(0, np.pi / 2, n))[:, None]
    return x


def master(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Gain to -14 LUFS with the look-ahead limiter; returns (audio, gain_db applied pre-limiter)."""
    lufs = pyln.Meter(SR).integrated_loudness(x)
    g = TARGET - lufs
    return limit(x * 10 ** (g / 20), SR), g


def join(a: np.ndarray, b: np.ndarray, ms: float = 10) -> np.ndarray:
    n = int(ms / 1000 * SR)
    t = np.linspace(0, np.pi / 2, n)[:, None]
    mid = a[-n:] * np.cos(t) + b[:n] * np.sin(t)
    return np.vstack([a[:-n], mid, b[n:]])


def beat_fit(x: np.ndarray, seconds: float = LEN) -> tuple[float, float]:
    """(period_s, phase_s) of a straight line fitted to tracked beats in the first `seconds`."""
    import librosa
    y = x[: int(seconds * SR)].mean(axis=1).astype(np.float32)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=128)
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=128, start_bpm=96,
                                       tightness=400, units="time")
    k = np.round(beats / (60 / 96))
    (period, phase), *_ = np.linalg.lstsq(np.vstack([k, np.ones_like(k)]).T, beats, rcond=None)
    return float(period), float(phase)


def align(x: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Make the beat grid exactly 96 BPM starting at t=0.

    1) tempo off by > 0.1 BPM -> pitch-preserving time stretch (Rubber Band via pedalboard)
    2) beats late by 0-80 ms -> drop that much from the head
    Returns (audio, measured_bpm, stretch_factor, head_trim_ms).
    """
    period, _ = beat_fit(x)
    bpm = 60 / period
    factor = 1.0
    if abs(bpm - 96) > 0.1:
        from pedalboard import time_stretch
        factor = bpm / 96  # >1 = slow down
        x = time_stretch(x.T.astype(np.float32), SR, stretch_factor=1 / factor,
                         high_quality=True).T.astype(np.float64)
    _, phase = beat_fit(x)
    trim_ms = phase * 1000 if 0 < phase * 1000 <= 80 else 0.0
    return x[int(trim_ms / 1000 * SR):], round(bpm, 2), round(factor, 4), round(trim_ms, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--loop-src", type=Path)
    args = ap.parse_args()
    out = ROOT / "output/cm/final" / args.src.stem
    (out / "stems").mkdir(parents=True, exist_ok=True)

    src, bpm_in, stretch, shift_ms = align(load48(args.src))
    raw = fit(src, LEN)
    full, gain_db = master(declick(raw))
    sf.write(out / "full_15s.wav", full, SR, subtype="PCM_24")

    bar = int(BAR * SR)
    short = join(full[:bar + int(0.005 * SR)], full[-bar:])  # bar 1 + bar 6
    sf.write(out / "short_6s.wav", fit(short, 2 * BAR), SR, subtype="PCM_24")

    if args.loop_src:
        loop, _ = master(fit(load48(args.loop_src), LEN))
        sf.write(out / "loop_15s.wav", loop, SR, subtype="PCM_24")
        # seam check: level jump across the wrap point (end -> start)
        w = int(0.02 * SR)
        seam = 20 * np.log10((np.sqrt((loop[:w] ** 2).mean()) + 1e-9) /
                             (np.sqrt((loop[-w:] ** 2).mean()) + 1e-9))
        sf.write(out / "loop_check_x3.wav", np.vstack([loop] * 3), SR, subtype="PCM_24")

    # Stems: separate the un-mastered audio, then apply the same gain as the full mix.
    tmp = out / "stems/_raw.wav"
    sf.write(tmp, raw, SR, subtype="FLOAT")
    subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", "mps",
                    "-o", str(out / "stems/_demucs"), str(tmp)], check=True,
                   stdout=subprocess.DEVNULL)
    sep = out / "stems/_demucs/htdemucs/_raw"
    for name in ["drums", "bass", "other", "vocals"]:
        s, ssr = sf.read(sep / f"{name}.wav", always_2d=True)
        if ssr != SR:  # Demucs writes at the model rate (44.1 kHz)
            import librosa
            s = librosa.resample(s.T, orig_sr=ssr, target_sr=SR).T
        s = fit(s, LEN) * 10 ** (gain_db / 20)
        sf.write(out / f"stems/{name}.wav", s, SR, subtype="FLOAT")
    import shutil
    shutil.rmtree(out / "stems/_demucs", ignore_errors=True)
    tmp.unlink(missing_ok=True)
    # Sync check: the mix with a 96 BPM click (accent on bar heads) starting at t=0.
    click = np.zeros(len(full))
    for k in range(int(LEN / (60 / 96))):
        i, n = int(k * 60 / 96 * SR), int(0.02 * SR)
        click[i:i + n] += np.sin(2 * np.pi * (2000 if k % 4 == 0 else 1200) * np.arange(n) / SR) \
            * np.exp(-np.arange(n) / (0.004 * SR)) * 0.5
    sf.write(out / "check_click_96bpm.wav", full * 0.7 + click[:, None], SR, subtype="PCM_24")
    vocal_db = 20 * np.log10(np.sqrt((sf.read(out / "stems/vocals.wav")[0] ** 2).mean()) + 1e-9)

    fp, fph = beat_fit(full)
    report = {"source": str(args.src), "measured_bpm_in": bpm_in, "stretch_factor": stretch,
              "head_trim_ms": shift_ms,
              "after_align": {"bpm": round(60 / fp, 2), "phase_ms": round(fph * 1000, 1),
                              "drift_at_15s_ms": round((fp - 0.625) * 24 * 1000, 1)},
              "gain_db": round(gain_db, 1),
              "vocals_stem_rms_dbfs": round(vocal_db, 1)}
    for f in ["full_15s.wav", "short_6s.wav"] + (["loop_15s.wav"] if args.loop_src else []):
        x, _ = sf.read(out / f, always_2d=True)
        report[f] = {**score(out / f),
                     "lufs": round(pyln.Meter(SR).integrated_loudness(x), 1),
                     "peak_dbfs": round(20 * np.log10(np.abs(x).max()), 2)}
    if args.loop_src:
        report["loop_seam_level_jump_db"] = round(float(seam), 1)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(report, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
