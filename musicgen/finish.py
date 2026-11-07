"""Produce the finished versions of one take (root venv).

    python -m musicgen.finish <item_dir> [full short loop stems]

final/full.wav     (+ .m4a, AAC 256 kbps, for full/short/loop) target length, -14 LUFS, peak <= -1 dBFS
final/short.wav    first bar(s) + last bar, <= ~8 s (Stories short cut / 6 s ads)
final/loop.wav     last bar replaced by a copy of the bar whose original successor sounds most
                   like bar 1 (repaint-based loops failed: ACE inserts silence before "bar 1")
final/loop_x3.wav  loop played 3 times, for listening to the seam
final/click.wav    full + click on the bpm grid (accent on bar heads)
final/stems/       drums / bass / other (Demucs htdemucs), same gain as full
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .analysis import load, measure
from .config import SR, TARGET_LUFS, bar_seconds
from .library import read_json, update_json


def limit(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Look-ahead peak limiter (5 ms look-ahead, 80 ms release), stereo-linked."""
    ceil = 10 ** (ceiling_db / 20)
    need = np.minimum(1.0, ceil / np.maximum(np.abs(x).max(axis=1), 1e-12))
    la = int(0.005 * SR)
    g = np.lib.stride_tricks.sliding_window_view(np.concatenate([need, np.ones(la)]), la + 1).min(axis=1)
    rel = np.exp(-1.0 / (0.08 * SR))
    out, cur = np.empty_like(g), 1.0
    for i, v in enumerate(g):
        cur = v if v < cur else v + (cur - v) * rel
        out[i] = cur
    return np.clip(x * out[:, None], -ceil, ceil)


def master(x: np.ndarray) -> tuple[np.ndarray, float]:
    gain = TARGET_LUFS - pyln.Meter(SR).integrated_loudness(x)
    return limit(x * 10 ** (gain / 20)), gain


def xfade(a: np.ndarray, b: np.ndarray, ms: float = 20) -> np.ndarray:
    """a + b with an equal-power overlap of `ms`; b must carry `ms` of lead-in."""
    n = int(ms / 1000 * SR)
    t = np.linspace(0, np.pi / 2, n)[:, None]
    return np.vstack([a[:-n], a[-n:] * np.cos(t) + b[:n] * np.sin(t), b[n:]])


def n_bars(x: np.ndarray, bar: int) -> int:
    return max(1, int(round(len(x) / bar)))  # lengths are snapped to whole bars (+- rounding)


def make_short(x: np.ndarray, bar: int, max_s: float = 8.0) -> np.ndarray:
    """Leading bars + the last bar, as long as it fits in max_s (at least 1 + 1 bars)."""
    nbars = n_bars(x, bar)
    k = max(1, min(nbars - 1, int(max_s * SR // bar) - 1))
    n = int(0.01 * SR)
    head = x[: k * bar + n]
    tail = x[(nbars - 1) * bar - n:]
    return xfade(head, tail, ms=10)[: (k + 1) * bar]


def make_loop(x: np.ndarray, bar: int) -> tuple[np.ndarray, int]:
    import librosa
    nbars = n_bars(x, bar)
    if nbars < 3:
        return x, 0

    def head_feat(i):
        y = x[i * bar: i * bar + bar // 4].mean(1).astype(np.float32)
        c = librosa.feature.chroma_stft(y=y, sr=SR).mean(1)
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=40)).mean(1)
        return np.concatenate([c / (np.linalg.norm(c) + 1e-9), m / (np.linalg.norm(m) + 1e-9)])

    ref = head_feat(0)
    cands = range(1, nbars - 1)  # successor bars (0-based) other than bar 1 and the last
    succ = max(cands, key=lambda i: float(head_feat(i) @ ref))
    n = int(0.02 * SR)
    copy = x[(succ - 1) * bar - n: succ * bar] if succ > 1 else x[:bar]  # bar before successor
    if succ == 1:  # copy of bar 1 itself: no lead-in available, pad
        copy = np.vstack([x[bar - n: bar], x[:bar]])
    loop = xfade(x[: (nbars - 1) * bar], copy)
    loop = np.vstack([loop, np.zeros((max(0, len(x) - len(loop)), x.shape[1]))])[: len(x)].copy()
    m = int(0.003 * SR)
    loop[-m:] *= np.linspace(1, 0, m)[:, None]
    loop[:m] *= np.linspace(0, 1, m)[:, None]
    return loop, succ  # bar `succ` (1-based) was copied into the last bar


def click_track(n: int, bpm: float) -> np.ndarray:
    c = np.zeros(n)
    beat = 60.0 / bpm
    for k in range(int(n / SR / beat) + 1):
        i, L = int(k * beat * SR), int(0.02 * SR)
        seg = np.sin(2 * np.pi * (2000 if k % 4 == 0 else 1200) * np.arange(L) / SR) \
            * np.exp(-np.arange(L) / (0.004 * SR)) * 0.5
        c[i:i + L] += seg[: max(0, min(L, n - i))]
    return c


def finish_item(d: Path, versions: list[str]) -> dict:
    meta = read_json(d / "meta.json")
    bpm, seconds = meta["bpm"], meta["seconds"]
    bar = int(round(bar_seconds(bpm) * SR))
    take = load(d / "take.wav")
    out = d / "final"
    out.mkdir(exist_ok=True)
    full, gain = master(take)
    files, info = {}, {"gain_db": round(gain, 1)}

    def write(name, y, m4a=False):
        sf.write(out / name, y.astype(np.float32), SR, subtype="PCM_24")
        files[name.removesuffix(".wav")] = f"final/{name}"
        if m4a:  # for phones / Instagram (re-encoded to AAC 128 kbps on upload anyway)
            m = name.replace(".wav", ".m4a")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / name), "-c:a", "aac_at",
                            "-b:a", "256k", str(out / m)], check=True)
            files[name.removesuffix(".wav") + "_m4a"] = f"final/{m}"

    write("full.wav", full, m4a=True)
    write("click.wav", full * 0.7 + click_track(len(full), bpm)[:, None])
    if "short" in versions:
        write("short.wav", make_short(full, bar), m4a=True)
    if "loop" in versions:
        loop, copied = make_loop(full, bar)
        write("loop.wav", loop, m4a=True)
        write("loop_x3.wav", np.vstack([loop] * 3))
        w = int(0.02 * SR)
        info["loop_copied_bar"] = copied
        info["loop_seam_jump_db"] = round(float(20 * np.log10(
            (np.sqrt((loop[:w] ** 2).mean()) + 1e-9) / (np.sqrt((loop[-w:] ** 2).mean()) + 1e-9))), 1)
    if "stems" in versions:
        tmp = out / "_sep"
        tmp.mkdir(exist_ok=True)
        sf.write(tmp / "in.wav", take.astype(np.float32), SR, subtype="FLOAT")
        subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", "mps", "-o", str(tmp),
                        str(tmp / "in.wav")], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        (out / "stems").mkdir(exist_ok=True)
        for name in ["drums", "bass", "other"]:
            s, ssr = sf.read(tmp / f"htdemucs/in/{name}.wav", always_2d=True)
            if ssr != SR:  # Demucs writes at the model rate (44.1 kHz)
                import librosa
                s = librosa.resample(s.T, orig_sr=ssr, target_sr=SR).T
            sf.write(out / f"stems/{name}.wav", (s[: len(take)] * 10 ** (gain / 20)).astype(np.float32),
                     SR, subtype="FLOAT")
            files[f"stem_{name}"] = f"final/stems/{name}.wav"
        shutil.rmtree(tmp, ignore_errors=True)
    info["full"] = measure(full, bpm, seconds)
    return update_json(d / "meta.json", final={"files": files, **info})


if __name__ == "__main__":
    d = Path(sys.argv[1])
    print(finish_item(d, sys.argv[2:] or ["full", "short", "loop"])["final"]["files"])
