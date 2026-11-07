"""Fallback loop maker (no model): replace bar 6 with a copy of bar b (2-4) whose original
successor (bar b+1) sounds most like bar 1, so that the wrap 'bar b copy -> bar 1' mimics a
transition that already exists in the track. Equal-power 20 ms joins on bar lines.

    uv run python cm/make_loop_xfade.py output/cm/final/<name>/full_15s.wav
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

BAR = 2.5


def main() -> None:
    src = Path(sys.argv[1])
    x, sr = sf.read(src, always_2d=True)
    nb = int(BAR * sr)
    bars = [x[i * nb:(i + 1) * nb] for i in range(6)]

    def head_feat(b):  # chroma + mel of the first beat of a bar
        y = b[: int(0.625 * sr)].mean(1).astype(np.float32)
        c = librosa.feature.chroma_stft(y=y, sr=sr).mean(1)
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)).mean(1)
        return np.concatenate([c / (np.linalg.norm(c) + 1e-9), m / (np.linalg.norm(m) + 1e-9)])

    ref = head_feat(bars[0])
    sims = {b: float(head_feat(bars[b]) @ ref) for b in (1, 2, 3)}  # successor of bar b (0-based b-1)
    succ = max(sims, key=sims.get)       # bar index (0-based) whose head is most like bar 1
    n = int(0.02 * sr)
    # the bar that originally preceded it, plus n samples of lead-in for the crossfade
    copy = x[(succ - 1) * nb - n:succ * nb]
    print("similarity of bar heads to bar 1:", {k + 1: round(v, 3) for k, v in sims.items()},
          f"-> bar 6 := copy of bar {succ}")

    t = np.linspace(0, np.pi / 2, n)[:, None]
    body = np.vstack(bars[:5])
    joined = np.vstack([body[:-n], body[-n:] * np.cos(t) + copy[:n] * np.sin(t), copy[n:]])
    # wrap point: 3 ms micro-fades at both ends keep the restart click-free
    joined = joined[: 6 * nb].copy()
    m = int(0.003 * sr)
    joined[-m:] *= np.linspace(1, 0, m)[:, None]
    joined[:m] *= np.linspace(0, 1, m)[:, None]
    out = src.parent.parent.parent / "loop" / f"{src.parent.name}_xfade.wav"
    sf.write(out, joined, sr, subtype="FLOAT")
    print(out)


if __name__ == "__main__":
    main()
