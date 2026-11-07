"""Robust beat-grid check against a fixed 96 BPM grid starting at t=0.

For each window (start / middle / end of the 15 s), find the grid offset (ms) that best matches
the onset envelope. Constant offset = sync shift; growing offset = tempo drift.

    uv run python cm/grid_check.py file.wav [...]
"""
import sys

import librosa
import numpy as np

BEAT, HOP, SR = 60 / 96, 64, 48000


def window_offset(env, t, a, b):
    offs = np.arange(-BEAT / 2, BEAT / 2, 0.002)
    best, val = 0.0, -1.0
    for o in offs:
        ts = np.arange(0, 15, BEAT) + o
        ts = ts[(ts >= a) & (ts < b)]
        idx = np.clip(np.searchsorted(t, ts), 0, len(env) - 1)
        v = np.mean([env[max(i - 2, 0):i + 3].max() for i in idx])
        if v > val:
            best, val = o, v
    return best * 1000


def check(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = y[: 15 * SR]
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    t = librosa.times_like(env, sr=SR, hop_length=HOP)
    return [round(window_offset(env, t, a, b)) for a, b in [(0, 5), (5, 10), (10, 15)]]


if __name__ == "__main__":
    for p in sys.argv[1:]:
        o = check(p)
        print(f"{p:60s} offset ms  0-5s:{o[0]:+5d}  5-10s:{o[1]:+5d}  10-15s:{o[2]:+5d}")
