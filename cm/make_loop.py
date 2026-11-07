"""Make a seamless-loop version: regenerate bar 6 so that it leads back into bar 1.

Builds [full 0-15 s] + [full 0-2.5 s] (17.5 s), repaints 12.5-15.0 s with ACE-Step (the
model sees the following bar 1 as context), then keeps 0-15 s.

    vendor/ACE-Step-1.5/.venv/bin/python cm/make_loop.py output/cm/candidates/<file>.wav
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ace_engine  # noqa: E402
from gen_candidates import BPM, CAPTIONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Use the finished (grid-aligned, 15.000 s) full mix as the source of the loop.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--start", type=float, default=12.5, help="repaint from here to 15.0 s")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    src = a.src
    x, sr = sf.read(src, always_2d=True)
    if src.name == "full_15s.wav":
        src = src.with_name(src.parent.name + ".wav")  # name outputs after the candidate
    n15, nbar = int(15.0 * sr), int(2.5 * sr)
    x = x[:n15]
    ctx = np.vstack([x, x[:nbar]])
    out_dir = ROOT / "output/cm/loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_path = out_dir / f"{src.stem}_ctx.wav"
    sf.write(ctx_path, ctx, sr, subtype="FLOAT")

    variant = src.stem.split("_")[0]
    caption = CAPTIONS.get(variant, CAPTIONS["plain"]).replace(
        "then resolves to a clean final chord with a tight ending",
        "keeps the groove going without stopping, seamless and continuous").replace(
        "Starts immediately, builds up, ", "")
    eng = ace_engine.load()
    tmp = ace_engine.generate(
        eng, out_dir / f"{src.stem}_repainted.wav", task_type="repaint", src_audio=str(ctx_path),
        repainting_start=a.start, repainting_end=15.0, chunk_mask_mode="explicit",
        caption=caption, lyrics="[Build-up]\n\n[Instrumental - continuous, no ending]", instrumental=True, bpm=BPM,
        timesignature="4", keyscale="D major", inference_steps=8, seed=a.seed, thinking=False,
    )
    y, ysr = sf.read(tmp, always_2d=True)
    sf.write(out_dir / f"{src.stem}.wav", y[: int(15.0 * ysr)], ysr, subtype="FLOAT")
    print(out_dir / f"{src.stem}.wav")


if __name__ == "__main__":
    main()
