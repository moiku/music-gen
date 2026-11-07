"""Generate candidate tracks for the Egatari 15-second CM brief
(egatari/handoff/cm-music-brief-2026-11-06.md).

    vendor/ACE-Step-1.5/.venv/bin/python cm/gen_candidates.py --n 12
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ace_engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/cm/candidates"

BPM = 96
SECONDS = 15.0  # 6 bars of 4/4 at 96 BPM

# Two caption variants: with and without Japanese instruments (brief: "候補").
CAPTIONS = {
    "wa": ("Bright, warm and playful instrumental for a short brand film. Koto plucks and a light "
           "shakuhachi-style flute melody over acoustic guitar, marimba and soft piano, crisp claps "
           "and small taiko accents, gentle bass. Starts immediately, builds up, then resolves to a "
           "clean final chord with a tight ending. Fresh and human, not a generic corporate track."),
    "plain": ("Bright, warm and playful instrumental for a short brand film. Acoustic guitar, "
              "marimba, glockenspiel sparkles and soft piano, crisp claps and light percussion, "
              "gentle bass. Starts immediately, builds up, then resolves to a clean final chord "
              "with a tight ending. Fresh and human, not a generic corporate track."),
}
# Structure tags (official songwriting guide); instrumental=True keeps it vocal-free.
LYRICS = "[Intro - bright, playful]\n\n[Build-up]\n\n[Outro - resolving final chord, tight ending]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="candidates per caption variant / 2")
    ap.add_argument("--durations", type=float, nargs="+", default=[SECONDS],
                    help="generation lengths; ACE tends to end ~2.5 s before the requested length")
    ap.add_argument("--variants", nargs="+", default=list(CAPTIONS))
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    eng = ace_engine.load()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    log = []
    seeds = [args.seed0 + i for i in range(args.n // 2)]
    for variant in args.variants:
        caption = CAPTIONS[variant]
        for seed, dur in [(s, d) for s in seeds for d in args.durations]:
            out = args.out / f"{variant}_s{seed}_d{dur:g}.wav"
            t = time.time()
            ace_engine.generate(eng, out, caption=caption, lyrics=LYRICS, instrumental=True,
                                bpm=BPM, timesignature="4", keyscale="D major",
                                duration=dur, inference_steps=8, seed=seed)
            log.append({"file": out.name, "variant": variant, "seed": seed, "duration": dur,
                        "gen_s": round(time.time() - t, 1)})
            print(log[-1], flush=True)
    (args.out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
