"""Generate comparison clips with Stable Audio 3 (official MLX runtime).

Run with any Python (it shells out to the SA3 CLI in vendor/):
    uv run python compare/run_sa3.py [--dit medium]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SA3 = ROOT / "vendor/stable-audio-3/optimized/mlx/sa3"
OUT = ROOT / "output/compare/raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default="medium", choices=["medium", "sm-music"])
    args = ap.parse_args()
    decoder = "same-l" if args.dit == "medium" else "same-s"
    cfg = json.loads((ROOT / "compare/prompts.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    for p in cfg["prompts"]:
        # SA3 has no BPM/key parameters, so they go into the text prompt.
        prompt = f"{p['caption']}, {p['bpm']} BPM, {p['key']}"
        for seed in cfg["seeds"]:
            out = OUT / f"sa3-{args.dit}__{p['id']}__s{seed}.wav"
            t = time.time()
            subprocess.run(
                [str(SA3), "--prompt", prompt, "--dit", args.dit, "--decoder", decoder,
                 "--seconds", str(cfg["seconds"]), "--seed", str(seed), "--out", str(out)],
                check=True, stdin=subprocess.DEVNULL,
            )
            # Wall time includes model loading (the CLI reloads per call).
            log.append({"model": f"sa3-{args.dit}", "prompt": p["id"], "seed": seed,
                        "path": str(out), "wall_s": round(time.time() - t, 1)})
            print(log[-1], flush=True)
    (OUT / f"log_sa3-{args.dit}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
