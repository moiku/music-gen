"""Generate comparison clips with ACE-Step 1.5 (official repo, MLX DiT + MLX LM).

Must run inside the ACE-Step venv:
    vendor/ACE-Step-1.5/.venv/bin/python compare/run_ace.py --dit acestep-v15-xl-turbo
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACE = ROOT / "vendor/ACE-Step-1.5"
OUT = ROOT / "output/compare/raw"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.chdir(ACE)

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402

# Official recommendation: turbo = 8 steps without CFG, sft/base = 50 steps with CFG.
STEPS = {"turbo": 8, "sft": 50, "base": 50}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default="acestep-v15-xl-turbo")
    ap.add_argument("--lm", default="acestep-5Hz-lm-1.7B")
    args = ap.parse_args()
    kind = args.dit.rsplit("-", 1)[-1]
    tag = args.dit.replace("acestep-v15-", "ace-")
    cfg = json.loads((ROOT / "compare/prompts.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"_tmp_{tag}"

    t0 = time.time()
    dit = AceStepHandler()
    print(dit.initialize_service(project_root=str(ACE), config_path=args.dit, device="auto",
                                 use_mlx_dit=True))
    llm = LLMHandler()
    print(llm.initialize(checkpoint_dir=str(ACE / "checkpoints"), lm_model_path=args.lm,
                         backend="mlx", device="auto"))
    load_s = round(time.time() - t0, 1)

    log = []
    for p in cfg["prompts"]:
        for seed in cfg["seeds"]:
            params = GenerationParams(
                caption=p["caption"], lyrics="[Instrumental]", instrumental=True,
                bpm=p["bpm"], keyscale=p["key"], timesignature="4",
                duration=cfg["seconds"], inference_steps=STEPS[kind], seed=seed,
            )
            conf = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed],
                                    audio_format="wav")
            t = time.time()
            res = generate_music(dit, llm, params, conf, save_dir=str(tmp))
            if not res.success:
                raise RuntimeError(res.error)
            out = OUT / f"{tag}__{p['id']}__s{seed}.wav"
            shutil.move(res.audios[0]["path"], out)
            log.append({"model": tag, "prompt": p["id"], "seed": seed, "path": str(out),
                        "gen_s": round(time.time() - t, 1), "load_s": load_s})
            print(log[-1], flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    (OUT / f"log_{tag}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
