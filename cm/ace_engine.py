"""Thin wrapper around the official ACE-Step 1.5 Python API (runs inside the ACE venv).

- Loads XL-turbo (MLX DiT) + 5Hz LM (MLX) once.
- Frees the PyTorch copy of the DiT decoder after the MLX copy is built
  (measured 2026-11-07: process footprint 43 GB -> 33 GB, generation unaffected).
"""

import gc
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACE = ROOT / "vendor/ACE-Step-1.5"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class Engine:
    dit: object
    llm: object


def load(dit_model: str = "acestep-v15-xl-turbo", lm_model: str = "acestep-5Hz-lm-1.7B") -> Engine:
    cwd = os.getcwd()
    os.chdir(ACE)
    try:
        import torch
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        dit = AceStepHandler()
        dit.initialize_service(project_root=str(ACE), config_path=dit_model, device="auto",
                               use_mlx_dit=True)
        if getattr(dit, "use_mlx_dit", False) and getattr(dit, "mlx_decoder", None) is not None:
            dit.model.decoder.to("meta")  # MLX path is used for diffusion; drop torch weights
            gc.collect()
            torch.mps.empty_cache()
        llm = LLMHandler()
        llm.initialize(checkpoint_dir=str(ACE / "checkpoints"), lm_model_path=lm_model,
                       backend="mlx", device="auto")
        return Engine(dit, llm)
    finally:
        os.chdir(cwd)


def generate(engine: Engine, out_path: Path, **params) -> Path:
    """Generate one clip. `params` are GenerationParams fields (caption, lyrics, bpm, ...)."""
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    seed = params.get("seed", -1)
    tmp = out_path.parent / f"_tmp_{out_path.stem}"
    res = generate_music(
        engine.dit, engine.llm, GenerationParams(**params),
        GenerationConfig(batch_size=1, use_random_seed=seed < 0,
                         seeds=None if seed < 0 else [seed], audio_format="wav"),
        save_dir=str(tmp),
    )
    if not res.success:
        raise RuntimeError(res.error)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(res.audios[0]["path"], out_path)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_path
