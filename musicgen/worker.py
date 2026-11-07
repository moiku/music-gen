"""Job worker (runs in the ACE venv): generate takes, then hand analysis/finishing to the root venv.

    vendor/ACE-Step-1.5/.venv/bin/python -m musicgen.worker

Loads ACE-Step lazily on the first generate job and exits after IDLE_UNLOAD_S without work,
which releases all model memory (~33 GB); the server starts it again on the next job.
"""

import subprocess
import sys
import time
import traceback
from pathlib import Path

from .config import GEN_MARGIN_S, IDLE_UNLOAD_S, ROOT, ROOT_PYTHON, ending_of
from .library import item_dir, new_id, next_job, read_json, request_dir, update_json, write_json

WA = ("Light Japanese instruments woven in: koto plucks, shakuhachi-style flute phrases, "
      "small taiko accents. ")
COMMON = "Starts immediately on the first beat with no intro. "
STYLE = {  # keyed by config.ENDINGS
    "loop": ("Steady groove throughout, loop-friendly, no fade-out. ",
             "[Instrumental - immediate start, steady loopable groove]"),
    "button": ("Builds up, then resolves to a clean final chord with a tight ending. ",
               "[Intro - immediate start]\n\n[Build-up]\n\n[Outro - resolving final chord, tight ending]"),
    "auto": ("", "[Instrumental - immediate start]"),
}


def build_prompt(req: dict) -> tuple[str, str]:
    style, lyrics = STYLE[ending_of(req)]
    caption = req.get("caption", "").strip()
    caption = (caption + (" " if caption else "") + (WA if req.get("wa") else "") + COMMON + style
               + "Instrumental, no vocals.")
    return caption.strip(), lyrics


def run_analysis(d: Path) -> None:
    subprocess.run([str(ROOT_PYTHON), "-m", "musicgen.analysis", str(d)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def do_generate(job_path: Path, job: dict, engine_box: dict) -> None:
    from . import ace_engine
    req = read_json(request_dir(job["request_id"]) / "request.json")
    if engine_box.get("engine") is None:
        update_json(job_path, status="running", message="AIモデルを読み込み中（約1分）")
        engine_box["engine"] = ace_engine.load()
    caption, lyrics = build_prompt(req)
    n = int(job.get("n", 4))
    seed0 = int(job.get("seed0", int(time.time()) % 100000))
    done = []
    for k in range(n):
        update_json(job_path, status="running", message=f"{k + 1}/{n} 曲目を作成中", progress=k / n)
        iid = new_id("t")
        d = item_dir(req["id"], iid)
        meta = {"id": iid, "request_id": req["id"], "parent": job.get("parent"),
                "bpm": req["bpm"], "seconds": req["seconds"], "bars": req["bars"],
                "seed": seed0 + k, "caption": caption, "lyrics": lyrics, "model": "acestep-v15-xl-turbo",
                "status": "generating", "created": time.time(), "rating": 0, "fav": False,
                "note": "", "tags": [], "archived": False}
        write_json(d / "meta.json", meta)
        t = time.time()
        ace_engine.generate(engine_box["engine"], d / "raw.wav", caption=caption, lyrics=lyrics,
                            instrumental=True, bpm=int(round(req["bpm"])), timesignature="4",
                            keyscale=req.get("key") or "", duration=req["seconds"] + GEN_MARGIN_S,
                            inference_steps=8, seed=seed0 + k)
        update_json(d / "meta.json", status="analyzing", gen_s=round(time.time() - t, 1))
        run_analysis(d)
        done.append(iid)
    update_json(job_path, status="done", message=f"{n} 曲を作成しました", progress=1.0, items=done)


def do_finish(job_path: Path, job: dict) -> None:
    update_json(job_path, status="running", message="書き出し中（ステムありは1分ほど）")
    subprocess.run([str(ROOT_PYTHON), "-m", "musicgen.finish", job["item_dir"], *job["versions"]],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    update_json(job_path, status="done", message="書き出しが完了しました", progress=1.0)


def main() -> None:
    engine_box: dict = {}
    last = time.time()
    while True:
        p = next_job(("generate", "finish"))
        if p is None:
            if time.time() - last > IDLE_UNLOAD_S:
                print("idle: exiting to free memory", flush=True)
                return
            time.sleep(1)
            continue
        job = read_json(p)
        try:
            if job["kind"] == "generate":
                do_generate(p, job, engine_box)
            else:
                do_finish(p, job)
        except Exception as e:  # keep the worker alive; show the error in the UI
            traceback.print_exc()
            update_json(p, status="error", message=f"{type(e).__name__}: {e}")
        last = time.time()


if __name__ == "__main__":
    sys.exit(main())
