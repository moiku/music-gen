"""Web UI + API (root venv).

    uv run python -m musicgen.server --host <IP to listen on> --port 8890

Serves the gallery page, stores ratings/notes in each take's meta.json, queues jobs, and
starts the ACE worker on demand (it exits by itself when idle, freeing ~33 GB).
"""

import argparse
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ACE_PYTHON, ENDINGS, JOBS, LENGTH_PRESETS, LIBRARY, ROOT, ending_of, snap_to_bars
from .library import (create_request, enqueue, find_item, list_jobs, list_requests, read_json,
                      update_json)

app = FastAPI(title="music-gen")
STATIC = Path(__file__).parent / "static"
LIBRARY.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)

_worker: subprocess.Popen | None = None
_lock = threading.Lock()


def ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or _worker.poll() is not None:
            log = open(LIBRARY / "_worker.log", "a")
            _worker = subprocess.Popen([str(ACE_PYTHON), "-m", "musicgen.worker"], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT)


# ---------- models ----------

class NewRequest(BaseModel):
    ending: str = "loop"
    seconds: float = 15
    bpm: float = 96
    key: str = ""
    caption: str = ""
    wa: bool = False
    n: int = 4
    title: str = ""
    order_text: str = ""     # the Japanese request, when settings came from /api/interpret
    explanation: str = ""


class Words(BaseModel):
    text: str
    n: int | None = None


class Rate(BaseModel):
    rating: int | None = None
    note: str | None = None
    fav: bool | None = None
    tags: list[str] | None = None
    archived: bool | None = None


class Finish(BaseModel):
    versions: list[str] = ["full", "short", "loop"]


class More(BaseModel):
    n: int = 4
    seconds: float | None = None  # set -> new request with another length


# ---------- API ----------

@app.get("/api/presets")
def presets():
    return {"endings": ENDINGS, "presets": LENGTH_PRESETS}


@app.get("/api/library")
def library():
    return list_requests()


@app.get("/api/jobs")
def jobs():
    alive = _worker is not None and _worker.poll() is None
    return {"worker_running": alive, "jobs": list_jobs()}


def _start(req: dict, n: int, parent: str | None = None) -> dict:
    bars, seconds = snap_to_bars(req["seconds"], req["bpm"])
    req = create_request({**req, "bars": bars, "seconds": seconds, "requested_seconds": req["seconds"]})
    job = enqueue("generate", request_id=req["id"], n=max(1, min(n, 12)), parent=parent)
    ensure_worker()
    return {"request": req, "job": job}


@app.post("/api/requests")
def new_request(r: NewRequest):
    if r.ending not in ENDINGS:
        raise HTTPException(400, "unknown ending")
    return _start(r.model_dump(exclude={"n"}), r.n)


def _item(iid: str) -> Path:
    d = find_item(iid)
    if d is None:
        raise HTTPException(404, "no such take")
    return d


@app.post("/api/items/{iid}/rate")
def rate(iid: str, r: Rate):
    fields = {k: v for k, v in r.model_dump().items() if v is not None}
    return update_json(_item(iid) / "meta.json", **fields)


@app.post("/api/items/{iid}/finish")
def finish(iid: str, f: Finish):
    d = _item(iid)
    job = enqueue("finish", item_dir=str(d), versions=f.versions, item=iid)
    ensure_worker()
    return job


@app.post("/api/interpret")
def interpret_order(w: Words):
    from .interpret import new_order
    try:
        return new_order(w.text)
    except Exception as e:
        raise HTTPException(502, f"解釈に失敗しました: {e}")


@app.post("/api/items/{iid}/refine")
def refine(iid: str, w: Words):
    """'この曲をもとに直す': Sonnet rewrites the settings, then new takes are generated."""
    from .interpret import change
    d = _item(iid)
    req, take = read_json(d.parent / "request.json"), read_json(d / "meta.json")
    try:
        p = change(w.text, req, take)
    except Exception as e:
        raise HTTPException(502, f"解釈に失敗しました: {e}")
    base = {k: p[k] for k in ("title", "ending", "seconds", "bpm", "key", "caption", "wa")}
    base.update(order_text=w.text, explanation=p["explanation"])
    return {**_start(base, w.n or p["n"], parent=iid), "interpretation": p}


@app.post("/api/items/{iid}/more")
def more(iid: str, m: More):
    d = _item(iid)
    req = read_json(d.parent / "request.json")
    base = {k: req[k] for k in ("bpm", "key", "caption", "wa", "title") if k in req}
    base["ending"] = ending_of(req)
    base["seconds"] = m.seconds if m.seconds else req["seconds"]
    base["title"] = (req.get("title") or "") + (f"（{base['seconds']:g}秒）" if m.seconds else "（バリエーション）")
    return _start(base, m.n, parent=iid)


# ---------- files ----------

app.mount("/files", StaticFiles(directory=LIBRARY), name="files")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8890)
    a = ap.parse_args()
    if any(j.get("status") in ("queued", "running") for j in list_jobs()):
        # jobs left from a previous run: mark running ones as interrupted, keep queued ones
        for p in JOBS.glob("j*.json"):
            j = read_json(p)
            if j and j.get("status") == "running":
                update_json(p, status="error", message="サーバーの再起動により中断しました")
        ensure_worker()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
