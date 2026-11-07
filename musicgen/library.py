"""On-disk gallery. Stdlib only.

library/
  <request_id>/request.json         the order (ending, bpm, seconds, caption, ...)
  <request_id>/<item_id>/meta.json   one generated take: params, analysis, rating, note, fav, ...
  <request_id>/<item_id>/raw.wav     as generated (longer than target)
  <request_id>/<item_id>/take.wav    cut to the target length (what the gallery plays)
  <request_id>/<item_id>/final/      finished versions (full / short / loop / stems / click check)
  _jobs/<job_id>.json                work queue (status: queued / running / done / error)
"""

import json
import os
import secrets
import time
from pathlib import Path

from .config import JOBS, LIBRARY


def new_id(prefix: str) -> str:
    return f"{prefix}{time.strftime('%y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=float))
    os.replace(tmp, p)  # atomic: server and worker read these concurrently


def update_json(p: Path, **fields) -> dict:
    data = read_json(p, {}) or {}
    data.update(fields)
    write_json(p, data)
    return data


# ---------- requests & items ----------

def create_request(req: dict) -> dict:
    rid = new_id("r")
    req = {**req, "id": rid, "created": time.time()}
    write_json(LIBRARY / rid / "request.json", req)
    return req


def request_dir(rid: str) -> Path:
    return LIBRARY / rid


def item_dir(rid: str, iid: str) -> Path:
    return LIBRARY / rid / iid


def find_item(iid: str) -> Path | None:
    for d in LIBRARY.glob(f"*/{iid}"):
        if (d / "meta.json").exists():
            return d
    return None


def list_requests() -> list[dict]:
    out = []
    for rd in sorted(LIBRARY.glob("r*/request.json"), reverse=True):
        req = read_json(rd)
        if not req:
            continue
        items = []
        for md in sorted(rd.parent.glob("*/meta.json")):
            meta = read_json(md)
            if meta:
                items.append(meta)
        out.append({**req, "items": items})
    return out


# ---------- jobs ----------

def enqueue(kind: str, **payload) -> dict:
    job = {"id": new_id("j"), "kind": kind, "status": "queued", "created": time.time(), **payload}
    write_json(JOBS / f"{job['id']}.json", job)
    return job


def list_jobs(limit: int = 30) -> list[dict]:
    jobs = [read_json(p) for p in sorted(JOBS.glob("j*.json"), reverse=True)[:limit]]
    return [j for j in jobs if j]


def next_job(kinds: tuple[str, ...]) -> Path | None:
    for p in sorted(JOBS.glob("j*.json")):
        j = read_json(p)
        if j and j.get("status") == "queued" and j.get("kind") in kinds:
            return p
    return None
