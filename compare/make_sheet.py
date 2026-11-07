"""Build a blind listening page from output/compare/raw/*.wav.

- Loudness-matches every clip to -14 LUFS (gain only; capped so sample peak <= -1 dBFS)
  so that louder clips do not win by loudness alone.
- Shuffles clips per prompt under neutral labels (A, B, C, ...).
- Computes simple objective numbers (sample rate, stereo width, high-frequency content).

    uv run python compare/make_sheet.py
    open output/compare/blind/index.html
"""

import html
import json
import random
import string
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "output/compare/raw"
BLIND = ROOT / "output/compare/blind"
TARGET_LUFS = -14.0


def limit(x: np.ndarray, sr: int, ceiling_db: float = -1.0) -> np.ndarray:
    """Simple look-ahead peak limiter (5 ms look-ahead/attack, 80 ms release), stereo-linked.

    Only touches peaks above the ceiling, so all clips can reach the same LUFS
    without clipping and without changing the overall tone.
    """
    ceil = 10 ** (ceiling_db / 20)
    peak = np.abs(x).max(axis=1)
    need = np.minimum(1.0, ceil / np.maximum(peak, 1e-12))
    la = int(0.005 * sr)
    # look-ahead: gain at t must already satisfy the requirement up to t+la
    padded = np.concatenate([need, np.ones(la)])
    g = np.lib.stride_tricks.sliding_window_view(padded, la + 1).min(axis=1)
    rel = np.exp(-1.0 / (0.08 * sr))
    out = np.empty_like(g)
    cur = 1.0
    for i, v in enumerate(g):  # instant attack (already look-ahead), smooth release
        cur = v if v < cur else v + (cur - v) * rel
        out[i] = cur
    return np.clip(x * out[:, None], -ceil, ceil)


def analyze(x: np.ndarray, sr: int) -> dict:
    if x.ndim == 1:
        x = x[:, None]
    stereo = x.shape[1] >= 2
    mono = x.mean(axis=1)
    out = {"sr": sr, "channels": x.shape[1]}
    if stereo:
        mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
        out["side_mid_db"] = round(20 * np.log10(np.std(side) / (np.std(mid) + 1e-12) + 1e-12), 1)
        out["lr_corr"] = round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 3)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    cum = np.cumsum(spec) / spec.sum()
    out["rolloff99_khz"] = round(float(freqs[np.searchsorted(cum, 0.99)]) / 1000, 1)
    out["energy_above_12k_db"] = round(10 * np.log10(spec[freqs > 12000].sum() / spec.sum() + 1e-12), 1)
    return out


def main() -> None:
    BLIND.mkdir(parents=True, exist_ok=True)
    clips = sorted(RAW.glob("*__*__s*.wav"))
    by_prompt: dict[str, list[Path]] = {}
    for c in clips:
        by_prompt.setdefault(c.stem.split("__")[1], []).append(c)

    rng = random.Random(20261106)
    key, rows = {}, []
    for prompt, files in sorted(by_prompt.items()):
        rng.shuffle(files)
        for label, f in zip(string.ascii_uppercase, files):
            x, sr = sf.read(f, always_2d=True)
            lufs = pyln.Meter(sr).integrated_loudness(x)
            gain_db = TARGET_LUFS - lufs
            y = limit(x * 10 ** (gain_db / 20), sr)
            gain_db = pyln.Meter(sr).integrated_loudness(y) - lufs
            name = f"{prompt}_{label}.wav"
            sf.write(BLIND / name, y, sr, subtype="PCM_24")
            model, _, seed = f.stem.split("__")
            info = {"prompt": prompt, "label": label, "file": name, "model": model,
                    "seed": seed, "orig_lufs": round(lufs, 1),
                    "out_lufs": round(lufs + gain_db, 1), **analyze(x, sr)}
            key[f"{prompt}_{label}"] = info
            rows.append(info)

    (BLIND / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    prompts = json.loads((ROOT / "compare/prompts.json").read_text())["prompts"]
    captions = {p["id"]: p["caption"] for p in prompts}

    sections = []
    for prompt in sorted(by_prompt):
        items = [r for r in rows if r["prompt"] == prompt]
        cells = "".join(
            f"""<div class="clip"><b>{r['label']}</b><audio controls preload="none" src="{r['file']}"></audio>
<select data-id="{prompt}_{r['label']}"><option value="">評価</option>{''.join(f'<option>{i}</option>' for i in range(1, 6))}</select>
<input data-note="{prompt}_{r['label']}" placeholder="メモ（奥行き・音質など）">
<span class="ans" hidden>{html.escape(r['model'])} / {r['seed']} · side/mid {r.get('side_mid_db', '-')} dB · 12kHz以上 {r['energy_above_12k_db']} dB · {r['sr']} Hz</span></div>"""
            for r in sorted(items, key=lambda r: r["label"]))
        sections.append(f"<h2>{prompt}</h2><p class=cap>{html.escape(captions.get(prompt, ''))}</p>{cells}")

    page = f"""<!doctype html><meta charset="utf-8"><title>BGM blind test</title>
<style>body{{font:15px system-ui;max-width:900px;margin:2em auto;padding:0 16px}}
.clip{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}}
audio{{height:32px}} input{{flex:1;min-width:200px}} .cap{{color:#666}} .ans{{color:#a33;font-size:13px}}</style>
<h1>BGM ブラインド聴き比べ</h1>
<p>全クリップを -14 LUFS に揃えてある。1（悪い）〜5（良い）で評価し、最後に「答え合わせ」。入力はこのブラウザに自動保存。</p>
{''.join(sections)}
<p><button id=reveal>答え合わせ</button> <button id=csv>CSVを保存</button></p>
<pre id=summary></pre>
<script>
const KEY={json.dumps({k: v['model'] for k, v in key.items()})};
const ls=(k,v)=>{{try{{return v===undefined?localStorage.getItem(k):localStorage.setItem(k,v)}}catch(e){{}}}};
document.querySelectorAll('select,input').forEach(el=>{{const k='bt:'+(el.dataset.id||'n:'+el.dataset.note);
 const v=ls(k); if(v) el.value=v; el.addEventListener('change',()=>ls(k,el.value));}});
document.getElementById('reveal').onclick=()=>{{document.querySelectorAll('.ans').forEach(e=>e.hidden=false);
 const agg={{}};document.querySelectorAll('select').forEach(s=>{{if(!s.value)return;const m=KEY[s.dataset.id];
 (agg[m]=agg[m]||[]).push(+s.value)}});
 document.getElementById('summary').textContent=Object.entries(agg).map(([m,a])=>m+': 平均 '+(a.reduce((x,y)=>x+y,0)/a.length).toFixed(2)+' (n='+a.length+')').join('\\n');}};
document.getElementById('csv').onclick=()=>{{let t='clip,model,score,note\\n';document.querySelectorAll('select').forEach(s=>{{
 const n=document.querySelector('[data-note="'+s.dataset.id+'"]').value.replaceAll('"','""');
 t+=s.dataset.id+','+KEY[s.dataset.id]+','+s.value+',"'+n+'"\\n'}});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([t],{{type:'text/csv'}}));a.download='ratings.csv';a.click();}};
</script>"""
    (BLIND / "index.html").write_text(page)

    # Objective summary per model (printed; also in key.json)
    print(f"{'model':28s} {'n':>2} {'sr':>6} {'side/mid dB':>11} {'LRcorr':>6} {'>12kHz dB':>9} {'LUFS':>6}")
    for m in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == m]
        avg = lambda k: np.mean([r[k] for r in rs if k in r]) if any(k in r for r in rs) else float("nan")
        print(f"{m:28s} {len(rs):>2} {rs[0]['sr']:>6} {avg('side_mid_db'):>11.1f} {avg('lr_corr'):>6.2f} "
              f"{avg('energy_above_12k_db'):>9.1f} {avg('orig_lufs'):>6.1f}")


if __name__ == "__main__":
    main()
