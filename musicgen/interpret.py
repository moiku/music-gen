"""Turn a Japanese request (or a change request about an existing take) into generation settings.

Uses Claude Sonnet via the Anthropic API (the only external API in this project; the owner
asked for it on 2026-11-07). The key is read from ~/.secrets/api_keys.env, never from the shell.
ACE-Step's built-in LM was tested first and ignored length/instrument/difference instructions.
"""

import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .config import ENDINGS, ending_of

load_dotenv(dotenv_path=Path.home() / ".secrets" / "api_keys.env")

MODEL = "claude-sonnet-5-5"

SYSTEM = """You translate music requests for a local BGM generator into generation settings.
The owner writes in Japanese. The generator is ACE-Step 1.5 (instrumental only).

Every take starts on the first beat with no intro. ending (pick one):
- loop: steady groove whose end flows back into the start. Instagram Stories (15 s per card)
  and Reels (7 / 15 / 30 s, they replay automatically).
- button: builds up and resolves to a clean final chord. Commercials / brand films (15 or 30 s),
  or whenever the owner wants it to "end properly".
- auto: let the model decide (long pieces, or no preference).
seconds: use the length the owner names (5-180). Otherwise 15 for Stories, Reels and
commercials, 30 if nothing suggests a use. The system snaps it to whole bars.

caption (English, 1-3 sentences) is what the model listens to. Describe genre, instruments,
mood, energy and production (e.g. "Warm acoustic pop with fingerpicked guitar, soft brushed
drums and upright bass, cozy morning cafe mood, clean intimate mix"). Rules:
- Do not put BPM, key, duration, use or structure words like "loop"/"intro"/"ending" in the caption;
  the system adds those itself.
- Never name real artists, songs, brands or franchises; describe the sound instead.
- No vocals, no lyrics.
- Japanese instruments (koto, shakuhachi, taiko, shamisen) are switched on with `wa`, not
  written in the caption, unless the request asks for a specific one prominently (then also
  mention it in the caption).

bpm: integer 60-180 that fits the style (lo-fi 70-90, cafe/acoustic 85-105, pop 100-120,
house 118-126, upbeat ad 110-128). key: "" (let the model choose) unless the mood clearly calls
for one, written like "D major" or "A minor". n: number of takes, 4 unless asked.

When a previous take is given, change only what the request asks for and keep everything else
(including ending, seconds, key) as it was. Use the owner's note and rating as hints about
what to fix.

title: a short Japanese title for the gallery (e.g. 「カフェの朝・尺八」).

explanation: 1-2 short Japanese sentences telling the owner how you read the request
(what you set or changed and why). Plain Japanese, no English except instrument names.
Refer to endings by their on-screen names: loop = 「ループ」, button = 「きっちり終わる」,
auto = 「おまかせ」. Never write the field names (ending, wa, caption) in the explanation."""

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "ending": {"type": "string", "enum": list(ENDINGS)},
        "seconds": {"type": "number"},
        "bpm": {"type": "integer"},
        "key": {"type": "string"},
        "caption": {"type": "string"},
        "wa": {"type": "boolean"},
        "n": {"type": "integer"},
        "explanation": {"type": "string"},
    },
    "required": ["title", "ending", "seconds", "bpm", "key", "caption", "wa", "n", "explanation"],
    "additionalProperties": False,
}

_client: anthropic.Anthropic | None = None


def _ask(user_content: str) -> dict:
    global _client
    _client = _client or anthropic.Anthropic()
    resp = _client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request")
    text = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(text)
    out["seconds"] = float(min(max(out["seconds"], 5), 180))
    out["bpm"] = int(min(max(out["bpm"], 60), 180))
    out["n"] = int(min(max(out["n"], 1), 12))
    out["_usage"] = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
    return out


def new_order(text: str) -> dict:
    return _ask(f"Request:\n{text}")


def change(text: str, request: dict, take: dict) -> dict:
    prev = {k: request.get(k) for k in ("title", "seconds", "bpm", "key", "caption", "wa")}
    prev["ending"] = ending_of(request)
    prev["caption"] = request.get("caption", "")
    hints = {"rating_1to5": take.get("rating") or None, "owner_note": take.get("note") or None,
             "auto_check": take.get("verdict")}
    return _ask("Previous settings:\n" + json.dumps(prev, ensure_ascii=False)
                + "\nAbout the take the owner is reacting to:\n" + json.dumps(hints, ensure_ascii=False)
                + f"\nChange request:\n{text}")


if __name__ == "__main__":
    import sys
    print(json.dumps(new_order(" ".join(sys.argv[1:])), ensure_ascii=False, indent=1))
