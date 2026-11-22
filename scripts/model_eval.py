"""Ask language models every CLadder v1.5 question and keep their yes/no answers (OpenRouter chat API).

    OPENROUTER_API_KEY=... python -B scripts/model_eval.py --model qwen/qwen3-32b [--workers 16] [--budget 15]

One request per task: the CLadder prompt followed by INSTRUCTION, temperature 0, seed 1. The answer is the reply's
first word when it is "yes" or "no", else its last standalone "yes" or "no" (case-insensitive); anything else is "invalid". Results are appended to
results/model_eval/<model>.jsonl (id, answer, raw reply, tokens, cost) and the run resumes from that file. The run stops
when the summed cost of all models' files reaches --budget (US dollars, as reported by OpenRouter).
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cladder" / "full_v1.5_default.csv"
OUT = ROOT / "results" / "model_eval"
INSTRUCTION = 'Answer the question with only "Yes" or "No".'
ANSWER = re.compile(r"\b(yes|no)\b", re.I)
MAX_TOKENS = {"qwen/qwen3-32b": 4096}
EXTRA = {"anthropic/claude-haiku-5.5": {"reasoning": {"enabled": False}}}  # answer directly, no thinking tokens


def parse(text: str) -> str:
    """The reply's first word when it is yes/no, else its last standalone yes/no, else 'invalid'."""
    first = re.match(r"\W*(yes|no)\b", text, re.I)
    if first:
        return first.group(1).lower()
    found = ANSWER.findall(text)
    return found[-1].lower() if found else "invalid"


assert [parse(s) for s in ("No.", " yes", "So P = 0.3, hence the answer is No", "Yesterday", "")] == \
    ["no", "yes", "no", "invalid", "invalid"]


def spent() -> float:
    total = 0.0
    for f in OUT.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            total += json.loads(line).get("cost") or 0.0
    return total


def ask(model: str, prompt: str, key: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": f"{prompt}\n{INSTRUCTION}"}],
            "temperature": 0, "seed": 1, "max_tokens": MAX_TOKENS.get(model, 256), "usage": {"include": True},
            **EXTRA.get(model, {})}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == 5:
                return {"error": str(exc)}
            time.sleep(2 ** attempt)
    if "choices" not in data:
        return {"error": json.dumps(data)[:300]}
    text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage", {})
    return {"answer": parse(text), "raw": text[:200],
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (args.model.replace("/", "__") + ".jsonl")
    done = {json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    csv.field_size_limit(10**8)
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["id"] not in done][: args.limit]
    lock, total = threading.Lock(), [spent()]
    print(f"{args.model}: {len(done)} done, {len(rows)} to ask, ${total[0]:.3f} spent so far", flush=True)
    stop = threading.Event()

    def one(row):
        if stop.is_set():
            return
        result = ask(args.model, row["prompt"], key)
        if "error" in result:
            print(f"{row['id']}: {result['error']}", flush=True)
            return
        with lock:
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"id": row["id"], **result}) + "\n")
            total[0] += result.get("cost") or 0.0
            if total[0] >= args.budget:
                stop.set()

    with ThreadPoolExecutor(args.workers) as pool:
        for n, _ in enumerate(as_completed([pool.submit(one, r) for r in rows]), 1):
            if n % 500 == 0:
                print(f"{n}/{len(rows)} ${total[0]:.3f}", flush=True)
    print(f"{args.model}: finished, ${total[0]:.3f} spent in all files; budget stop: {stop.is_set()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
