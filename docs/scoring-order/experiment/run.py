"""Run the traversal experiment (thorwhalen/essays#28): the same matrix scored row-wise,
column-wise, as one table, and one isolated call per cell, by several models, with the
alternatives and criteria presented in a shuffled order that is the same for every
traversal within a repetition.

    python run.py --models haiku sonnet --reps 6 [--cell-reps 3] [--jobs 4]

Results append to results.jsonl (one line per scored matrix or cell), so a rerun resumes.
Claude models run through the Claude Code subscription (`claude -p`, cost as reported by
the CLI); Ollama models run locally through aix (no cost).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from matrix import ALTERNATIVES, COMPANY, CRITERIA

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
CONDITIONS = ("row", "column", "table", "cell")
CLAUDE_MODELS = {"haiku", "sonnet", "opus"}
LOCAL_MODELS = {"mistral": "ollama/mistral:7b-instruct-q6_K", "llama3.2": "ollama/llama3.2:latest"}
SYSTEM = "You are a careful analyst scoring business opportunities. Follow the output format exactly."


def presentation(rep: int) -> tuple[list[str], list[str]]:
    rng = random.Random(1000 + rep)
    alts, crits = list(ALTERNATIVES), list(CRITERIA)
    rng.shuffle(alts)
    rng.shuffle(crits)
    return alts, crits


def _context(alts, crits) -> str:
    cases = "\n".join(f"{a}. {ALTERNATIVES[a][0]}: {ALTERNATIVES[a][1]}" for a in alts)
    rubric = "\n".join(f"- {c}: {CRITERIA[c]}" for c in crits)
    return f"{COMPANY}\n\nCandidate use cases:\n{cases}\n\nCriteria, each scored 1 to 5:\n{rubric}\n"


def matrix_prompt(condition: str, alts, crits) -> str:
    order = {
        "row": "Work use case by use case: score one use case on every criterion before moving to the next use case. "
               "Write one line per score, in this order: " + "; ".join(f"{a} on all criteria" for a in alts) + ".",
        "column": "Work criterion by criterion: score every use case on one criterion before moving to the next criterion. "
                  "Write one line per score, in this order: " + "; ".join(f"all use cases on {c}" for c in crits) + ".",
    }
    if condition == "table":
        return (_context(alts, crits) + "\nFill in the whole scoring matrix as a markdown table: one row per use case ("
                + ", ".join(alts) + "), one column per criterion (" + ", ".join(crits) + "), each cell a whole number 1 to 5. "
                "Output only the table.")
    return (_context(alts, crits) + "\n" + order[condition] + "\nEach line has exactly this form, nothing else: "
            "<letter> | <criterion> | <score>\nfor example: A | pain | 3")


def cell_prompt(alt: str, crit: str) -> str:
    return (f"{COMPANY}\n\nCandidate use case: {ALTERNATIVES[alt][0]}: {ALTERNATIVES[alt][1]}\n\n"
            f"Score it on one criterion, 1 to 5. {CRITERIA[crit]}\nAnswer with the number only.")


def ask(model: str, prompt: str) -> tuple[str, float, dict]:
    """The reply, its cost in USD (as the CLI reports it; 0 locally), and usage."""
    if model in CLAUDE_MODELS:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        out = subprocess.run(
            ["claude", "-p", "--setting-sources", "project", "--strict-mcp-config", "--disable-slash-commands",
             "--model", model, "--output-format", "json", "--no-session-persistence", "--tools", "",
             "--system-prompt", SYSTEM],
            input=prompt, capture_output=True, text=True, env=env, cwd=HERE, timeout=600,
        )
        d = json.loads(out.stdout)
        if d.get("is_error") or d.get("terminal_reason") not in (None, "completed"):
            raise RuntimeError(f"{model}: {d.get('terminal_reason')} {d.get('result')!r:.200}")
        return d["result"], float(d.get("total_cost_usd") or 0), d.get("usage", {})
    from aix.chat import chat

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    return chat(messages, model=LOCAL_MODELS[model], temperature=1.0, api_key="none"), 0.0, {}


LINE = re.compile(r"^\W*([A-H])\W*\|\s*\**([a-z]+)\**\s*\|\W*([1-5])\b", re.I | re.M)


def parse_lines(text: str) -> list[tuple[str, str, int]]:
    """(alt, crit, score) in the order written; a repeated cell keeps its first score."""
    seen, out = set(), []
    for a, c, s in LINE.findall(text):
        a, c = a.upper(), c.lower()
        if c in CRITERIA and (a, c) not in seen:
            seen.add((a, c))
            out.append((a, c, int(s)))
    return out


def parse_table(text: str, crits) -> list[tuple[str, str, int]]:
    rows = [r for r in text.splitlines() if r.strip().startswith("|")]
    header = [h.strip().strip("*").lower() for h in rows[0].strip().strip("|").split("|")] if rows else []
    cols = {i: h for i, h in enumerate(header) if h in CRITERIA}
    out = []
    for r in rows[1:]:
        cells = [x.strip().strip("*") for x in r.strip().strip("|").split("|")]
        m = re.match(r"([A-H])\b", cells[0]) if cells else None
        if not m:
            continue
        for i, c in cols.items():
            if i < len(cells) and re.fullmatch(r"[1-5]", cells[i]):
                out.append((m.group(1), c, int(cells[i])))
    return out


def done_keys() -> set:
    if not RESULTS.exists():
        return set()
    return {tuple(json.loads(l)["key"]) for l in RESULTS.read_text().splitlines() if l.strip()}


def job(key: tuple) -> dict:
    model, condition, rep, *cell = key
    alts, crits = presentation(rep)
    prompt = cell_prompt(*cell) if condition == "cell" else matrix_prompt(condition, alts, crits)
    started = time.time()
    for attempt in range(3):
        try:
            reply, cost, usage = ask(model, prompt)
            break
        except Exception as e:  # noqa: BLE001 - a flaky call is retried, then recorded
            reply, cost, usage, err = "", 0.0, {}, str(e)
            time.sleep(5 * (attempt + 1))
    if condition == "cell":
        m = re.search(r"[1-5]", reply)
        scores = [(cell[0], cell[1], int(m.group()))] if m else []
    elif condition == "table":
        scores = parse_table(reply, crits)
    else:
        # Some small models answer with a table whatever they are asked; the realised order
        # is recorded, and the analysis checks it against the instructed traversal.
        scores = parse_lines(reply) or parse_table(reply, crits)
    return {"key": list(key), "model": model, "condition": condition, "rep": rep, "scores": scores,
            "order": [f"{a}:{c}" for a, c, _ in scores], "cost_usd": cost, "seconds": round(time.time() - started, 1),
            "usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")},
            "reply": reply if condition != "cell" else reply[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku"])
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--cell-reps", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    keys = []
    for model in args.models:
        for rep in range(args.reps):
            keys += [(model, cond, rep) for cond in ("row", "column", "table")]
            if rep < args.cell_reps:
                keys += [(model, "cell", rep, a, c) for a in ALTERNATIVES for c in CRITERIA]
    todo = [k for k in keys if k not in done_keys()]
    print(f"{len(todo)} calls to make ({len(keys) - len(todo)} done)", flush=True)
    local = [k for k in todo if k[0] in LOCAL_MODELS]
    remote = [k for k in todo if k[0] not in LOCAL_MODELS]
    with RESULTS.open("a") as f, ThreadPoolExecutor(args.jobs) as pool:
        def record(rows):
            for row in rows:
                f.write(json.dumps(row) + "\n")
                f.flush()
        futures = pool.map(job, remote)
        record(job(k) for k in local)  # Ollama serves one request at a time anyway
        record(futures)


if __name__ == "__main__":
    main()
