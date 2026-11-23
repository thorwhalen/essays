"""Analyse results.jsonl (thorwhalen/essays#28) and write results.md.

Per model and traversal: complete matrices, whether the realised order matched the
instructed one, the mean score, error against the designed scores, two halo measures
(inter-criterion correlation across use cases, and spread of one use case's scores across
criteria), ranking stability between repetitions, and agreement with the other traversals.
Row against column is tested by permutation over matrices.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from matrix import ALTERNATIVES, CRITERIA, DESIGNED

HERE = Path(__file__).resolve().parent
ALTS, CRITS = list(ALTERNATIVES), list(CRITERIA)
PERMUTATIONS = 10_000


def as_array(scores) -> np.ndarray | None:
    m = np.full((len(ALTS), len(CRITS)), np.nan)
    for a, c, s in scores:
        m[ALTS.index(a), CRITS.index(c)] = s
    return None if np.isnan(m).any() else m


DESIGN = as_array([(a, c, s) for a, row in DESIGNED.items() for c, s in row.items()])


def halo_corr(m: np.ndarray) -> float:
    """Mean correlation between criteria across use cases (constant columns skipped)."""
    vals = []
    for i, j in combinations(range(m.shape[1]), 2):
        x, y = m[:, i], m[:, j]
        if x.std() and y.std():
            vals.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(vals)) if vals else float("nan")


def spread(m: np.ndarray) -> float:
    """Mean standard deviation of a use case's scores across criteria (low = halo)."""
    return float(m.std(axis=1).mean())


def realised(order: list[str]) -> str:
    """'row' when scores came grouped by use case, 'column' when grouped by criterion."""
    alts = [o.split(":")[0] for o in order]
    crits = [o.split(":")[1] for o in order]
    changes = lambda seq: sum(a != b for a, b in zip(seq, seq[1:]))
    if changes(alts) == len(ALTS) - 1:
        return "row"
    if changes(crits) == len(CRITS) - 1:
        return "column"
    return "mixed"


def load():
    rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]
    matrices = defaultdict(list)  # (model, condition) -> [(rep, array, realised)]
    cells = defaultdict(list)
    cost = Counter()
    calls = Counter()
    for r in rows:
        cost[r["model"]] += r["cost_usd"]
        calls[r["model"]] += 1
        if r["condition"] == "cell":
            cells[(r["model"], r["rep"])] += r["scores"]
        else:
            m = as_array(r["scores"])
            matrices[(r["model"], r["condition"])].append((r["rep"], m, realised(r["order"]) if m is not None else None))
    for (model, rep), scores in cells.items():
        matrices[(model, "cell")].append((rep, as_array(scores), "cell"))
    return matrices, cost, calls


def perm_test(a: list[float], b: list[float]) -> float:
    rng = random.Random(0)
    observed = abs(np.mean(a) - np.mean(b))
    pooled = a + b
    hits = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pooled)
        hits += abs(np.mean(pooled[: len(a)]) - np.mean(pooled[len(a):])) >= observed
    return (hits + 1) / (PERMUTATIONS + 1)


def mean_tau(ms: list[np.ndarray]) -> float:
    taus = [kendalltau(x.sum(1), y.sum(1)).statistic for x, y in combinations(ms, 2)]
    return float(np.nanmean(taus)) if taus else float("nan")


def main():
    matrices, cost, calls = load()
    models = sorted({m for m, _ in matrices}, key=lambda m: ["haiku", "sonnet", "mistral", "llama3.2"].index(m) if m in ["haiku", "sonnet", "mistral", "llama3.2"] else 9)
    conds = ["row", "column", "table", "cell"]
    out = ["# Results: does traversal order change LLM scores?", "",
           f"Designed matrix: mean inter-criterion correlation {halo_corr(DESIGN):.2f}, spread {spread(DESIGN):.2f}, totals "
           + ", ".join(f"{a} {int(t)}" for a, t in zip(ALTS, DESIGN.sum(1))) + ".", ""]
    out += ["| model | traversal | complete | as instructed | mean | MAE vs design | halo r | spread | rank tau within | winner |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for model in models:
        for cond in conds:
            got = matrices.get((model, cond), [])
            ok = [(rep, m, how) for rep, m, how in got if m is not None]
            if not got:
                continue
            arrays = [m for _, m, _ in ok]
            instructed = sum(how == cond for _, _, how in ok) if cond in ("row", "column") else len(ok)
            summary[(model, cond)] = arrays
            if not arrays:
                out.append(f"| {model} | {cond} | 0/{len(got)} | | | | | | | |")
                continue
            winners = Counter(ALTS[int(np.argmax(m.sum(1)))] for m in arrays)
            out.append(
                f"| {model} | {cond} | {len(arrays)}/{len(got)} | {instructed}/{len(arrays)} | {np.mean(arrays):.2f} | "
                f"{np.mean([np.abs(m - DESIGN).mean() for m in arrays]):.2f} | {np.nanmean([halo_corr(m) for m in arrays]):.2f} | "
                f"{np.mean([spread(m) for m in arrays]):.2f} | {mean_tau(arrays):.2f} | "
                + ", ".join(f"{a}×{n}" for a, n in winners.most_common(3)) + " |")
    out += ["", "## Row against column (permutation test over matrices, two-sided)", "",
            "| model | halo r row | halo r column | p | spread row | spread column | p | MAE row | MAE column | p | tau(row mean, column mean) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in models:
        r, c = summary.get((model, "row"), []), summary.get((model, "column"), [])
        if len(r) < 2 or len(c) < 2:
            continue
        hr, hc = [halo_corr(m) for m in r], [halo_corr(m) for m in c]
        sr, sc = [spread(m) for m in r], [spread(m) for m in c]
        er, ec = [np.abs(m - DESIGN).mean() for m in r], [np.abs(m - DESIGN).mean() for m in c]
        tau = kendalltau(np.mean(r, axis=0).sum(1), np.mean(c, axis=0).sum(1)).statistic
        out.append(f"| {model} | {np.mean(hr):.2f} | {np.mean(hc):.2f} | {perm_test(hr, hc):.3f} | {np.mean(sr):.2f} | {np.mean(sc):.2f} | "
                   f"{perm_test(sr, sc):.3f} | {np.mean(er):.2f} | {np.mean(ec):.2f} | {perm_test(er, ec):.3f} | {tau:.2f} |")
    out += ["", "## Agreement between traversals (Kendall tau of mean totals) and with the design", ""]
    for model in models:
        means = {c: np.mean(summary[(model, c)], axis=0) for c in conds if summary.get((model, c))}
        pairs = [f"{a}-{b} {kendalltau(means[a].sum(1), means[b].sum(1)).statistic:.2f}" for a, b in combinations(means, 2)]
        design = [f"{c}-design {kendalltau(means[c].sum(1), DESIGN.sum(1)).statistic:.2f}" for c in means]
        out.append(f"- **{model}**: " + "; ".join(pairs + design))
    out += ["", "## Calls and cost", "", "| model | calls | cost (USD, as reported) |", "|---|---|---|"]
    out += [f"| {m} | {calls[m]} | {cost[m]:.3f} |" for m in models]
    (HERE / "results.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
