"""Pool runs into conditions and report means with bootstrap CIs and paired tests.

    python -m goalnet_general.evaluate.aggregate goalnet_results/*/ [--compare LABEL_A LABEL_B] [--json OUT]

Each directory is one unit of analysis: a construction run (scores.json) on one
experience set, averaged over its repeats, a maintenance replay (trajectory.json),
or a closed-loop run (loop.json, one per seed).
Variation therefore comes from different experience sets (different seeds),
not from re-asking the API at temperature 0.

Units are grouped by their label (run_construct --label, default mode/condition/
vocab/recipes) and task. A group must share one prompt fingerprint: results from
different prompt versions are never pooled. Test-task runs made with an unfrozen
prompt are left out unless --include-unfrozen (see splits.py).

--compare A B tests B - A per task. Units with the same task and experience runs
are paired (Wilcoxon signed-rank, bootstrap CI of the mean paired difference);
otherwise the groups are compared unpaired (Mann-Whitney U, bootstrap CI of the
difference of means). Zero-experience units have no runs, so comparisons against
them are unpaired.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

BOOTSTRAP = 10_000
SEED = 0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def construction_metrics(summary):
    """Per-unit metrics: each is the mean over the run's repeats of the final graph."""
    finals = [r["final"] for r in summary["results"]]

    def get(state, *path):
        for p in path:
            if not isinstance(state, dict) or p not in state:
                return None
            state = state[p]
        return state

    m = {
        "valid_initial": summary["validity_rate_initial"],
        "valid_final": summary["validity_rate_final"],
        "tree_valid": _mean([float(get(s, "tree_valid") or False) for s in finals]),
        "node_f1": _mean([get(s, "fidelity_full", "nodes", "f1") for s in finals]),
        "edge_f1": _mean([get(s, "fidelity_full", "edges", "f1") for s in finals]),
        "closure_f1": _mean([get(s, "fidelity_full", "closure", "f1") for s in finals]),
        "valid_node_recall": _mean([get(s, "valid_node_recall") for s in finals]),
        "valid_closure_recall": _mean([get(s, "valid_closure_recall") for s in finals]),
        "goal_satisfaction": _mean([get(s, "plan_quality", "goal_satisfaction") or 0 for s in finals]),
        "completeness": _mean([get(s, "plan_quality", "completeness") or 0 for s in finals]),
        "observed_precision": _mean([get(s, "provenance", "observed", "precision") for s in finals]),
        "prior_precision": _mean([get(s, "provenance", "prior", "precision") for s in finals]),
        "evidence_verified": _mean([get(s, "provenance", "evidence_verified_rate") for s in finals]),
    }
    return {k: v for k, v in m.items() if v is not None}


def maintenance_metrics(traj):
    closure = traj["closure_f1"]
    return {
        "valid_final": float(traj["valid"][-1]),
        "tree_valid_final": float(traj["tree_valid"][-1]),
        "closure_f1_final": closure[-1],
        "closure_f1_mean": sum(closure) / len(closure),     # area under the learning curve
        "forgetting_total": float(sum(traj["forgetting"])),
        "churn_total": float(sum(traj["churn"])),
        "unsupported_changes": float(len(traj["unsupported_changes"])),
        "episodes_to_threshold": (float(traj["episodes_to_threshold"])
                                  if traj["episodes_to_threshold"] is not None else float(traj["episodes"] + 1)),
    }


def loop_metrics(loop):
    """Closed loop: how the agent did in its last round, and the graph it ended with."""
    first, last = loop["rounds"][0], loop["rounds"][-1]
    return {
        "depth_first_round": float(np.mean(first["depth"])),
        "depth_last_round": float(np.mean(last["depth"])),
        "success_last_round": last["success_rate"],
        "rounds_acting_with_graph": float(sum(r["acted_with"] == "graph" for r in loop["rounds"])),
        "closure_f1_final_task": last["state_final_task"]["fidelity_full"]["closure"]["f1"],
        "tree_valid_final_task": float(last["state_final_task"]["tree_valid"]),
        "forgetting_total": float(sum(loop.get("forgetting", []))),
        "unsupported_changes": float(len(loop.get("unsupported_changes", []))),
    }


def load_units(dirs, include_unfrozen=False):
    units, skipped = [], []
    for d in dirs:
        if os.path.exists(os.path.join(d, "scores.json")):
            with open(os.path.join(d, "scores.json")) as f:
                s = json.load(f)
            label = s.get("label") or f"{s.get('mode', 'edges')}/{s['condition']}"
            if s.get("split") == "test" and not s.get("prompt_frozen") and not include_unfrozen:
                skipped.append(d)
                continue
            units.append({"dir": d, "label": label, "task": s["task"], "runs": tuple(s.get("runs") or []),
                          "fingerprint": s.get("prompt_fingerprint", "unrecorded"),
                          "metrics": construction_metrics(s)})
        elif os.path.exists(os.path.join(d, "trajectory.json")):
            with open(os.path.join(d, "trajectory.json")) as f:
                t = json.load(f)
            units.append({"dir": d, "label": t.get("label") or f"maintain/{t['strategy']}/vocab-{t['vocab']}",
                          "task": t["task"], "runs": tuple(t.get("runs") or []),
                          "fingerprint": t.get("prompt_fingerprint", "unrecorded"),
                          "metrics": maintenance_metrics(t)})
        elif os.path.exists(os.path.join(d, "loop.json")):
            with open(os.path.join(d, "loop.json")) as f:
                lp = json.load(f)
            if not lp["rounds"]:
                continue
            label = lp.get("label") or f"loop/{lp['strategy']}/init-{lp['init']}/{'+'.join(lp['tasks'])}"
            units.append({"dir": d, "label": label, "task": lp["tasks"][-1], "runs": (),
                          "fingerprint": lp["prompt_fingerprint"], "metrics": loop_metrics(lp)})
    return units, skipped


def group(units):
    groups = defaultdict(list)
    for u in units:
        groups[(u["label"], u["task"])].append(u)
    for (label, task), us in groups.items():
        fps = {u["fingerprint"] for u in us}
        if len(fps) > 1:
            raise ValueError(f"{label} / {task} mixes prompt versions {sorted(fps)}: "
                             f"{[u['dir'] for u in us]}. Give each version its own --label.")
    return groups


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    means = rng.choice(values, size=(BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(groups):
    rng = np.random.default_rng(SEED)
    out = {}
    for key, us in sorted(groups.items()):
        metrics = sorted({m for u in us for m in u["metrics"]})
        row = {"n": len(us)}
        for m in metrics:
            vals = [u["metrics"][m] for u in us if m in u["metrics"]]
            row[m] = {"mean": float(np.mean(vals)), "ci95": bootstrap_ci(vals, rng), "n": len(vals)}
        out[key] = row
    return out


def compare(groups, label_a, label_b):
    rng = np.random.default_rng(SEED)
    out = {}
    tasks = {t for (l, t) in groups if l == label_a} & {t for (l, t) in groups if l == label_b}
    if not tasks:
        raise ValueError(f"no task has units under both {label_a!r} and {label_b!r}")
    for task in sorted(tasks):
        a, b = groups[(label_a, task)], groups[(label_b, task)]
        a_by_runs = {u["runs"]: u for u in a if u["runs"]}
        pairs = [(a_by_runs[u["runs"]], u) for u in b if u["runs"] in a_by_runs]
        res = {}
        for m in sorted(set(a[0]["metrics"]) & set(b[0]["metrics"])):
            if len(pairs) >= 2:
                diffs = [ub["metrics"][m] - ua["metrics"][m] for ua, ub in pairs
                         if m in ua["metrics"] and m in ub["metrics"]]
                test = stats.wilcoxon(diffs) if any(diffs) else None
                res[m] = {"paired": True, "n_pairs": len(diffs), "mean_diff": float(np.mean(diffs)),
                          "ci95": bootstrap_ci(diffs, rng),
                          "p": float(test.pvalue) if test is not None else 1.0}
            else:
                va = np.array([u["metrics"][m] for u in a if m in u["metrics"]], dtype=float)
                vb = np.array([u["metrics"][m] for u in b if m in u["metrics"]], dtype=float)
                if len(va) == 0 or len(vb) == 0:
                    continue
                boots = (rng.choice(vb, (BOOTSTRAP, len(vb))).mean(1) - rng.choice(va, (BOOTSTRAP, len(va))).mean(1))
                test = stats.mannwhitneyu(vb, va) if len(va) > 1 and len(vb) > 1 else None
                res[m] = {"paired": False, "n_a": len(va), "n_b": len(vb),
                          "mean_diff": float(vb.mean() - va.mean()),
                          "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
                          "p": float(test.pvalue) if test is not None else None}
        out[task] = res
    return out


def _fmt(cell):
    if cell["ci95"] is None:
        return f"{cell['mean']:.2f}"
    lo, hi = cell["ci95"]
    return f"{cell['mean']:.2f} [{lo:.2f},{hi:.2f}]"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="run directories (globs allowed)")
    p.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"))
    p.add_argument("--include-unfrozen", action="store_true")
    p.add_argument("--json", help="write the full summary here")
    args = p.parse_args(argv)

    dirs = sorted({d for pattern in args.dirs for d in glob.glob(pattern) if os.path.isdir(d)})
    units, skipped = load_units(dirs, args.include_unfrozen)
    if skipped:
        print(f"left out {len(skipped)} test-task runs with an unfrozen prompt: {skipped}")
    if not units:
        raise SystemExit("no scores.json or trajectory.json found in the given directories")
    groups = group(units)
    summary = summarise(groups)
    for (label, task), row in summary.items():
        print(f"\n{label}  [{task}]  n={row['n']}")
        for m, cell in row.items():
            if m != "n":
                print(f"  {m:24s} {_fmt(cell)}")
    result = {"groups": {f"{l} | {t}": r for (l, t), r in summary.items()}}
    if args.compare:
        cmp = compare(groups, *args.compare)
        result["compare"] = {"a": args.compare[0], "b": args.compare[1], "by_task": cmp}
        print(f"\n{args.compare[1]}  minus  {args.compare[0]}")
        for task, res in cmp.items():
            print(f"  [{task}]")
            for m, r in res.items():
                ci = f"[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]" if r["ci95"] else ""
                p_txt = f"p={r['p']:.3f}" if r["p"] is not None else "p=n/a"
                print(f"    {m:24s} {r['mean_diff']:+.2f} {ci} {p_txt} ({'paired' if r['paired'] else 'unpaired'})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
