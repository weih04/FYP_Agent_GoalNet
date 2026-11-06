"""Shared by the metric modules: goal-name normalisation, node and edge sets,
precision/recall/F1, and transitive closure.

Node names are compared after normalisation, optionally through an alias map. The
GoG paper merges goals by embedding similarity; exact matching is used here because
the reference is fixed, which keeps the metrics reproducible. Supply `aliases`
(the adapter's goal_aliases) when the constructor uses different wording.
"""

from collections import defaultdict


def _norm(name, aliases=None):
    # Underscores are spacing, not meaning: "make iron_pickaxe" is "make iron pickaxe".
    key = " ".join(str(name).replace("_", " ").strip().lower().split())
    return (aliases or {}).get(key, key)


def _nodes(gog, aliases=None):
    return {_norm(n["name"], aliases) for n in gog.get("nodes", [])}


def _edges(gog, aliases=None):
    return {(_norm(e["from"], aliases), _norm(e["to"], aliases)) for e in gog.get("edges", [])}


def _prf(pred, gold):
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positives": tp, "false_positives": sorted(pred - gold), "false_negatives": sorted(gold - pred)}


def _closure(edges):
    """All (a, b) such that b is reachable from a; self-pairs from cycles are dropped."""
    succ = defaultdict(set)
    for a, b in edges:
        succ[a].add(b)
    out = set()
    for start in list(succ):
        stack, seen = list(succ[start]), set()
        while stack:
            m = stack.pop()
            if m in seen:
                continue
            seen.add(m)
            stack.extend(succ.get(m, ()))
        out |= {(start, m) for m in seen if m != start}
    return out
