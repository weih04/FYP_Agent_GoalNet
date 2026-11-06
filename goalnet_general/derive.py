"""Prerequisite edges computed from goal facts (Leung, Wang & Shen 2026, §3.1.2).

In the paper the LLM extracts goals and their conditions; edges are added by
matching one goal's preconditions to another goal's postconditions. The same
split is used here: the model states what each goal needs and what it gives,
and this module wires the graph. Edge/precondition agreement then holds by
construction, and every remaining structural fault (a cycle, an unreachable
goal, a requirement nothing provides) traces back to a stated fact.

The rule is domain-agnostic: a goal needs every item named in its
required_tools or required_materials, and a goal provides every item in its
postconditions plus the thing it places ("nearby"). A consumed material must come
from the inventory, so it matches only postcondition items; a tool may be held or
stand nearby, so it also matches placed things. Names are matched after
the same normalisation as metrics._norm, with no aliases, so a requirement
spelled differently from the product it means stays unmatched and is reported.
"""

from collections import defaultdict


def _key(name):
    return " ".join(str(name).replace("_", " ").strip().lower().split())


def _names(field):
    return list(field) if isinstance(field, (list, dict)) else []


def requirements(node):
    """[(item, kind)] a goal needs; kind is "tool" (kept) or "material" (consumed)."""
    return ([(t, "tool") for t in _names(node.get("required_tools"))]
            + [(m, "material") for m in _names(node.get("required_materials"))])


def provisions(node):
    """[(item, kind)] a goal provides; kind is "item" (inventory) or "placed"."""
    post = node.get("postconditions") or {}
    if not isinstance(post, dict):
        return []
    return [(v, "placed") if k == "nearby" else (k, "item") for k, v in post.items()]


# Which provision kinds can satisfy which requirement kinds.
SATISFIES = {"tool": ("item", "placed"), "material": ("item",)}


def derive_edges(gog):
    """Return (graph, diagnostics): `gog` with preconditions and edges recomputed.

    diagnostics:
      unprovided  - [(goal, item)]: a requirement no goal in the graph provides
      multiple    - [(goal, item, [providers])]: more than one goal provides it;
                    every provider is linked, since nothing in the facts says which
    A goal is never linked to itself, so a goal that needs what it also gives
    (a station reused by its own recipe) does not create a self-loop.
    """
    nodes = [dict(n) for n in gog.get("nodes", [])]
    providers = defaultdict(list)          # (item key, provision kind) -> goal names
    for n in nodes:
        for item, kind in provisions(n):
            if n["name"] not in providers[(_key(item), kind)]:
                providers[(_key(item), kind)].append(n["name"])

    edges, unprovided, multiple = [], [], []
    for n in nodes:
        pre = []
        for item, kind in requirements(n):
            able = [p for k in SATISFIES[kind] for p in providers.get((_key(item), k), [])]
            found = list(dict.fromkeys(p for p in able if p != n["name"]))
            if not found:
                if n["name"] not in able:
                    unprovided.append((n["name"], item))
                continue
            if len(found) > 1:
                multiple.append((n["name"], item, found))
            for p in found:
                if p not in pre:
                    pre.append(p)
                    edges.append({"from": n["name"], "to": p,
                                  "description": f"{n['name']!r} needs {item}, which {p!r} provides."})
                else:
                    # A second item from the same provider: keep one edge, list both reasons.
                    for e in edges:
                        if e["from"] == n["name"] and e["to"] == p:
                            e["description"] = e["description"][:-1] + f"; also {item}."
        n["preconditions"] = pre
    out = {k: v for k, v in gog.items() if k not in ("nodes", "edges")}
    out["nodes"], out["edges"] = nodes, edges
    return out, {"unprovided": unprovided, "multiple": multiple}
