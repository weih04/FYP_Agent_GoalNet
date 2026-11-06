"""Structural validity checks for a Goal-Oriented Graph (GoG).

Every check derives from the GoG definition in Leung, Wang & Shen (2026): a
directed acyclic graph whose nodes are goals carrying Name / Description /
Required Tools / Required Materials / Preconditions / Postconditions, and whose
edges encode goal -> subgoal prerequisite relations, so that a depth-first search
from a root goal yields a well-defined goal tree.

Violations are returned, never raised: a constructed graph is data to be scored,
and an invalid one is a result rather than an error.

Each message names the problem and the possible fixes without choosing between
them, because the messages are also sent back to the model in the repair loop.
They mention only structure and the known vocabulary, never the reference graph.
An edge is written "A -> B", meaning goal A has prerequisite B.

`unprovided_requirement` holds in both construction modes: a goal needing an item
that no goal in the graph produces or places cannot be planned for, whichever way
the edges were made (derive.SATISFIES says which provisions meet which needs).
`details` carries the cycles and unreachable goals as data, so facts-mode repair
can restate them as facts instead of edges.

Grounding is checked against the environment's environment.Environment,
and the messages never name an environment's actions or items beyond the ones in
the graph: they are sent to the model, so they are part of the frozen prompt and
are fingerprinted with it (prompts.fingerprint).
"""

from collections import defaultdict

from ...derive import SATISFIES, _key, provisions, requirements

REQUIRED_ATTRS = ("name", "description", "required_tools", "required_materials",
                  "preconditions", "postconditions")


def _as_names(field):
    """Required tools/materials are a list or a {name: qty} mapping; both appear."""
    if isinstance(field, dict):
        return list(field)
    if isinstance(field, list):
        return list(field)
    return []


def validate(gog, root=None, env=None):
    """Check one graph. Returns {"valid": bool, "violations": {category: [...]}}.

    root: goal name to check reachability from. Defaults to every node being
    reachable from at least one node with no incoming edges.
    env: the environment.Environment whose names and actions ground the graph.
    """
    v = defaultdict(list)
    nodes = {n.get("name"): n for n in gog.get("nodes", [])}
    edges = [(e.get("from"), e.get("to")) for e in gog.get("edges", [])]
    if env is None:
        raise ValueError("validate needs the environment's Environment (adapter.environment())")
    vocab = {"tools": env.tool_names, "all": env.names}

    # --- schema -----------------------------------------------------------
    for i, n in enumerate(gog.get("nodes", [])):
        for attr in REQUIRED_ATTRS:
            if attr not in n:
                v["schema"].append(f"node {n.get('name', i)!r} missing {attr!r}")
    counts = defaultdict(int)
    for n in gog.get("nodes", []):
        counts[n.get("name")] += 1
    for name, c in counts.items():
        if c > 1:
            v["schema"].append(f"{c} goals are named {name!r}: merge them into one goal or rename all but one")

    # --- referential integrity -------------------------------------------
    for a, b in edges:
        if a not in nodes:
            v["dangling_edge"].append(f"edge {a!r} -> {b!r} starts at {a!r}, which is not a goal: "
                                      f"add a goal named {a!r} or remove the edge")
        if b not in nodes:
            v["dangling_edge"].append(f"edge {a!r} -> {b!r} points to {b!r}, which is not a goal: "
                                      f"add a goal named {b!r} or remove the edge and the precondition")

    # --- preconditions must agree with edges ------------------------------
    declared = {(n["name"], p) for n in nodes.values() for p in n.get("preconditions", [])}
    actual = set(edges)
    for a, b in sorted(declared - actual):
        if a == b:
            continue  # reported once below as a self-loop
        v["precondition_without_edge"].append(
            f"{a!r} lists {b!r} as a precondition but there is no edge {a!r} -> {b!r}: "
            f"add that edge, or remove {b!r} from the preconditions of {a!r}")
    for a, b in sorted(actual - declared):
        if a == b:
            continue
        v["edge_without_precondition"].append(
            f"edge {a!r} -> {b!r} has no matching precondition: add {b!r} to the preconditions "
            f"of {a!r}, or remove the edge")

    # --- self-loops -------------------------------------------------------
    for a in sorted({a for a, b in actual | declared if a == b}):
        v["self_loop"].append(f"{a!r} is its own prerequisite: remove the edge {a!r} -> {a!r} "
                              f"and remove {a!r} from its own preconditions")

    # --- acyclicity -------------------------------------------------------
    succ = defaultdict(list)
    for a, b in edges:
        if a != b:  # self-loops are reported above
            succ[a].append(b)
    colour, cycles = {}, []

    def visit(n, trail):
        colour[n] = 1
        for m in succ[n]:
            if colour.get(m) == 1:
                loop = trail[trail.index(m):] + [m]
                cycles.append(loop)
                v["cycle"].append(
                    f"cycle {' -> '.join(loop)}: each goal needs the next, so none can be achieved "
                    "first; remove or reverse one edge in this loop and change the matching precondition")
            elif m not in colour:
                visit(m, trail + [m])
        colour[n] = 2

    for n in nodes:
        if n not in colour:
            visit(n, [n])

    # --- requirements something provides ---------------------------------
    provided = {(_key(item), kind) for n in nodes.values() for item, kind in provisions(n)}
    for n in nodes.values():
        for item, kind in requirements(n):
            if not any((_key(item), k) in provided for k in SATISFIES[kind]):
                where = "produces" if kind == "material" else "produces or places"
                v["unprovided_requirement"].append(
                    f"{n['name']!r} requires {item!r} but no goal {where} it: add a goal that "
                    f"{where} {item!r}, or remove {item!r} from the requirements of {n['name']!r}")

    # --- reachability -----------------------------------------------------
    unreachable = []
    if root is not None:
        if root not in nodes:
            v["reachability"].append(f"root {root!r} is not a node")
        else:
            seen, stack = {root}, [root]
            while stack:
                for m in succ[stack.pop()]:
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)
            unreachable = sorted(set(nodes) - seen)
            for n in unreachable:
                v["reachability"].append(
                    f"{n!r} cannot be reached from the root {root!r} by following edges: make it a "
                    "prerequisite of a goal that needs it, or remove it if the task does not need it")

    # --- grounding --------------------------------------------------------
    for n in nodes.values():
        for t in _as_names(n.get("required_tools")):
            if t not in vocab["tools"]:
                v["grounding"].append(f"{n['name']!r} requires unknown tool {t!r}: use the exact known "
                                      "name of a tool, or remove it")
        for m in _as_names(n.get("required_materials")):
            if m not in vocab["all"]:
                v["grounding"].append(f"{n['name']!r} requires unknown material {m!r}: use the exact "
                                      "known name, or remove it")
        post = n.get("postconditions") or {}
        for k in (post if isinstance(post, dict) else []):
            if k != "nearby" and k not in vocab["all"]:
                v["grounding"].append(f"{n['name']!r} produces unknown item {k!r}: use the exact known "
                                      "name, or remove it")
        for step in n.get("task_list", []):
            if not env.is_valid_step(step):
                v["grounding"].append(f"{n['name']!r} has task {step!r}, which is not a valid action: "
                                      "use a valid action or an allowed shorthand")

    # Rates, not just pass/fail: the thesis reports reachability and grounding as
    # percentages so partially-correct graphs are distinguishable from empty ones.
    reach = None
    if root is not None and root in nodes:
        seen, stack = {root}, [root]
        while stack:
            for m in succ[stack.pop()]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        reach = len(seen) / len(nodes) if nodes else 0.0

    checked = grounded = 0
    for n in nodes.values():
        refs = ([(t, vocab["tools"]) for t in _as_names(n.get("required_tools"))]
                + [(m, vocab["all"]) for m in _as_names(n.get("required_materials"))])
        post = n.get("postconditions") or {}
        refs += [(k, vocab["all"]) for k in (post if isinstance(post, dict) else []) if k != "nearby"]
        for name, allowed in refs:
            checked += 1
            grounded += name in allowed
        for step in n.get("task_list", []):
            checked += 1
            grounded += env.is_valid_step(step)

    return {
        "valid": not v,
        "violations": dict(v),
        "details": {"cycles": cycles, "unreachable": unreachable},
        "rates": {
            "acyclic": not v.get("cycle") and not v.get("self_loop"),
            "reachability": reach,
            "grounding_rate": grounded / checked if checked else None,
            "nodes": len(nodes),
            "edges": len(edges),
        },
    }


def validity_rate(graphs, env, root=None):
    """Fraction of repeated construction runs that pass every check."""
    if not graphs:
        raise ValueError("validity_rate needs at least one graph")
    return sum(validate(g, root, env)["valid"] for g in graphs) / len(graphs)


def gated_recall(valid, recall):
    """Recall that counts only for a valid graph. Validity alone rewards building
    less (a three-goal chain is easy to keep consistent); recall gated on validity
    cannot be gamed that way."""
    return recall if valid else 0.0
