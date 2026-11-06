"""The problems sent back to the model for repair: structural ones, and (in
maintenance) observed goals whose facts disagree with the evidence they cite, and
goals whose action failed while everything they list was available.

These messages are prompt text: the model reads them and acts on them. They are
kept here, apart from the scoring code in scoring.py, so that prompts.fingerprint()
covers them, and they are worded without naming any environment. They use only
`validate` (structure and names) and the agent's own experience, never the
reference graph. Code reports a disagreement; the model decides the fix.
"""

from .evaluate.metrics.validity import validate


# Per-category cap on problems sent back, so one runaway category cannot flood the prompt.
MAX_PROBLEMS_PER_CATEGORY = 10


def edge_problems(graph, root_name, task_text, env):
    """Edges mode: validate's own messages, plus a missing root."""
    validity = validate(graph, root=root_name, env=env)
    out = []
    if root_name is None:
        out.append(f"no goal in the graph achieves the task '{task_text}'; "
                   "the graph needs a root goal that completes the task")
    for items in validity["violations"].values():
        out += items[:MAX_PROBLEMS_PER_CATEGORY]
    return out


# validate categories that cannot occur when derive_edges made the edges.
_IMPOSSIBLE_IN_FACTS = ("dangling_edge", "precondition_without_edge", "edge_without_precondition", "self_loop")


def fact_problems(graph, env, knowledge_base=False):
    """Facts mode: the problems of a derived graph, stated as facts the model wrote.

    Reachability is checked from the model's own task_goal, never from the reference
    root, so nothing here depends on what the scorer knows. With knowledge_base
    (maintenance), goals off the current task's path are knowledge kept for other
    tasks, as in GoG's knowledge base, so unreachable goals are not reported; what
    disconnects a task's tree is a requirement nothing provides, which still is.
    """
    root = graph.get("_root_goal")
    names = {n.get("name") for n in graph.get("nodes", [])}
    out = []
    if root not in names:
        out.append(f"task_goal {root!r} is not one of the goals: name the goal that achieves the task")
        root = None
    validity = validate(graph, root=root, env=env)
    v = validity["violations"]
    for cat in _IMPOSSIBLE_IN_FACTS:
        if v.get(cat):
            raise RuntimeError(f"derived graph has {cat} violations, which derive_edges should prevent: {v[cat]}")
    for cat in ("schema", "unprovided_requirement", "grounding"):
        out += v.get(cat, [])[:MAX_PROBLEMS_PER_CATEGORY]
    why = {(e["from"], e["to"]): e["description"] for e in graph.get("edges", [])}
    for loop in validity["details"]["cycles"][:MAX_PROBLEMS_PER_CATEGORY]:
        steps = "; ".join(why[(a, b)] for a, b in zip(loop, loop[1:]))
        out.append(f"these goals need each other in a loop, so none can be achieved first: {steps} "
                   "One of these requirements or products is wrong.")
    unreachable = [] if knowledge_base else validity["details"]["unreachable"]
    for n in unreachable[:MAX_PROBLEMS_PER_CATEGORY]:
        out.append(f"{n!r} is not needed on the way to the task goal {root!r}: no goal that leads to the "
                   f"task requires what {n!r} produces or places. Either a requirement is missing or "
                   f"spelled differently from what {n!r} produces, or {n!r} is not needed and should be removed")
    return out


def _items(d):
    return ", ".join(f"{k} {v}" for k, v in d.items()) or "nothing"


def _change(rec):
    parts = [f"{k} -{v}" for k, v in rec["consumed"].items()] + [f"{k} +{v}" for k, v in rec["gained"].items()]
    if rec["placed"]:
        parts.append(f"{rec['placed']} placed")
    return ", ".join(parts) or "no item change"


def _mismatches(node, rec):
    """How `node`'s facts differ from one recorded success; [] if they agree."""
    out = []
    mats = node.get("required_materials") or {}
    if mats != rec["consumed"]:
        out.append(f"it lists materials {_items(mats)}, but {_items(rec['consumed'])} was used up")
    products = {k: v for k, v in (node.get("postconditions") or {}).items() if k != "nearby"}
    if products != rec["gained"]:
        out.append(f"it lists products {_items(products)}, but {_items(rec['gained'])} was gained")
    available = set(rec["present"]) | set(rec["held"])
    absent = [t for t in node.get("required_tools") or [] if t not in available]
    if absent:
        out.append(f"it lists {', '.join(absent)} as required, but none of that was present or held")
    return out


def unsupported_observed(graph, episodes, records):
    """Observed goals whose cited episodes record no success that agrees with their
    facts, as (goal name, message). `episodes` maps episode number to summary;
    `records(episode, node)` is the adapter's evidence_records."""
    out = []
    for node in graph.get("nodes", []):
        if node.get("basis") != "observed":
            continue
        cites = node.get("evidence") or []
        found = [(c, rec) for c in cites if c.get("episode") in episodes
                 for rec in records(episodes[c["episode"]], node)]
        if not found:
            where = ", ".join(f"episode {c.get('episode')} step {c.get('step')}" for c in cites) or "no episode"
            out.append((node["name"], f"{node['name']!r} is marked observed and cites {where}, but no interaction "
                                      f"there achieved it. Mark it prior, or cite an episode and step where it succeeded."))
            continue
        if any(not _mismatches(node, rec) for _, rec in found):
            continue
        cite, rec = min(found, key=lambda f: abs((f[1]["step"] or 0) - (f[0].get("step") or 0)))
        context = (f" with {', '.join(rec['present'])} present" if rec["present"] else "") + \
                  f", holding {', '.join(rec['held']) or 'nothing'}"
        out.append((node["name"], f"{node['name']!r} is marked observed, citing episode {cite['episode']} step "
                                  f"{rec['step']}, where {rec['action']}{context} recorded: {_change(rec)}. But "
                                  f"{'; '.join(_mismatches(node, rec))}. Correct its facts to what was recorded, "
                                  f"or mark it prior."))
    return out


def evidence_problems(graph, episodes, records):
    """Maintenance: the messages of unsupported_observed, capped like the structural ones."""
    return [msg for _, msg in unsupported_observed(graph, episodes, records)][:MAX_PROBLEMS_PER_CATEGORY]


def _requirements(node):
    tools = node.get("required_tools") or []
    mats = node.get("required_materials") or {}
    parts = ([f"tools {', '.join(tools)}"] if tools else []) + ([f"materials {_items(mats)}"] if mats else [])
    return " and ".join(parts) or "no requirements"


def _available(node, rec):
    held = rec["held"]
    return (all(t in rec["present"] or held.get(t, 0) > 0 for t in node.get("required_tools") or [])
            and all(held.get(m, 0) >= q for m, q in (node.get("required_materials") or {}).items()))


def contradicted_by_failures(graph, episodes, failures):
    """Goals, prior or observed, whose action changed nothing although everything they
    list was available, as (goal name, message). This is how experience challenges a
    belief: a recipe that is wrong fails while it looks satisfied. `failures(episodes,
    node)` is the adapter's failure_records; it keeps only failures that are evidence."""
    out = []
    for node in graph.get("nodes", []):
        found = [r for r in failures(episodes, node) if _available(node, r)]
        if not found:
            continue
        rec = found[-1]
        total = sum(r["times"] for r in found)
        where = ", ".join(sorted({f"episode {r['episode']}" for r in found}, key=lambda w: int(w.split()[1])))
        context = (f" with {', '.join(rec['present'])} present" if rec["present"] else "") + \
                  f", holding {_items(rec['held'])}"
        mats = node.get("required_materials") or {}
        most = {m: max(r["held"].get(m, 0) for r in found) for m in mats}
        more = (f" It failed even holding {_items(most)}, so listing more of a material only helps above "
                f"those amounts." if mats else "")
        out.append((node["name"], f"{node['name']!r} lists {_requirements(node)}, but its action changed nothing "
                                  f"{total} time(s) while all of that was available ({where}; e.g. episode "
                                  f"{rec['episode']} step {rec['step']}, {rec['action']}{context}).{more} So it "
                                  f"needs something it does not list. The attempts show the goal is wanted: "
                                  f"correct its requirements rather than removing it, and mark anything the "
                                  f"experience does not show as prior."))
    return out


def failure_problems(graph, episodes, failures):
    """Maintenance: the messages of contradicted_by_failures, capped like the others."""
    return [msg for _, msg in contradicted_by_failures(graph, episodes, failures)][:MAX_PROBLEMS_PER_CATEGORY]
