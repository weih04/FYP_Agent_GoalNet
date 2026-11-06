"""Plan quality - Leung, Wang & Shen (2026) §4.4.1: goal satisfaction, soundness,
completeness and efficiency, computed on the plan a graph yields rather than on the
graph itself. A graph can score well structurally and still produce an unexecutable
plan, so both are reported.

Also here: plan_from_gog (the dependency-first plan a graph yields, the paper's DFS
retrieval) and the bill of materials (total resources a goal needs), which plan
quality is built on and which the acting agent is shown.
"""

from collections import defaultdict

from ._names import _norm


def plan_from_gog(gog, target, aliases=None):
    """Dependency-first plan for `target`, mirroring the paper's DFS retrieval.

    Returns the subtree rooted at `target` in an order where every goal appears
    after all of its prerequisites.
    """
    nodes = {_norm(n["name"], aliases): n for n in gog.get("nodes", [])}
    if _norm(target, aliases) not in nodes:
        raise KeyError(f"target goal {target!r} is not in the graph")
    succ = defaultdict(list)
    for e in gog.get("edges", []):
        succ[_norm(e["from"], aliases)].append(_norm(e["to"], aliases))

    order, visiting, done = [], set(), set()

    def visit(n):
        if n in done:
            return
        if n in visiting:
            return  # cycle: validate() reports it; emit what we can
        visiting.add(n)
        for m in succ.get(n, []):
            visit(m)
        visiting.discard(n)
        done.add(n)
        order.append(n)

    visit(_norm(target, aliases))
    return order


def _requirements(node):
    out = defaultdict(int)
    for field in ("required_materials", "required_tools"):
        f = node.get(field)
        if isinstance(f, dict):
            for k, qty in f.items():
                out[k] += qty
        elif isinstance(f, list):
            for k in f:
                out[k] += 1
    return out


def _producers(reference, aliases=None):
    """resource -> goal that yields it (an item, or a station via 'nearby')."""
    out = {}
    for n in reference["nodes"]:
        post = n.get("postconditions") or {}
        if not isinstance(post, dict):
            continue
        for k, val in post.items():
            out[val if k == "nearby" else k] = _norm(n["name"], aliases)
    return out


def _persistent(graph):
    """Resources that are not used up: anything the graph uses as a tool, or places."""
    out = set()
    for n in graph.get("nodes", []):
        out |= set(n.get("required_tools") or [])
        placed = (n.get("postconditions") or {}).get("nearby")
        if placed:
            out.add(placed)
    return out


def graph_bill_of_materials(graph, target):
    """bill_of_materials for a constructed graph, from the graph's own facts only, so
    the totals shown to the acting agent carry nothing the graph does not state."""
    return bill_of_materials(graph, target)


def bill_of_materials(reference, target, aliases=None, persistent=None):
    """Total quantity of each resource needed to achieve `target` once.

    Materials are used up, so a resource may be required several times over a
    chain (wood is spent on the table and on all three pickaxes). Tools and placed
    things persist once obtained, so they count once however often they are used -
    otherwise a pickaxe needed by two mining goals would be double-counted. Which
    resources persist is read from the graph itself (_persistent) unless given.
    """
    ref = {_norm(n["name"], aliases): n for n in reference["nodes"]}
    producers = _producers(reference, aliases)
    if persistent is None:
        persistent = _persistent(reference)
    order = plan_from_gog(reference, target, aliases)
    runs = defaultdict(int)
    runs[_norm(target, aliases)] = 1
    needed = defaultdict(int)

    for goal in reversed(order):          # target first, prerequisites after
        n = runs[goal]
        if not n:
            continue
        for res, qty in _requirements(ref[goal]).items():
            total = qty if res in persistent else qty * n
            needed[res] += total if res not in persistent else 0
            if res in persistent:
                needed[res] = 1
            producer = producers.get(res)
            if producer:
                runs[producer] = max(runs[producer], 1) if res in persistent else runs[producer] + total
    return dict(needed)


def plan_quality(plan, reference, target, aliases=None):
    """Goal satisfaction, soundness, completeness and efficiency for one plan.

    Follows Leung, Wang & Shen (2026) §4.4.1. `reference` supplies the true
    preconditions and effects, so the plan is judged against the environment
    rather than against the graph that produced it.
    """
    ref = {_norm(n["name"], aliases): n for n in reference["nodes"]}
    producers = _producers(reference, aliases)
    plan = [_norm(s, aliases) for s in plan]
    target = _norm(target, aliases)

    # Soundness: every step is a real goal and follows its prerequisites.
    seen, sound = set(), 1
    for step in plan:
        node = ref.get(step)
        if node is None:
            sound = 0
            continue
        if any(_norm(p, aliases) not in seen for p in node.get("preconditions", [])):
            sound = 0
        seen.add(step)

    # Completeness: of the quantities the task needs, how much the plan obtains.
    # A plan step is assumed to obtain the quantity its goal is needed for, as in
    # the paper's plans, which pair each instruction with an amount.
    needed = bill_of_materials(reference, target, aliases)
    obtained = sum(qty for res, qty in needed.items() if producers.get(res) in plan)
    total_needed = sum(needed.values())
    completeness = min(obtained / total_needed, 1.0) if total_needed else 1.0

    minimal = len(plan_from_gog(reference, target, aliases))
    efficiency = minimal / len(plan) if len(plan) >= minimal else 0.0

    satisfaction = int(bool(sound) and completeness == 1.0 and target in plan)
    return {"goal_satisfaction": satisfaction, "soundness": sound,
            "completeness": completeness, "efficiency": efficiency,
            "plan_length": len(plan), "minimal_length": minimal}
