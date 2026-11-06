"""Scoring shared by construction, maintenance, the closed loop and the baseline.

One `score` measures every condition - edges or facts mode, one-shot or maintained,
LLM or symbolic - in every environment. What differs between environments (goal
name aliases, how an episode shows a goal, the reference) comes from the adapter.
"""

from ..derive import derive_edges
from .metrics import (gated_recall, graph_bill_of_materials, graph_fidelity, plan_from_gog, plan_quality,
                      provenance, validate)
from .metrics._names import _norm


def experienced_reference(reference, metas, adapter):
    """Reference restricted to goals some episode actually achieved."""
    reached = {g for m in metas for g in adapter.achieved_goals(m)}
    return {
        "nodes": [n for n in reference["nodes"] if n["name"] in reached],
        "edges": [e for e in reference["edges"] if e["from"] in reached and e["to"] in reached],
    }


def find_root(graph, ref_root, aliases):
    """The graph's goal for the task, found through the alias map (scoring only).

    In facts mode the model names its own task_goal; it is used when it is a node and
    means the reference root. Otherwise the alias lookup decides, as in edges mode.
    """
    names = {_norm(n.get("name", ""), aliases): n.get("name") for n in graph.get("nodes", [])}
    own = graph.get("_root_goal")
    if own in {n.get("name") for n in graph.get("nodes", [])} and _norm(own, aliases) == ref_root:
        return own
    return names.get(ref_root)


def score(graph, reference, experienced, ref_root, adapter, experience=None):
    """Validity, fidelity (full, experienced, rooted), provenance and plan quality."""
    aliases, env = adapter.goal_aliases, adapter.environment()
    graph_root = find_root(graph, ref_root, aliases)
    validity = validate(graph, root=graph_root, env=env)
    valid = validity["valid"] and graph_root is not None
    fid = graph_fidelity(graph, reference, aliases)
    out = {
        # A graph with no goal for the task cannot plan for it, so it is not valid.
        "valid": valid,
        "root_found": graph_root is not None,
        "validity": validity,
        "fidelity_full": fid,
        "tree_valid": False,
        "valid_node_recall": gated_recall(valid, fid["nodes"]["recall"]),
        "valid_closure_recall": gated_recall(valid, fid["closure"]["recall"]),
    }
    if experienced["nodes"]:
        out["fidelity_experienced"] = graph_fidelity(graph, experienced, aliases)
    if any("basis" in n for n in graph.get("nodes", [])):
        out["provenance"] = provenance(graph, reference, experience or [], adapter.episode_shows, aliases)
    if graph_root is not None:
        # Goals not reachable from the root play no part in the plan; they are reported
        # as off-task and pruned before a second fidelity score, so they cannot mask how
        # well the task chain itself was recovered.
        succ = {}
        for e in graph.get("edges", []):
            succ.setdefault(e.get("from"), []).append(e.get("to"))
        seen, stack = {graph_root}, [graph_root]
        while stack:
            for m in succ.get(stack.pop(), []):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        ref_names = {_norm(n["name"], aliases) for n in reference["nodes"]}
        unreachable = sorted(n.get("name") for n in graph.get("nodes", []) if n.get("name") not in seen)
        out["disconnected_task_goals"] = [n for n in unreachable if _norm(n, aliases) in ref_names]
        out["off_task_goals"] = [n for n in unreachable if _norm(n, aliases) not in ref_names]
        rooted = {
            "nodes": [n for n in graph.get("nodes", []) if n.get("name") in seen],
            "edges": [e for e in graph.get("edges", []) if e.get("from") in seen and e.get("to") in seen],
        }
        out["fidelity_rooted"] = graph_fidelity(rooted, reference, aliases)
        # The goal tree under the root is what the agent is shown (GoG retrieves it by
        # DFS); a knowledge base may hold other goals and still give a sound tree.
        out["tree_valid"] = validate(rooted, root=graph_root, env=env)["valid"]
        plan = plan_from_gog(graph, graph_root, aliases)
        out["plan"] = plan
        out["plan_quality"] = plan_quality(plan, reference, ref_root, aliases)
    return out


def summary_line(state):
    if "parse_error" in state:
        return "valid=False (reply could not be parsed)"
    fid = state["fidelity_full"]
    text = (f"valid={state['valid']} tree_valid={state['tree_valid']} root={state['root_found']} nodes F1={fid['nodes']['f1']:.2f} "
            f"edges F1={fid['edges']['f1']:.2f} closure F1={fid['closure']['f1']:.2f}")
    if "provenance" in state:
        p = state["provenance"]
        text += (f" | observed {p['observed']['goals']} (verified {p['evidence_verified_rate']}) "
                 f"prior {p['prior']['goals']}")
    return text


def finalise(graph, root):
    """A graph ready for the acting agent: root recorded, totals from its own facts."""
    out = dict(graph)
    out["_root_goal"] = root
    out["_bill_of_materials"] = graph_bill_of_materials(graph, root)
    return out


def derive(graph):
    """derive_edges, keeping the diagnostics alongside the graph."""
    derived, diag = derive_edges(graph)
    derived["_derivation"] = {"unprovided": diag["unprovided"], "multiple": diag["multiple"]}
    return derived
