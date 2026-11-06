"""A non-LLM Goal Net built from the same experience: classical precondition induction.

The counterpart of GoG's HKG baseline: what can be recovered from the experience
by symbolic means alone, so the thesis can say what the LLM adds. In the spirit of
action-model learning (e.g. ARMS, LOCM), each successful interaction is a positive
example of a goal. The environment's adapter finds the successes (`successes`);
the induction here is the same in every environment:

  produces     the item gained, or the thing placed
  materials    the items used up, at the largest quantity seen
  tools        the items held in *every* success and not used up, plus, where the
               surroundings matter, the things around the agent in every success
               that some observed goal places (requirements must be achievable)

Intersection over successes is the whole induction: with few successes it keeps
coincidental items (holding a sapling while mining stone), and it can never name
a goal the agent did not achieve. Both weaknesses are the point of comparison.
Edges come from derive.derive_edges, exactly as in facts mode, and the graph is
scored by the same scoring.score.

    python -m goalnet_general.baseline_symbolic --env crafter --task collect_diamond --runs <run> --out OUT
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from .adapter import load as load_adapter
from .evaluate.scoring import derive, experienced_reference, finalise, find_root, score, summary_line


def induce(records):
    """GoG dict (edges derived) from success records (see Adapter.successes)."""
    examples = defaultdict(list)
    for r in records:
        examples[r["name"]].append(r)
    placed = {recs[0]["product"] for recs in examples.values() if recs[0]["places"]}

    nodes = []
    for name, recs in examples.items():
        first = recs[0]
        product = first["product"]
        used = defaultdict(int)
        for r in recs:
            for k, v in r["used"].items():
                used[k] = max(used[k], v)
        tools = set.intersection(*(set(r["held"]) for r in recs)) - set(used) - {product}
        if first["surroundings_matter"]:
            tools |= (set.intersection(*(set(r["near"]) for r in recs)) & placed) - {product}
        post = {"nearby": product} if first["places"] else {product: first["gained"].get(product, 1)}
        nodes.append({
            "name": name,
            "description": f"Induced from {len(recs)} successful interactions.",
            "required_tools": sorted(tools),
            "required_materials": dict(sorted(used.items())),
            "preconditions": [],
            "postconditions": post,
            "task_list": list(first["task_list"]),
            "examples": len(recs),
        })
    return derive({"nodes": sorted(nodes, key=lambda n: n["name"]), "edges": []})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="environment package goalnet_<env>, e.g. crafter")
    p.add_argument("--task", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--experience-task", help="read episodes played under this task (default: --task)")
    p.add_argument("--variant", "--recipes", dest="variant", help="world variant the episodes were played in")
    args = p.parse_args(argv)

    adapter = load_adapter(args.env)
    reference, root = adapter.reference(args.task, args.variant)
    variant_id = adapter.variant_id(args.variant)
    experience, metas, paths = adapter.load_experience(args.runs, args.experience_task or args.task)
    adapter.check_variant(metas, args.variant)
    graph = induce(adapter.successes(paths))
    state = score(graph, reference, experienced_reference(reference, metas, adapter), root, adapter, experience)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "graph_0.json"), "w") as f:
        json.dump(graph, f, indent=2)
    if state["valid"]:
        with open(os.path.join(args.out, "final_0.json"), "w") as f:
            json.dump(finalise(graph, find_root(graph, root, adapter.goal_aliases)), f, indent=2)
    summary = {
        "env": args.env, "task": args.task, "label": f"{args.env}/symbolic/experience/variant-{variant_id}",
        "condition": "experience", "mode": "symbolic", "variant": args.variant, "variant_id": variant_id,
        "runs": args.runs, "episodes": len(experience), "repeats": 1,
        "prompt_fingerprint": "none", "prompt_frozen": True,
        "validity_rate_initial": float(state["valid"]), "validity_rate_final": float(state["valid"]),
        "results": [{"repeat": 0, "valid_initial": state["valid"], "valid_final": state["valid"],
                     "initial": state, "final": state, "repair_rounds": 0}],
    }
    with open(os.path.join(args.out, "scores.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for n in graph["nodes"]:
        print(f"  {n['name']:24s} tools={n['required_tools']} materials={n['required_materials']} "
              f"({n['examples']} examples) -> {n['preconditions']}")
    print(summary_line(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
