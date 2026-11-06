"""Reference Goal Nets: what a correct graph is, for every task and recipe set.

The reference graph is generated from the recipe set, never written by hand, so a
counterfactual reference cannot drift from the world the agent played in.
`python -m goalnet_crafter.evaluate.references` checks that the generator
reproduces the hand-authored ground_truth_gog_crafter.json (in this folder) for the
default recipes; `--recipes FILE` prints a variant's reference.

Generation follows what crafter/objects.py actually checks, not what data.yaml
lists: `collect` needs `require` held, `place` consumes `uses` (its `nearby`, if
any, is never checked), `make` needs `nearby` stations and consumes `uses`.
"""

import json
import os
import sys

from goalnet_general.derive import derive_edges

from ..recipes import load_overrides, recipes

HERE = os.path.dirname(os.path.abspath(__file__))

# Goal names follow the hand-authored reference's "<action> <item>" convention.
# They name goals, not recipes, so they are the same in every recipe set.
COLLECT_NAMES = {"wood": "collect wood", "stone": "mine stone", "coal": "mine coal",
                 "iron": "mine iron", "diamond": "mine diamond", "drink": "collect drink",
                 "sapling": "collect sapling"}
PLACE_NAMES = {"table": "place crafting table", "furnace": "place furnace",
               "stone": "place stone", "plant": "place plant"}
MAKE_NAMES = {"wood_pickaxe": "craft wooden pickaxe", "stone_pickaxe": "craft stone pickaxe",
              "iron_pickaxe": "craft iron pickaxe", "wood_sword": "craft wooden sword",
              "stone_sword": "craft stone sword", "iron_sword": "craft iron sword"}

TASK_ROOTS = {
    "make_wood_pickaxe": "craft wooden pickaxe",
    "make_stone_pickaxe": "craft stone pickaxe",
    "place_furnace": "place furnace",
    "collect_iron": "mine iron",
    "make_iron_pickaxe": "craft iron pickaxe",
    "collect_diamond": "mine diamond",
}


def _action(name):
    return " ".join(w.capitalize() for w in name.split("_"))


def full_reference(recipe_set):
    """Every collect/place/make goal of a recipe set, wired by derive_edges."""
    nodes = []
    for material, info in recipe_set["collect"].items():
        for item, qty in info["receive"].items():
            if item not in COLLECT_NAMES:
                raise ValueError(f"no goal name for collecting {item!r}; add it to COLLECT_NAMES")
            nodes.append({
                "name": COLLECT_NAMES[item],
                "description": f"Face {material} and use Do to obtain {item}.",
                "required_tools": list(info.get("require") or {}),
                "required_materials": [],
                "preconditions": [],
                "postconditions": {item: qty},
                "task_list": [f"Move toward {material}", "Do"],
            })
    for thing, info in recipe_set["place"].items():
        nodes.append({
            "name": PLACE_NAMES[thing],
            "description": f"Place a {thing}, consuming {', '.join(info['uses'])}.",
            "required_tools": [],
            "required_materials": dict(info["uses"]),
            "preconditions": [],
            "postconditions": {"nearby": thing},
            "task_list": [f"Place {_action(thing)}"],
        })
    for item, info in recipe_set["make"].items():
        nodes.append({
            "name": MAKE_NAMES[item],
            "description": f"Make a {item} next to {', '.join(info['nearby'])}.",
            "required_tools": list(info["nearby"]),
            "required_materials": dict(info["uses"]),
            "preconditions": [],
            "postconditions": {item: info["gives"]},
            "task_list": [f"Make {_action(item)}"],
        })
    graph, diag = derive_edges({"nodes": nodes, "edges": []})
    if diag["unprovided"]:
        raise ValueError(f"recipe set has requirements nothing provides: {diag['unprovided']}")
    if diag["multiple"]:
        raise ValueError(f"recipe set has requirements with several providers: {diag['multiple']}")
    return graph


def task_reference(recipe_set, task):
    """The reference for one task: its root goal and everything it transitively needs."""
    from goalnet_general.evaluate.metrics import plan_from_gog

    full = full_reference(recipe_set)
    root = TASK_ROOTS[task]
    keep = set(plan_from_gog(full, root))
    return {
        "_root_goal": root,
        "nodes": [n for n in full["nodes"] if n["name"] in keep],
        "edges": [e for e in full["edges"] if e["from"] in keep and e["to"] in keep],
    }


def check_against_hand_authored():
    """Differences between the generated default reference and the hand-authored one."""
    with open(os.path.join(HERE, "ground_truth_gog_crafter.json")) as f:
        hand = json.load(f)
    gen = task_reference(recipes(), "collect_diamond")
    gen_nodes = {n["name"]: n for n in gen["nodes"]}
    problems = []
    for n in hand["nodes"]:
        g = gen_nodes.get(n["name"])
        if g is None:
            problems.append(f"missing node {n['name']!r}")
            continue
        for field in ("required_tools", "required_materials", "postconditions"):
            a, b = n[field], g[field]
            if (sorted(a) if isinstance(a, list) else a) != (sorted(b) if isinstance(b, list) else b):
                problems.append(f"{n['name']!r} {field}: hand {a} != generated {b}")
        if sorted(n["preconditions"]) != sorted(g["preconditions"]):
            problems.append(f"{n['name']!r} preconditions: hand {n['preconditions']} != generated {g['preconditions']}")
    extra = set(gen_nodes) - {n["name"] for n in hand["nodes"]}
    if extra:
        problems.append(f"generated nodes not in hand-authored: {sorted(extra)}")
    hand_edges = {(e["from"], e["to"]) for e in hand["edges"]}
    gen_edges = {(e["from"], e["to"]) for e in gen["edges"]}
    if hand_edges != gen_edges:
        problems.append(f"edges only in hand: {sorted(hand_edges - gen_edges)}; "
                        f"only generated: {sorted(gen_edges - hand_edges)}")
    return problems


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Check the generator, or print a recipe set's reference.")
    p.add_argument("--recipes", help="override YAML; omit to check the default against the hand-authored file")
    p.add_argument("--task", default="collect_diamond")
    args = p.parse_args(argv)
    if not args.recipes:
        problems = check_against_hand_authored()
        for line in problems:
            print(line)
        print("generated default reference matches ground_truth_gog_crafter.json" if not problems
              else f"{len(problems)} differences")
        return 1 if problems else 0
    ref = task_reference(recipes(load_overrides(args.recipes)), args.task)
    for n in ref["nodes"]:
        print(f"{n['name']:22s} tools={n['required_tools']} materials={n['required_materials']} "
              f"-> {n['preconditions']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
