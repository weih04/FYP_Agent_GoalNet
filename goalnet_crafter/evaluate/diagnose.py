"""Why did an episode end where it did? Death cause and stalled progress, from BALROG logs.

Separates the two limits on how far an agent gets along a plan:

* survival - it died from attacks or from letting food, drink or energy run out;
* wandering - it stopped advancing along the plan long before it died.

Uses only the logged observations and status, no model calls.

    python -m goalnet_crafter.evaluate.diagnose BALROG/results/<run> --task collect_diamond
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

from ..experience import parse_adjacent, parse_facing, parse_inventory
from ..vocabulary import ACTIONS

# The collect_diamond reference plan as achievements; the next goal's target is what
# the agent must find or hold to advance.
CHAIN = ["collect_wood", "place_table", "make_wood_pickaxe", "collect_stone", "place_furnace",
         "collect_coal", "make_stone_pickaxe", "collect_iron", "make_iron_pickaxe", "collect_diamond"]
NEEDS_TO_SEE = {"collect_wood": "tree", "collect_stone": "stone", "collect_coal": "coal",
                "collect_iron": "iron", "collect_diamond": "diamond"}
THREATS = {"zombie", "skeleton", "arrow"}
_VITAL = re.compile(r"- (health|food|drink|energy): (\d+)/9")
_SEEN = re.compile(r"^- (\w+) ", re.M)


def vitals(status):
    return {k: int(v) for k, v in _VITAL.findall(status)}


def diagnose(csv_path, json_path, task):
    with open(json_path) as f:
        meta = json.load(f)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chain = CHAIN[:CHAIN.index(task) + 1]

    # unlock_step counts from 1, CSV steps from 0.
    unlocked = {a: s - 1 for a, s in (meta.get("unlock_step") or {}).items() if a in chain}
    reached = [a for a in chain if a in unlocked]
    last_progress = max(unlocked.values(), default=-1)
    next_goal = next((a for a in chain if a not in unlocked), None)

    # Health drops, each tagged with what was within 1 tile and the vitals at the time.
    drops, prev = [], vitals(rows[0]["Status"])
    first_zero = {}
    prev_obs = rows[0]["Observation"]
    for row in rows:
        v = vitals(row["Status"])
        step = int(row["Step"])
        for k in ("food", "drink", "energy"):
            if v.get(k) == 0 and k not in first_zero:
                first_zero[k] = step
        lost = prev.get("health", 9) - v.get("health", 9)
        if lost > 0:
            # An arrow that hits is removed, so it shows only in the observation before the hit.
            near = (parse_adjacent(row["Observation"]) | parse_adjacent(prev_obs)) & (THREATS | {"lava"})
            drops.append({"step": step, "lost": lost, "near": sorted(near),
                          "empty_vitals": sorted(k for k in ("food", "drink", "energy") if v.get(k) == 0),
                          "action": row["Action"]})
        prev, prev_obs = v, row["Observation"]

    final = vitals(rows[-1]["Status"])
    if meta.get("dead"):
        window = [d for d in drops if d["step"] >= len(rows) - 30]
        attacked = sum(d["lost"] for d in window if d["near"])
        starved = sum(d["lost"] for d in window if not d["near"] and d["empty_vitals"])
        if "lava" in parse_adjacent(rows[-1]["Observation"]) and not attacked:
            cause = "lava"
        elif attacked and starved:
            cause = "attacked while food/drink/energy were empty"
        elif attacked:
            cause = "attacked"
        elif starved:
            cause = "food/drink/energy ran out"
        else:
            cause = "unclear"
    else:
        cause = "did not die"

    # How the steps after the last plan progress were spent.
    after = rows[last_progress + 1:]
    spent, prev_inv, prev_face = Counter(), parse_inventory(rows[last_progress]["Status"]) if last_progress >= 0 else {}, None
    for row in after:
        a = row["Action"]
        inv = parse_inventory(row["Status"])
        if a not in ACTIONS:
            spent["invalid output (Noop)"] += 1
        elif a.startswith("Move"):
            spent["move"] += 1
        elif a == "Do":
            spent["Do, inventory changed" if inv != prev_inv else "Do, nothing gained"] += 1
        else:
            spent[a] += 1
        prev_inv = inv

    target = NEEDS_TO_SEE.get(next_goal)
    seen_steps = [int(r["Step"]) for r in after if target and target in _SEEN.findall(r["Observation"])]

    return {
        "seed": meta.get("seed"),
        "steps": len(rows),
        "dead": meta.get("dead"),
        "cause": cause,
        "final_vitals": final,
        "vital_first_zero": first_zero,
        "health_lost_to": {
            "threat nearby": sum(d["lost"] for d in drops if d["near"]),
            "empty food/drink/energy": sum(d["lost"] for d in drops if not d["near"] and d["empty_vitals"]),
            "other": sum(d["lost"] for d in drops if not d["near"] and not d["empty_vitals"]),
        },
        "plan_reached": reached,
        "last_progress_step": last_progress,
        "steps_without_progress": len(rows) - 1 - last_progress,
        "next_goal": next_goal,
        "next_goal_target_seen_after_progress": (f"{target}: seen in {len(seen_steps)} steps" if target
                                                 else "needs crafting, not a visible target"),
        "steps_since_progress_spent_on": dict(spent.most_common()),
        "final_inventory": parse_inventory(rows[-1]["Status"]),
        "last_facing": parse_facing(rows[-1]["Observation"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--task", default="collect_diamond")
    ap.add_argument("--json", action="store_true", help="print full per-episode records")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.run_dir, "crafter", args.task, f"{args.task}_run_*.csv")))
    if not csvs:
        raise FileNotFoundError(f"no episodes for {args.task} in {args.run_dir}")
    results = [diagnose(c, c[:-4] + ".json", args.task) for c in csvs]

    if args.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        print(f"seed {r['seed']}: died at {r['steps'] - 1} - {r['cause']}")
        print(f"  reached {r['plan_reached'] or 'nothing'}; next {r['next_goal']}; "
              f"{r['steps_without_progress']} steps since progress ({r['next_goal_target_seen_after_progress']})")
        print(f"  health lost to {r['health_lost_to']}; vitals first hit 0 at {r['vital_first_zero'] or 'never'}")
        print(f"  since progress: {r['steps_since_progress_spent_on']}; final inventory {r['final_inventory'] or 'empty'}")
    causes = Counter(r["cause"] for r in results)
    print(f"\ncauses: {dict(causes)}")
    print(f"mean steps without progress: {sum(r['steps_without_progress'] for r in results) / len(results):.0f} "
          f"of {sum(r['steps'] for r in results) / len(results):.0f}")


if __name__ == "__main__":
    main()
