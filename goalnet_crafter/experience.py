"""Experience summaries for Goal-Net construction, built from BALROG episode logs.

A summary states what happened in the environment and nothing else: which
interaction the agent attempted, in what state, and what changed afterwards. The
agent's reasoning text is deliberately left out. It carries hints from the acting
prompt and the model's pre-training, so including it would let `construct` read
off structure that the experience itself does not show, and would make naive and
CoT experience incomparable.

Failed attempts are kept alongside successful ones: "Do while facing stone,
holding nothing -> no change" is the evidence from which a missing prerequisite
can be inferred.

Crafting and placing interactions record what was held with counts
({"wood": 3}), because they use up quantities: "failed holding 1 wood" and
"failed holding 4 wood" say different things about a recipe. Do and Sleep record
only the names held, so collecting wood nine times stays one group.

Each BALROG CSV row holds the action the agent output and the observation and
status *after* that action ran. The state an action was taken from is therefore
the previous row's; for the first step it is the reset state (empty inventory,
facing unknown).
"""

import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

from .vocabulary import ACTIONS

# Actions that act on the world; movement and Noop are only counted.
INTERACTIONS = [a for a in ACTIONS if a == "Do" or a == "Sleep"
                or a.startswith("Place ") or a.startswith("Make ")]

_INV_LINE = re.compile(r"^- (\w+): (\d+)$")
_FACE = re.compile(r"You face (\w+) at your front")
_SEEN = re.compile(r"^- (\w+) (.+)$")
_OFFSET = re.compile(r"(\d+) steps? (north|south|east|west)")


def parse_inventory(status):
    """{"wood": 2, ...} from the status block; vitals (health etc.) are excluded."""
    inv = {}
    if "Your inventory:" not in status:
        return inv
    for line in status.split("Your inventory:", 1)[1].splitlines():
        m = _INV_LINE.match(line.strip())
        if m:
            inv[m.group(1)] = int(m.group(2))
    return inv


def parse_facing(observation):
    m = _FACE.search(observation)
    return m.group(1) if m else "unknown"


def parse_adjacent(observation):
    """Things within 1 tile, diagonals included - the radius Crafter uses for `nearby`."""
    out = set()
    for line in observation.splitlines():
        m = _SEEN.match(line.strip())
        if not m:
            continue
        offsets = _OFFSET.findall(m.group(2))
        if offsets and all(int(n) <= 1 for n, _ in offsets):
            out.add(m.group(1))
    return out


def _delta(before, after):
    keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(keys) if after.get(k, 0) != before.get(k, 0)}


def _result_text(delta, facing_before, facing_after):
    parts = [f"{k} {v:+d}" for k, v in delta.items()]
    if facing_after != facing_before:
        parts.append(f"now facing {facing_after}")
    return ", ".join(parts) if parts else "no change"


def transitions(rows):
    """Each logged step as the state it was taken from, the action, and what changed.

    Yields {step, output, executed, inv, facing, adjacent, inv_after, facing_after, delta}.
    `inv`, `facing` and `adjacent` are the state before the action (the previous row's,
    or the reset state for the first step); see the module docstring.
    """
    inv, facing, adjacent = {}, "unknown", set()
    for row in rows:
        output = row["Action"]
        inv_after = parse_inventory(row["Status"])
        facing_after = parse_facing(row["Observation"])
        yield {
            "step": int(row["Step"]),
            "output": output,
            # An output outside the action space was executed as Noop (env_wrapper.check_action_validity).
            "executed": output if output in ACTIONS else "Noop",
            "inv": inv, "facing": facing, "adjacent": adjacent,
            "inv_after": inv_after, "facing_after": facing_after,
            "delta": _delta(inv, inv_after),
        }
        inv, facing, adjacent = inv_after, facing_after, parse_adjacent(row["Observation"])


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no steps")
    return rows


def parse_seen(observation):
    """Names of everything the observation lists as visible, at any distance."""
    return {m.group(1) for line in observation.splitlines() if (m := _SEEN.match(line.strip()))}


def observed_names(csv_paths):
    """Item and object names the agent saw: inventory keys, faced and visible things.

    This is the vocabulary="observed" condition: the prompt names only what the
    agent's own observations contain, so it tells the model nothing more.
    """
    names = set()
    for path in csv_paths:
        for row in read_rows(path):
            names |= set(parse_inventory(row["Status"]))
            names |= parse_seen(row["Observation"])
            facing = parse_facing(row["Observation"])
            if facing != "unknown":
                names.add(facing)
    return names


def summarise_episode(csv_path, json_path):
    """One episode as {outcome, achievements, interactions, action counts}."""
    with open(json_path) as f:
        meta = json.load(f)
    rows = read_rows(csv_path)
    groups = defaultdict(lambda: {"count": 0, "first_step": None, "results": Counter()})
    moves, invalid = Counter(), Counter()

    for t in transitions(rows):
        if t["executed"] != t["output"]:
            invalid[t["output"]] += 1
        if t["executed"] in INTERACTIONS:
            # Crafting and placing depend on what is nearby; Do and Sleep act only on the
            # faced tile, so recording passers-by (cows, arrows) would only split groups.
            crafting = t["executed"].startswith(("Place ", "Make "))
            nearby = tuple(sorted(t["adjacent"] - {t["facing"]})) if crafting else None
            held = tuple(sorted(t["inv"].items())) if crafting else tuple(sorted(t["inv"]))
            key = (t["executed"], t["facing"], held, nearby)
            g = groups[key]
            g["count"] += 1
            if g["first_step"] is None:
                g["first_step"] = t["step"]
            g["results"][_result_text(t["delta"], t["facing"], t["facing_after"])] += 1
        else:
            moves[t["executed"]] += 1

    interactions = []
    for (action, face, held, nxt), g in sorted(groups.items(), key=lambda kv: kv[1]["first_step"]):
        if nxt is not None:
            entry = {"action": action, "facing": face, "holding": dict(held) or ["nothing"]}
        else:
            entry = {"action": action, "facing": face, "holding": list(held) or ["nothing"]}
        if nxt is not None:
            entry["next_to"] = list(nxt)
        entry.update(times=g["count"], first_step=g["first_step"], results=dict(g["results"].most_common()))
        interactions.append(entry)

    # unlock_step counts steps from 1; CSV steps count from 0.
    achievements = sorted(
        ({"achievement": name.replace("_", " "), "step": s - 1} for name, s in (meta.get("unlock_step") or {}).items()),
        key=lambda a: a["step"],
    )
    last = rows[-1]
    if meta.get("success"):
        outcome = f"completed the task at step {meta['steps_to_success'] - 1}"
    elif meta.get("dead"):
        outcome = f"died at step {len(rows) - 1}, next to: {', '.join(sorted(parse_adjacent(last['Observation']))) or 'nothing'}"
    else:
        outcome = f"stopped after {len(rows)} steps without completing the task"

    summary = {
        "steps": len(rows),
        "outcome": outcome,
        "achievements": achievements,
        "final_inventory": parse_inventory(last["Status"]) or "empty",
        "interactions": interactions,
        "other_actions": dict(moves.most_common()),
    }
    if invalid:
        summary["invalid_outputs_executed_as_noop"] = dict(invalid.most_common())
    return summary, meta


def episode_csvs(run_dir, task):
    """The episode CSVs of `task` in a BALROG results directory, in episode order."""
    pattern = os.path.join(run_dir, "crafter", task, f"{task}_run_*.csv")
    csvs = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"_run_(\d+)\.csv$", p).group(1)))
    if not csvs:
        raise FileNotFoundError(f"no episodes match {pattern}")
    return csvs


def load_run(run_dir, task):
    """Summaries for every episode of `task` in a BALROG results directory, in episode order.

    Returns (summaries, metas); metas are the raw episode JSONs (seed, unlock_step, ...).
    """
    csvs = episode_csvs(run_dir, task)
    summaries, metas = [], []
    for i, csv_path in enumerate(csvs):
        json_path = csv_path[:-4] + ".json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} missing for {csv_path}")
        s, meta = summarise_episode(csv_path, json_path)
        summaries.append({"episode": i + 1, **s})
        metas.append(meta)
    return summaries, metas
