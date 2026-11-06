"""Crafter, played through BALROG: everything goalnet_general needs to know about it.

To run the experiments in another environment, write goalnet_<env>/adapter.py with
the same methods (goalnet_general.adapter.Adapter); nothing in goalnet_general changes.

World variants are recipe override files (recipes.py, recipes/*.yaml). Tasks are
Crafter achievements; reference graphs are the hand-authored ground truth for the
default recipes and are generated from the recipe set otherwise.
"""

import os
import re
import subprocess
import sys

from goalnet_general.adapter import Adapter
from goalnet_general.environment import Environment, Shorthand

from . import experience as exp
from .evaluate.goals import ACHIEVEMENT_GOALS, aliases
from .evaluate.references import task_reference
from .recipes import fingerprint as recipe_fingerprint
from .recipes import load_overrides, recipes
from .vocabulary import ACTIONS, vocabulary

HERE = os.path.dirname(os.path.abspath(__file__))
BALROG_DIR = os.path.join(HERE, "..", "BALROG")
REFERENCE_DIR = os.path.join(HERE, "evaluate")

# The shallow chain is for prompt development; the deep tasks are held out.
DEV_TASKS = ["make_wood_pickaxe", "make_stone_pickaxe", "place_furnace"]
TEST_TASKS = ["collect_iron", "make_iron_pickaxe", "collect_diamond"]

# --env-option shortcuts for BALROG's crafter settings; any other key is passed to
# eval.py as a Hydra override unchanged. `agent` picks how a graph guides acting:
# "gog" (default) replicates the GoG paper's goal inference + planning pipeline
# (BALROG agents/gog.py); "goalnet" lists the goal tree in the prompt (the ablation).
OPTION_KEYS = {
    "action_hints": "envs.crafter_kwargs.action_hints",
    "survival_rules": "envs.crafter_kwargs.survival_rules",
    "max_steps": "envs.crafter_kwargs.max_episode_steps",
}


def _held(interaction):
    """Names held: crafting interactions record {name: count}, others a list of names."""
    held = interaction.get("holding", [])
    return list(held) if isinstance(held, dict) else held


class CrafterAdapter(Adapter):
    name = "crafter"
    secrets_path = os.path.join(BALROG_DIR, "SECRETS")

    def __init__(self):
        self._aliases = aliases()
        self._env = None

    # --- what the model is told -----------------------------------------------------

    def environment(self):
        if self._env is None:
            v = vocabulary()
            self._env = Environment(
                name="crafter",
                actions=tuple(ACTIONS),
                shorthands=(Shorthand("Move toward ", "Move toward <name>",
                                      "use movement actions until you face <name>"),),
                names=frozenset(v["all"]),
                tool_names=frozenset(v["tools"]),
            )
        return self._env

    def task_text(self, task):
        return task.replace("_", " ")

    # --- tasks and references ---------------------------------------------------

    @property
    def tasks(self):
        return {"dev": DEV_TASKS, "test": TEST_TASKS}

    @property
    def goal_aliases(self):
        return self._aliases

    def reference(self, task, variant=None):
        """Default recipes use the hand-authored ground truth (which recipes.py checks the
        generator against); a recipe override uses the generated reference."""
        import json

        if variant:
            reference = task_reference(recipes(load_overrides(variant)), task)
        else:
            with open(os.path.join(REFERENCE_DIR, f"ground_truth_gog_{task}.json")) as f:
                reference = json.load(f)
        root = reference["_root_goal"]
        if ACHIEVEMENT_GOALS.get(task) != root:
            raise ValueError(f"reference root {root!r} does not match task {task!r}")
        return reference, root

    def variant_id(self, variant):
        return recipe_fingerprint(load_overrides(variant))

    # --- experience -------------------------------------------------------------

    def load_experience(self, runs, task):
        """Episodes numbered 1..N across all runs, so cited evidence is unambiguous."""
        summaries, metas, paths = [], [], []
        for run in runs or []:
            s, m = exp.load_run(run, task)
            for e in s:
                summaries.append({**e, "episode": len(summaries) + 1,
                                  "source": os.path.basename(os.path.normpath(run))})
            metas += m
            paths += exp.episode_csvs(run, task)
        return summaries, metas, paths

    def check_variant(self, metas, variant):
        """Episodes record BALROG's crafter_kwargs; a missing `recipes` (older runs, or
        null) is the default set. A relative path is resolved from BALROG/."""
        expected = self.variant_id(variant)
        for m in metas:
            path = (m.get("env_kwargs") or {}).get("recipes")
            if path and not os.path.isabs(path):
                path = os.path.join(BALROG_DIR, path)
            played = self.variant_id(path)
            if played != expected:
                raise ValueError(f"episode seed {m.get('seed')} was played with recipes {played} ({path}), "
                                 f"but scoring uses {expected} ({variant}); pass the matching --variant")

    def achieved_goals(self, meta):
        return {ACHIEVEMENT_GOALS[a] for a in (meta.get("unlock_step") or {}) if a in ACHIEVEMENT_GOALS}

    def episode_succeeded(self, meta):
        return bool(meta.get("success"))

    def observed_names(self, paths):
        return exp.observed_names(paths)

    def episode_shows(self, episode, node):
        """An interaction gaining one of the node's products, or placing what it places,
        read from experience._result_text ("wood +1, now facing table")."""
        post = node.get("postconditions") or {}
        gains = [k for k in post if k != "nearby"]
        placed = post.get("nearby")
        for it in episode.get("interactions", []):
            for result in it.get("results", {}):
                parts = result.split(", ")
                if any(p.startswith(f"{g} +") for p in parts for g in gains):
                    return True
                if placed and it.get("action", "").startswith("Place ") and f"now facing {placed}" in parts:
                    return True
        return False

    def evidence_records(self, episode, node):
        """The interactions episode_shows would accept, with their result text
        ("sapling -1, wood -1, wood_pickaxe +1") read into used-up and gained items."""
        post = node.get("postconditions") or {}
        gains = {k for k in post if k != "nearby"}
        placed = post.get("nearby")
        out = []
        for it in episode.get("interactions", []):
            for result in it.get("results", {}):
                parts = result.split(", ")
                delta = {m.group(1): int(m.group(2)) for m in map(re.compile(r"^(\w+) ([+-]\d+)$").match, parts) if m}
                put = placed if placed and it.get("action", "").startswith("Place ") and f"now facing {placed}" in parts else None
                if not (gains & {k for k, v in delta.items() if v > 0} or put):
                    continue
                held = [h for h in _held(it) if h != "nothing"]
                out.append({"step": it.get("first_step"), "action": it.get("action"),
                            "consumed": {k: -v for k, v in delta.items() if v < 0},
                            "gained": {k: v for k, v in delta.items() if v > 0}, "placed": put,
                            "present": [t for t in [it.get("facing")] + list(it.get("next_to", [])) if t],
                            "held": held})
        return out

    def failure_records(self, episodes, node):
        """Attempts at one of the node's Make or Place actions that changed nothing.

        Only these are evidence about a recipe: Crafter's making and placing are
        deterministic, while Do is not (grass gives a sapling only sometimes, a cow
        takes several hits). A Place also fails when the faced tile cannot take it,
        which a goal's facts do not describe, so a failed Place counts only when the
        agent has placed something on that kind of tile before: then the tile was
        not the reason."""
        actions = {a for a in node.get("task_list") or [] if a.startswith(("Make ", "Place "))}
        if not actions:
            return []
        placeable = {it.get("facing") for e in episodes.values() for it in e.get("interactions", [])
                     if it.get("action", "").startswith("Place ")
                     and any(r != "no change" and "now facing " in r for r in it.get("results", {}))}
        out = []
        for n, e in sorted(episodes.items()):
            for it in e.get("interactions", []):
                action, times = it.get("action", ""), it.get("results", {}).get("no change", 0)
                if action not in actions or not times or not isinstance(it.get("holding"), (dict, list)):
                    continue
                if action.startswith("Place ") and it.get("facing") not in placeable:
                    continue
                held = it["holding"]
                if isinstance(held, list):
                    if held != ["nothing"]:
                        continue  # names without counts: what was used up cannot be judged
                    held = {}
                out.append({"episode": n, "step": it.get("first_step"), "action": action, "times": times,
                            "present": [t for t in [it.get("facing")] + list(it.get("next_to", [])) if t],
                            "held": dict(held)})
        return out

    def successes(self, paths):
        """Do gaining X -> "collect X"; Place T succeeding -> "place T"; Make X -> "make X"."""
        out = []
        for path in paths:
            for t in exp.transitions(exp.read_rows(path)):
                act = t["executed"]
                gained = {k: v for k, v in t["delta"].items() if v > 0}
                used = {k: -v for k, v in t["delta"].items() if v < 0}
                held = {k for k, v in t["inv"].items() if v > 0}
                near = set(t["adjacent"]) | ({t["facing"]} if t["facing"] != "unknown" else set())
                base = {"held": held, "near": near, "used": used, "gained": gained}
                if act == "Do" and gained:
                    for item in gained:
                        out.append(dict(base, name=f"collect {item}", places=False, product=item,
                                        surroundings_matter=False, task_list=[f"Move toward {t['facing']}", "Do"]))
                elif act.startswith("Place "):
                    thing = act[len("Place "):].lower()
                    if t["facing_after"] == thing and t["facing"] != thing:
                        out.append(dict(base, name=f"place {thing}", places=True, product=thing,
                                        surroundings_matter=True, task_list=[act]))
                elif act.startswith("Make "):
                    item = act[len("Make "):].lower().replace(" ", "_")
                    if gained.get(item, 0) > 0:
                        out.append(dict(base, name=f"make {item}", places=False, product=item,
                                        surroundings_matter=True, task_list=[act]))
        return out

    # --- acting (closed loop) ---------------------------------------------------

    def run_episodes(self, task, graph_path, out_dir, episodes, seed, actor_model, variant, options):
        """One BALROG evaluation (cot without a graph; with one, the `agent` option, default
        gog); returns its run dir."""
        options = dict(options or {})
        guided = options.pop("agent", "gog")
        if guided not in ("gog", "goalnet"):
            raise ValueError(f"--env-option agent must be gog or goalnet, got {guided!r}")
        os.makedirs(out_dir, exist_ok=True)
        before = set(os.listdir(out_dir))
        overrides = [
            f"agent.type={guided if graph_path else 'cot'}",
            f"agent.goalnet_path={graph_path or 'null'}",
            "envs.names=crafter",
            f"tasks.crafter_tasks=[{task}]",
            f"eval.num_episodes.crafter={episodes}",
            f"eval.output_dir={out_dir}",
            f"client.model_id={actor_model}",
            f"envs.env_kwargs.seed={seed if seed is not None else 'null'}",
            f"envs.crafter_kwargs.recipes={os.path.abspath(variant) if variant else 'null'}",
        ]
        overrides += [f"{OPTION_KEYS.get(k, k)}={v}" for k, v in options.items()]
        env = dict(os.environ, OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES")
        print(f"  BALROG: {' '.join(overrides)}", flush=True)
        proc = subprocess.run([sys.executable, "eval.py", *overrides], cwd=BALROG_DIR, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"BALROG exited with {proc.returncode}:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
        new = sorted(set(os.listdir(out_dir)) - before)
        if len(new) != 1:
            raise RuntimeError(f"expected one new BALROG run in {out_dir}, found {new}")
        return os.path.join(out_dir, new[0])
