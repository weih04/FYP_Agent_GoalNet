"""Closed loop: act with the current Goal Net, maintain it from the new episodes, repeat.

    python -m goalnet_general.run_loop --env crafter --tasks collect_diamond \
        --iterations 4 --episodes 5 --out OUT
        [--init empty|zero|<graph.json>] [--variant FILE] [--actor-model M] [--model M]
        [--vocab full|observed] [--seed S] [--env-option key=value ...]

Each iteration has the environment's adapter play episodes (guided by the current
graph, or with no graph while none is usable), then feeds every new episode to
maintain.Maintainer, one at a time. The graph acting improves the experience, and
the experience improves the graph.

--tasks with several tasks is a curriculum: the same graph is carried from task to
task (e.g. make_wood_pickaxe make_stone_pickaxe collect_diamond), as GoG carries
one graph across all its chunks. At each switch the model names the new task's goal
and adds any goals it lacks (Maintainer.retarget); nothing existing is changed.

--init zero seeds the graph from the zero-experience construction (pure prior);
--init empty starts with nothing, so the first iteration acts without a graph.

Whether a graph is usable is decided without the reference: its own task goal must
be a node and the goal tree under it must pass validate. When the current graph is
not usable, the last usable graph for that task is kept (with a warning); if there
is none, the iteration acts without a graph. --actor-model sets the acting model
separately from the constructing one (--model).

--maps fresh (default) gives iteration i the base seed S + 1000*i, so every iteration
plays new maps; --maps fixed replays the same maps (base seed S) every iteration, so
a change between iterations comes from the graph, not from easier or harder maps, and
the agent meets the situations it failed in again. Without --seed, fresh draws a
random seed per iteration and fixed draws one base seed for the whole run; the
adapter records each (BALROG: seed.json in the run). Test the final graph on maps
the loop never played.
--env-option key=value passes environment settings to the adapter (Crafter:
action_hints=false, survival_rules=true, max_steps=5, or any BALROG Hydra override).

Outputs in OUT: episodes/ (the runs), graphs/ (every version), acting/ (the graph
each iteration acted with), maintain/ (raw replies), loop.json (per-iteration
achievements and scores, rewritten after every iteration), changelog.json.
"""

import argparse
import json
import os
import random
import sys

from .adapter import load as load_adapter
from .llm import DEFAULT_MODEL, call_model
from .maintain import Maintainer, empty_graph, merge
from .evaluate.metrics import (churn, depth, forgetting, plan_from_gog, success_rate, unsupported_changes,
                      validate)
from .prompts import construct_facts, fingerprint
from .schema import DeltaReply, FactsReply
from .evaluate.scoring import experienced_reference, finalise, score, summary_line


def own_tree_valid(graph, env):
    """Usable for acting, judged from the graph alone: its task goal is a node and the
    goal tree under it passes every structural check."""
    root = graph.get("_root_goal")
    names = {n["name"] for n in graph.get("nodes", [])}
    if root not in names:
        return False
    by_key = {" ".join(n["name"].replace("_", " ").lower().split()): n["name"] for n in graph["nodes"]}
    keep = {by_key[k] for k in plan_from_gog(graph, root)}
    tree = {"nodes": [n for n in graph["nodes"] if n["name"] in keep],
            "edges": [e for e in graph["edges"] if e["from"] in keep and e["to"] in keep]}
    return validate(tree, root=root, env=env)["valid"]


def seed_from_prior(m, task_text, names):
    """--init zero: the zero-experience construction, merged in as episode 0."""
    raw, t_in, t_out, _ = call_model(construct_facts(task_text, [], names, env=m.env), m.model, m.temperature,
                                     FactsReply, m.base_url, m.api_key_env, m.adapter.secrets_path)
    m.tokens[0] += t_in
    m.tokens[1] += t_out
    with open(os.path.join(m.out, "raw_e0.txt"), "w") as f:
        f.write(raw)
    facts = FactsReply.model_validate_json(raw)
    m.graph, log = merge(empty_graph(), DeltaReply(task_goal=facts.task_goal, goals=facts.goals, remove=[]),
                         {"episode": 0})
    m.changelog += log


def _options(pairs):
    out = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--env-option needs key=value, got {pair!r}")
        out[key] = value
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="environment package goalnet_<env>, e.g. crafter")
    p.add_argument("--tasks", nargs="+", required=True, help="one task, or a curriculum in order")
    p.add_argument("--iterations", type=int, default=4, help="act/maintain rounds per task")
    p.add_argument("--episodes", type=int, default=5, help="episodes per iteration")
    p.add_argument("--out", required=True)
    p.add_argument("--init", default="empty", help="'empty', 'zero', or a graph JSON")
    p.add_argument("--strategy", choices=["merge", "regenerate"], default="merge")
    p.add_argument("--variant", "--recipes", dest="variant", help="world variant to play in")
    p.add_argument("--vocab", choices=["full", "observed"], default="full")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model that maintains the graph")
    p.add_argument("--actor-model", default=DEFAULT_MODEL, help="model that plays")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--seed", type=int, help="base seed (default: random)")
    p.add_argument("--maps", choices=["fresh", "fixed"], default="fresh",
                   help="fresh: new maps each iteration (seed + 1000*i); fixed: the same maps every iteration")
    p.add_argument("--env-option", nargs="*", help="environment settings as key=value, passed to the adapter")
    args = p.parse_args(argv)

    adapter = load_adapter(args.env)
    env = adapter.environment()
    options = _options(args.env_option)
    out = os.path.abspath(args.out)
    for sub in ("episodes", "graphs", "acting", "maintain"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    if args.maps == "fixed" and args.seed is None:
        args.seed = random.randrange(1, 2 ** 31)
    final_task = args.tasks[-1]
    final_ref, final_root = adapter.reference(final_task, args.variant)
    variant_id = adapter.variant_id(args.variant)

    m = Maintainer(adapter, adapter.task_text(args.tasks[0]), os.path.join(out, "maintain"), args.strategy,
                   model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                   max_repairs=args.max_repairs)
    if args.init == "zero":
        seed_from_prior(m, m.task_text, None if args.vocab == "full" else set())
    elif args.init != "empty":
        with open(args.init) as f:
            m.graph = json.load(f)

    opts = "".join(f"/{k}-{v}" for k, v in sorted(options.items()))
    record = {"env": args.env, "tasks": args.tasks, "iterations": args.iterations,
              "episodes_per_iteration": args.episodes, "init": args.init, "strategy": args.strategy,
              "variant": args.variant, "variant_id": variant_id, "vocab": args.vocab, "model": args.model,
              "actor_model": args.actor_model, "env_options": options, "prompt_fingerprint": fingerprint(),
              "maps": args.maps, "seed": args.seed,
              "label": (f"{args.env}/loop/{args.strategy}/init-{args.init if args.init in ('empty', 'zero') else 'file'}"
                        f"/{'+'.join(args.tasks)}/vocab-{args.vocab}/variant-{variant_id}/maps-{args.maps}{opts}"),
              "rounds": []}
    versions, experience, metas, paths = [m.graph], [], [], []
    last_usable = {}
    with open(os.path.join(out, "graphs", "v0.json"), "w") as f:
        json.dump(m.graph, f, indent=2)

    def save_version():
        versions.append(m.graph)
        with open(os.path.join(out, "graphs", f"v{len(versions) - 1}.json"), "w") as f:
            json.dump(m.graph, f, indent=2)

    it = 0
    for task in args.tasks:
        reference, root = adapter.reference(task, args.variant)
        # A new task (after the first, or after a supplied graph) needs its own root in
        # the graph before acting; the seeded zero graph was built for the first task.
        seeded_for_this = task == args.tasks[0] and args.init in ("empty", "zero")
        if m.graph.get("nodes") and not seeded_for_this:
            m.names = None if args.vocab == "full" else adapter.observed_names(paths)
            m.retarget(adapter.task_text(task), len(experience))
            save_version()
        m.task_text = adapter.task_text(task)
        for i in range(args.iterations):
            it += 1
            # Act: the current graph if it is usable for this task, else the last usable one, else none.
            acting = None
            if own_tree_valid(m.graph, env):
                last_usable[task] = finalise(m.graph, m.graph["_root_goal"])
            elif task in last_usable:
                print(f"WARNING: graph after {len(experience)} episodes has no valid goal tree for {task}; "
                      "acting with the last usable graph", file=sys.stderr)
            if task in last_usable:
                acting = os.path.join(out, "acting", f"it{it}.json")
                with open(acting, "w") as f:
                    json.dump(last_usable[task], f, indent=2)
            print(f"[{task} iteration {i + 1}] acting with {'graph' if acting else 'no graph (none usable)'}",
                  flush=True)
            seed = None if args.seed is None else args.seed + (0 if args.maps == "fixed" else 1000 * it)
            run_dir = adapter.run_episodes(task, acting, os.path.join(out, "episodes"), args.episodes, seed,
                                           args.actor_model, args.variant, options)

            summaries, run_metas, run_paths = adapter.load_experience([run_dir], task)
            adapter.check_variant(run_metas, args.variant)
            for e, meta, path in zip(summaries, run_metas, run_paths):
                e = {**e, "episode": len(experience) + 1}
                experience.append(e)
                metas.append(meta)
                paths.append(path)
                m.names = None if args.vocab == "full" else adapter.observed_names(paths)
                m.apply(e)
                save_version()

            state = score(m.graph, reference, experienced_reference(reference, metas, adapter), root, adapter,
                          experience)
            final_state = score(m.graph, final_ref, experienced_reference(final_ref, metas, adapter), final_root,
                                adapter, experience)
            record["rounds"].append({
                "iteration": it, "task": task, "acted_with": "graph" if acting else "none", "run": run_dir,
                "success_rate": success_rate(run_metas, adapter),
                "depth": [depth(x, reference, root, adapter) for x in run_metas],
                "achieved": [sorted(adapter.achieved_goals(x)) for x in run_metas],
                "graph_version": len(versions) - 1,
                "state": state, "state_final_task": final_state,
            })
            print(f"  success {record['rounds'][-1]['success_rate']:.2f}, depth "
                  f"{[round(d, 2) for d in record['rounds'][-1]['depth']]}; graph: {summary_line(state)}", flush=True)
            record.update(
                churn=churn(versions, adapter.goal_aliases),
                forgetting=forgetting(versions, final_ref, adapter.goal_aliases),
                unsupported_changes=unsupported_changes(m.changelog),
                input_tokens=m.tokens[0], output_tokens=m.tokens[1])
            with open(os.path.join(out, "loop.json"), "w") as f:
                json.dump(record, f, indent=2)
            with open(os.path.join(out, "changelog.json"), "w") as f:
                json.dump(m.changelog, f, indent=2)
    print(f"done: {it} iterations, {len(experience)} episodes; loop.json in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
