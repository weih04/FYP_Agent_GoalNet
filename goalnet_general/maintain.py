"""Incremental Goal-Net maintenance: one episode at a time, extract a delta and merge it.

This adapts GoG's Algorithm 1 from text chunks to episodes. For each episode the
model returns only the goals that episode bears on (prompts.extract_delta); `merge`
folds them into the graph in code, and derive_edges rewires it. The model never
rewrites goals the episode says nothing about, so they cannot drift.

Merging a reported goal (GoG matches by name embedding plus condition similarity;
here goals carry structured facts, so identity is decided exactly):
  1. same name, or a recorded alias          -> the same goal
  2. else same provisions (what it produces   -> the same goal under another name;
     or places), when it provides anything       the new name becomes an alias
  3. else                                     -> a new goal
For the same goal, an "observed" report replaces the facts (the newest evidence
wins), and a "prior" report replaces them only if the existing goal is prior too:
belief never overwrites observation. An observed goal can be revised but never
removed, and neither can the task goal, nor a goal whose failed attempts contradict
its facts (removing it would make the contradiction vanish rather than fix it), nor
a goal the same reply reports (a reply that corrects a goal and removes it means the
correction).
Evidence accumulates. After each merge,
repair also reports observed goals whose cited episodes record a different change
(problems.evidence_problems); one still contradicted after the last repair round is
downgraded to prior. It also reports any goal, prior or observed, whose action
failed while everything it lists was available (problems.failure_problems): this is
the only way experience challenges a prior belief. One still contradicted after
repair is logged as "contested". Code only reports the disagreement; the model writes the fix. Every change is logged, with the episode that
caused it, so churn and unsupported edits can be measured.

The maintained graph is a knowledge base, as GoG's is: it may hold observed goals
the current task does not need (eating, saplings). What the agent receives is the
goal tree under the task root, so readiness is judged by `tree_valid` and repair
never reports a goal for being off the current task's path (in a curriculum, that
would delete what earlier tasks taught).

    python -m goalnet_general.maintain --env crafter --task collect_diamond --runs BALROG/results/<run> --out OUT
        [--init empty|<graph.json>] [--strategy merge|regenerate] [--variant FILE]

--strategy regenerate is the ablation: the model rewrites every goal after each
episode (prompts.maintain_regenerate); the task goal is still never removed. Outputs in OUT: graph_v<t>.json per version,
changelog.json, raw replies, and trajectory.json with per-version scores and trends.
"""

import argparse
import copy
import json
import os
import sys

from .adapter import load as load_adapter
from .derive import _key, provisions
from .llm import DEFAULT_MODEL, call_model
from .problems import (contradicted_by_failures, evidence_problems, fact_problems, failure_problems,
                       unsupported_observed)
from .prompts import extract_delta, fingerprint, maintain_regenerate, repair_delta, retarget_delta
from .schema import DeltaReply, FactsReply, facts_to_node, parse_delta
from .evaluate.scoring import derive, experienced_reference, finalise, find_root, score, summary_line
from .evaluate.metrics import churn, episodes_to_threshold, forgetting, unsupported_changes

FACT_FIELDS = ("description", "required_tools", "required_materials", "postconditions", "task_list")


def empty_graph():
    return {"_root_goal": None, "nodes": [], "edges": []}


def _identity(node):
    """What a goal provides, as a comparable key; None if it provides nothing."""
    prov = frozenset((_key(item), kind) for item, kind in provisions(node))
    return prov or None


def _facts(node):
    return {f: copy.deepcopy(node.get(f)) for f in FACT_FIELDS}


def _same_facts(a, b):
    fa, fb = _facts(a), _facts(b)
    fa["required_tools"] = sorted(fa["required_tools"] or [])
    fb["required_tools"] = sorted(fb["required_tools"] or [])
    fa.pop("description"), fb.pop("description")
    return fa == fb


def _find(nodes, incoming):
    key = _key(incoming["name"])
    for n in nodes:
        if key == _key(n["name"]) or key in {_key(a) for a in n.get("aliases", [])}:
            return n, "name"
    ident = _identity(incoming)
    if ident is not None:
        for n in nodes:
            if _identity(n) == ident:
                return n, "provisions"
    return None, None


def merge(graph, delta, source, contradicted=()):
    """Fold a DeltaReply into `graph`. Returns (new graph with edges derived, changelog).

    `source` labels the changes ({"episode": n} or {"repair": r, "episode": n}).
    `contradicted` names goals whose failures contradict them; they are not removed.
    """
    nodes = copy.deepcopy([n for n in graph.get("nodes", [])])
    log = []
    for g in delta.goals:
        incoming = facts_to_node(g)
        existing, how = _find(nodes, incoming)
        if existing is None:
            incoming["aliases"] = []
            nodes.append(incoming)
            log.append({**source, "op": "add", "goal": incoming["name"], "basis": incoming["basis"],
                        "evidence": incoming["evidence"]})
            continue
        if how == "provisions" and _key(incoming["name"]) != _key(existing["name"]):
            existing.setdefault("aliases", [])
            if incoming["name"] not in existing["aliases"]:
                existing["aliases"].append(incoming["name"])
        new_evidence = [e for e in incoming["evidence"] if e not in existing.get("evidence", [])]
        existing["evidence"] = existing.get("evidence", []) + new_evidence
        if incoming["basis"] == "prior" and existing.get("basis") == "observed":
            if not _same_facts(existing, incoming):
                log.append({**source, "op": "kept", "goal": existing["name"],
                            "reason": "a prior report does not override observed facts",
                            "proposed": _facts(incoming)})
            continue
        if _same_facts(existing, incoming):
            if new_evidence or incoming["basis"] != existing.get("basis"):
                log.append({**source, "op": "confirm", "goal": existing["name"], "basis": incoming["basis"],
                            "evidence": new_evidence})
            if incoming["basis"] == "observed":
                existing["basis"] = "observed"
            continue
        before = _facts(existing)
        for f in FACT_FIELDS:
            existing[f] = copy.deepcopy(incoming[f])
        existing["reasoning"] = incoming.get("reasoning", "")
        if incoming["basis"] == "observed":
            existing["basis"] = "observed"
        log.append({**source, "op": "revise", "goal": existing["name"], "matched_by": how,
                    "basis": incoming["basis"], "evidence": new_evidence,
                    "before": before, "after": _facts(existing)})
    task_keys = {_key(t) for t in (delta.task_goal, graph.get("_root_goal")) if t}
    reported = {_key(g.name) for g in delta.goals}
    for r in delta.remove:
        existing, _ = _find(nodes, {"name": r.name, "postconditions": {}})
        if existing is None:
            log.append({**source, "op": "remove-missing", "goal": r.name, "reason": r.reason})
            continue
        if _key(r.name) in reported:
            # Models list a goal they just corrected under remove, citing the fixed problem as the reason.
            log.append({**source, "op": "kept", "goal": existing["name"], "reason": r.reason,
                        "policy": "a goal the same reply reports is corrected, not removed"})
            continue
        if task_keys & {_key(x) for x in [existing["name"]] + existing.get("aliases", [])}:
            # Without its task goal the graph cannot guide acting at all; a wrong recipe is revised instead.
            log.append({**source, "op": "kept", "goal": existing["name"], "reason": r.reason,
                        "policy": "the task goal is revised, never removed"})
            continue
        if _key(existing["name"]) in {_key(c) for c in contradicted}:
            log.append({**source, "op": "kept", "goal": existing["name"], "reason": r.reason,
                        "policy": "a goal its failed attempts contradict is corrected, never removed"})
            continue
        if existing.get("basis") == "observed":
            # An observed goal happened; its facts can be corrected, but it cannot be undone.
            log.append({**source, "op": "kept", "goal": existing["name"], "reason": r.reason,
                        "policy": "observed goals are revised, never removed"})
            continue
        nodes.remove(existing)
        log.append({**source, "op": "remove", "goal": existing["name"], "reason": r.reason,
                    "before": _facts(existing), "basis": existing.get("basis")})

    root = delta.task_goal
    match, _ = _find(nodes, {"name": root, "postconditions": {}})
    out = derive({"_root_goal": match["name"] if match else root, "nodes": nodes, "edges": []})
    return out, log


def regenerate(graph, reply, source):
    """Ablation: the reply replaces every goal. Evidence of goals kept under the same
    name is carried over, so provenance is comparable with merge. A task goal the
    rewrite omits is kept as it was, as merge keeps it."""
    old = {_key(n["name"]): n for n in graph.get("nodes", [])}
    nodes = []
    for g in reply.goals:
        n = facts_to_node(g)
        prev = old.get(_key(n["name"]))
        n["aliases"] = list(prev.get("aliases", [])) if prev else []
        if prev:
            n["evidence"] = prev.get("evidence", []) + [e for e in n["evidence"] if e not in prev.get("evidence", [])]
        nodes.append(n)
    new = {_key(n["name"]) for n in nodes}
    log = []
    old_root = graph.get("_root_goal")
    if old_root and _key(old_root) in old and _key(old_root) not in new:
        nodes.append(copy.deepcopy(old[_key(old_root)]))
        new.add(_key(old_root))
        log.append({**source, "op": "kept", "goal": old[_key(old_root)]["name"], "reason": "omitted from the rewrite",
                    "policy": "the task goal is revised, never removed"})
    log += [{**source, "op": "add", "goal": n["name"], "basis": n["basis"], "evidence": n["evidence"]}
           for n in nodes if _key(n["name"]) not in old]
    log += [{**source, "op": "remove", "goal": n["name"], "reason": "omitted from the rewrite",
             "before": _facts(n), "basis": n.get("basis")} for k, n in old.items() if k not in new]
    for n in nodes:
        prev = old.get(_key(n["name"]))
        if prev and not _same_facts(prev, n):
            log.append({**source, "op": "revise", "goal": n["name"], "matched_by": "name", "basis": n["basis"],
                        "evidence": n["evidence"], "before": _facts(prev), "after": _facts(n)})
    return derive({"_root_goal": reply.task_goal, "nodes": nodes, "edges": []}), log


class Maintainer:
    """Holds one graph and applies episodes to it, calling the model for each delta."""

    def __init__(self, adapter, task_text, out, strategy="merge", names=None, model=DEFAULT_MODEL,
                 temperature=0.0, base_url=None, api_key_env="OPENAI_API_KEY", max_repairs=2, graph=None):
        self.adapter, self.env = adapter, adapter.environment()
        self.task_text, self.out, self.strategy, self.names = task_text, out, strategy, names
        self.model, self.temperature, self.base_url, self.api_key_env = model, temperature, base_url, api_key_env
        self.max_repairs = max_repairs
        self.graph = derive(graph) if graph and graph.get("nodes") else empty_graph()
        self.changelog, self.tokens = [], [0, 0]
        self.episodes = {}    # every summary applied so far: observed goals may cite any of them
        os.makedirs(out, exist_ok=True)

    def _call(self, prompt, fmt, name):
        raw, t_in, t_out, served = call_model(prompt, self.model, self.temperature, fmt,
                                              self.base_url, self.api_key_env, self.adapter.secrets_path)
        self.tokens[0] += t_in
        self.tokens[1] += t_out
        with open(os.path.join(self.out, name), "w") as f:
            f.write(raw)
        return raw

    def apply(self, episode):
        """One maintenance step for one episode summary (its "episode" number is global)."""
        n = episode["episode"]
        self.episodes[n] = episode
        source = {"episode": n}
        if self.strategy == "merge":
            raw = self._call(extract_delta(self.task_text, self.graph, episode, self.names, env=self.env), DeltaReply,
                             f"raw_e{n}.txt")
            graph, log = merge(self.graph, parse_delta(raw), source, self._contradicted(self.graph))
        else:
            raw = self._call(maintain_regenerate(self.task_text, self.graph, episode, self.names, env=self.env), FactsReply,
                             f"raw_e{n}.txt")
            graph, log = regenerate(self.graph, FactsReply.model_validate_json(raw), source)
        graph, log = self._repair(graph, log, n, f"e{n}")
        self.graph = graph
        self.changelog += log
        return graph, log

    def _repair(self, graph, log, n, tag):
        # Repairs come back as deltas and merge the same way, in both strategies, so a
        # repair never rewrites goals it was not asked about.
        for r in range(1, self.max_repairs + 1):
            problems = fact_problems(graph, self.env, knowledge_base=True) + \
                evidence_problems(graph, self.episodes, self.adapter.evidence_records) + \
                failure_problems(graph, self.episodes, self.adapter.failure_records)
            if not problems:
                break
            raw = self._call(repair_delta(self.task_text, graph, problems, self.names, env=self.env), DeltaReply,
                             f"raw_{tag}_repair{r}.txt")
            graph, rlog = merge(graph, parse_delta(raw), {"episode": n, "repair": r}, self._contradicted(graph))
            log = log + rlog
        graph, log = self._downgrade_unsupported(graph, log, n)
        return graph, log + self._contested(graph, n)

    def _contradicted(self, graph):
        return {name for name, _ in contradicted_by_failures(graph, self.episodes, self.adapter.failure_records)}

    def _contested(self, graph, n):
        """Goals whose failures still contradict them after repair. Nothing is changed:
        code may not write a recipe, and a prior goal has no label left to lose."""
        return [{"episode": n, "op": "contested", "goal": name, "reason": msg}
                for name, msg in contradicted_by_failures(graph, self.episodes, self.adapter.failure_records)]

    def _downgrade_unsupported(self, graph, log, n):
        """An observed goal its evidence still contradicts after repair loses the label
        (and with it, protection from removal); its facts stay as the model wrote them."""
        unsupported = {name for name, _ in unsupported_observed(graph, self.episodes, self.adapter.evidence_records)}
        if not unsupported:
            return graph, log
        graph = copy.deepcopy(graph)
        for node in graph["nodes"]:
            if node["name"] in unsupported:
                node["basis"] = "prior"
                log = log + [{"episode": n, "op": "downgrade", "goal": node["name"], "basis": "prior",
                              "reason": "its cited evidence still disagrees with its facts after repair"}]
        return graph, log

    def retarget(self, task_text, episodes_so_far):
        """Curriculum: switch to a new task, keeping the graph; the model names the new root."""
        self.task_text = task_text
        tag = f"retarget_{task_text.replace(' ', '_')}"
        raw = self._call(retarget_delta(task_text, self.graph, self.names, env=self.env), DeltaReply, f"raw_{tag}.txt")
        source = {"episode": episodes_so_far, "retarget": task_text}
        graph, log = merge(self.graph, parse_delta(raw), source, self._contradicted(self.graph))
        graph, log = self._repair(graph, log, episodes_so_far, tag)
        self.graph = graph
        self.changelog += log
        return graph, log


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="environment package goalnet_<env>, e.g. crafter")
    p.add_argument("--task", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="run directories, replayed in the order given")
    p.add_argument("--out", required=True)
    p.add_argument("--experience-task", help="read episodes played under this task (default: --task); the world is the same, so another task's episodes are valid experience")
    p.add_argument("--init", default="empty", help="'empty' or a graph JSON (e.g. a zero-experience final_<k>.json)")
    p.add_argument("--strategy", choices=["merge", "regenerate"], default="merge")
    p.add_argument("--vocab", choices=["full", "observed"], default="full")
    p.add_argument("--variant", "--recipes", dest="variant", help="world variant the episodes were played in")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.8, help="closure F1 for episodes-to-threshold")
    args = p.parse_args(argv)

    adapter = load_adapter(args.env)
    reference, root = adapter.reference(args.task, args.variant)
    variant_id = adapter.variant_id(args.variant)
    experience, metas, paths = adapter.load_experience(args.runs, args.experience_task or args.task)
    adapter.check_variant(metas, args.variant)
    aliases = adapter.goal_aliases
    init = empty_graph()
    if args.init != "empty":
        with open(args.init) as f:
            init = json.load(f)

    m = Maintainer(adapter, adapter.task_text(args.task), args.out, args.strategy, model=args.model,
                   temperature=args.temperature, base_url=args.base_url, api_key_env=args.api_key_env,
                   max_repairs=args.max_repairs, graph=init)
    versions = [m.graph]
    states = [score(m.graph, reference, experienced_reference(reference, [], adapter), root, adapter, [])]
    with open(os.path.join(args.out, "graph_v0.json"), "w") as f:
        json.dump(m.graph, f, indent=2)
    for i, episode in enumerate(experience, 1):
        # The vocabulary grows with what has been seen so far, never ahead of it.
        m.names = None if args.vocab == "full" else adapter.observed_names(paths[:i])
        graph, log = m.apply(episode)
        versions.append(graph)
        state = score(graph, reference, experienced_reference(reference, metas[:i], adapter), root, adapter,
                      experience[:i])
        states.append(state)
        with open(os.path.join(args.out, f"graph_v{i}.json"), "w") as f:
            json.dump(graph, f, indent=2)
        if state["tree_valid"]:
            with open(os.path.join(args.out, "final.json"), "w") as f:
                json.dump(finalise(graph, find_root(graph, root, aliases)), f, indent=2)
        print(f"episode {i}: {len(log)} changes; {summary_line(state)}")

    trajectory = {
        "task": args.task, "strategy": args.strategy, "vocab": args.vocab, "init": args.init,
        "env": args.env,
        "label": (f"{args.env}/maintain/{args.strategy}/init-{'empty' if args.init == 'empty' else 'file'}"
                  f"/vocab-{args.vocab}/variant-{variant_id}"),
        "prompt_fingerprint": fingerprint(),
        "variant": args.variant, "variant_id": variant_id, "runs": args.runs, "model": args.model,
        "episodes": len(experience),
        "valid": [s["valid"] for s in states],
        "tree_valid": [s["tree_valid"] for s in states],
        "closure_f1": [s["fidelity_full"]["closure"]["f1"] for s in states],
        "edge_f1": [s["fidelity_full"]["edges"]["f1"] for s in states],
        "node_f1": [s["fidelity_full"]["nodes"]["f1"] for s in states],
        "churn": churn(versions, aliases),
        "forgetting": forgetting(versions, reference, aliases),
        "unsupported_changes": unsupported_changes(m.changelog),
        "episodes_to_threshold": episodes_to_threshold(
            [s["fidelity_full"]["closure"]["f1"] for s in states], args.threshold),
        "input_tokens": m.tokens[0], "output_tokens": m.tokens[1],
        "states": states,
    }
    with open(os.path.join(args.out, "changelog.json"), "w") as f:
        json.dump(m.changelog, f, indent=2)
    with open(os.path.join(args.out, "trajectory.json"), "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"closure F1 by version {[round(x, 2) for x in trajectory['closure_f1']]}; "
          f"forgetting {trajectory['forgetting']}; unsupported changes {len(trajectory['unsupported_changes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
