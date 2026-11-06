"""Construct Goal Nets from an agent's experience and score them against the reference.

    python -m goalnet_general.run_construct --env crafter --task collect_diamond --zero-experience --out OUT
    python -m goalnet_general.run_construct --env crafter --task collect_diamond --runs <run dir> --out OUT

--env NAME loads goalnet_NAME's adapter (see adapter.py); everything about the
environment - its actions and names, its experience files, its reference graphs and
world variants - comes from there.

--mode facts (default): the model states each goal's facts with basis and evidence;
derive.derive_edges computes the edges (GoG §3.1.2). --mode edges: the model writes
the edges itself, as in the pilots; kept as the ablation.

--vocab full names everything the environment defines; --vocab observed names only the
items in the agent's own observations (none for zero-experience), so the prompt
carries nothing the agent has not seen.

--variant scores against a world variant's reference (for Crafter, a recipe override
file; --recipes is the same flag) and refuses experience played in another variant.

Replies use structured outputs, so they always parse; graph structure is still left
to the model (edges mode) or to its facts (facts mode). Each condition is called
`--repeats` times. `--dry-run` writes the prompt and stops before any API call.

After construction an invalid graph goes through up to `--max-repairs` rounds of
check-and-repair. The check never sees the reference graph. Both the one-shot
(initial) and the repaired (final) graph are scored.

The prompt fingerprint and dev/test split are recorded; a test-task run with an
unfrozen prompt is refused unless --unfrozen-ok (see splits.py).

Outputs in OUT: prompt.txt, experience.json, raw_<k>[_repair<r>].txt,
graph_<k>[_repair<r>].json, final_<k>.json (valid graphs, ready for agent.goalnet_path)
and scores.json. A reply that cannot be parsed is scored as an invalid graph and
reported, not dropped: a failed construction is a result.
"""

import argparse
import json
import os
import sys

from .evaluate import splits
from .adapter import load as load_adapter
from .llm import DEFAULT_MODEL, call_model
from .problems import edge_problems, fact_problems
from .prompts import construct, construct_facts, repair, repair_facts
from .schema import FactsReply, GoalNetReply, parse_facts, parse_reply
from .evaluate.scoring import derive, experienced_reference, finalise, find_root, score, summary_line

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="environment package goalnet_<env>, e.g. crafter")
    p.add_argument("--task", required=True, help="task id of that environment, e.g. collect_diamond")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--zero-experience", action="store_true", help="empty experience log (control)")
    src.add_argument("--runs", nargs="+", help="run directories whose episodes form the experience")
    p.add_argument("--out", required=True)
    p.add_argument("--experience-task", help="read episodes played under this task (default: --task); the world is the same, so another task's episodes are valid experience")
    p.add_argument("--mode", choices=["facts", "edges"], default="facts")
    p.add_argument("--vocab", choices=["full", "observed"], default="full")
    p.add_argument("--variant", "--recipes", dest="variant", help="world variant the episodes were played in")
    p.add_argument("--label", help="condition label for aggregate.py; default derived from the settings")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", help="OpenAI-compatible endpoint for another model family")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true", help="write the prompt and stop before calling the API")
    p.add_argument("--max-repairs", type=int, default=3,
                   help="check-and-repair rounds after construction; 0 = one-shot only")
    p.add_argument("--rescore", action="store_true",
                   help="re-score the raw_<k>.txt responses already in --out instead of calling the API")
    p.add_argument("--unfrozen-ok", action="store_true", help="allow a test-task run with an unfrozen prompt")
    args = p.parse_args(argv)

    adapter = load_adapter(args.env)
    env = adapter.environment()
    split = splits.check(args.task, adapter, args.unfrozen_ok or args.rescore or args.dry_run)
    reference, root = adapter.reference(args.task, args.variant)
    variant_id = adapter.variant_id(args.variant)
    experience, metas, paths = adapter.load_experience(args.runs, args.experience_task or args.task)
    adapter.check_variant(metas, args.variant)
    experienced = experienced_reference(reference, metas, adapter)
    names = None if args.vocab == "full" else adapter.observed_names(paths)

    os.makedirs(args.out, exist_ok=True)
    task_text = adapter.task_text(args.task)
    build = construct_facts if args.mode == "facts" else construct
    prompt = build(task_text, experience, names, env=env)
    # A rescore keeps the prompt that produced the saved responses, which may predate
    # the current prompts.py.
    if not args.rescore:
        with open(os.path.join(args.out, "prompt.txt"), "w") as f:
            f.write(prompt)
        with open(os.path.join(args.out, "experience.json"), "w") as f:
            json.dump(experience, f, indent=2)
    print(f"prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens), {len(experience)} episodes, "
          f"env {args.env}, mode {args.mode}, vocab {args.vocab}, variant {variant_id}, split {split['split']}, "
          f"experienced goals: {[n['name'] for n in experienced['nodes']]}")
    if args.dry_run:
        print(f"dry run: prompt written to {args.out}/prompt.txt; no API call made")
        return 0

    reply_format = FactsReply if args.mode == "facts" else GoalNetReply
    results, tokens_in, tokens_out, previous = [], 0, 0, None
    if args.rescore:
        # Rescoring makes no calls; keep the token counts of the run that made the replies.
        with open(os.path.join(args.out, "scores.json")) as f:
            previous = json.load(f)
        tokens_in, tokens_out = previous["input_tokens"], previous["output_tokens"]

    def respond(name, prompt_text, k):
        """Model reply for one call; in --rescore mode, the saved reply (None once they run out)."""
        nonlocal tokens_in, tokens_out
        path = os.path.join(args.out, name)
        if args.rescore:
            if not os.path.exists(path):
                return None, None
            with open(path) as f:
                return f.read(), previous["results"][k]["served_model"]
        raw, t_in, t_out, served = call_model(prompt_text, args.model, args.temperature, reply_format,
                                              args.base_url, args.api_key_env, adapter.secrets_path)
        tokens_in, tokens_out = tokens_in + t_in, tokens_out + t_out
        with open(path, "w") as f:
            f.write(raw)
        return raw, served

    def parse(raw):
        return derive(parse_facts(raw)) if args.mode == "facts" else parse_reply(raw)

    def problems_of(graph):
        if args.mode == "facts":
            return fact_problems(graph, env)
        return edge_problems(graph, find_root(graph, root, adapter.goal_aliases), task_text, env)

    def repair_prompt(graph, problems):
        if args.mode == "facts":
            return repair_facts(task_text, graph, problems, names, env=env)
        return repair(task_text, graph, problems, names, env=env)

    for k in range(args.repeats):
        history, current_prompt, initial, final_graph = [], prompt, None, None
        for r in range(args.max_repairs + 1):
            suffix = "" if r == 0 else f"_repair{r}"
            raw, served = respond(f"raw_{k}{suffix}.txt", current_prompt, k)
            if raw is None:
                break
            try:
                graph = parse(raw)
            except ValueError as e:  # json.JSONDecodeError and pydantic ValidationError are ValueErrors
                print(f"WARNING: repeat {k} round {r} reply could not be parsed ({e})", file=sys.stderr)
                state = {"valid": False, "parse_error": str(e)}
                problems = [f"reply could not be parsed: {e}"]
                graph = None
            else:
                with open(os.path.join(args.out, f"graph_{k}{suffix}.json"), "w") as f:
                    json.dump(graph, f, indent=2)
                state = score(graph, reference, experienced, root, adapter, experience)
                problems = problems_of(graph)
                final_graph = graph
            history.append({"round": r, "valid": state["valid"], "problems": problems})
            if initial is None:
                initial = state
            final = state
            print(f"repeat {k} round {r}: {summary_line(state)}")
            if state["valid"] or r == args.max_repairs:
                break
            # With no parsed graph there is nothing to repair, so the round asks again.
            current_prompt = repair_prompt(graph, problems) if graph is not None else current_prompt
        if final["valid"]:
            with open(os.path.join(args.out, f"final_{k}.json"), "w") as f:
                json.dump(finalise(final_graph, find_root(final_graph, root, adapter.goal_aliases)), f, indent=2)
        results.append({"repeat": k, "served_model": served, "repair_rounds": len(history) - 1,
                        "valid_initial": initial["valid"], "valid_final": final["valid"],
                        "initial": initial, "final": final, "history": history})

    condition = "zero_experience" if args.zero_experience else "experience"
    summary = {
        "task": args.task,
        "env": args.env,
        "label": args.label or f"{args.env}/{args.mode}/{condition}/vocab-{args.vocab}/variant-{variant_id}",
        "condition": condition,
        "mode": args.mode,
        "vocab": args.vocab,
        "variant": args.variant,
        "variant_id": variant_id,
        "runs": args.runs or [],
        "episodes": len(experience),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "repeats": args.repeats,
        "max_repairs": args.max_repairs,
        "experienced_goals": [n["name"] for n in experienced["nodes"]],
        **split,
        # Initial = one-shot construction, final = after the automatic check-and-repair loop.
        "validity_rate_initial": sum(r["valid_initial"] for r in results) / len(results),
        "validity_rate_final": sum(r["valid_final"] for r in results) / len(results),
        "mean_repair_rounds": sum(r["repair_rounds"] for r in results) / len(results),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "results": results,
    }
    if args.rescore and previous is not None:
        # The replies were produced by the prompt recorded then, not by today's prompts.py.
        summary["prompt_fingerprint"] = previous.get("prompt_fingerprint", "unrecorded")
        summary["prompt_frozen"] = previous.get("prompt_frozen", False)
        if "label" in previous and not args.label:
            summary["label"] = previous["label"]
    with open(os.path.join(args.out, "scores.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"validity rate one-shot {summary['validity_rate_initial']:.2f}, after repair "
          f"{summary['validity_rate_final']:.2f} (mean {summary['mean_repair_rounds']:.1f} rounds); "
          f"tokens {tokens_in} in / {tokens_out} out; scores in {args.out}/scores.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
