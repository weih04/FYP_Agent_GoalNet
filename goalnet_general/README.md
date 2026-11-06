# goalnet_general

Environment-agnostic Goal Net (GoG) construction, maintenance and evaluation.
Nothing here imports an environment package (a test enforces it); every command
takes `--env NAME` and loads `goalnet_NAME.ADAPTER`.

| Module | Role |
|---|---|
| `prompts.py`, `schema.py`, `problems.py`, `evaluate/metrics/validity.py` | everything the model reads; fingerprinted and frozen together |
| `environment.py` | `Environment`: actions, task_list shorthands, names (data, not prompt) |
| `adapter.py` | the interface an environment package implements |
| `derive.py` | edges from goal facts (GoG §3.1.2) |
| `evaluate/metrics/` | one module per metric family: `validity`, `fidelity`, `provenance`, `plan_quality`, `maintenance`, `acting` |
| `evaluate/scoring.py` | combines the metrics into one score per graph, used by every command |
| `evaluate/aggregate.py` | bootstrap CIs, paired tests, refuses to pool prompt versions |
| `evaluate/splits.py` | dev/test split check and the prompt freeze |
| `llm.py` | structured-output calls to any OpenAI-compatible endpoint |
| `run_construct.py` | one-shot construction (facts mode / edges-mode ablation) |
| `maintain.py` | per-episode delta + merge maintenance (regenerate ablation) |
| `run_loop.py` | act → maintain → act closed loop, curriculum |
| `baseline_symbolic.py` | non-LLM precondition induction |

## Adding an environment

Write a package `goalnet_<env>/` beside this one, starting from a copy of
`goalnet_crafter/`. The only thing goalnet_general uses is `ADAPTER` in its
`__init__.py`, an instance of a subclass of `adapter.Adapter`. Implement:

| Method | What it supplies |
|---|---|
| `environment()` | actions, task_list shorthands, every name, which names persist when used |
| `task_text(task)` | how a task id is stated to the model |
| `tasks` | `{"dev": [...], "test": [...]}` |
| `reference(task, variant)` | the reference GoG and its root goal |
| `variant_id(variant)` | a stable id of a world variant (`"default"` if unmodified) |
| `goal_aliases` | other spellings of reference goal names, for scoring |
| `load_experience(runs, task)` | episode summaries numbered 1..N, metas, per-episode paths |
| `check_variant(metas, variant)` | raise if episodes were played in another variant |
| `achieved_goals(meta)`, `episode_succeeded(meta)` | what an episode achieved |
| `observed_names(paths)` | names the agent saw (the `--vocab observed` condition) |
| `episode_shows(episode, node)` | whether a summary shows a goal achieved (evidence checks) |
| `successes(paths)` | success records for the symbolic baseline |
| `run_episodes(...)` | play episodes, optionally guided by a graph (closed loop) |

The frozen prompts don't change. The prompts assume GoG's goal schema: goals that
need tools, use up materials, and produce or place things. That fits crafting and
procedural domains, but not pure navigation.

Tests: `python -m pytest goalnet_general/tests goalnet_<env>/tests -q`.
