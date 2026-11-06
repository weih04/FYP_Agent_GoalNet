"""Prompts for constructing and maintaining a Goal-Oriented Graph.

Two construction modes, compared as an ablation:

* facts (default) - the model states each goal's facts (what it needs, what it
  gives) and derive.derive_edges wires the graph, as GoG §3.1.2 derives subgoal
  edges by matching preconditions to postconditions. The model never writes edges.
* edges - the model writes goals, preconditions and edges itself (the pilots).

Maintenance is facts-only: each episode yields a *delta* (goals added, corrected or
removed) that maintain.merge folds into the graph, as GoG's Algorithm 1 folds each
chunk's goals into the existing graph. `maintain_regenerate` (the model rewrites
every goal each time) is the ablation for that choice.

No prompt states a dependency, recipe or ordering: supplying those would hand the
model the structure it is supposed to recover. Repair prompts return only the
problems `validate` found; they add no domain knowledge.

Prompt structure borrowed from the GoG extraction prompt (Fig. A.4): numbered steps,
a grounding constraint ("use only what the experience shows"), and reasoning about
where each goal's facts come from written before the facts themselves.

The prompts are written to be environment-agnostic, so they can be frozen once and
reused in other environments: nothing in the wording names Crafter, its actions or
its items. Those come from an environment.Environment profile (actions, task_list
shorthands, item names) built by the environment's adapter; it is data and not part
of the fingerprint. The worked
examples use an invented clay/kiln domain with invented actions, so they show the
format without supplying any structure that is scored in any environment.
"""

import json

from .schema import from_gog, gog_to_facts

# --- shared blocks -------------------------------------------------------------


def _env(env):
    if env is None:
        raise ValueError("prompts need the environment's Environment (adapter.environment())")
    return env


def _vocab_block(names=None, env=None):
    """Valid actions, task_list shorthands, and the names the model may use.

    names=None gives every name the environment defines. A collection gives only those
    (vocabulary="observed": names that appeared in the agent's own observations), so the
    prompt itself tells the model nothing the agent has not seen.
    """
    env = _env(env)
    lines = [f"Valid actions: {', '.join(env.actions)}"]
    for sh in env.shorthands:
        lines.append(f'A task_list step may also be "{sh.form}": {sh.meaning}.')
    if names is None:
        lines.append(f"Known names: {', '.join(sorted(env.names))}")
    elif not names:
        lines.append("Names seen in your experience: (none yet)")
    else:
        lines.append(f"Names seen in your experience: {', '.join(sorted(names))}")
        lines.append("Something you have not seen may still exist; if you need it, name it as you expect "
                     "the environment to, in the same style as the names above.")
    return "\n".join(lines)


def _evidence_preamble(experience_log):
    if experience_log:
        return ("Below is a log of your own attempts in the environment. Each interaction records "
                "the state you acted from, the action, and what changed afterwards.",
                json.dumps(experience_log, indent=2))
    return ("You have not yet acted in this environment.", "(no episodes have been played yet)")


# --- facts mode ----------------------------------------------------------------

_FACTS_FIELDS = """Each goal has these fields, in this order:
  reasoning            first, say where this goal's facts come from: which episodes and steps
                       show it, or that the experience does not show it
  name                 "<verb> <thing>", e.g. "collect clay"
  description          one sentence saying what the goal involves
  required_tools       things that must be available - held, or present nearby - and are not used up
  required_materials   items used up, as a list of {"item": <name>, "quantity": <n>}
  produces             items gained, as a list of {"item": <name>, "quantity": <n>}
  places               the name of the thing the goal puts in the world, or null
  task_list            the actions performed, each a valid action or an allowed shorthand
  basis                "observed" if the experience shows this goal achieved; "prior" otherwise
  evidence             for an observed goal, where it is shown: [{"episode": <n>, "step": <n>}];
                       for a prior goal, []
and task_goal is the name of the goal that achieves the task.

How the graph is built from these facts: a goal that requires an item is linked to every goal
that produces it (for materials) or produces or places it (for tools). You do not write these
links. So a requirement only counts if it is spelled exactly like the item another goal produces
or places, and every item a goal requires must be produced or placed by some goal.

Example (format only; an invented environment, not the one you are acting in):
{"task_goal": "fire brick",
 "goals": [
  {"reasoning": "Episode 2 step 14: dig while at clay gave clay +1.",
   "name": "collect clay", "description": "Go to clay and dig to obtain clay.",
   "required_tools": [], "required_materials": [],
   "produces": [{"item": "clay", "quantity": 1}], "places": null,
   "task_list": ["go to clay", "dig"],
   "basis": "observed", "evidence": [{"episode": 2, "step": 14}]},
  {"reasoning": "Not shown in the experience; a kiln is usually built from clay.",
   "name": "build kiln", "description": "Build a kiln, using clay.",
   "required_tools": [], "required_materials": [{"item": "clay", "quantity": 2}],
   "produces": [], "places": "kiln", "task_list": ["build kiln"],
   "basis": "prior", "evidence": []},
  {"reasoning": "Not shown in the experience.",
   "name": "fire brick", "description": "Fire clay into a brick next to a kiln.",
   "required_tools": ["kiln"], "required_materials": [{"item": "clay", "quantity": 1}],
   "produces": [{"item": "brick", "quantity": 1}], "places": null, "task_list": ["fire"],
   "basis": "prior", "evidence": []}]}
Here "fire brick" is linked to "build kiln" (it needs the kiln in place) and to "collect clay"."""

_FACTS_RULES = """Rules:
  - for an observed goal, state only requirements and products the experience shows: what was
    available and used up when it succeeded, and what was gained. Attempts that changed nothing
    show what was missing. Do not add requirements the experience does not show
  - goals the task needs but the experience does not show may be added; mark them "prior"
  - include every goal needed to achieve the task, and only those
  - write each goal once, under one name; do not repeat a goal
  - if several tools would do, list only the least advanced one that is enough
  - the graph must not be circular: no goal may need, directly or through other goals, an item
    that only it provides
  - every name must be written exactly as it appears in the names above"""


def construct_facts(task, experience_log, names=None, env=None):
    """Facts-mode construction from experience alone; [] is the zero-experience control."""
    preamble, evidence = _evidence_preamble(experience_log)
    return f"""You are building a Goal-Oriented Graph: the goals an agent must achieve to complete a
task, each with what it needs and what it gives.

Task: {task}

{preamble}

Experience:
{evidence}

{_vocab_block(names, env)}

Steps:
1. Go through the experience. For every interaction that gained something or put something in
   place, identify the goal it achieved and what was available when it succeeded.
2. Decide which goals the task needs, including any the experience does not show.
3. Write each goal: reasoning first, then its facts, basis and evidence.

{_FACTS_FIELDS}

{_FACTS_RULES}"""


def _show_facts(graph):
    return json.dumps(gog_to_facts(graph), indent=2)


def _show_links(graph):
    lines = [f"  {e['from']!r} needs {e['to']!r}" for e in graph.get("edges", [])]
    return "\n".join(lines) or "  (none)"


def repair_facts(task, graph, problems, names=None, env=None):
    """Return the structural problems of a facts-mode graph; the model restates the facts.

    The links shown are the ones derived from the model's own facts, so a cycle or an
    unreachable goal can be traced to the requirement or product that causes it.
    """
    listed = "\n".join(f"  - {p}" for p in problems)
    return f"""You built the Goal-Oriented Graph below for a task. Its links were computed from your
facts, and an automatic structural check found problems.

Task: {task}

Your goals:
{_show_facts(graph)}

Links computed from them:
{_show_links(graph)}

Problems found:
{listed}

Return the corrected goals. Fix these problems by correcting the facts that cause them and change
nothing else: keep every goal and fact that is not involved in a problem as it is.

{_vocab_block(names, env)}

{_FACTS_FIELDS}

{_FACTS_RULES}"""


_DELTA_FIELDS = """Reply with:
  task_goal  the name of the goal that achieves the task
  goals      only the goals this evidence adds or corrects, each written in full
  remove     existing goals the evidence shows cannot be right, each with the reason. A goal this
             evidence merely does not show is not wrong: leave it. Goals marked "observed"
             happened and are never removed; correct their facts instead
Goals you do not mention are kept exactly as they are."""


def extract_delta(task, graph, episode, names=None, env=None):
    """One maintenance step: what a single new episode adds to or corrects in the graph.

    The model returns only the goals the episode bears on; maintain.merge folds them
    into the graph. Showing the current goals lets it reuse their names, so a goal
    seen again merges rather than appearing twice.
    """
    return f"""You maintain a Goal-Oriented Graph for an agent: the goals it must achieve to complete a
task, each with what it needs and what it gives.

Task: {task}

Current goals:
{_show_facts(graph) if graph.get("nodes") else "(none yet)"}

A new episode has been played:
{json.dumps(episode, indent=2)}

{_vocab_block(names, env)}

Steps:
1. Go through the episode. For every interaction that gained something or put something in place,
   identify the goal it achieved and what was available when it succeeded.
2. Compare with the current goals. Report a goal if the episode shows it for the first time, or
   shows that its facts are wrong or incomplete: it succeeded without something listed as
   required, it failed while everything listed was available, or it gave something not listed.
3. Report a goal the task needs but no episode has shown only if it is missing from the current
   goals; mark it "prior".
4. Use the existing name when reporting on an existing goal.

{_FACTS_FIELDS}

{_FACTS_RULES}

{_DELTA_FIELDS}"""


def retarget_delta(task, graph, names=None, env=None):
    """A curriculum step: the agent's task changes, and no new episode has been played.

    The graph is kept as knowledge; the model names the goal for the new task and adds
    any goals that task needs which the graph lacks. Nothing existing is changed.
    """
    return f"""You maintain a Goal-Oriented Graph for an agent: the goals it must achieve to complete a
task, each with what it needs and what it gives. The agent's task has just changed.

New task: {task}

Current goals:
{_show_facts(graph) if graph.get("nodes") else "(none yet)"}

{_vocab_block(names, env)}

Name the goal that achieves the new task as task_goal, reusing an existing goal if one achieves it.
If goals the new task needs are missing, add them, marked "prior". Do not change or remove
existing goals: no new experience has arrived.

{_FACTS_FIELDS}

{_FACTS_RULES}

{_DELTA_FIELDS}"""


def repair_delta(task, graph, problems, names=None, env=None):
    """Maintenance-time repair: the fix comes back as a delta and is merged like an episode."""
    listed = "\n".join(f"  - {p}" for p in problems)
    return f"""You maintain the Goal-Oriented Graph below. Its links were computed from the goals'
facts, and automatic checks of its structure and of its evidence found problems.

Task: {task}

Current goals:
{_show_facts(graph)}

Links computed from them:
{_show_links(graph)}

Problems found:
{listed}

Fix these problems by correcting the goals that cause them. Return only the goals you change or
add; keep the basis and evidence of a goal unless the fix itself is not shown by the experience.

{_vocab_block(names, env)}

{_FACTS_FIELDS}

{_FACTS_RULES}

{_DELTA_FIELDS}"""


# --- edges mode (ablation: the model writes the links itself) --------------------

_SCHEMA = """Each node is one goal, with these fields:
  name                 "<verb> <thing>", e.g. "collect clay"
  description          one sentence saying what the goal involves
  required_tools       things that must be available (held or nearby) and are not used up
  required_materials   materials consumed, as a list of {"item": <known name>, "quantity": <n>}
  preconditions        names of the goals that must be achieved before this one
  produces             items gained by completing the goal, as a list of {"item", "quantity"}
  places               the known name of what the goal places in the world, or null
  task_list            the actions performed, each a valid action or an allowed shorthand

Each edge says that one goal needs another to be achieved first:
  {"goal": <goal that needs it>, "prerequisite": <goal that must be achieved first>, "description": <why>}

Example of two nodes and their edge (format only; an invented environment, not the one you are acting in):
{"nodes": [
  {"name": "collect clay", "description": "Go to clay and dig to obtain clay.",
   "required_tools": [], "required_materials": [], "preconditions": [],
   "produces": [{"item": "clay", "quantity": 1}], "places": null,
   "task_list": ["go to clay", "dig"]},
  {"name": "build kiln", "description": "Build a kiln, using clay.",
   "required_tools": [], "required_materials": [{"item": "clay", "quantity": 2}],
   "preconditions": ["collect clay"], "produces": [], "places": "kiln",
   "task_list": ["build kiln"]}],
 "edges": [
  {"goal": "build kiln", "prerequisite": "collect clay", "description": "A kiln is built from clay."}]}

Rules:
  - include every goal needed to achieve the task, and only those
  - write each goal once, under one name; do not repeat a goal
  - if several tools would do, list only the least advanced one that is enough
  - the goal that achieves the task is the root: every other goal must be reachable from it
    by following edges from goal to prerequisite
  - the graph must be acyclic, and no goal is its own prerequisite
  - preconditions and edges must agree exactly: for every precondition P of goal G there is
    an edge {"goal": G, "prerequisite": P}, and there are no other edges
  - every item a goal requires must be produced or placed by some goal in the graph
  - every tool, material and item must be written exactly as one of the names above"""


def _show(graph):
    """A GoG dict rendered in the edges-mode reply format, so the model reads what it writes."""
    return json.dumps(from_gog(graph), indent=2)


def construct(task, experience_log, names=None, env=None):
    """Edges-mode construction from experience alone; [] is the zero-experience control."""
    if experience_log:
        preamble = ("Below is a log of your own attempts at this task in the environment. "
                    "Infer the goal structure from what actually happened - which attempts "
                    "succeeded, which failed, and what changed afterwards.")
    else:
        preamble = ("You have not yet acted in this environment. Build the graph from what "
                    "you can infer about the task as stated.")
    evidence = json.dumps(experience_log, indent=2) if experience_log else "(no episodes have been played yet)"

    return f"""You are building a Goal-Oriented Graph: a directed acyclic graph of goals and the
prerequisite relations between them, used to plan how to achieve a task.

Task: {task}

{preamble}

Experience:
{evidence}

{_vocab_block(names, env)}

{_SCHEMA}"""


def maintain_regenerate(task, graph, episode, names=None, env=None):
    """Ablation for maintenance: the model rewrites every goal after each episode.

    Same facts format and evidence rules as extract_delta, so the only difference
    between the two maintenance conditions is delta-and-merge vs whole rewrite.
    """
    return f"""You maintain a Goal-Oriented Graph for an agent: the goals it must achieve to complete a
task, each with what it needs and what it gives.

Task: {task}

Current goals:
{_show_facts(graph) if graph.get("nodes") else "(none yet)"}

A new episode has been played:
{json.dumps(episode, indent=2)}

{_vocab_block(names, env)}

Revise the goals so that they account for this episode. Change only what the episode
supports: add a goal it shows, correct one it contradicts, and leave the rest untouched.
Return every goal, changed or not.

{_FACTS_FIELDS}

{_FACTS_RULES}"""


def repair(task, graph, problems, names=None, env=None):
    """Edges mode: ask the model to fix the structural problems an automatic check found.

    `problems` comes from `validate`, which checks structure and names only and never
    sees the reference graph: it can say a graph is broken, never that it is wrong.
    No experience is resent, so a repair cannot learn anything new about the
    environment; it can only make the graph it already built consistent.
    """
    listed = "\n".join(f"  - {p}" for p in problems)
    return f"""You built the Goal-Oriented Graph below for a task. An automatic structural check
found problems in it.

Task: {task}

Your graph:
{_show(graph)}

Problems found (an edge 'A' -> 'B' means {{"goal": "A", "prerequisite": "B"}}):
{listed}

Return the corrected graph. Fix these problems and change nothing else: keep every goal,
precondition and attribute that is not involved in a problem as it is.

{_vocab_block(names, env)}

{_SCHEMA}"""


def fingerprint():
    """Hash of every source the model reads: identical prompts <=> identical fingerprint.

    Recorded in every scores.json so results produced by different prompt versions are
    never pooled (aggregate.py refuses to), and checked against the frozen fingerprint
    before a test-split run (splits.py).
    """
    import hashlib
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    # Everything the model reads: the prompts, the reply schemas (their field descriptions
    # are sent), the repair messages, and validate's violation messages they include.
    for name in ("prompts.py", "schema.py", "problems.py", os.path.join("evaluate", "metrics", "validity.py")):
        with open(os.path.join(here, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]
