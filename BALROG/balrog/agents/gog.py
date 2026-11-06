"""GoG-guided agent: Leung, Wang & Shen (2026) §3.2 and §4.1, replicated for BALROG Crafter.

The paper's pipeline, step by step, and what stands in for each part here:

1. Goal retrieval: embed the task, retrieve the top-k goals by cosine similarity of
   goal names (k = 3; paper: nomic-embed-text-v1.5; here `agent.gog_embedding`,
   OpenAI text-embedding-3-small by default, or "nomic" with sentence-transformers).

2. Goal inference (Fig. A.5): the LLM picks the goal matching the task from the k
   candidates and describes the game state ("visual inference"). Crafter is played
   from text, so the game state is the text observation rather than a screenshot.

3. DFS from the chosen goal over subgoal edges, pruning cycles, gives the goal tree where 
   the list of materials and tools with amounts is computed from it.
   Deviation: goals producing a tool already held or a placed thing within reach are
   pruned first, so a replan does not restart the chain (the paper starts every plan
   from an empty inventory and names pruning achieved goals as future work, §5).

4. Planning (Fig. A.6): one LLM call turns the goal, game state, goal hierarchy and
   materials list into a JSON plan of steps {task, goal: [item, amount]}.

5. Execution: the paper hands each step to STEVE-1. Here the chain-of-thought actor
   plays that role and is given only the current step's instruction; a step is done
   when its target item is in the inventory in the planned amount (or, for a placed
   thing, stands within reach). The instruction and any failure feedback go next to
   the current observation (`agent.gog_guidance=last`): appended to the instruction
   prompt instead (`first`), gpt-4o-mini ignored both in replayed stuck states.
   
6. Replanning: when a crafting or placing action fails, the missing tools and
   materials are found by checking the attempted goal against the goal tree (the
   paper's environment reports them; Crafter does not). If that goal is the current
   step's, a plan for them is generated from the goal tree and inserted before the
   current step, as in the paper (only crafting steps replan there; mining does not).
   Deviation: STEVE-1 only ever attempts the current step, but the LLM actor also
   attempts others; for those the missing items are told to the actor for its next
   action instead, the way Minecraft reports them, with no plan change. A Make/Place has
   failed when the inventory did not change (each one uses up or gains an item).
   Deviation: when it fails although the goal lists nothing missing, the graph's recipe
   is incomplete, and the actor is told so and asked to try it in a different state
   (holding or near something new) rather than repeat it; the paper's environment
   names what is missing, Crafter does not. When no progress is made for
   `agent.gog_replan_after` steps (the paper's x, unstated there), goal inference and
   planning are repeated (Table C.2).

Goal inference and planning run at temperature 0, as in the paper. Their tokens are
added to the step's token counts, and plan events are prefixed to the step's
reasoning, so both appear in the episode logs. The worked examples use an invented
domain so that no recipe of any Crafter variant reaches the agent through them.
"""

import json
import logging
import math
import re
from collections import defaultdict

from hydra.utils import to_absolute_path

from balrog.prompt_builder.history import Message

from .chain_of_thought import ChainOfThoughtAgent

logger = logging.getLogger(__name__)

_INV_LINE = re.compile(r"^- (\w+): (\d+)$")
_SEEN = re.compile(r"^- (\w+) (.+)$")
_OFFSET = re.compile(r"(\d+) steps? (north|south|east|west)")

GOAL_INFERENCE_PROMPT = """You are a {env} game expert and you can guide agents to complete complex tasks. For a given game state, task, and context information, you need to complete "goal inference" and "visual inference".
The context information is a set of possible goals to choose from for "goal inference".
"goal inference": According to the task, you need to select the goal from given options that best matches the given query.
"visual inference": According to the game state, you need to infer the following aspects: health, food, inventory, environment.
{examples}
Here is a game state and task, you MUST respond in JSON format as shown in the example outputs WITHOUT further explanation, introduction, or extra text. Complete "goal inference" by setting it to the value of the "name" of the option that best matches the given task as shown in the example. Other fields should be completed based on the given game state.
<task>: {task}
<game state>:
{state}
<context>:
{context}
Output:"""

GOAL_INFERENCE_EXAMPLES = """Example:
<task>: make a brick
<game state>:
You see grass, a river 2 steps west, clay 1 step north. Health 9/9, food 8/9. Inventory: clay 2.
<context>:
[{"name": "fire brick", "description": "Fire clay into a brick next to a kiln.", "preconditions": ["kiln nearby", "clay 1"], "postconditions": {"brick": 1}},
 {"name": "build kiln", "description": "Build a kiln, using clay.", "preconditions": ["clay 2"], "postconditions": {"nearby": "kiln"}}]
Output:
{"goal inference": "fire brick", "visual inference": {"health": "full", "food": "high", "inventory": "clay 2", "environment": "grass, a river to the west, clay to the north"}}"""

PLANNING_PROMPT = """You are a {env} game expert and you can guide agents to complete complex tasks. For a given overall goal, game state, hierarchy of goals, and list of needed materials, construct an ordered plan that completes the given task. The goal hierarchy is structured as a JSON object whose keys are names of goals, and values are information about the goal and its subgoals. You will be given a list of tools and materials and amounts needed for you to obtain and craft to complete the overall goal. Based on the information from the goal hierarchy and the list of tools and materials, create a plan in JSON format as shown in the following example:
#####################
-Example-
#####################
{example}
#####################
-Real Task-
#####################
<goal>
{goal}
<visual info>
{visual_info}
<goal hierarchy>
{goal_hierarchy}
<materials and tools>
{materials_and_tools}
<planning>
Complete <planning> for the given overall <goal> with valid JSON as instructed and in the format shown in the example. Use the information in the goal hierarchy, game state, and list of tools and materials and their amounts to generate "task" and "goal" in each step of the plan. Use the same wording styles and patterns for the "task" in each step as shown in the example plan. Write the item in each step's "goal" exactly as it is named in the materials and tools list. Only output the plan as a valid JSON object with no additional text, introduction, or explanation. Do not use Markdown."""

PLAN_EXAMPLE = """<goal>
fire brick
<materials and tools>
{"clay": 3, "kiln": 1}
<planning>
{"plan": [{"task": "dig clay", "goal": ["clay", 3]},
          {"task": "build a kiln", "goal": ["kiln", 1]},
          {"task": "fire a brick", "goal": ["brick", 1]}]}"""

# Fewest steps between two full replans triggered by a plan running out, so a plan
# whose steps are all already satisfied cannot cause a planning call on every step.
MIN_STEPS_BETWEEN_REPLANS = 10

STEP_HEADER = """CURRENT INSTRUCTION (step {i} of {n} of your plan):
{task}
This step is complete when you have {amount} {item}{where}. Choose your next action to carry out this instruction."""


def _key(name):
    return " ".join(str(name).replace("_", " ").strip().lower().split())


def _item(name):
    return str(name).strip().lower().replace(" ", "_")


def parse_inventory(status):
    inv = {}
    if "Your inventory:" in status:
        for line in status.split("Your inventory:", 1)[1].splitlines():
            m = _INV_LINE.match(line.strip())
            if m:
                inv[m.group(1)] = int(m.group(2))
    return inv


def parse_within_reach(observation):
    """Things within 1 tile, diagonals included - the radius Crafter uses for nearby."""
    out = set()
    for line in observation.splitlines():
        m = _SEEN.match(line.strip())
        if m:
            offsets = _OFFSET.findall(m.group(2))
            if offsets and all(int(n) <= 1 for n, _ in offsets):
                out.add(m.group(1))
    face = re.search(r"You face (\w+) at your front", observation)
    if face:
        out.add(face.group(1))
    return out


def _parse_json(text):
    """The first JSON object in a reply, tolerating Markdown fences around it."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


# --- goal tree and materials (mirrors goalnet_general's derive and bill_of_materials) ----


def _requirements(node):
    out = defaultdict(int)
    for field in ("required_materials", "required_tools"):
        f = node.get(field)
        if isinstance(f, dict):
            for k, qty in f.items():
                out[k] += qty
        elif isinstance(f, list):
            for k in f:
                out[k] += 1
    return out


def _producers(graph):
    out = {}
    for n in graph["nodes"]:
        for k, v in (n.get("postconditions") or {}).items():
            out[v if k == "nearby" else k] = _key(n["name"])
    return out


def _persistent(graph):
    out = set()
    for n in graph["nodes"]:
        out |= set(n.get("required_tools") or [])
        placed = (n.get("postconditions") or {}).get("nearby")
        if placed:
            out.add(placed)
    return out


def goal_tree(graph, root):
    """DFS from `root` over subgoal edges; a goal already visited is not expanded again,
    so cyclic subgoal relations cannot loop. Returns goal keys, prerequisites first."""
    nodes = {_key(n["name"]): n for n in graph["nodes"]}
    if _key(root) not in nodes:
        raise KeyError(f"goal {root!r} is not a node of the Goal Net")
    sub = defaultdict(list)
    for e in graph.get("edges", []):
        sub[_key(e["from"])].append(_key(e["to"]))
    order, seen = [], set()

    def visit(k):
        if k in seen or k not in nodes:
            return
        seen.add(k)
        for m in sub[k]:
            visit(m)
        order.append(k)

    visit(_key(root))
    return order


def bill_of_materials(graph, root):
    """Total of each resource needed to achieve `root` once; tools and placed things count
    once (same rule as goalnet_general.evaluate.metrics.plan_quality.bill_of_materials)."""
    nodes = {_key(n["name"]): n for n in graph["nodes"]}
    producers, persistent = _producers(graph), _persistent(graph)
    runs = defaultdict(int)
    runs[_key(root)] = 1
    needed = defaultdict(int)
    for goal in reversed(goal_tree(graph, root)):
        n = runs[goal]
        if not n:
            continue
        for res, qty in _requirements(nodes[goal]).items():
            total = qty if res in persistent else qty * n
            needed[res] += total if res not in persistent else 0
            if res in persistent:
                needed[res] = 1
            producer = producers.get(res)
            if producer:
                runs[producer] = max(runs[producer], 1) if res in persistent else runs[producer] + total
    return dict(needed)


def prune_achieved(graph, root, achieved):
    """`graph` without the goals that produce an `achieved` tool or placed thing (the root
    excepted), so goals needed only by them fall out of the goal tree too."""
    drop = {_key(n["name"]) for n in graph["nodes"] if _key(n["name"]) != _key(root)
            and any((v if k == "nearby" else k) in achieved for k, v in (n.get("postconditions") or {}).items())}
    return {**graph, "nodes": [n for n in graph["nodes"] if _key(n["name"]) not in drop],
            "edges": [e for e in graph.get("edges", []) if _key(e["from"]) not in drop and _key(e["to"]) not in drop]}


def hierarchy(graph, root):
    """The goal tree as the paper's goal hierarchy: {goal name: its information and subgoals}."""
    nodes = {_key(n["name"]): n for n in graph["nodes"]}
    sub = defaultdict(list)
    for e in graph.get("edges", []):
        sub[_key(e["from"])].append(e["to"])
    out = {}
    for k in goal_tree(graph, root):
        n = nodes[k]
        out[n["name"]] = {
            "description": n.get("description", ""),
            "required_tools": n.get("required_tools") or [],
            "required_materials": n.get("required_materials") or {},
            "postconditions": n.get("postconditions") or {},
            "subgoals": sub[k],
        }
    return out


def _listed(node):
    """A goal's requirements as the actor is told them, e.g. "table, wood 1"."""
    parts = list(node.get("required_tools") or []) + \
        [f"{k} {v}" for k, v in (node.get("required_materials") or {}).items()]
    return ", ".join(p.replace("_", " ") for p in parts) or "nothing"


class GoGAgent(ChainOfThoughtAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder, config)
        if not config.agent.get("goalnet_path"):
            raise ValueError("agent.type=gog needs agent.goalnet_path (a GoG JSON file)")
        self.goalnet_path = to_absolute_path(config.agent.goalnet_path)
        self.top_k = int(config.agent.get("gog_top_k", 3))
        self.replan_after = int(config.agent.get("gog_replan_after", 100))
        self.embedding = config.agent.get("gog_embedding", "openai:text-embedding-3-small")
        self.guidance = config.agent.get("gog_guidance", "last")
        if self.guidance not in ("first", "last"):
            raise ValueError(f"agent.gog_guidance must be 'first' or 'last', not {self.guidance!r}")
        self.env_name = "Crafter"
        self.task = None
        self._embedder = None
        self._name_vectors = {}
        self._reset_state()

    # --- episode state ----------------------------------------------------------

    def _reset_state(self):
        with open(self.goalnet_path) as f:
            self.graph = json.load(f)    # re-read each episode: the loop may rewrite it
        self.plan, self.step = None, 0
        self.root, self.visual = None, {}
        self.no_progress, self.best = 0, 0
        self.since_plan = 0
        self.replanned_step = None
        self.feedback = None
        self.prev_inv = None
        self.tokens = [0, 0]
        self.events = []

    def reset(self):
        super().reset()
        self._reset_state()

    def set_task(self, task):
        """The evaluator's task id (e.g. collect_diamond); the retrieval query."""
        self.task = task.replace("_", " ")

    # --- model calls -------------------------------------------------------------

    def _ask(self, prompt):
        """One planner call at temperature 0, as in the paper; tokens are tallied."""
        kwargs = self.client.client_kwargs
        saved = kwargs.get("temperature")
        kwargs["temperature"] = 0.0
        try:
            response = self.client.generate([Message(role="user", content=prompt)])
        finally:
            kwargs["temperature"] = saved
        self.tokens[0] += response.input_tokens or 0
        self.tokens[1] += response.output_tokens or 0
        return response.completion

    def _embed(self, texts, query):
        if self.embedding.startswith("openai:"):
            if self._embedder is None:
                from openai import OpenAI
                self._embedder = OpenAI()
            resp = self._embedder.embeddings.create(model=self.embedding.split(":", 1)[1], input=texts)
            return [d.embedding for d in resp.data]
        if self.embedding == "nomic":
            if self._embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise ImportError("agent.gog_embedding=nomic needs `pip install sentence-transformers einops`") from e
                self._embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
            prefix = "search_query: " if query else "search_document: "
            return [list(v) for v in self._embedder.encode([prefix + t for t in texts])]
        raise ValueError(f"unknown agent.gog_embedding {self.embedding!r}")

    def _retrieve(self):
        """Top-k goals by cosine similarity between the task and the goal names."""
        names = [n["name"] for n in self.graph["nodes"]]
        missing = [n for n in names if n not in self._name_vectors]
        if missing:
            self._name_vectors.update(zip(missing, self._embed(missing, query=False)))
        q = self._embed([self.task], query=True)[0]

        def cos(a, b):
            return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

        return sorted(names, key=lambda n: -cos(q, self._name_vectors[n]))[: self.top_k]

    # --- the pipeline ------------------------------------------------------------

    def _state_text(self, obs):
        return obs["text"].get("long_term_context", "") + "\n" + obs["text"].get("short_term_context", "")

    def _infer_goal(self, obs):
        candidates = self._retrieve()
        nodes = {n["name"]: n for n in self.graph["nodes"]}
        context = [{"name": c, "description": nodes[c].get("description", ""),
                    "preconditions": [f"{t} nearby or held" for t in nodes[c].get("required_tools") or []]
                    + [f"{k} {v}" for k, v in (nodes[c].get("required_materials") or {}).items()],
                    "postconditions": nodes[c].get("postconditions") or {}} for c in candidates]
        reply = self._ask(GOAL_INFERENCE_PROMPT.format(
            env=self.env_name, examples=GOAL_INFERENCE_EXAMPLES, task=self.task,
            state=self._state_text(obs), context=json.dumps(context, indent=1)))
        try:
            out = _parse_json(reply)
            chosen = out.get("goal inference")
            self.visual = out.get("visual inference") or {}
        except ValueError as e:
            logger.warning("goal inference reply was not JSON (%s); using the top retrieved goal", e)
            chosen = None
        if chosen not in candidates:
            logger.warning("goal inference chose %r, not one of %s; using the top retrieved goal", chosen, candidates)
            chosen = candidates[0]
        self.events.append(f"goal inference: {chosen} (candidates {candidates})")
        return chosen

    def _plan_for(self, goal_text, graph, root, materials):
        reply = self._ask(PLANNING_PROMPT.format(
            env=self.env_name, example=PLAN_EXAMPLE, goal=goal_text,
            visual_info=json.dumps(self.visual), goal_hierarchy=json.dumps(hierarchy(graph, root), indent=1),
            materials_and_tools=json.dumps(materials)))
        try:
            steps = _parse_json(reply)["plan"]
            plan = [{"task": str(s["task"]), "item": _item(s["goal"][0]), "amount": int(s["goal"][1])} for s in steps]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.warning("plan reply could not be read (%s): %r", e, reply[:300])
            plan = []
        return plan

    def _make_plan(self, obs):
        if self.task is None:
            raise RuntimeError("GoGAgent needs set_task(task) before acting (evaluator calls it)")
        self.root = self._infer_goal(obs)
        inv = parse_inventory(obs["text"].get("short_term_context", ""))
        reach = parse_within_reach(obs["text"].get("long_term_context", ""))
        achieved = {p for p in _persistent(self.graph) if inv.get(p, 0) > 0 or p in reach}
        graph = prune_achieved(self.graph, self.root, achieved)
        materials = {k: v for k, v in bill_of_materials(graph, self.root).items() if k not in achieved}
        if achieved:
            self.events.append(f"plan skips achieved: {', '.join(sorted(achieved))}")
        self.plan = self._plan_for(self.root, graph, self.root, materials)
        self.step, self.no_progress, self.best, self.since_plan = 0, 0, 0, 0
        self.events.append("plan: " + "; ".join(f"{s['task']} -> {s['item']} {s['amount']}" for s in self.plan))

    def _attempted_goal(self, action):
        """The goal node a Make/Place action tries to achieve, or None if the graph has none."""
        action = str(action)
        for prefix, placed in (("Make ", False), ("Place ", True)):
            if action.startswith(prefix):
                item = _item(action[len(prefix):])
                for n in self.graph["nodes"]:
                    post = n.get("postconditions") or {}
                    if (post.get("nearby") == item) if placed else (item != "nearby" and item in post):
                        return n
        return None

    def _missing_for(self, node, inv, reach):
        """Tools and materials `node` needs but the agent lacks, checked against the goal
        tree (the paper's environment reports these)."""
        missing = {}
        for t in node.get("required_tools") or []:
            if inv.get(t, 0) < 1 and t not in reach:
                missing[t] = 1
        for m, q in (node.get("required_materials") or {}).items():
            if inv.get(m, 0) < q:
                missing[m] = q - inv.get(m, 0)
        return missing

    def _insert_plan_for_missing(self, missing):
        """Plan for the missing items from the goal tree, inserted before the current step."""
        need = {"name": "obtain missing items", "description": "", "required_tools":
                [k for k in missing if k in _persistent(self.graph)],
                "required_materials": {k: v for k, v in missing.items() if k not in _persistent(self.graph)},
                "postconditions": {}}
        producers = _producers(self.graph)
        graph = {"nodes": self.graph["nodes"] + [need],
                 "edges": self.graph.get("edges", []) + [{"from": need["name"], "to": next(
                     n["name"] for n in self.graph["nodes"] if _key(n["name"]) == producers[k])}
                     for k in missing if k in producers]}
        materials = bill_of_materials(graph, need["name"])
        steps = self._plan_for("obtain " + ", ".join(f"{k} {v}" for k, v in missing.items()),
                               graph, need["name"], materials)
        self.plan[self.step:self.step] = steps
        self.events.append(f"missing {missing}: inserted " + "; ".join(f"{s['task']} -> {s['item']} {s['amount']}"
                                                                      for s in steps))

    def _done(self, step, inv, reach):
        if inv.get(step["item"], 0) >= step["amount"]:
            return True
        return step["item"] in reach and step["item"] in _persistent(self.graph) and step["amount"] <= 1

    def _advance(self, obs, prev_action):
        inv = parse_inventory(obs["text"].get("short_term_context", ""))
        reach = parse_within_reach(obs["text"].get("long_term_context", ""))
        prev_inv, self.prev_inv = self.prev_inv, inv
        self.feedback = None
        if self.plan is None:
            self._make_plan(obs)
            return
        self.since_plan += 1
        progressed = False
        while self.step < len(self.plan) and self._done(self.plan[self.step], inv, reach):
            done = self.plan[self.step]
            self.events.append(f"step {self.step + 1}/{len(self.plan)} done: {done['task']} -> {done['item']} {done['amount']}")
            self.step += 1
            progressed = True
        if self.step < len(self.plan):
            count = inv.get(self.plan[self.step]["item"], 0)
            if count > self.best:
                progressed = True
            self.best = count if progressed else max(self.best, count)
        self.no_progress = 0 if progressed else self.no_progress + 1
        if self.step >= len(self.plan):
            # An empty plan (unreadable reply) waits for the no-progress limit; a plan that
            # ran out is replaced, but not more often than MIN_STEPS_BETWEEN_REPLANS.
            limit = self.replan_after if not self.plan else MIN_STEPS_BETWEEN_REPLANS
            if self.since_plan >= limit:
                self.events.append("plan ran out without completing the task: replanning")
                self._make_plan(obs)
            return
        failed = (prev_action and str(prev_action).startswith(("Make ", "Place "))
                  and prev_inv is not None and inv == prev_inv)
        node = self._attempted_goal(prev_action) if failed else None
        missing = self._missing_for(node, inv, reach) if node else {}
        if missing:
            on_step = _key(node["name"]) == _producers(self.graph).get(self.plan[self.step]["item"])
            if not on_step:
                self.feedback = (f"Your last action, {prev_action}, failed: it needs "
                                 + ", ".join(f"{k} {v}" for k, v in missing.items()) + " that you do not have.")
                self.events.append(f"feedback: {self.feedback}")
            elif self.replanned_step != self.step:
                self.replanned_step = self.step
                self._insert_plan_for_missing(missing)
                return
        elif node is not None:
            self.feedback = (f"Your last action, {prev_action}, failed although you have everything the plan "
                             f"lists for it ({_listed(node)}). Something the plan does not list must also be "
                             "needed. Do not repeat it as you are: first get or go near something you did not "
                             "have with you when it failed, then try again.")
            self.events.append(f"feedback: {self.feedback}")
        if self.no_progress >= self.replan_after:
            self.events.append(f"no progress for {self.no_progress} steps: replanning")
            self._make_plan(obs)

    # --- acting -------------------------------------------------------------------

    def act(self, obs, prev_action=None):
        self.tokens, self.events = [0, 0], []
        self._advance(obs, prev_action)
        response = super().act(obs, prev_action=prev_action)
        notes = "".join(f"[GoG] {e}\n" for e in self.events)
        return response._replace(
            input_tokens=(response.input_tokens or 0) + self.tokens[0],
            output_tokens=(response.output_tokens or 0) + self.tokens[1],
            reasoning=notes + (response.reasoning or ""))

    def _messages(self):
        messages = super()._messages()
        if not messages or messages[0].content != self.prompt_builder.system_prompt:
            raise RuntimeError("expected the instruction prompt as the first message")
        target = messages[0] if self.guidance == "first" else messages[-1]
        if self.plan and self.step < len(self.plan):
            s = self.plan[self.step]
            where = " (or a placed one within reach)" if s["item"] in _persistent(self.graph) else ""
            target.content += "\n\n" + STEP_HEADER.format(
                i=self.step + 1, n=len(self.plan), task=s["task"], amount=s["amount"],
                item=s["item"].replace("_", " "), where=where)
        if self.feedback:
            target.content += "\n\n" + self.feedback
        return messages
