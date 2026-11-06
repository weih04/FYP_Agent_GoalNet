"""Chain-of-thought agent guided by a Goal-Oriented Graph (Leung, Wang & Shen, 2026).

The only difference from `ChainOfThoughtAgent` is that the instruction prompt is
followed by the plan the graph yields: its goals in dependency-first DFS order, each
with what it requires, what it achieves and the actions it involves. The agent is
not told which goal it is on; working that out from its inventory is left to it, so
the graph is the single thing that differs from the CoT baseline.

The graph is read from `agent.goalnet_path` at every reset, so a graph rewritten
between episodes (the dynamic condition) is picked up without restarting the run.
A constructed graph may carry `_bill_of_materials`; it is shown after the plan.
"""

import json
from collections import defaultdict

from hydra.utils import to_absolute_path

from .chain_of_thought import ChainOfThoughtAgent


def _key(name):
    # Same normalisation as goalnet_general.evaluate.metrics._names._norm, so edges written as
    # "make iron_pickaxe" still match a node named "make iron pickaxe".
    return " ".join(str(name).replace("_", " ").strip().lower().split())


def plan_order(graph, root):
    """Goal nodes reachable from `root`, every goal after its prerequisites.

    Mirrors goalnet_general.evaluate.metrics.plan_from_gog (duplicated so BALROG does not depend on
    the goalnet package), but returns the nodes themselves rather than their names.
    """
    nodes = {_key(n["name"]): n for n in graph["nodes"]}
    if _key(root) not in nodes:
        raise KeyError(f"root goal {root!r} is not a node of the Goal Net")
    prerequisites = defaultdict(list)
    for e in graph.get("edges", []):
        prerequisites[_key(e["from"])].append(_key(e["to"]))

    order, visiting, done = [], set(), set()

    def visit(k):
        if k in done:
            return
        if k in visiting:
            raise ValueError(f"the Goal Net has a cycle through {k!r}")
        if k not in nodes:
            raise KeyError(f"an edge names {k!r}, which is not a node of the Goal Net")
        visiting.add(k)
        for m in prerequisites[k]:
            visit(m)
        visiting.discard(k)
        done.add(k)
        order.append(nodes[k])

    visit(_key(root))
    return order


def _amounts(field):
    """'wood 2, stone 1' from {"wood": 2, ...} or a list of names."""
    if isinstance(field, dict):
        return [f"{k} {v}" for k, v in field.items()]
    return [str(k) for k in field or []]


def render_plan(graph, root):
    # A required tool that some goal places in the world (a table, a furnace) must be
    # nearby; any other tool must be in the inventory. Read from the graph, not hard-coded.
    placed_things = {(n.get("postconditions") or {}).get("nearby") for n in graph["nodes"]} - {None}
    lines = []
    for i, n in enumerate(plan_order(graph, root), 1):
        needs = [f"{t} nearby" if t in placed_things else f"{t} in inventory"
                 for t in n.get("required_tools") or []]
        needs += _amounts(n.get("required_materials"))
        post = dict(n.get("postconditions") or {})
        placed = post.pop("nearby", None)
        gives = _amounts(post) + ([f"a {placed} placed next to you"] if placed else [])
        lines.append(f"{i}. {n['name']}: {n.get('description', '').strip()}")
        if n.get("preconditions"):
            lines.append(f"   after: {', '.join(n['preconditions'])}")
        lines.append(f"   needs: {', '.join(needs) or 'nothing'}")
        lines.append(f"   achieves: {', '.join(gives) or 'nothing'}")
        lines.append(f"   actions: {', '.join(n.get('task_list') or [])}")
    return "\n".join(lines)


GOALNET_HEADER = """GOAL NET:
The goals below lead to your main goal, listed so that every goal comes after the goals it
depends on. Each goal says which goals must be achieved first (after), what it needs, what it
achieves and which actions it involves. "Move toward X" means using Move actions until you
face X; it is not an action itself. Use the Goal Net to decide what to do next."""


TOTALS_HEADER = """MATERIALS AND TOOLS NEEDED IN TOTAL for the main goal (tools and placed stations count once):"""


class GoalNetAgent(ChainOfThoughtAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder, config)
        if not config.agent.get("goalnet_path"):
            raise ValueError("agent.type=goalnet needs agent.goalnet_path (a GoG JSON file)")
        self.goalnet_path = to_absolute_path(config.agent.goalnet_path)
        self.goalnet_root = config.agent.get("goalnet_root")
        self.goalnet_text = self._load()

    def _load(self):
        with open(self.goalnet_path) as f:
            graph = json.load(f)
        root = self.goalnet_root or graph.get("_root_goal")
        if not root:
            raise ValueError(f"{self.goalnet_path} has no _root_goal; set agent.goalnet_root")
        text = f"{GOALNET_HEADER}\n{render_plan(graph, root)}"
        # Totals computed from the graph's own facts (goalnet_general.evaluate.metrics.graph_bill_of_materials);
        # GoG found this list mattered more for plan quality than the goal tree (Table 6).
        # Hand-authored reference graphs carry none, so their rendering is unchanged.
        totals = graph.get("_bill_of_materials")
        if totals:
            text += "\n\n" + TOTALS_HEADER + "\n" + ", ".join(f"{k} {v}" for k, v in totals.items())
        return text

    def reset(self):
        super().reset()
        self.goalnet_text = self._load()

    def _messages(self):
        messages = super()._messages()
        # get_prompt rebuilds every message from stored strings, so this addition does not
        # accumulate across steps. The first message is the instruction prompt.
        if not messages or messages[0].content != self.prompt_builder.system_prompt:
            raise RuntimeError("expected the instruction prompt as the first message")
        messages[0].content += "\n\n" + self.goalnet_text
        return messages
