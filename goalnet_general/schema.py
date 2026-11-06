"""Pydantic schemas the model must answer in, and conversion to and from the GoG dict.

Three reply formats: GoalNetReply (edges mode: goals plus edges), FactsReply (facts
mode: goals with basis/evidence, edges derived in code) and DeltaReply (maintenance:
only the goals one episode adds, corrects or removes).

OpenAI structured outputs guarantee the reply parses into `GoalNetReply`, which
removes malformed JSON, missing fields and invented keys. They do not touch graph
structure: cycles, edge/precondition agreement and reachability are still the
model's to get right and are still measured by `validate`.

Names are plain strings, not enums of the vocabulary. An enum would make grounding
true by construction and so remove it from what the thesis measures.

Strict structured outputs cannot express a free-form {name: quantity} object, so
quantities are lists of {item, quantity}; `to_gog` turns them back into the dict
form used by the reference graphs, validate and metrics. Edges name their ends
`goal` and `prerequisite` rather than `from` and `to`, whose direction the model
reversed in the first pilots.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Quantity(_Strict):
    item: str
    quantity: int


class Goal(_Strict):
    name: str = Field(description='"<verb> <thing>", e.g. "collect clay"')
    description: str
    required_tools: list[str] = Field(description="things that must be available (held or nearby); not used up")
    required_materials: list[Quantity] = Field(description="materials consumed")
    preconditions: list[str] = Field(description="goals that must be achieved before this one")
    produces: list[Quantity] = Field(description="items gained by completing the goal")
    places: Optional[str] = Field(description="what the goal places in the world, or null")
    task_list: list[str] = Field(description="actions performed, each a valid action or an allowed shorthand")


class Edge(_Strict):
    goal: str = Field(description="the goal that needs the prerequisite")
    prerequisite: str = Field(description="the goal that must be achieved first")
    description: str


class GoalNetReply(_Strict):
    nodes: list[Goal]
    edges: list[Edge]


# --- facts mode -------------------------------------------------------------
# The model states each goal's facts; derive.derive_edges computes preconditions and
# edges from them (as GoG §3.1.2 does). `reasoning` comes first so that, generated in
# schema order, the model says where a goal's facts come from before stating them
# (the GoG extraction prompt asks for the same). `basis` and `evidence` separate what
# the experience shows from what the model brings from pre-training.


class Evidence(_Strict):
    episode: int
    step: int


class GoalFacts(_Strict):
    reasoning: str = Field(description="where this goal's facts come from, written before them")
    name: str = Field(description='"<verb> <thing>", e.g. "collect clay"')
    description: str
    required_tools: list[str] = Field(description="things that must be available (held or nearby); not used up")
    required_materials: list[Quantity] = Field(description="items used up")
    produces: list[Quantity] = Field(description="items gained by completing the goal")
    places: Optional[str] = Field(description="what the goal places in the world, or null")
    task_list: list[str] = Field(description="actions performed, each a valid action or an allowed shorthand")
    basis: Literal["observed", "prior"] = Field(
        description='"observed" if the experience shows this goal achieved, else "prior"')
    evidence: list[Evidence] = Field(description="episode and step where the experience shows it; empty if prior")


class FactsReply(_Strict):
    task_goal: str = Field(description="name of the goal that achieves the task")
    goals: list[GoalFacts]


class Removal(_Strict):
    name: str
    reason: str


class DeltaReply(_Strict):
    """A maintenance step: only the goals one episode (or one repair) changes."""
    task_goal: str = Field(description="name of the goal that achieves the task")
    goals: list[GoalFacts] = Field(description="goals added or corrected; unmentioned goals are kept")
    remove: list[Removal] = Field(description="existing goals the evidence shows are wrong")


def facts_to_node(g):
    """GoalFacts -> GoG node, with no preconditions yet (derive_edges adds them)."""
    post = {q.item: q.quantity for q in g.produces}
    if g.places:
        post["nearby"] = g.places
    return {
        "name": g.name,
        "description": g.description,
        "required_tools": list(g.required_tools),
        "required_materials": {q.item: q.quantity for q in g.required_materials},
        "preconditions": [],
        "postconditions": post,
        "task_list": list(g.task_list),
        "basis": g.basis,
        "evidence": [{"episode": e.episode, "step": e.step} for e in g.evidence],
        "reasoning": g.reasoning,
    }


def facts_to_gog(reply):
    """FactsReply -> GoG dict with the model's root recorded; edges still empty."""
    return {"_root_goal": reply.task_goal, "nodes": [facts_to_node(g) for g in reply.goals], "edges": []}


def gog_to_facts(gog):
    """GoG dict -> the facts format, for showing a graph back to the model."""
    out = []
    for n in from_gog(gog)["nodes"]:
        src = next(m for m in gog["nodes"] if m.get("name") == n["name"])
        out.append({
            "name": n["name"],
            "description": n["description"],
            "required_tools": n["required_tools"],
            "required_materials": n["required_materials"],
            "produces": n["produces"],
            "places": n["places"],
            "task_list": n["task_list"],
            "basis": src.get("basis", "prior"),
            "evidence": src.get("evidence", []),
        })
    return {"task_goal": gog.get("_root_goal"), "goals": out}


def parse_facts(raw):
    """Raw facts-mode reply -> GoG dict (edges not yet derived)."""
    import json

    return facts_to_gog(FactsReply.model_validate(json.loads(raw)))


def parse_delta(raw):
    """Raw maintenance reply -> DeltaReply."""
    import json

    return DeltaReply.model_validate(json.loads(raw))


def to_gog(reply):
    """GoalNetReply -> GoG dict (the format of the reference graphs)."""
    nodes = []
    for g in reply.nodes:
        post = {q.item: q.quantity for q in g.produces}
        if g.places:
            post["nearby"] = g.places
        nodes.append({
            "name": g.name,
            "description": g.description,
            "required_tools": list(g.required_tools),
            "required_materials": {q.item: q.quantity for q in g.required_materials},
            "preconditions": list(g.preconditions),
            "postconditions": post,
            "task_list": list(g.task_list),
        })
    edges = [{"from": e.goal, "to": e.prerequisite, "description": e.description} for e in reply.edges]
    return {"nodes": nodes, "edges": edges}


def from_gog(gog):
    """GoG dict -> the reply format, so a graph shown back to the model (repair,
    maintain) reads exactly like the format it is asked to answer in."""
    nodes = []
    for n in gog.get("nodes", []):
        post = dict(n.get("postconditions") or {})
        places = post.pop("nearby", None)
        mats = n.get("required_materials") or {}
        if isinstance(mats, list):
            mats = {m: 1 for m in mats}
        nodes.append({
            "name": n.get("name"),
            "description": n.get("description", ""),
            "required_tools": list(n.get("required_tools") or []),
            "required_materials": [{"item": k, "quantity": v} for k, v in mats.items()],
            "preconditions": list(n.get("preconditions") or []),
            "produces": [{"item": k, "quantity": v} for k, v in post.items()],
            "places": places,
            "task_list": list(n.get("task_list") or []),
        })
    edges = [{"goal": e.get("from"), "prerequisite": e.get("to"), "description": e.get("description", "")}
             for e in gog.get("edges", [])]
    return {"nodes": nodes, "edges": edges}


def parse_reply(raw):
    """Raw reply text -> GoG dict.

    Replies in the GoalNetReply format are converted. Replies saved by the pilots
    that predate structured outputs are already GoG dicts and are returned as is,
    so those runs can still be rescored. Anything else raises.
    """
    import json

    data = json.loads(raw)
    try:
        return to_gog(GoalNetReply.model_validate(data))
    except ValidationError as structured_error:
        edges = data.get("edges") if isinstance(data, dict) else None
        if isinstance(data, dict) and "nodes" in data and all("from" in e for e in edges or []):
            return data
        raise ValueError(f"reply is neither a GoalNetReply nor a GoG dict: {structured_error}") from structured_error
