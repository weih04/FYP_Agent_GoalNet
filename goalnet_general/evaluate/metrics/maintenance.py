"""Maintenance metrics: a Goal-Oriented Graph tracked across successive revisions.

Construction metrics score one graph; these score a *sequence* of graphs, which
is what the incremental-maintenance half of the thesis measures. Churn says how
much a revision changed; the distance trend says whether those changes moved the
graph toward the reference or away from it. A method can have low churn and still
be wrong (it stopped changing early), or high churn and be converging, so both
are reported.
"""

from ._names import _closure, _edges
from .fidelity import graph_fidelity


def churn(versions, aliases=None):
    """Edit distance between each consecutive pair of graph versions."""
    if len(versions) < 2:
        return []
    return [graph_fidelity(b, a, aliases)["graph_edit_distance"]
            for a, b in zip(versions, versions[1:])]


def distance_trend(versions, reference, aliases=None):
    """Edit distance from each version to the reference graph."""
    return [graph_fidelity(v, reference, aliases)["graph_edit_distance"] for v in versions]


def fidelity_trend(versions, reference, aliases=None):
    """Edge precision/recall/F1 of each version against the reference."""
    return [graph_fidelity(v, reference, aliases)["edges"] for v in versions]


def forgetting(versions, reference, aliases=None):
    """For each revision, the reference dependencies the graph had and then lost.

    Counted on transitive closures (metrics.graph_fidelity's "closure"), so rerouting
    a correct dependency through an intermediate goal is not counted as forgetting.
    """
    gold = _closure(_edges(reference, aliases))
    held = [_closure(_edges(v, aliases)) & gold for v in versions]
    return [len(a - b) for a, b in zip(held, held[1:])]


def episodes_to_threshold(values, threshold):
    """First version index whose value reaches `threshold`, or None if none does.

    Version 0 is the initial graph, so the index is the number of episodes applied.
    """
    return next((i for i, v in enumerate(values) if v >= threshold), None)


def unsupported_changes(changelog):
    """Additions and revisions not backed by the episode that caused them.

    A change made while processing episode n is supported when it cites evidence from
    episode n. Changes reported as "prior" and changes made by a repair round are
    listed with their kind rather than hidden, because they are the edits the
    experience did not ask for. Removals carry a stated reason and are not counted.
    """
    out = []
    for c in changelog:
        if c["op"] not in ("add", "revise"):
            continue
        cites = any(e.get("episode") == c["episode"] for e in c.get("evidence", []))
        if cites:
            continue
        kind = ("repair" if "repair" in c else "retarget" if "retarget" in c
                else "prior" if c.get("basis") == "prior" else "uncited")
        out.append({"episode": c["episode"], "op": c["op"], "goal": c["goal"], "kind": kind})
    return out
