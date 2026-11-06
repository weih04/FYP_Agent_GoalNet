"""Provenance, for facts-mode graphs: how the goals the model marked "observed" and
"prior" each compare with the reference, and whether each observed goal's cited
evidence actually shows it. It separates what the graph learned from the agent's
experience from what the model brought from pre-training.
"""

from ._names import _nodes, _norm


def provenance(graph, reference, experience, shows, aliases=None):
    """Observed vs prior goals: how many, how many are in the reference, and whether
    an observed goal's cited episode shows it (its evidence is "verified").

    `shows(episode, node)` is the environment adapter's episode_shows: whether an
    experience summary shows the goal achieved, read from that environment's format.

    A cited step must also lie inside that episode. Steps are not matched exactly,
    because summaries group repeated interactions and keep only the first step.
    """
    ref = _nodes(reference, aliases)
    episodes = {e["episode"]: e for e in experience}
    out = {}
    for basis in ("observed", "prior"):
        goals = [n for n in graph.get("nodes", []) if n.get("basis") == basis]
        in_ref = [n for n in goals if _norm(n["name"], aliases) in ref]
        out[basis] = {"goals": len(goals), "in_reference": len(in_ref),
                      "precision": len(in_ref) / len(goals) if goals else None}
    observed = [n for n in graph.get("nodes", []) if n.get("basis") == "observed"]
    verified = []
    for n in observed:
        ok = False
        for cite in n.get("evidence") or []:
            ep = episodes.get(cite.get("episode"))
            if ep and 0 <= cite.get("step", -1) < ep["steps"] and shows(ep, n):
                ok = True
                break
        verified.append(ok)
    out["evidence_verified_rate"] = sum(verified) / len(verified) if verified else None
    out["unverified_observed"] = [n["name"] for n, ok in zip(observed, verified) if not ok]
    return out
