"""Acting metrics: how far an agent got in the environment, per episode.

Achievement depth is the fraction of the reference goal tree an episode achieved.
Task success is rare for small actors on deep tasks, so depth is the measure that
separates conditions; success rate is reported alongside it.
"""

from .plan_quality import plan_from_gog


def depth(meta, reference, root, adapter):
    """Fraction of the reference goal tree the episode achieved (achievement depth)."""
    goals = plan_from_gog(reference, root)
    got = adapter.achieved_goals(meta)
    return sum(g in got for g in goals) / len(goals)


def success_rate(metas, adapter):
    """Fraction of episodes that completed their task."""
    if not metas:
        raise ValueError("success_rate needs at least one episode")
    return sum(adapter.episode_succeeded(m) for m in metas) / len(metas)
