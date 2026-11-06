"""How Crafter's goals are named for scoring: achievement -> reference goal name, and
the other spellings a constructor plausibly uses for the same goal."""


# Crafter achievement -> reference goal name. Fixed from Crafter's own achievement
# names before any constructed graph was seen, so the alias map cannot be tuned to
# flatter a result.
ACHIEVEMENT_GOALS = {
    "collect_wood": "collect wood",
    "place_table": "place crafting table",
    "make_wood_pickaxe": "craft wooden pickaxe",
    "collect_stone": "mine stone",
    "collect_coal": "mine coal",
    "make_stone_pickaxe": "craft stone pickaxe",
    "place_furnace": "place furnace",
    "collect_iron": "mine iron",
    "make_iron_pickaxe": "craft iron pickaxe",
    "collect_diamond": "mine diamond",
    "collect_sapling": "collect sapling",
}


def aliases():
    """Spellings a constructor plausibly uses for the same goal: systematic verb swaps
    (collect/mine, make/craft) crossed with noun variants (wood/wooden pickaxe,
    table/crafting table)."""
    verbs = {"collect": ["mine"], "mine": ["collect"], "craft": ["make"], "make": ["craft"]}
    nouns = {"wooden pickaxe": ["wood pickaxe"], "crafting table": ["table"]}
    out = {}
    for goal in ACHIEVEMENT_GOALS.values():
        verb, noun = goal.split(" ", 1)
        for v in [verb] + verbs.get(verb, []):
            for n in [noun] + nouns.get(noun, []):
                if f"{v} {n}" != goal:
                    out[f"{v} {n}"] = goal
    return out
