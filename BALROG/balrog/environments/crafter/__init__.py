from .env import ACTIONS, CrafterLanguageWrapper

ACTION_DICT = {
    "Noop": "do nothing",
    "Move West": "move west on flat ground",
    "Move East": "move east on flat ground",
    "Move North": "move north on flat ground",
    "Move South": "move south on flat ground",
    "Do": "Multiuse action to collect material, drink from lake and hit creature in front",
    "Sleep": "sleep when energy level is below maximum",
    "Place Stone": "place a stone in front",
    "Place Table": "place a table",
    "Place Furnace": "place a furnace",
    "Place Plant": "place a plant",
    "Make Wood Pickaxe": "craft a wood pickaxe with a nearby table and wood in inventory",
    "Make Stone Pickaxe": "craft a stone pickaxe with a nearby table, wood, and stone in inventory",
    "Make Iron Pickaxe": "craft an iron pickaxe with a nearby table and furnace, wood, coal, and iron in inventory",
    "Make Wood Sword": "craft a wood sword with a nearby table and wood in inventory",
    "Make Stone Sword": "craft a stone sword with a nearby table, wood, and stone in inventory",
    "Make Iron Sword": "craft an iron sword with a nearby table and furnace, wood, coal, and iron in inventory",
}


# Crafter achievement names (as in info["achievements"]). Any of these is a valid task;
# "default" keeps BALROG's original open-ended objective.
ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie", "collect_stone",
    "place_stone", "eat_plant", "defeat_skeleton", "make_stone_pickaxe", "make_stone_sword",
    "wake_up", "place_furnace", "collect_coal", "collect_iron", "make_iron_pickaxe",
    "make_iron_sword", "collect_diamond",
]
DEFAULT_TASK = "default"


def task_goal_sentence(task):
    """Goal line for the instruction prompt. "default" = BALROG's original objective."""
    if task in (None, "", DEFAULT_TASK):
        return "Your main goal is to get as far as possible by completing all the achievements."
    if task not in ACHIEVEMENTS:
        raise ValueError(f"Unknown crafter task {task!r}; expected 'default' or one of {ACHIEVEMENTS}")
    verb, *obj = task.split("_")
    return (
        "Take the next best action in the game based on the current game state, the history "
        "and your goal.\n"
        f"Your main goal is to {verb} {' '.join(obj)}. The episode ends as soon as you achieve it."
    )


# Survival mechanics as stated in crafter's source (objects.Player/Zombie/Skeleton, env._balance_chunk).
# Facts only: no crafting recipes or ordering, which the Goal Net must recover itself.
SURVIVAL_RULES = """
SURVIVAL RULES:
- Your status shows your health, food, drink and energy. If food, drink or energy reaches 0, you lose health over time. Otherwise, health slowly recovers.
- Restore food by facing a cow and using Do repeatedly until you eat it. Restore drink by facing water and using Do. Restore energy with Sleep.
- Do interacts only with the tile you are facing. Check the observation ("You face ...") to confirm which tile you face before interacting.
- Move turns you toward the chosen direction first. If that tile is blocked (for example by a creature, tree, stone or water), you turn without changing position. Check your observations to determine whether movement succeeded.
- Zombies can approach and attack you at close range. A zombie 1 step north, south, east or west of you is next to you and can attack. Zombies deal much more damage while you are asleep.
- Skeletons can attack from a distance with arrows.
- Avoid remaining near threats when your health is low. If you choose to fight a zombie or skeleton next to you, first Move toward it so you face it, then use Do repeatedly.
- Avoid lava, as entering it causes instant death.
- Dying ends the episode and prevents completion of the current task.
"""

def _make_description(action, hints):
    """A Make action's description, stated from the recipes currently in force.

    BALROG's fixed ACTION_DICT text describes crafter's default recipes; under
    recipe overrides it would mislead the agent, so it is generated from
    crafter.constants.make instead. With hints off only the crafted item is named.
    """
    import crafter.constants

    item = action[len("Make "):].lower().replace(" ", "_")
    article = "an" if item[0] in "aeiou" else "a"
    if not hints:
        return f"craft {article} {item.replace('_', ' ')}"
    info = crafter.constants.make[item]
    stations = " and ".join(info["nearby"])
    materials = ", ".join(info["uses"])
    return f"craft {article} {item.replace('_', ' ')} with a nearby {stations} and {materials} in inventory"


def action_descriptions(action_hints=True, recipes_overridden=False):
    """ACTION_DICT, with Make descriptions regenerated when they could be wrong or are unwanted."""
    out = dict(ACTION_DICT)
    if recipes_overridden or not action_hints:
        for action in out:
            if action.startswith("Make "):
                out[action] = _make_description(action, action_hints)
    return out


def get_instruction_prompt(task=None, survival_rules=False, action_hints=True, recipes_overridden=False):
    descriptions = action_descriptions(action_hints, recipes_overridden)
    action_strings = ",\n".join(f"{action}: {descriptions[action]}" for action in ACTIONS)
    instruction_prompt = f"""
You are an agent playing Crafter. The following are the ONLY valid actions you can take in the game, 
followed by a short description of each action:

{action_strings}.

STRICT RULES:
- There are no other possible actions in the game. 
- "North", "South", "East", and "West" are the only four cardinal directions in the game world.
- There are NO DIAGONAL MOVES such as "Move NorthEast" or "Move SouthWest".
- You can only take one action at a time, and you must wait for the next observation before taking another action.
- Achievements are goals, not actions. There is no 'Collect' action. To collect wood, stone, coal, iron or diamond, face it and output Do.
- A "nearby" table or furnace means one within 1 tile of you, including diagonally.

These survival rules are only present to help you achieve main goals better. Follow them only if necessary.
{SURVIVAL_RULES if survival_rules else ""}

These are the game achievements you can get:
1. Collect Wood
2. Place Table
3. Eat Cow
4. Collect Sapling
5. Collect Drink
6. Make Wood Pickaxe
7. Make Wood Sword
8. Place Plant
9. Defeat Zombie
10. Collect Stone
11. Place Stone
12. Eat Plant
13. Defeat Skeleton
14. Make Stone Pickaxe
15. Make Stone Sword
16. Wake Up
17. Place Furnace
18. Collect Coal
19. Collect Iron
20. Make Iron Pickaxe
21. Make Iron Sword
22. Collect Diamond

In a moment I will present a history of actions and observations from the game.
{task_goal_sentence(task)}


PLAY!
""".strip()

    return instruction_prompt
