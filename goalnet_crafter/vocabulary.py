"""Crafter vocabulary, read from the environment itself.

Grounding checks must compare a constructed graph against what Crafter actually
defines, not against a hand-copied list that can drift. Items and materials come
from crafter/data.yaml; the action names are BALROG's language action space.
"""

import functools
import os

import crafter
import yaml

# BALROG's Crafter action space (balrog/environments/crafter/env.py::ACTIONS).
# Duplicated here so goalnet stays importable without BALROG on the path.
ACTIONS = [
    "Noop", "Move West", "Move East", "Move North", "Move South", "Do", "Sleep",
    "Place Stone", "Place Table", "Place Furnace", "Place Plant",
    "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe",
    "Make Wood Sword", "Make Stone Sword", "Make Iron Sword",
]


@functools.lru_cache(maxsize=1)
def crafter_data():
    """Parsed crafter/data.yaml - the authoritative source for recipes."""
    path = os.path.join(os.path.dirname(crafter.__file__), "data.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def vocabulary():
    """Names a constructed graph is allowed to reference."""
    d = crafter_data()
    items = set(d["items"])
    materials = set(d["materials"])
    # Tools are the craftable items plus the two placeable stations.
    tools = set(d["make"]) | {"table", "furnace"}
    return {
        "items": items,
        "materials": materials,
        "tools": tools,
        "achievements": set(d["achievements"]),
        "all": items | materials | tools,
    }
