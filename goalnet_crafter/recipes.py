"""Crafter recipe sets: the world variants the experiments run in.

A counterfactual world changes some recipes (crafter/data.yaml `collect`, `place`,
`make`) so that what the model believes from pre-training is wrong and only
experience can say what is right. The gap between a zero-experience graph and an
experience-built graph is then a measure of learning rather than recall. The
reference graph each recipe set implies is generated in evaluate/references.py.

An override file uses data.yaml's layout and replaces whole fields of the
entries it names, e.g.
    make:
      wood_pickaxe: {uses: {wood: 1, sapling: 1}}
The merge mirrors BALROG's crafter_env.apply_recipe_overrides; they must agree.
"""

import copy
import hashlib
import json

import yaml

from .vocabulary import crafter_data

RECIPE_KEYS = ("collect", "place", "make")


def load_overrides(path):
    """Override dict from a YAML file; None or "" means the default recipes."""
    if not path:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - set(RECIPE_KEYS) - {"description"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}; expected {RECIPE_KEYS}")
    return data


def recipes(overrides=None):
    """{collect, place, make} with `overrides` merged over crafter's data.yaml."""
    base = {k: copy.deepcopy(crafter_data()[k]) for k in RECIPE_KEYS}
    for section, entries in (overrides or {}).items():
        if section not in RECIPE_KEYS:
            continue
        for name, fields in entries.items():
            if name not in base[section]:
                raise ValueError(f"override names unknown {section} entry {name!r}")
            base[section][name].update(fields)
    return base


def fingerprint(overrides):
    """Short stable id of a recipe set, recorded in run outputs."""
    text = json.dumps(overrides or {}, sort_keys=True)
    return "default" if not overrides else hashlib.sha256(text.encode()).hexdigest()[:12]
