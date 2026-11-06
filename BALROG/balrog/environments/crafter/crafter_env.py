import copy
from typing import Optional

import crafter
import crafter.constants
import yaml
from hydra.utils import to_absolute_path
from balrog.environments.crafter import CrafterLanguageWrapper
from balrog.environments.wrappers import GymV21CompatibilityV0


# crafter reads recipes from module globals (crafter.constants.collect/place/make) each
# time an action runs, so a world's recipes are set by patching those globals. The
# pristine copy lets every env start from data.yaml, whatever an earlier env applied.
_PRISTINE = {k: copy.deepcopy(getattr(crafter.constants, k)) for k in ("collect", "place", "make")}


def apply_recipe_overrides(path):
    """Reset crafter's recipes to data.yaml, then merge the overrides in `path` (if any).

    Same merge as goalnet_crafter.recipes.recipes: each named entry's listed fields are replaced.
    """
    for key, value in _PRISTINE.items():
        setattr(crafter.constants, key, copy.deepcopy(value))
    if not path:
        return
    with open(to_absolute_path(path)) as f:
        overrides = yaml.safe_load(f) or {}
    for section, entries in overrides.items():
        if section == "description":
            continue
        if section not in _PRISTINE:
            raise ValueError(f"{path}: unknown recipe section {section!r}")
        table = getattr(crafter.constants, section)
        for name, fields in entries.items():
            if name not in table:
                raise ValueError(f"{path}: unknown {section} entry {name!r}")
            table[name].update(fields)


def make_crafter_env(env_name, task, config, render_mode: Optional[str] = None, seed: Optional[int] = None):
    crafter_kwargs = dict(config.envs.crafter_kwargs)
    max_episode_steps = crafter_kwargs.pop("max_episode_steps", 2)
    unique_items = crafter_kwargs.pop("unique_items", True)
    precise_location = crafter_kwargs.pop("precise_location", False)
    skip_items = crafter_kwargs.pop("skip_items", [])
    edge_only_items = crafter_kwargs.pop("edge_only_items", [])
    crafter_kwargs.pop("survival_rules", None)  # prompt option, read by make_env
    crafter_kwargs.pop("action_hints", None)    # prompt option, read by make_env
    apply_recipe_overrides(crafter_kwargs.pop("recipes", None))

    for param in ["area", "view", "size"]:
        if param in crafter_kwargs:
            crafter_kwargs[param] = tuple(crafter_kwargs[param])

    # Seed the world explicitly: with seed=None crafter.Env draws it from the global numpy RNG,
    # which the first import of this package consumes, breaking per-episode reproducibility.
    # An explicit crafter_kwargs.seed still wins (same world every episode).
    if crafter_kwargs.get("seed") is None:
        crafter_kwargs["seed"] = seed

    env = crafter.Env(**crafter_kwargs)
    env = CrafterLanguageWrapper(
        env,
        task,
        max_episode_steps=max_episode_steps,
        unique_items=unique_items,
        precise_location=precise_location,
        skip_items=skip_items,
        edge_only_items=edge_only_items,
    )
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
