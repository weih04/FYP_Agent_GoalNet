"""The interface between the general Goal Net code and one environment.

goalnet_general never imports an environment package directly. Every command takes
--env NAME and loads goalnet_NAME, which must expose ADAPTER, an instance of a
subclass of Adapter. Moving the experiments to another environment means writing
goalnet_<env>/ with its own Adapter; nothing in goalnet_general changes, so the
frozen prompts stay frozen.

Data the adapter hands over:

* experience summaries - one dict per episode, numbered 1..N across all runs given:
      {"episode": n, "steps": int, "outcome": str, "achievements": [...],
       "interactions": [...], ...}
  The model reads these as they are, so their keys should describe themselves.
  "steps" is required (evidence checks use it); the rest is the adapter's choice.
* metas - the raw per-episode records, passed back to the adapter
  (achieved_goals, episode_succeeded, check_variant); the general code never reads them.
* paths - per-episode files, passed back to the adapter (observed_names, successes).
* reference graphs - GoG dicts with "_root_goal" (see derive.py for node fields).
* success records for the symbolic baseline - see `successes`.
"""

import importlib
from abc import ABC, abstractmethod


class Adapter(ABC):
    name = None                      # e.g. "crafter"
    secrets_path = None              # optional KEY=value file for API keys

    # --- what the model is told -----------------------------------------------------

    @abstractmethod
    def environment(self):
        """environment.Environment: actions, task_list shorthands, names, tool names."""

    @abstractmethod
    def task_text(self, task):
        """How a task id is stated to the model, e.g. "collect_diamond" -> "collect diamond"."""

    # --- tasks and references ---------------------------------------------------

    @property
    @abstractmethod
    def tasks(self):
        """{"dev": [...], "test": [...]}: prompts are tuned on dev and frozen before test."""

    @abstractmethod
    def reference(self, task, variant=None):
        """(reference GoG, root goal name) for a task in a world variant."""

    @abstractmethod
    def variant_id(self, variant):
        """A short stable id of a world variant ("default" for the unmodified world)."""

    @property
    @abstractmethod
    def goal_aliases(self):
        """{other spelling: reference goal name}, for scoring only; fixed before any result."""

    # --- experience -------------------------------------------------------------

    @abstractmethod
    def load_experience(self, runs, task):
        """(summaries numbered 1..N across runs, metas, paths) for `task` in the given runs."""

    @abstractmethod
    def check_variant(self, metas, variant):
        """Raise unless every episode was played in `variant`."""

    @abstractmethod
    def achieved_goals(self, meta):
        """Reference goal names an episode achieved."""

    @abstractmethod
    def episode_succeeded(self, meta):
        """Did the episode complete its task?"""

    @abstractmethod
    def observed_names(self, paths):
        """Names the agent saw in these episodes (the vocabulary="observed" condition)."""

    @abstractmethod
    def episode_shows(self, episode, node):
        """Does an experience summary show `node` (a GoG node) achieved? For provenance."""

    @abstractmethod
    def evidence_records(self, episode, node):
        """The interactions in an experience summary that achieved `node`, for checking an
        observed goal against its evidence during maintenance. One dict each:
            step        the interaction's (first) step
            action      the action taken
            consumed    {item: quantity} used up
            gained      {item: quantity} gained
            placed      the thing put in the world, or None
            present     things around the agent when it acted
            held        items held when it acted"""

    @abstractmethod
    def failure_records(self, episodes, node):
        """Attempts at `node`'s action that changed nothing and say something about its
        requirements, across `episodes` ({number: summary}). The adapter leaves out
        failures its environment explains otherwise (chance, a blocked target). One dict each:
            episode     the episode number
            step        the interaction's (first) step
            action      the action taken
            times       how many times it changed nothing in that state
            present     things around the agent when it acted
            held        {item: quantity} held when it acted"""

    @abstractmethod
    def successes(self, paths):
        """Successful interactions, for the symbolic baseline. One dict each:
            name        goal name, e.g. "collect wood"
            places      True if the goal puts `product` in the world rather than gaining it
            product     the item gained or the thing placed
            gained      {item: qty} gained
            used        {item: qty} used up
            held        set of items held before the action
            near        set of things around the agent before the action
            surroundings_matter  True if `near` can hold requirements (crafting, placing)
            task_list   the actions, as a GoG task_list
        """

    # --- acting (closed loop) ---------------------------------------------------

    @abstractmethod
    def run_episodes(self, task, graph_path, out_dir, episodes, seed, actor_model, variant, options):
        """Play `episodes` episodes of `task`, guided by the GoG at `graph_path` (None = no
        graph), and return the directory of the new run. `options` are environment-specific
        settings ({key: value strings}) from --env-option."""


def load(name):
    """The ADAPTER of package goalnet_<name>."""
    try:
        module = importlib.import_module(f"goalnet_{name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"no environment package goalnet_{name}") from e
    adapter = getattr(module, "ADAPTER", None)
    if not isinstance(adapter, Adapter):
        raise TypeError(f"goalnet_{name}.ADAPTER must be an instance of goalnet_general.adapter.Adapter")
    return adapter
