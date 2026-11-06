"""What the frozen prompts need to know about an environment, kept out of the prompts.

The prompts (prompts.py, schema.py, problems.py, evaluate/metrics/validity.py) are fingerprinted and
frozen; they must work unchanged in another environment. Everything
environment-specific they show the model comes from an Environment instead: the
valid actions, any shorthands a task_list may use in place of a sequence of
actions, and the names the environment defines. Each environment package builds
one in its adapter (goalnet_<env>/adapter.py).

An Environment is data, not prompt wording, so it is not part of the fingerprint.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Shorthand:
    """A task_list step that stands for several actions, e.g. "Move toward <name>"."""
    prefix: str                 # what a step starts with, e.g. "Move toward "
    form: str                   # how it is shown to the model, e.g. "Move toward <name>"
    meaning: str                # one line explaining it


@dataclass(frozen=True)
class Environment:
    name: str
    actions: tuple
    shorthands: tuple = ()
    names: frozenset = field(default_factory=frozenset)       # every name the environment defines
    tool_names: frozenset = field(default_factory=frozenset)  # names that persist when used (tools, stations)

    def is_valid_step(self, step):
        return step in self.actions or any(step.startswith(s.prefix) for s in self.shorthands)
