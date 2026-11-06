"""Crafter (through BALROG) for the Goal Net experiments: the environment adapter,
experience reader, vocabulary, recipe variants and reference graphs.

goalnet_general loads ADAPTER via --env crafter; see adapter.py."""

from .adapter import CrafterAdapter

ADAPTER = CrafterAdapter()
