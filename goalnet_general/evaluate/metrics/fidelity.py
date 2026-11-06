"""Graph fidelity: how closely a constructed graph matches the reference structure.

Node and edge precision/recall/F1, and an edit distance. Edge agreement is also
scored on transitive closures ("closure"): a reference may list redundant edges (a
goal linked to a prerequisite and also to that prerequisite's own prerequisite), so
a correct but more compact graph loses edge recall that it does not lose on
what-depends-on-what.
"""

from ._names import _closure, _edges, _nodes, _prf


def graph_fidelity(constructed, reference, aliases=None):
    """Node- and edge-level agreement, plus an edit distance.

    The edit distance is the symmetric difference over nodes and edges (each
    insertion or deletion costs 1). True graph edit distance is NP-hard; this
    bound is exact for the labelled-node case used here, where correspondence is
    given by the names rather than searched for.
    """
    pn, gn = _nodes(constructed, aliases), _nodes(reference, aliases)
    pe, ge = _edges(constructed, aliases), _edges(reference, aliases)
    return {
        "nodes": _prf(pn, gn),
        "edges": _prf(pe, ge),
        "closure": _prf(_closure(pe), _closure(ge)),
        "graph_edit_distance": len(pn ^ gn) + len(pe ^ ge),
    }
