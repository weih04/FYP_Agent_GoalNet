"""Every metric used to evaluate a Goal Net, one family per module:

  validity      structural checks (well-formed and plannable, never "correct")
  fidelity      agreement with the reference: nodes, edges, closure, edit distance
  provenance    observed vs prior goals, and whether cited evidence shows them
  plan_quality  goal satisfaction, soundness, completeness, efficiency (GoG §4.4.1)
  maintenance   churn, forgetting, episodes-to-threshold, unsupported changes
  acting        achievement depth and success rate of an agent's episodes

No metric looks at anything but graphs, reference graphs and episode records, and
none depends on an environment beyond what the adapter passes in.
"""

from .acting import depth, success_rate
from .fidelity import graph_fidelity
from .maintenance import (churn, distance_trend, episodes_to_threshold, fidelity_trend, forgetting,
                          unsupported_changes)
from .plan_quality import bill_of_materials, graph_bill_of_materials, plan_from_gog, plan_quality
from .provenance import provenance
from .validity import gated_recall, validate, validity_rate
