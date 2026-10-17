# Reproducibility-only ablations

The primary model is `model/SARBF_Net.py`.  This directory retains mechanisms
from earlier controlled comparisons:

- `no_mass_bcif`: removes external history confidence only inside BCIF;
- `unbalanced_bcif`: removes BCIF's balanced cancellation term.

They remain callable through `tools.sa_lcm_transport.run_pilot`, but are not
part of the primary model package or recommended deployment path.  Historical
procedures are stored under `ablations/docs/`.
