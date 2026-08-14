"""P76 density sweep — one cell (the card, locked at launch;
sub-claims (b) and (d) of the ORIGINAL P76 registration, instrument =
the F32-supported expectation channel).

PREDICTIONS (registered 2026-08-10): (b) the serial margin GROWS with
edge density — expectations are edge-derived, so a sparse graph has
little to stage; (d) receipts-discarded stays inert at every density
(S+D == P+D identically, by construction in this implementation —
reported as design confirmation).

CELLS: warmup length W in {2, 6, 12} worlds (low/mid/high density;
densities measured, not assumed — reported per cell). Same four
accountants, stride 1, expectation channel, EDGE_FLOOR relaxed to 0
for the low cell (the treatment, not a deficiency; every other check
carries). Endpoint per cell: margin = P+D tightness − S+C tightness
(positive = staging helps), plus the P+C damage figure.
VERDICT (pooled by the analyzer inline next session or by inspection):
(b) SUPPORTED iff margin is monotone non-decreasing in measured
density with the high cell positive; NOT SUPPORTED iff flat or
decreasing. UNTESTED: any cell's fit floors unmet.

Usage: python p76_density.py <n_warmup_worlds>
"""

import os
import sys

import staged_fit_experiment as sfe

W = int(sys.argv[1])
sfe.WARMUP_WORLDS = [(96500 + i, (4, 3)[i % 2]) for i in range(W)]
sfe.TREAT_WORLDS = [(96600 + i, (4, 3)[i % 2]) for i in range(6)]
sfe.CONSTRAIN_STRIDE = 1
sfe.CONSUME_MODE = 'expectation'
sfe.EDGE_FLOOR = 0
sfe.RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', f'p76_density_w{W}.json')

if __name__ == '__main__':
    sfe.main()
