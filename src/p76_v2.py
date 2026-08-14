"""P76-v2 — the dose-iterated staged-fit card (locked at launch;
licensed by F30's mediator data, required by check-6's quantitative
corollary to state its mass estimate IN ADVANCE).

ONE KNOB from v1 (staged_fit_experiment.py, whose card, arms, endpoints
and verdict rules carry over verbatim): CONSTRAIN_STRIDE 50 -> 5.

THE MASS ESTIMATE (the clause v1 lacked): stride 5 yields ~900-1000
consumption events across ~33 slots ~= 30 per slot. Constrain acts on
geometry DIRECTLY (no EMA damping — the reason it is the right lawful
channel; v1's failure was count, not mechanism): per event ~4.5%
radius contraction and ~9% centroid displacement toward the predicting
slot. Cumulative per slot: radius x ~0.25, displacement ~1 sigma of
the fit distribution — two orders above v1's dose and well above the
endpoint's noise floor (v1 measured tightness deltas of 1e-4; the
projected geometric motion is ~1e-1). If S+C's expectations are
directionally true, tightness must move; if they are false, tightness
must WORSEN measurably — either way the endpoint now hears the
mechanism (check 6 satisfied quantitatively, both directions stated).

Note: runs under the net-per-fire economy (2026-08-12) — arms compare
policies WITHIN one economy, so the economy change is common-mode.

Verdicts, floors, endpoints: identical to v1's card. Results ->
results/p76_v2_staged_fit.json.
"""

import os

import staged_fit_experiment as sfe

sfe.CONSTRAIN_STRIDE = 5
sfe.RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'p76_v2_staged_fit.json')

if __name__ == '__main__':
    sfe.main()
