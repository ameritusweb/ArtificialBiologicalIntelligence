"""P76-v4 (card locked at launch): consumption via EXPECTATION
RECEIPTS — first-class evidence (sov.expectation_receipt, battery
165/165), stride 1.

MASS PROJECTION FROM MEASURED ADMISSION: v3 measured the admission
ceiling at ~300 events (TAU_E-limited). At EXPECT_ALPHA=0.05, ~300
events / ~33 slots ~= 9 per slot -> cumulative geometric weight
~9 x 5% ~= 37% of each consuming slot's centroid/radius — three
orders above the fit-EMA's per-event 1/n and well above the 1e-4
noise floor. The channel is truth-directed by construction (updates
use the ACTUAL lived embedding; the prediction only licenses the
boosted weight), Law-3 grounded through the confirming fit.
Endpoints, verdicts, floors: v1 card verbatim (constrains counter now
counts expectation receipts).
"""

import os

import staged_fit_experiment as sfe

sfe.CONSTRAIN_STRIDE = 1
sfe.CONSUME_MODE = 'expectation'
sfe.RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'p76_v4_staged_fit.json')

if __name__ == '__main__':
    sfe.main()
