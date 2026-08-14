"""P76-v3 (card locked at launch): CONSTRAIN_STRIDE 5 -> 1.

MASS PROJECTION FROM MEASURED ADMISSION (the check-6 quantitative way):
v2 MEASURED 213 consumption events at stride 5 (events scaled ~2.4x
from stride 50 -> 5, so stride remained partially binding alongside
the TAU_E admission rate). Stride 1 projects ~5x v2 = ~1000 events
~= 30 per slot -> cumulative radius x ~0.25 and ~1 sigma displacement
— the v2 card's intended dose, now projected from MEASURED admission
instead of assumption. Endpoints, verdicts, floors: v1 card verbatim.
"""

import os

import staged_fit_experiment as sfe

sfe.CONSTRAIN_STRIDE = 1
sfe.RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'p76_v3_staged_fit.json')

if __name__ == '__main__':
    sfe.main()
