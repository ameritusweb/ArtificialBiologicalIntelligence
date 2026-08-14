"""P105@BAND pooled analyzer — same card as p105_band.py (verdict
rules fixed there; this pools R replicates and prints the bill).

Floors (pooled): oracle mass >= 10 strain events on paired slots
(dNM events + neg_frac > 0.05 slots + reopens on either side);
pairs >= 8. VOID-BY-ORACLE / UNTESTED / SUPPORTED (mean + wins>=0.7)
/ NOT SUPPORTED / PARTIAL as the card states.
"""

import glob
import json
import os

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'results', 'p105_band_r*.json')
ORACLE_FLOOR = 10
PAIR_FLOOR = 8
WIN_FRAC = 0.7
NEG_EVENT = 0.05


def main():
    rows, reps = [], []
    for path in sorted(glob.glob(BASE)):
        with open(path) as f:
            d = json.load(f)
        reps.append(d['replicate'])
        rows.extend(d['pairs'])
    print('pooled: %d pairs from reps %s' % (len(rows), reps))

    oracle = 0
    wins = ties = losses = 0
    for r in rows:
        sn, sc = r['novel'], r['control']
        oracle += sn['d_nm'] + sc['d_nm']
        oracle += sn['d_reopen'] + sc['d_reopen']
        oracle += int(sn['neg_frac'] > NEG_EVENT)
        oracle += int(sc['neg_frac'] > NEG_EVENT)
        if sn['d_nm'] > sc['d_nm']:
            wins += 1
        elif sn['d_nm'] == sc['d_nm']:
            ties += 1
        else:
            losses += 1
    mean_n = (float(np.mean([r['novel']['d_nm'] for r in rows]))
              if rows else None)
    mean_c = (float(np.mean([r['control']['d_nm'] for r in rows]))
              if rows else None)
    decided = wins + losses
    wf = wins / decided if decided else None

    if oracle < ORACLE_FLOOR:
        verdict = ('VOID-BY-ORACLE (pooled strain events %d < %d)'
                   % (oracle, ORACLE_FLOOR))
    elif len(rows) < PAIR_FLOOR:
        verdict = 'UNTESTED (pairs %d < %d)' % (len(rows), PAIR_FLOOR)
    elif mean_n > mean_c and decided and wf >= WIN_FRAC:
        verdict = ('SUPPORTED: dNM %.2f vs %.2f, wins %.2f, oracle '
                   'mass %d' % (mean_n, mean_c, wf, oracle))
    elif mean_n <= mean_c:
        verdict = ('NOT SUPPORTED: dNM %.2f vs %.2f with live oracle '
                   '(mass %d)' % (mean_n, mean_c, oracle))
    else:
        verdict = ('PARTIAL: dNM %.2f vs %.2f, wins %.2f < %.2f '
                   '(oracle %d)' % (mean_n, mean_c, wf or 0.0,
                                    WIN_FRAC, oracle))
    print('P105@BAND POOLED VERDICT: %s' % verdict)
    print('  wins=%d ties=%d losses=%d' % (wins, ties, losses))
    for r in rows:
        print('  N %s %s dNM=%s re=%s neg=%s | C %s dNM=%s re=%s '
              'neg=%s' % (r['novel_slot'], r['novel_classes'],
                          r['novel']['d_nm'], r['novel']['d_reopen'],
                          r['novel']['neg_frac'], r['control_slot'],
                          r['control']['d_nm'],
                          r['control']['d_reopen'],
                          r['control']['neg_frac']))


if __name__ == '__main__':
    main()
