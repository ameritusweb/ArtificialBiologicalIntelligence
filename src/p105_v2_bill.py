"""P105-v2 — the pairing-exhaustion amendment (pre-registered BLIND,
2026-08-13, before any pair outcome was read).

SEQUENCE OF RECORD: the v1 run completed UNTESTED because the matched
control pool (12) < |N| (14) — the telemetry set covers 31 of ~40
active slots in this world class, so unflagged controls are scarce by
the world's nature, not by design deficiency. The v1 card's verdict
stands as printed. This amendment was written and committed BEFORE the
saved pair rows (means, wins) were inspected; only the floor line
printed by the v1 run (set sizes) was seen.

AMENDMENT (the only change): when the unflagged pool is exhausted,
bill on the matched subset — pairs are already formed in deterministic
order (N sorted by slot id, nearest-fit-count matching without
replacement) — with floor PAIRS >= 8. Endpoints, primary (delta
near-miss), win fraction 0.7, and all thresholds are UNCHANGED from
the v1 card.

VERDICTS: SUPPORTED iff pairs >= 8 AND mean dNM(N) > mean dNM(ctl)
AND wins/(wins+losses) >= 0.7. NOT SUPPORTED: pairs >= 8 and
mean(N) <= mean(ctl). PARTIAL: mean elevated, wins < 0.7.
UNTESTED: pairs < 8.
"""

import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'p105_edge_detector.json')
PAIR_FLOOR = 8
WIN_FRAC = 0.7


def main():
    with open(RESULTS) as f:
        data = json.load(f)
    rows = data['pairs']
    m = data['means']
    mean_n, mean_c = m['novel_d_nm'], m['control_d_nm']
    wins, ties, losses = m['wins'], m['ties'], m['losses']
    decided = wins + losses
    win_frac = wins / decided if decided else None

    if len(rows) < PAIR_FLOOR:
        verdict = 'UNTESTED (pairs=%d < %d)' % (len(rows), PAIR_FLOOR)
    elif mean_n > mean_c and decided and win_frac >= WIN_FRAC:
        verdict = ('SUPPORTED: dNM %.2f vs %.2f, wins %.2f (n=%d)'
                   % (mean_n, mean_c, win_frac, len(rows)))
    elif mean_n <= mean_c:
        verdict = ('NOT SUPPORTED: dNM %.2f vs %.2f (n=%d)'
                   % (mean_n, mean_c, len(rows)))
    else:
        verdict = ('PARTIAL: mean elevated (%.2f vs %.2f) but wins '
                   '%.2f < %.2f (n=%d)'
                   % (mean_n, mean_c, win_frac or 0.0, WIN_FRAC,
                      len(rows)))
    print('P105-v2 VERDICT: %s' % verdict)
    print('  wins=%d ties=%d losses=%d' % (wins, ties, losses))
    for r in rows:
        print('  N slot %s %s dNM=%s neg=%s | ctl %s dNM=%s neg=%s'
              % (r['novel_slot'], r['novel_classes'],
                 r['novel']['d_nm'], r['novel']['neg_frac'],
                 r['control_slot'], r['control']['d_nm'],
                 r['control']['neg_frac']))
    data['v2_amendment_verdict'] = verdict
    with open(RESULTS, 'w') as f:
        json.dump(data, f, indent=1, default=str)
    print('billed into %s' % RESULTS)


if __name__ == '__main__':
    main()
