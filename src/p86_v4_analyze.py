"""P86 v4 pooled analyzer — the successor card F38 pre-registered
(locked 2026-08-13, before any rep >= 1 results existed; rep-0 pair
receipts stand and pool as members).

CHANNELS: world-sensitive ONLY (court_pending + lifecycle). The
genesis channel is excluded as a treatment-independent constant
(F38: 2.0/gen everywhere — the check-8 cousin clause).

VERDICTS (pooled over R replicate pairs): floors — pooled DOSED P2
world-sensitive events >= 8 (from F28's ~0.25/gen x 12 gens x R at
R>=3, floor set below expectation), else VOID-BY-BAND; both arms of
every pair closed in P1, else that pair is dropped (UNTESTED if < 2
pairs remain). SUPPORTED iff (i) STARVATION: pooled dosed P2 rate
>= 3x pooled spent P2 rate; (ii) STOCK PARITY: every pair ends P2
with |spent assertable - dosed assertable| <= 1; (iii) RESTART:
pooled spent P3 rate >= 3x pooled spent P2 rate AND pooled spent P3
rate > pooled dosed P3 rate (migration-specific, not time).
NOT SUPPORTED (endogeny): spent P2 within 1.5x of dosed P2.
NOT SUPPORTED (irreversibility): (i)+(ii) hold, restart fails.
PARTIAL between.
"""

import glob
import json
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'results')
DOSED_FLOOR = 8


def load(arm):
    out = {}
    for p in glob.glob(os.path.join(BASE, 'p86_v4_%s*.json' % arm)):
        name = os.path.basename(p)
        if arm == 'spent' and 'dosed' in name:
            continue
        with open(p) as f:
            d = json.load(f)
        rep = 0
        if '_r' in name:
            rep = int(name.rsplit('_r', 1)[1].split('.')[0])
        out[rep] = d
    return out


def ws_events(d, phase):
    rows = [f for f in d['flow_per_gen'] if f['phase'] == phase]
    ev = sum(f['court_pending'] + f['lifecycle'] for f in rows)
    return ev, len(rows)


def main():
    spent, dosed = load('spent'), load('dosed')
    reps = sorted(set(spent) & set(dosed))
    print('pairs found: %s' % reps)
    kept = []
    for r in reps:
        if spent[r]['closure_gen_p1'] is None \
                or dosed[r]['closure_gen_p1'] is None:
            print('  rep %d dropped: no P1 closure' % r)
            continue
        kept.append(r)
    if len(kept) < 2:
        print('P86 v4 POOLED: UNTESTED (pairs=%d < 2)' % len(kept))
        return

    tot = {('spent', 'p2'): [0, 0], ('spent', 'p3'): [0, 0],
           ('dosed', 'p2'): [0, 0], ('dosed', 'p3'): [0, 0]}
    parity_ok = True
    for r in kept:
        for arm, d in (('spent', spent[r]), ('dosed', dosed[r])):
            for ph in ('p2', 'p3'):
                ev, n = ws_events(d, ph)
                tot[(arm, ph)][0] += ev
                tot[(arm, ph)][1] += n
        sa = [s for s in spent[r]['stock_per_gen']
              if s['phase'] == 'p2'][-1]['assertable']
        da = [s for s in dosed[r]['stock_per_gen']
              if s['phase'] == 'p2'][-1]['assertable']
        if abs(sa - da) > 1:
            parity_ok = False
        print('  rep %d: spent p2=%s p3=%s | dosed p2=%s p3=%s | '
              'assertable %d vs %d'
              % (r, ws_events(spent[r], 'p2'),
                 ws_events(spent[r], 'p3'),
                 ws_events(dosed[r], 'p2'),
                 ws_events(dosed[r], 'p3'), sa, da))

    rate = {k: (v[0] / v[1] if v[1] else 0.0)
            for k, v in tot.items()}
    dosed_p2_events = tot[('dosed', 'p2')][0]
    sp2, sp3 = rate[('spent', 'p2')], rate[('spent', 'p3')]
    dp2, dp3 = rate[('dosed', 'p2')], rate[('dosed', 'p3')]
    print('pooled rates: spent p2=%.3f p3=%.3f | dosed p2=%.3f '
          'p3=%.3f | dosed p2 events=%d'
          % (sp2, sp3, dp2, dp3, dosed_p2_events))

    if dosed_p2_events < DOSED_FLOOR:
        verdict = ('VOID-BY-BAND (pooled dosed P2 world-sensitive '
                   'events %d < %d)' % (dosed_p2_events, DOSED_FLOOR))
    elif sp2 * 1.5 > dp2:
        verdict = ('NOT SUPPORTED (endogeny): spent P2 rate %.3f '
                   'within 1.5x of dosed %.3f' % (sp2, dp2))
    elif dp2 >= 3 * max(sp2, 1e-9) and parity_ok \
            and sp3 >= 3 * max(sp2, 1e-9) and sp3 > dp3:
        verdict = ('SUPPORTED: starvation (dosed %.3f >= 3x spent '
                   '%.3f), stock parity all pairs, migration-specific '
                   'restart (spent p3 %.3f > dosed p3 %.3f)'
                   % (dp2, sp2, sp3, dp3))
    elif dp2 >= 3 * max(sp2, 1e-9) and parity_ok:
        verdict = ('NOT SUPPORTED (irreversibility): starvation + '
                   'parity hold, restart fails (spent p3 %.3f, '
                   'needed >= 3x %.3f and > dosed %.3f)'
                   % (sp3, sp2, dp3))
    else:
        verdict = ('PARTIAL: starvation=%s parity=%s restart=%s'
                   % (dp2 >= 3 * max(sp2, 1e-9), parity_ok,
                      sp3 >= 3 * max(sp2, 1e-9) and sp3 > dp3))
    print('P86 v4 POOLED VERDICT (R=%d): %s' % (len(kept), verdict))


if __name__ == '__main__':
    main()
