"""Split-knob analyzer — the acceptance verdict (same card as
split_knob.py, fixed before launch).

For each web's FIRST split: rate-window discriminator (thresholds
locked since v1/v3 cards):
  parent rate  = parent reopen delta / parent fit delta over the final
                 8 gens pre-split;
  child rate   = combined child (reopens, fits) deltas from split-time
                 snapshots to run end; floor: child fits >= 8000.
  R = child/parent: <= 0.5 CONFLATION; >= 0.8 NON-STATIONARITY (with
  reopen-timing clustering reported, unbilled); else INCONCLUSIVE.
Pooled: majority over floor-met webs; UNTESTED if none.
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RATE_WINDOW = 8
CHILD_FITS_FLOOR = 8000


def arm_verdict(d):
    if not d['splits']:
        return {'verdict': 'UNTESTED (no split fired)',
                'churn': d['total_churn_events']}
    s = d['splits'][0]
    reopens = [{int(k): v for k, v in snap.items()}
               for snap in d['per_gen_reopens']]
    fits = [{int(k): v for k, v in snap.items()}
            for snap in d['fit_snapshots']]
    sg = s['gen']
    sid = s['slot_id']
    w0 = max(0, sg - RATE_WINDOW)
    parent_reopens = reopens[sg].get(sid, 0) - reopens[w0].get(sid, 0)
    parent_fits = fits[sg].get(sid, 0) - fits[w0].get(sid, 0)
    child_fits = child_reopens = 0
    for cid_s, snap in s['children'].items():
        cid = int(cid_s)
        child_fits += fits[-1].get(cid, 0) - snap['fits_at_split']
        child_reopens += (reopens[-1].get(cid, 0)
                          - snap['reopens_at_split'])
    detail = {'parent': s['name'], 'split_gen': sg,
              'parent_reopens_w': parent_reopens,
              'parent_fits_w': parent_fits,
              'child_reopens': child_reopens,
              'child_fits': child_fits}
    if child_fits < CHILD_FITS_FLOOR:
        detail['verdict'] = 'UNTESTED (child exposure floor unmet)'
        return detail
    if parent_fits <= 0 or parent_reopens <= 0:
        detail['verdict'] = ('UNTESTED (no parent churn in the rate '
                             'window — the split targeted stale churn)')
        return detail
    parent_rate = parent_reopens / parent_fits
    child_rate = child_reopens / child_fits
    R = child_rate / parent_rate
    detail['R'] = round(R, 4)
    if R <= 0.5:
        detail['verdict'] = ('CONFLATION SUPPORTED: the split named a '
                             'real distinction')
    elif R >= 0.8:
        detail['verdict'] = ('NON-STATIONARITY: churn is a '
                             'world-regime clock (T153 measurement)')
    else:
        detail['verdict'] = 'INCONCLUSIVE (0.5 < R < 0.8)'
    return detail


def main():
    arms = []
    for path in sorted(glob.glob(os.path.join(
            HERE, 'results', 'split_knob_r*.json'))):
        d = json.load(open(path))
        v = arm_verdict(d)
        v['replicate'] = d['replicate']
        arms.append(v)
    billed = [a for a in arms if a.get('verdict', '').startswith(
        ('CONFLATION', 'NON-STATIONARITY', 'INCONCLUSIVE'))]
    counts = {}
    for a in billed:
        key = a['verdict'].split(':')[0].split(' (')[0]
        counts[key] = counts.get(key, 0) + 1
    if not billed:
        pooled = 'UNTESTED (no web produced a floor-met split verdict)'
    else:
        top = max(counts.items(), key=lambda kv: kv[1])
        pooled = (f'{top[0]} ({top[1]}/{len(billed)} floor-met webs; '
                  f'{len(arms)} arms)')
    out = {'pooled_verdict': pooled, 'counts': counts, 'arms': arms}
    path = os.path.join(HERE, 'results', 'split_knob_pooled.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f'saved {path}')


if __name__ == '__main__':
    main()
