"""Dose-sweep analyzer — the three curves and the band rule (same card
as dose_sweep.py, fixed before launch).

Pools results/dose_sweep_<dose>_r*.json per dose:
  reopen rate   — P(standing K reopens within 3 gens of a dose event),
                  orphaned events excluded from the denominator and
                  counted separately.
  orphan rate   — P(no contact within 3 gens | standing K at event).
  proposal rate — court proposals per phase-2 generation.
Band rule (F25 impl. 1): BAND EXISTS iff >= 1 dose has pooled
reopens >= 1 AND pooled p2-proposals >= 1 AND orphan rate <= 0.5.
NO BAND iff both single-sided responses exist but at disjoint doses.
UNTESTED per the arm card's rules.
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 3       # gens post-event for contact/reopen assessment


def analyze_arm(d):
    closed_by_gen = [set(map(int, g)) for g in d['per_gen_closed']]
    fits = [{int(k): v for k, v in s.items()} for s in d['fit_snapshots']]
    gens = len(closed_by_gen)
    events = []
    for ev in d['world_events']:
        sg = ev['gen']
        for K in ev['standing']:
            sid = K['sid']
            horizon = min(sg + WINDOW, gens - 1)
            contact = any(fits[gg].get(sid, 0) > fits[gg - 1].get(sid, 0)
                          for gg in range(sg + 1, horizon + 1))
            reopened = any(sid not in closed_by_gen[gg]
                           for gg in range(sg + 1, horizon + 1))
            events.append({'slot': K['name'], 'contact': contact,
                           'reopened': reopened})
    return events


def main():
    by_dose = {}
    for path in sorted(glob.glob(os.path.join(
            HERE, 'results', 'dose_sweep_*_r*.json'))):
        d = json.load(open(path))
        dose = d['dose']
        e = by_dose.setdefault(dose, {'events': [], 'p2_props': 0,
                                      'p2_gens': 0, 'arms': 0,
                                      'closures': 0, 'churn': 0})
        e['events'].extend(analyze_arm(d))
        e['p2_props'] += d['proposals_by_phase']['p2']
        e['p2_gens'] += sum(1 for g in d['gen_records']
                            if g['phase'] == 'p2')
        e['arms'] += 1
        e['closures'] += 1 if d['closure_gen_p1'] is not None else 0
        e['churn'] += d['total_churn_events']

    curves = {}
    for dose, e in sorted(by_dose.items(),
                          key=lambda kv: (kv[0] == 'RESEED',
                                          kv[0].rjust(3, '0'))):
        evs = e['events']
        contacted = [x for x in evs if x['contact']]
        orphan_events = [x for x in evs if not x['contact']]
        reopens = sum(1 for x in contacted if x['reopened'])
        curves[dose] = {
            'arms': e['arms'], 'closures_p1': e['closures'],
            'standing_K_events': len(evs),
            'orphan_rate': (round(len(orphan_events) / len(evs), 3)
                            if evs else None),
            'reopen_rate': (round(reopens / len(contacted), 3)
                            if contacted else None),
            'reopens': reopens,
            'p2_proposals': e['p2_props'],
            'proposal_rate': (round(e['p2_props'] / e['p2_gens'], 3)
                              if e['p2_gens'] else None),
            'churn': e['churn']}

    any_standing = any(c['standing_K_events'] > 0 for c in curves.values())
    all_silent = all(c['p2_proposals'] == 0 for c in curves.values())
    band = [d for d, c in curves.items()
            if c['reopens'] >= 1 and c['p2_proposals'] >= 1
            and c['orphan_rate'] is not None and c['orphan_rate'] <= 0.5]
    web_doses = [d for d, c in curves.items() if c['reopens'] >= 1]
    court_doses = [d for d, c in curves.items() if c['p2_proposals'] >= 1]
    if not any_standing:
        verdict = 'UNTESTED (no standing K at any dose event)'
    elif all_silent:
        verdict = ('UNTESTED-COURT (zero proposals at every dose incl. '
                   'reseed — the court is silent in this harness at all '
                   'doses; a court-side instrument finding, billable)')
    elif band:
        verdict = f'BAND EXISTS at dose(s) {band} — the treadmill has a gear'
    elif web_doses and court_doses:
        verdict = (f'NO BAND: web responds at {web_doses}, court at '
                   f'{court_doses}, disjoint — different world-change '
                   f'species needed')
    else:
        verdict = ('PARTIAL-RESPONSE: only one membrane responded '
                   f'(web: {web_doses}, court: {court_doses})')

    out = {'curves': curves, 'verdict': verdict}
    path = os.path.join(HERE, 'results', 'dose_sweep_pooled.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f'saved {path}')


if __name__ == '__main__':
    main()
