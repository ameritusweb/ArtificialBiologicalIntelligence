"""Law-dose analyzer — same card as law_dose.py (fixed before launch).

Pools results/law_dose_<L>_r*.json. Uses dose_sweep_analyze.analyze_arm
(contact/reopen within 3 gens of each event, per standing K).
  SUPPORTED   — pooled reopens >= 1 among contacted events AND pooled
                orphan rate <= 0.5 (the corridor is enterable by depth).
  NOT SUPPORTED — 0 reopens across >= 4 contacted events (the K is
                deeper than the world's parameterization).
  UNTESTED    — < 4 contacted events pooled.
Secondary: court proposals per phase-2 gen (law species vs the
furniture threshold ~8).
"""

import glob
import json
import os

from dose_sweep_analyze import analyze_arm

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    events, p2_props, p2_gens = [], 0, 0
    per_dose = {}
    for path in sorted(glob.glob(os.path.join(
            HERE, 'results', 'law_dose_*_r*.json'))):
        d = json.load(open(path))
        evs = analyze_arm(d)
        events.extend(evs)
        p2_props += d['proposals_by_phase']['p2']
        p2_gens += sum(1 for g in d['gen_records'] if g['phase'] == 'p2')
        e = per_dose.setdefault(d['dose'], {'events': 0, 'reopens': 0,
                                            'orphans': 0, 'props': 0})
        e['events'] += len(evs)
        e['reopens'] += sum(1 for x in evs
                            if x['contact'] and x['reopened'])
        e['orphans'] += sum(1 for x in evs if not x['contact'])
        e['props'] += d['proposals_by_phase']['p2']

    contacted = [x for x in events if x['contact']]
    reopens = sum(1 for x in contacted if x['reopened'])
    orphan_rate = (round((len(events) - len(contacted)) / len(events), 3)
                   if events else None)
    if len(contacted) < 4:
        verdict = (f'UNTESTED ({len(contacted)} contacted events < 4)')
    elif reopens >= 1 and orphan_rate <= 0.5:
        verdict = ('SUPPORTED: law-mutations reopen rule-Ks while '
                   'keeping contact — the corridor is enterable by '
                   'depth, not amplitude')
    elif reopens == 0:
        verdict = ('NOT SUPPORTED (deeper-than-parameterization): the '
                   'K survives law-mutations in contact — closure is '
                   'effectively permanent in-lineage')
    else:
        verdict = 'PARTIAL (reopens present but orphaning > 0.5)'

    out = {'verdict': verdict,
           'pooled': {'standing_K_events': len(events),
                      'contacted': len(contacted),
                      'reopens': reopens,
                      'orphan_rate': orphan_rate,
                      'p2_proposals': p2_props,
                      'proposal_rate': (round(p2_props / p2_gens, 3)
                                        if p2_gens else None)},
           'per_dose': per_dose}
    path = os.path.join(HERE, 'results', 'law_dose_pooled.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f'saved {path}')


if __name__ == '__main__':
    main()
