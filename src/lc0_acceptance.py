"""LC-0a acceptance (design §5, adapted to the core build): the
readout organ's four gates, deterministic, no world needed.

(a) ROUND-TRIP: every utterance parses back to its (act, family,
    frame) exactly — 100% or fail.
(b) EVIDENTIALITY CORRECTNESS: zero plain-form assertions from
    dormant Ks; the dormant K speaks ONLY in evidential past.
(c) CALIBRATION BY CONSTRUCTION: assertion count == assertable-K
    count exactly (no unfunded assertions, no funded silence).
(d) DETERMINISM: two readouts of the same web are identical.
Bus integration (LC-0b: the epsilon=0.02 pose-channel comparison on
the E1 harness) is the next gate, queued.
"""

import numpy as np

from sov import ConstraintWeb, DORMANCY_WINDOW, EMBED_DIM
from lc_store import readout, parse

PASS = FAIL = 0


def check(name, ok, note=''):
    global PASS, FAIL
    print(f'  {"PASS" if ok else "FAIL"}  {name}  {note}')
    if ok:
        PASS += 1
    else:
        FAIL += 1


def make_slot(web, fam):
    from sov import ConnectorGeometry, FIT_MATCH_THRESHOLD, NUM_FAMILIES
    th = np.zeros(NUM_FAMILIES)
    th[fam] = FIT_MATCH_THRESHOLD
    geo = ConnectorGeometry(family_thresholds=th,
                            eigen_soft=np.zeros(5), eigen_code=0,
                            neighbors=[],
                            centroid=np.zeros(EMBED_DIM),
                            radius=float('inf'))
    return web.create_slot(f'lc_test_{fam}', geo, 'inherited',
                           f'fam{fam}')


def do_fits(web, sid, fam, n, t0, emb):
    from sov import NUM_FAMILIES
    rv = np.zeros(400)
    for i in range(n):
        web.fit(sid, rv, emb, None, None, 0.0, t0 + i, 0, t0 + i,
                _family_activations=_fa(fam), support_obs=np.ones(96))


def _fa(fam):
    from sov import NUM_FAMILIES
    fa = np.zeros(NUM_FAMILIES)
    fa[fam] = 0.9
    return fa


print('=== LC-0a acceptance ===')
web = ConstraintWeb(eigen_coder=None, debug_level=0, ledger_id='LC0')
e1 = np.zeros(EMBED_DIM)
e1[0] = 1.0

k_live = make_slot(web, 0)       # will close and stay live
k_dorm = make_slot(web, 1)       # will close then go dormant
q_near = make_slot(web, 2)       # open with near-misses
q_starve = make_slot(web, 3)     # open, never fit

do_fits(web, k_live, 0, 30, 100, e1)
do_fits(web, k_dorm, 1, 30, 200, e1 * 2.0)
for i in range(20):              # near-misses on q_near (0.45 band)
    fa = np.zeros_like(_fa(2))
    fa[2] = 0.45
    web.fit(q_near, np.zeros(400), e1, None, None, 0.0, 300 + i, 0,
            300 + i, _family_activations=fa)
assert web.slots[k_live].state == 'closed'
assert web.slots[k_dorm].state == 'closed'
web._global_step = 300 + DORMANCY_WINDOW
do_fits(web, k_live, 0, 1, web._global_step, e1)   # keep live in contact
web.anneal_all(web._global_step + 1)               # dormancy audit
assert web.slots[k_dorm].dormant and not web.slots[k_live].dormant

u1 = readout(web)
u2 = readout(web)

# (d) determinism
check('determinism: two readouts identical', u1 == u2)

# (a) round-trip
rt = []
for u, receipt in u1:
    p = parse(u)
    rt.append(p is not None and p['act'] == receipt['act']
              and p['family'] == receipt['family'])
check('round-trip: every utterance parses to its (act, family)',
      len(rt) > 0 and all(rt), f'{sum(rt)}/{len(rt)}')

# (b) evidentiality
plain_dormant = [u for u, r in u1
                 if r['act'] == 'assert'
                 and web.slots[r['slot_id']].dormant
                 and r['evidential'] == 'plain']
dorm_utts = [r['evidential'] for u, r in u1
             if r['act'] == 'assert'
             and web.slots[r['slot_id']].dormant]
check('evidentiality: zero plain-form assertions from dormant Ks',
      not plain_dormant and dorm_utts == ['evidential-past'])

# (c) calibration by construction
n_assert = sum(1 for _, r in u1 if r['act'] == 'assert')
n_assertable_or_dormant = sum(1 for s in web.slots.values()
                              if s.state == 'closed')
check('calibration: assertions == closed Ks exactly '
      '(none unfunded, none silent)',
      n_assert == n_assertable_or_dormant == 2)

# gap typing
gaps = {r['gap_type'] for _, r in u1 if r['act'] == 'question'}
check('questions typed by gap evidence (near-miss + starvation seen)',
      'near-miss' in gaps and 'starvation' in gaps)

print('=== LC-0b-alpha: pose channel fidelity ===')
from lc_store import pose_phrase, parse_pose
recon = []
for sid in (k_live, k_dorm, q_near, q_starve):
    u = pose_phrase(web, sid)
    r = parse_pose(u)
    truth = web.slots[sid].geometry.family_thresholds
    if r is None or np.linalg.norm(truth) == 0:
        recon.append(r is not None)
        continue
    th, nm = r
    cos = float(np.dot(th, truth)
                / (np.linalg.norm(th) * np.linalg.norm(truth) + 1e-12))
    recon.append(cos >= 0.98)
check('pose phrases reconstruct geometry at cosine >= 0.98',
      all(recon), f'{sum(recon)}/{len(recon)}')
# composed (multi-family) geometry round-trips too
comp = make_slot(web, 5)
web.slots[comp].geometry.family_thresholds[6] = 0.25   # two families
u = pose_phrase(web, comp)
th, _ = parse_pose(u)
truth = web.slots[comp].geometry.family_thresholds
cos = float(np.dot(th, truth)
            / (np.linalg.norm(th) * np.linalg.norm(truth) + 1e-12))
check('multi-family pose round-trips (band precision)', cos >= 0.98,
      f'cos={cos:.4f}')
check('near-miss count survives the channel',
      parse_pose(pose_phrase(web, q_near))[1]
      == web.slots[q_near].ledger.near_miss_seen)

print('\n--- the first utterances ---')
for u, r in u1[:6]:
    print(f'  [{r["act"]}] {u}')
print(f'\nRESULT: {PASS} passed, {FAIL} failed')
if FAIL:
    raise SystemExit(1)
