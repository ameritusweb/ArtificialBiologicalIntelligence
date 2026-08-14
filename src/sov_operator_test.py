"""Operator-correctness battery for sov.py — every named operator, every
audit fix, and the conservation laws.

Not an organism experiment: this is a code-correctness instrument for the
real ConstraintWeb (no toy reimplementation). Synthetic receptor vectors are
crafted per FAMILY_GROUPS so each test controls exactly which boundary fires.
"""

import numpy as np

import sov
from sov import (ConstraintWeb, ConnectorGeometry, Receipt,
                 FIT_MATCH_THRESHOLD, NUM_FAMILIES, EMBED_DIM,
                 SUPPORT_RING, NEAR_MISS_STRIDE, REOPEN_WINDOW,
                 RENT_CREDIT_PER_FIT, CLOSURE_RADIUS)
from receptor_eigen_coder import FAMILY_GROUPS

PASS, FAIL = 0, 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def make_rv(fam_indices, val=0.9):
    """Receptor vector activating the given families above threshold."""
    rv = np.zeros(86, dtype=np.float64)
    for fam in fam_indices:
        for idx in FAMILY_GROUPS[fam][1]:
            rv[idx] = val
    return rv


def make_slot(web, fams, name=None, thr=FIT_MATCH_THRESHOLD):
    thresholds = np.zeros(NUM_FAMILIES, dtype=np.float64)
    for f in fams:
        thresholds[f] = thr
    geo = ConnectorGeometry(
        family_thresholds=thresholds,
        eigen_soft=np.zeros(5), eigen_code=0, neighbors=[],
        centroid=np.zeros(EMBED_DIM, dtype=np.float64),
        radius=float('inf'))
    return web.create_slot(name or f"slot{fams}", geo,
                           origin_operator='Compose')  # evictable by default


def do_fit(web, sid, fams, emb, t, support=None, val=0.9):
    web._global_step = t
    return web.fit(sid, make_rv(fams, val), emb, None, None, 0.0,
                   log_offset=t, episode=0, time_step=t,
                   support_obs=support)


class StubEncoder:
    """Deterministic linear encoder for rebase tests."""

    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(96, EMBED_DIM) * 0.1

    def embed_batch(self, obs_array):
        return np.asarray(obs_array, dtype=np.float64) @ self.W


def fresh_web():
    return ConstraintWeb(eigen_coder=None, debug_level=1)


e = np.zeros(EMBED_DIM)
e1 = np.zeros(EMBED_DIM); e1[0] = 1.0


# ---------------------------------------------------------------------------
print("== Fit: receipt, mass, rent credit, support ==")
web = fresh_web()
s = make_slot(web, [0])
r = do_fit(web, s, [0], e1.copy(), 1, support=np.ones(96))
slot = web.slots[s]
check("positive fit creates receipt", len(r) == 1 and r[0].kind == 'fit'
      and r[0].sign == 1)
check("fit carries log provenance", r[0].log_offset == 1
      and r[0].provenance == 'LIVED')
check("mass accrues", slot.ledger.mass > 0)
check("rent credited by fit (info-priced net: x0.25 at the "
      "uninformative prior)",
      abs(slot.ledger.rent_balance - RENT_CREDIT_PER_FIT * 0.25) < 1e-9)
check("support ring grows", len(slot.geometry.support) == 1)
check("embedding stamped with epoch", r[0].embed_epoch == 0
      and r[0].embedding is not None)

print("== Fit: near-miss boundary channel (strided) ==")
web = fresh_web()
s = make_slot(web, [0])
nms = []
for i in range(NEAR_MISS_STRIDE * 2):
    nms += do_fit(web, s, [0], e1, 10 + i, val=0.45)  # 0.45 in [0.4, 0.5)
check("near-miss receipts strided",
      len(nms) == 2 and all(x.kind == 'boundary' and x.sign == -1
                            for x in nms))
check("near-miss adds no mass", web.slots[s].ledger.mass == 0.0)
check("no-activation is silent",
      do_fit(web, s, [0], e1, 100, val=0.0) == [])

print("== Closure: outside-in via tight evidence ==")
web = fresh_web()
s = make_slot(web, [0])
other = make_slot(web, [1])
web._add_edge(s, other, 'constraint')
for i in range(25):
    do_fit(web, s, [0], e1.copy(), 200 + i, support=np.ones(96))
slot = web.slots[s]
check("tight evidence closes", slot.state == 'closed')
check("resolution booked with radius",
      slot.resolution is not None and 'radius' in slot.resolution)
check("contingent liability = connectivity", slot.posit_liability == 1.0)

print("== Closed K stays in Fit stream; 404 window triggers Reopen ==")
far = e1 * 50.0
for i in range(REOPEN_WINDOW):
    do_fit(web, s, [0], far.copy(), 300 + i)
slot = web.slots[s]
check("systematic 404s reopen the K", slot.state == 'open'
      and web._op_counts['reopen'] == 1)
check("reopen logged with retraction receipt",
      any(r.kind == 'retraction' and r.parent_receipt_ids
          for r in slot.ledger.receipts))
check("feasible set restored from lived receipts",
      slot.geometry.radius >= CLOSURE_RADIUS * 2)
check("fail window cleared", slot.ledger.fail_window == [])

print("== Closed K: confirming evidence does NOT reopen ==")
web = fresh_web()
s = make_slot(web, [0])
for i in range(25):
    do_fit(web, s, [0], e1.copy(), 400 + i, support=np.ones(96))
for i in range(REOPEN_WINDOW * 2):
    do_fit(web, s, [0], e1.copy(), 500 + i)
check("confirming K stays closed", web.slots[s].state == 'closed'
      and web._op_counts['reopen'] == 0)

print("== Constrain: Law 6 (narrows only) ==")
web = fresh_web()
a = make_slot(web, [0])
b = make_slot(web, [1])
do_fit(web, a, [0], e1.copy(), 600)
rb = do_fit(web, b, [1], e1 * 3, 601)
before = web.slots[a].geometry.radius
web.constrain(a, [rb[0].receipt_id])
check("constrain never widens", web.slots[a].geometry.radius <= before)
fake = Receipt(receipt_id=999999, kind='fit', source_operator='Fit',
               parent_receipt_ids=[], log_offset=-1, episode=-1,
               time_step=-1, slot_id=b, channel_indices=[], family_id=-1,
               magnitude=1.0, sign=1, provenance='IMAGINED',
               source_ledger='X')
web._receipts_by_id[999999] = fake
r_before = web.slots[a].geometry.radius
web.constrain(a, [999999])
check("IMAGINED constraint refused",
      web.slots[a].geometry.radius == r_before)
del web._receipts_by_id[999999]

print("== Compose: funded-parents guard + grounding ==")
web = fresh_web()
a = make_slot(web, [0])
b = make_slot(web, [1])
cid, _ = web.compose(a, b)
check("compose refuses unfunded parents", cid == -1)
do_fit(web, a, [0], e1.copy(), 700)
do_fit(web, b, [1], e1.copy(), 701)
cid, crs = web.compose(a, b)
check("funded compose creates slot", cid >= 0 and len(crs) == 1)
check("structural receipt grounds in Fit",
      crs[0].parent_receipt_ids and web._trace_grounding(crs[0]))
check("composition edges created",
      web._connectivity(cid) == 2)

print("== Rent: selection, not a TTL clock ==")
starved = make_slot(web, [2], name='starved')
do_fit(web, starved, [2], e1.copy(), 710)     # born funded, then starves
fed = cid                                     # composed slot, will be fed
t = 1000
for i in range(1200):
    t += 10
    if i % 5 == 0:
        # varied embeddings keep the slot open (still paying rent)
        do_fit(web, fed, [0, 1], e1 * float(i % 5 - 2), t, val=0.9)
    web.anneal_all(t)
check("starved non-inherited slot archaized",
      web.slots[starved].state == 'archaized')
check("earning slot survives same span", web.slots[fed].state != 'archaized')
inh_web = fresh_web()
inh_web.populate_from_families()
t = 0
for i in range(400):
    t += 10
    inh_web.anneal_all(t)
check("inherited trunk never rent-evicted",
      all(s.state == 'open' for s in inh_web.slots.values()))

print("== Differentiate: exact deterministic partition ==")


def build_bimodal():
    w = fresh_web()
    sid = make_slot(w, [0])
    for i in range(5):
        do_fit(w, sid, [0], e1 * 4.0, 800 + 2 * i)
        do_fit(w, sid, [0], e1 * -4.0, 801 + 2 * i)
    return w, sid


web, s = build_bimodal()
n_before = len(web.slots[s].ledger.receipts)
ca, cb, parts = web.differentiate(s)
check("split succeeds on bimodal evidence", ca >= 0 and cb >= 0)
parent_ids = sorted(r.receipt_id for r in web.slots[s].ledger.receipts)
child_parent_ids = sorted(p.parent_receipt_ids[0] for p in parts)
placed_parent_ids = sorted(
    r.parent_receipt_ids[0]
    for cid in (ca, cb)
    for r in web.slots[cid].ledger.receipts)
check("partition exact: parent-id multiset conserved",
      child_parent_ids == parent_ids
      and placed_parent_ids == parent_ids)
check("both pulls receive their receipts",
      len(web.slots[ca].ledger.receipts) == 5
      and len(web.slots[cb].ledger.receipts) == 5)
web2, s2 = build_bimodal()
ca2, cb2, parts2 = web2.differentiate(s2)
check("split is deterministic (no RNG)",
      [p.slot_id for p in parts]
      == [p.slot_id for p in parts2])
web3 = fresh_web()
s3 = make_slot(web3, [0])
do_fit(web3, s3, [0], e1, 900)
check("unfunded split refused",
      web3.differentiate(s3) == (-1, -1, []))

print("== Unify: pool; Exclude: cached and blocking ==")
web = fresh_web()
a = make_slot(web, [3])
b = make_slot(web, [3])
do_fit(web, a, [3], e1.copy(), 1000)
do_fit(web, b, [3], e1.copy(), 1001)
na = len(web.slots[a].ledger.receipts)
nb = len(web.slots[b].ledger.receipts)
mid, pooled = web.unify(a, b)
check("identical geometry unifies", mid >= 0)
check("pool is the union of histories", len(pooled) == na + nb)
check("parents archaized after unify",
      web.slots[a].state == 'archaized' and web.slots[b].state == 'archaized')
c = make_slot(web, [4])
d = make_slot(web, [5])
do_fit(web, c, [4], e1.copy(), 1010)
do_fit(web, d, [5], e1.copy(), 1011)
m2, _ = web.unify(c, d)
check("divergent geometry refuses and excludes",
      m2 == -1 and (min(c, d), max(c, d)) in web.exclusions)
m3, _ = web.unify(c, d)
check("exclusion cache blocks re-unification", m3 == -1)
check("exclusion funded by divergent receipts",
      any(r.kind == 'negative' and r.parent_receipt_ids
          for r in web.slots[c].ledger.receipts))

print("== Abstract: lien (claim, not copy) + funded guard ==")
web = fresh_web()
ga = np.zeros(NUM_FAMILIES); ga[2] = 0.5; ga[3] = 0.5
gb = np.zeros(NUM_FAMILIES); gb[3] = 0.5; gb[4] = 0.5
a = web.create_slot('A', ConnectorGeometry(ga, np.zeros(5), 0, [],
                                           np.zeros(EMBED_DIM), float('inf')),
                    origin_operator='Compose')
b = web.create_slot('B', ConnectorGeometry(gb, np.zeros(5), 0, [],
                                           np.zeros(EMBED_DIM), float('inf')),
                    origin_operator='Compose')
check("abstract refuses unfunded children", web.abstract(a, b)[0] == -1)
do_fit(web, a, [2, 3], e1.copy(), 1100)
do_fit(web, b, [3, 4], e1.copy(), 1101)
pid, liens = web.abstract(a, b)
check("abstract opens superordinate slot", pid >= 0 and len(liens) == 2)
check("lien holds shared sub-pattern only",
      web.slots[pid].geometry.family_thresholds[3] == 0.5
      and web.slots[pid].geometry.family_thresholds[2] == 0.0)
check("liens add no mass (claim, not copy)", web.slots[pid].ledger.mass == 0.0)
check("lattice links set",
      web.slots[a].geometry.parent_slot_id == pid
      and web.slots[pid].geometry.child_slot_ids == [a, b])

print("== Posit: imagination register only ==")
web = fresh_web()
a = make_slot(web, [0])
b = make_slot(web, [1])
do_fit(web, a, [0], e1.copy(), 1200)
do_fit(web, b, [1], e1.copy(), 1201)
web._add_edge(a, b, 'constraint')
ety_before = len(web.etymology)
rec_before = len(web._receipts_by_id)
ripple = web.posit(a, e1 * 2)
check("posit propagates over neighbors", b in ripple)
check("posit writes no etymology", len(web.etymology) == ety_before)
check("posit creates no receipts", len(web._receipts_by_id) == rec_before)
check("posit logged to imagination register",
      len(web.imagination_log) == 1
      and web.imagination_log[0]['provenance'] == 'IMAGINED')
tr = web.transpose(b, a, e1 * 2)
check("transpose = reverse-mode posit", isinstance(tr, dict))

print("== Pose / Attest: two-stage discount ==")
web = fresh_web()
a = make_slot(web, [0])
do_fit(web, a, [0], e1.copy(), 1300)
posed = web.pose(a)
check("pose serializes shape only",
      'family_thresholds' in posed and 'receipts' not in posed)
ext = Receipt(receipt_id=1, kind='fit', source_operator='Fit',
              parent_receipt_ids=[], log_offset=5, episode=0, time_step=5,
              slot_id=0, channel_indices=[0], family_id=0, magnitude=1.0,
              sign=1, provenance='LIVED', source_ledger='PEER_1')
t1 = web.attest(a, posed, [ext], discount=0.8)
check("attest imports at discount", len(t1) == 1
      and t1[0].provenance == 'ATTESTED' and t1[0].magnitude < 1.0)
check("attested receipt grounds across ledgers",
      web._trace_grounding(t1[0]))
for _ in range(10):
    web.bill_attest_outcome('PEER_1', survived=False)
t2 = web.attest(a, posed, [ext], discount=0.8)
check("reliability posterior lowers unreliable exporter",
      t2[0].magnitude < t1[0].magnitude * 0.6)
check("attest credits rent (funded evidence, info-priced)",
      web.slots[a].ledger.rent_balance > RENT_CREDIT_PER_FIT * 0.25)

print("== Quote: meta-slot funded by Fit statistics ==")
web = fresh_web()
a = make_slot(web, [0])
for i in range(5):
    do_fit(web, a, [0], e1.copy(), 1400 + i * 3)
mid, mrs = web.quote(a)
check("quote opens meta-slot", mid >= 0 and len(mrs) == 1)
parents = [web._find_receipt(p) for p in mrs[0].parent_receipt_ids]
check("meta-accrual grounds in Fit receipts only",
      all(p is not None and p.kind == 'fit' for p in parents))
check("meta receipt grounded", web._trace_grounding(mrs[0]))

print("== Anneal: certainty decays without receipts ==")
web = fresh_web()
a = make_slot(web, [0])
do_fit(web, a, [0], e1.copy(), 1500)
c0 = web.slots[a].ledger.certainty
web.anneal_all(1500 + 2000)
check("stale certainty decays", web.slots[a].ledger.certainty < c0)

print("== Rebase: geometry survives encoder epochs ==")
web = fresh_web()
a = make_slot(web, [0])
for i in range(25):
    do_fit(web, a, [0], e1.copy(), 1600 + i,
           support=np.ones(96) * (1.0 + 0.01 * i))
check("closed before rebase", web.slots[a].state == 'closed')
enc = StubEncoder(seed=7)
n = web.rebase(enc)
expected = enc.embed_batch(
    np.asarray(web.slots[a].geometry.support, dtype=np.float32)).mean(axis=0)
check("rebase recomputes centroid in new space",
      n >= 1 and np.allclose(web.slots[a].geometry.centroid, expected,
                             atol=1e-6))
check("resolution follows the rebase",
      np.allclose(web.slots[a].resolution['centroid'],
                  web.slots[a].geometry.centroid))
check("epoch advanced", web._embed_epoch == 1)
web4, s4 = build_bimodal()
web4.rebase(StubEncoder(seed=8))
check("stale-epoch embeddings refuse exact split",
      web4.differentiate(s4) == (-1, -1, []))

print("== Transfer (derived): Abstract then Constrain ==")
web = fresh_web()
ga = np.zeros(NUM_FAMILIES); ga[2] = 0.5; ga[3] = 0.5
gb = np.zeros(NUM_FAMILIES); gb[3] = 0.5; gb[4] = 0.5
a = web.create_slot('A', ConnectorGeometry(ga, np.zeros(5), 0, [],
                                           np.zeros(EMBED_DIM), float('inf')),
                    origin_operator='Compose')
b = web.create_slot('B', ConnectorGeometry(gb, np.zeros(5), 0, [],
                                           np.zeros(EMBED_DIM), float('inf')),
                    origin_operator='Compose')
do_fit(web, a, [2, 3], e1.copy(), 1700)
do_fit(web, b, [3, 4], e1.copy(), 1701)
tid, _ = web.transfer(a, [b])
check("transfer produces re-specialized slot", tid >= 0)

print("== Conservation laws over a full mixed history ==")
web = fresh_web()
web.populate_from_families()
t = 0
made = []
for i in range(60):
    t += 1
    fams = [i % 6]
    for sid in list(web.get_active_slots()):
        pass
    do_fit(web, i % 33, fams, e1 * (1 + (i % 3)), t, support=np.ones(96))
    if i % 20 == 19:
        web.anneal_all(t)
f1 = [r for r in web.slots[0].ledger.receipts if r.kind == 'fit']
f2 = [r for r in web.slots[1].ledger.receipts if r.kind == 'fit']
if f1 and f2:
    web.compose(0, 1)
web.unify(3, 4) if (web.slots[3].ledger.fit_count and
                    web.slots[4].ledger.fit_count) else None
violations = web.check_conservation_laws()
check("conservation laws PASS on mixed history", violations == [],
      str(violations))
stats = web.get_stats()
check("stats report new instruments",
      'embed_epoch' in stats and 'imagination_events' in stats)


# ---------------------------------------------------------------------------
print("== Review fixes: anneal schedule-independence ==")
web_a = fresh_web()
sa_ = make_slot(web_a, [0])
do_fit(web_a, sa_, [0], e1.copy(), 10)
web_b = fresh_web()
sb_ = make_slot(web_b, [0])
do_fit(web_b, sb_, [0], e1.copy(), 10)
web_a.anneal_all(210)                       # one anneal, dt=200
for t in (60, 110, 160, 210):               # four anneals, same span
    web_b.anneal_all(t)
check("anneal is schedule-independent",
      abs(web_a.slots[sa_].ledger.certainty
          - web_b.slots[sb_].ledger.certainty) < 1e-12,
      f"{web_a.slots[sa_].ledger.certainty} vs "
      f"{web_b.slots[sb_].ledger.certainty}")

print("== Review fixes: attest admissibility ==")
web = fresh_web()
a = make_slot(web, [0])
do_fit(web, a, [0], e1.copy(), 300)
posed = web.pose(a)
bad_imagined = Receipt(receipt_id=101, kind='fit', source_operator='Fit',
                       parent_receipt_ids=[], log_offset=5, episode=0,
                       time_step=5, slot_id=0, channel_indices=[0],
                       family_id=0, magnitude=1.0, sign=1,
                       provenance='IMAGINED', source_ledger='PEER_X')
bad_negative = Receipt(receipt_id=102, kind='fit', source_operator='Fit',
                       parent_receipt_ids=[], log_offset=5, episode=0,
                       time_step=5, slot_id=0, channel_indices=[0],
                       family_id=0, magnitude=1.0, sign=-1,
                       provenance='LIVED', source_ledger='PEER_X')
bad_kind = Receipt(receipt_id=103, kind='anneal', source_operator='Anneal',
                   parent_receipt_ids=[], log_offset=-1, episode=-1,
                   time_step=-1, slot_id=0, channel_indices=[],
                   family_id=-1, magnitude=1.0, sign=1,
                   provenance='LIVED', source_ledger='PEER_X')
check("attest rejects IMAGINED / negative / non-fit evidence",
      web.attest(a, posed, [bad_imagined, bad_negative, bad_kind],
                 discount=0.8) == [])
good = Receipt(receipt_id=104, kind='fit', source_operator='Fit',
               parent_receipt_ids=[], log_offset=5, episode=0, time_step=5,
               slot_id=0, channel_indices=[0], family_id=0, magnitude=1.0,
               sign=1, provenance='LIVED', source_ledger='PEER_X')
mixed = web.attest(a, posed, [bad_imagined, good, bad_negative],
                   discount=0.8)
check("attest filters to the admissible subset", len(mixed) == 1
      and mixed[0].parent_receipt_ids == [104])

print("== Review fixes: negative fits do not fund ==")
web = fresh_web()
tri = make_slot(web, [0, 1, 2])       # 3 active families
partner = make_slot(web, [3])
do_fit(web, partner, [3], e1.copy(), 400)
r = do_fit(web, tri, [0], e1.copy(), 401)  # score 1/3 -> mismatch receipt
check("partial match records a negative fit",
      len(r) == 1 and r[0].sign == -1)
check("negative fits leave fit_count at zero",
      web.slots[tri].ledger.fit_count == 0)
check("negative-only slot cannot fund Compose",
      web.compose(tri, partner)[0] == -1)
check("negative-only slot cannot fund Abstract",
      web.abstract(tri, partner)[0] == -1)

print("== Review fixes: unify preserves identical geometry ==")
web = fresh_web()
u1 = make_slot(web, [3])
u2 = make_slot(web, [3])
do_fit(web, u1, [3], e1.copy(), 500)
do_fit(web, u2, [3], e1.copy(), 501)
mid, _ = web.unify(u1, u2)
check("identity merge preserves thresholds exactly",
      mid >= 0 and web.slots[mid].geometry.family_thresholds[3] == 0.5)

print("== Review fixes: rent tracks topology ==")
web = fresh_web()
r1 = make_slot(web, [0])
r2 = make_slot(web, [1])
r3 = make_slot(web, [2])
base_rate = web.slots[r1].ledger.rent_rate
web._add_edge(r1, r2, 'constraint')
web._add_edge(r1, r3, 'constraint')
check("edge additions raise rent on both endpoints",
      abs(web.slots[r1].ledger.rent_rate - 2 * base_rate) < 1e-12
      and abs(web.slots[r2].ledger.rent_rate - base_rate) < 1e-12)

# ---------------------------------------------------------------------------
print("== Individuate: slot genesis from unassigned evidence ==")
from sov import (UNASSIGNED_MIN_ACTIVATION, INDIVIDUATE_MIN_CLUSTER)


def do_fit_all(web, fams, emb, t, val=0.9, support=None):
    web._global_step = t
    return web.fit_all(make_rv(fams, val), emb, None, None, 0.0,
                       log_offset=t, episode=0, time_step=t,
                       support_obs=support)


web = fresh_web()
s = make_slot(web, [0])
do_fit_all(web, [0], e1.copy(), 10)          # matches slot 0 -> not pooled
check("matched observations are not pooled", len(web.unassigned_pool) == 0)
do_fit_all(web, [5], e1.copy(), 11)          # no slot for family 5
check("unmatched observation enters the pool",
      len(web.unassigned_pool) == 1)
do_fit_all(web, [5], e1.copy(), 12, val=0.1)  # dead air, below floor
check("dead air is not pooled", len(web.unassigned_pool) == 1)

for i in range(INDIVIDUATE_MIN_CLUSTER):
    do_fit_all(web, [5], e1 * (1.0 + 0.01 * i), 20 + i,
               support=np.ones(96))
nid, opening = web.individuate()
check("coherent cluster carves a new slot", nid >= 0
      and len(opening) >= INDIVIDUATE_MIN_CLUSTER)
check("carved geometry targets the cluster's family",
      web.slots[nid].geometry.family_thresholds[5] > 0
      and web.slots[nid].geometry.family_thresholds[0] == 0)
check("opening receipts ground in lived log offsets",
      all(r.log_offset >= 0 and web._trace_grounding(r) for r in opening))
check("consumed entries leave the pool",
      len(web.unassigned_pool) <= 1)
check("genesis never closes", web.slots[nid].state == 'open')
r_new = do_fit_all(web, [5], e1.copy(), 100)
check("the new slot now fits what nothing could",
      any(sid == nid and any(x.sign > 0 for x in rs)
          for sid, rs in r_new))

web2 = fresh_web()
make_slot(web2, [0])
for i in range(4):
    do_fit_all(web2, [5 + i * 3], e1.copy(), 10 + i)  # incoherent profiles
check("incoherent / small pool refuses genesis",
      web2.individuate() == (-1, []))

wA = fresh_web(); make_slot(wA, [0])
wB = fresh_web(); make_slot(wB, [0])
for w in (wA, wB):
    for i in range(INDIVIDUATE_MIN_CLUSTER):
        do_fit_all(w, [7], e1 * 2.0, 30 + i, support=np.ones(96))
na_, oa_ = wA.individuate()
nb_, ob_ = wB.individuate()
check("individuate is deterministic",
      na_ == nb_ and [r.log_offset for r in oa_]
      == [r.log_offset for r in ob_])
check("conservation holds after genesis",
      web.check_conservation_laws() == [],
      str(web.check_conservation_laws()))

print("== Retract: licensed compensation (Law 6 exception) ==")
web = fresh_web()
a = make_slot(web, [0])
b = make_slot(web, [1])
do_fit(web, a, [0], e1.copy(), 200)
rb = do_fit(web, b, [1], e1 * 3, 201)
pre_constrain = web.slots[a].geometry.radius
web.constrain(a, [rb[0].receipt_id])
narrowed = web.slots[a].geometry.radius
neg = do_fit(web, a, [0], e1, 210, val=0.45)  # near-miss = failing evidence
for i in range(NEAR_MISS_STRIDE):
    neg = do_fit(web, a, [0], e1, 210 + i, val=0.45) or neg
failing = [r.receipt_id for r in web.slots[a].ledger.receipts
           if r.sign < 0][:1]
check("retract refuses without testimony",
      web.retract(a, [], target='constraint') == [])
rr = web.retract(a, failing, target='constraint')
check("licensed retract restores pre-constraint feasibility",
      len(rr) == 1 and web.slots[a].geometry.radius == pre_constrain
      and narrowed <= pre_constrain)
check("retraction receipt parented on the testimony",
      rr[0].parent_receipt_ids == failing)

# Attestation revocation
posed = web.pose(a)
ext = Receipt(receipt_id=201, kind='fit', source_operator='Fit',
              parent_receipt_ids=[], log_offset=5, episode=0, time_step=5,
              slot_id=0, channel_indices=[0], family_id=0, magnitude=1.0,
              sign=1, provenance='LIVED', source_ledger='PEER_2')
imported = web.attest(a, posed, [ext], discount=0.8)
rent_after_import = web.slots[a].ledger.rent_balance
rel_before = web.attest_reliability.get('PEER_2', 1.0)
rv = web.retract(a, failing, target='attestation', exporter_id='PEER_2')
check("attestation revoked: receipts marked, rent compensated",
      len(rv) == 1
      and imported[0].receipt_id in web.retracted_receipt_ids
      and web.slots[a].ledger.rent_balance < rent_after_import)
check("revocation bills the exporter's reliability down",
      web.attest_reliability.get('PEER_2', 1.0) < rel_before)
check("history preserved (revoked receipt still in ledger)",
      web._find_receipt(imported[0].receipt_id) is not None)
check("conservation holds after retracts",
      web.check_conservation_laws() == [])

# ---------------------------------------------------------------------------
print("== Suspend/Counterposit: rationally closed, hypothetically open ==")
web = fresh_web()
k = make_slot(web, [0])
o = make_slot(web, [1])
d = make_slot(web, [2])
m = make_slot(web, [3])
for i in range(25):                                  # close K on tight evidence
    do_fit(web, k, [0], e1.copy(), 600 + i, support=np.ones(96))
do_fit(web, o, [1], e1.copy(), 700)
do_fit(web, d, [2], e1.copy(), 701)
do_fit(web, m, [3], e1.copy(), 702)                  # m: own lived support
check("K is closed", web.slots[k].state == 'closed')
cid, _ = web.compose(k, o)                           # c: AND(K, O) — K-required
web._add_edge(k, d, 'constraint')                    # d: neighbor, independent
# p: PARTIAL dependence — abstract needs shared geometry, so give m and k
# overlapping thresholds via a fresh two-family pair
ga = np.zeros(NUM_FAMILIES); ga[4] = 0.5; ga[5] = 0.5
gb = np.zeros(NUM_FAMILIES); gb[5] = 0.5; gb[6] = 0.5
pa_ = web.create_slot('PA', ConnectorGeometry(ga, np.zeros(5), 0, [],
                                              np.zeros(EMBED_DIM),
                                              float('inf')),
                      origin_operator='Compose')
do_fit(web, pa_, [4, 5], e1.copy(), 710)
pb_ = web.create_slot('PB', ConnectorGeometry(gb, np.zeros(5), 0, [],
                                              np.zeros(EMBED_DIM),
                                              float('inf')),
                      origin_operator='Compose')
do_fit(web, pb_, [5, 6], e1.copy(), 711)
pid_, _ = web.abstract(pa_, pb_)                     # liens on BOTH children

check("posit still refuses the closed K", web.posit(k, e1 * 2) == {})
check("compose receipt carries AND justification",
      any(r.justification == 'AND'
          for r in web.slots[cid].ledger.receipts))

ety_before = len(web.etymology)
rec_before = len(web._receipts_by_id)
pkg = web.suspend(k)
check("suspend returns a hypothesis package",
      pkg['op'] == 'suspend' and pkg['provenance'] == 'IMAGINED')
check("anti-dogma (all): K-required conclusion is gone",
      cid in pkg['gone']
      and pkg['context_certainty'][cid] == 0.5)
check("anti-dogma (only): independent slots keep their testimony",
      d not in pkg['gone'] and m not in pkg['gone']
      and pkg['context_certainty'][d]
      == web.slots[d].ledger.certainty)
check("K stays rationally closed (no retraction paid)",
      web.slots[k].state == 'closed'
      and web._op_counts.get('reopen', 0) == 0)
check("suspend writes no ledger",
      len(web.etymology) == ety_before
      and len(web._receipts_by_id) == rec_before)

# Partial support: suspend PA — the abstract parent must survive on PB's
# liens at a discounted fraction, not be erased.
pkg_pa = web.suspend(pa_)
check("partial dependence survives discounted (K's contribution removed)",
      pid_ not in pkg_pa['gone']
      and pid_ in pkg_pa['discounted']
      and 0.0 < pkg_pa['discounted'][pid_] < 1.0)

cp = web.counterposit(k, e1 * -3.0)
check("counterposit = posit after suspend",
      cp['op'] == 'counterposit' and cid in cp['ripple']
      and d in cp['ripple'])
check("counterfactual certainties come from the context",
      cp['ripple'][cid]['current_certainty'] == 0.5
      and cp['ripple'][cid]['support_fraction'] == 0.0
      and cp['ripple'][d]['support_fraction'] == 1.0)
cp2 = web.counterposit(k, e1 * -3.0)
check("counterposit is deterministic", cp == cp2)

print("== The voucher: per-query counterfactual license ==")
v = pkg['voucher']
check("suspension ships its voucher",
      v is not None and v['receipts_examined'] > 0
      and v['surface_slots'] >= 2)
check("pure-lived surface fully vouched on provenance",
      v['lived_fraction'] == 1.0)
check("compose's AND formula counted as declared",
      v['declared_and'] >= 1)
check("unaudited web is flagged, honestly",
      v['last_audit_clean'] is None
      and 'no_clean_audit_on_record' in v['flags'])
web.check_conservation_laws()
v2 = web.suspend(k)['voucher']
check("clean audit cited after it exists",
      v2['last_audit_clean'] is True and v2['last_audit_step'] >= 0
      and 'no_clean_audit_on_record' not in v2['flags'])
web.quote(k)   # meta receipt: multi-parent, defaulted OR, on the surface
v3 = web.suspend(k)['voucher']
check("defaulted multi-parent OR surfaces as the known vouching hole",
      v3['defaulted_or_multiparent'] >= 1
      and 'or_defaulted_multiparent' in v3['flags'])
web.rebase(StubEncoder(seed=11))
v4 = web.suspend(k)['voucher']
check("stale embeddings on the surface are flagged after rebase",
      v4['embedding_currency'] < 1.0
      and 'stale_embeddings_on_surface' in v4['flags'])
check("counterposit inherits the voucher",
      cp['voucher'] == pkg['voucher'])

print("== Abduction-by-replay (rung 3 via the lived log) ==")
rec_before = len(web._receipts_by_id)   # re-baseline: quote(k) above accrued
window = ([{'receptor_values': make_rv([0], 0.9)}] * 5    # only K explains
          + [{'receptor_values': make_rv([2], 0.9)}] * 3  # d still explains
          + [{'receptor_values': make_rv([9], 0.9)}] * 2)  # nobody explains
rr = web.replay_through_context(k, window)
check("replay finds the discriminating observations",
      rr['discriminating'] == 5 and rr['explained_by_masked'] == 5)
check("independently explained history survives the context",
      rr['context_fit_counts'].get(d, 0) == 3)
check("context 404s counted",
      rr['unassigned_in_context'] == 7)  # K's 5 + the 2 novel
check("replay is read-only",
      len(web._receipts_by_id) == rec_before
      and web.imagination_log[-1]['op'] == 'replay_context')
check("conservation after counterfactuals",
      web.check_conservation_laws() == [])

# ---------------------------------------------------------------------------
print("== Social ledger: Pose/Attest bus ==")
import random as _random
from social_ledger import PoseAttestBus, build_masks


def build_pair_of_webs():
    """Two webs: org 0 struggles on family 0 (near misses), org 1 has
    strong lived receipts on family 0."""
    w0 = fresh_web(); w0.ledger_id = 'ORG_0'
    w1 = fresh_web(); w1.ledger_id = 'ORG_1'
    a0 = make_slot(w0, [0], name='fam0_org0')
    a1 = make_slot(w1, [0], name='fam0_org1')
    # org1: funded on fam 0 with tight embeddings
    for i in range(10):
        do_fit(w1, a1, [0], e1 * 1.0, 100 + i, support=np.ones(96))
    # org0: near misses on fam 0 (structured gap, no fits)
    for i in range(NEAR_MISS_STRIDE * 3):
        do_fit(w0, a0, [0], e1, 100 + i, val=0.45)
    # charge some rent so starvation term is live
    w0.anneal_all(400)
    return w0, w1, a0, a1


w0, w1, a0, a1 = build_pair_of_webs()
webs = {0: w0, 1: w1}
bus = PoseAttestBus(blind_families={0: set(), 1: set()}, q=3)
cert_before = w0.slots[a0].ledger.certainty
radius_before = w0.slots[a0].geometry.radius
np_state_before = np.random.get_state()
py_state_before = _random.getstate()
bus.episode_boundary(webs, generation=0, episode=0)
np_state_after = np.random.get_state()
imported = [r for r in w0.slots[a0].ledger.receipts if r.kind == 'transfer']
check("gap-selected pose triggers attest", len(imported) > 0)
check("imports carry ATTESTED provenance + exporter id",
      all(r.provenance == 'ATTESTED' and r.source_ledger == 'ORG_1'
          for r in imported))
check("bus draws no RNG (numpy + python states untouched)",
      np_state_before[0] == np_state_after[0]
      and np.array_equal(np_state_before[1], np_state_after[1])
      and np_state_before[2:] == np_state_after[2:]
      and py_state_before == _random.getstate())

print("== Bus: attest firewall ==")
slot0 = w0.slots[a0]
check("radius untouched by testimony",
      slot0.geometry.radius == radius_before)
check("certainty untouched by testimony (exact)",
      slot0.ledger.certainty == cert_before)
check("fit_count untouched by testimony", slot0.ledger.fit_count == 0)
check("centroid aimed by testimony",
      float(np.linalg.norm(slot0.geometry.centroid)) > 0)

print("== Bus: corroboration billing ==")
# org0 now LIVES a fit near the attested embedding -> corroboration
do_fit(w0, a0, [0], e1 * 1.0, 500, val=0.9, support=np.ones(96))
bus.episode_boundary(webs, generation=0, episode=1)
check("lived confirmation corroborates the import",
      bus.corroborated[(0, 1)] > 0)
check("reliability posterior billed upward",
      w0.attest_reliability.get('ORG_1', 0) >= 1.0 - 1e-9)

# Expiry path: fresh pair, no confirming fit, advance past the window
w0b, w1b, a0b, a1b = build_pair_of_webs()
webs_b = {0: w0b, 1: w1b}
bus_b = PoseAttestBus(blind_families={0: set(), 1: set()}, q=3)
bus_b.episode_boundary(webs_b, generation=0, episode=0)
w0b._global_step += 500  # past CORR_WINDOW without lived confirmation
bus_b.episode_boundary(webs_b, generation=0, episode=1)
check("unconfirmed imports expire and bill downward",
      bus_b.expired[(0, 1)] > 0
      and w0b.attest_reliability.get('ORG_1', 1.0) < 1.0)

print("== Bus: zero-budget identity (Q=0 is a no-op) ==")
w0c, w1c, a0c, a1c = build_pair_of_webs()
w0d, w1d, a0d, a1d = build_pair_of_webs()
bus_q0 = PoseAttestBus(blind_families={0: set(), 1: set()}, q=0)
bus_q0.episode_boundary({0: w0c, 1: w1c}, generation=0, episode=0)
check("Q=0 bus leaves webs bit-identical",
      len(w0c.etymology) == len(w0d.etymology)
      and w0c.slots[a0c].ledger.receipt_count
      == w0d.slots[a0d].ledger.receipt_count
      and float(np.linalg.norm(
          w0c.slots[a0c].geometry.centroid
          - w0d.slots[a0d].geometry.centroid)) == 0.0)

print("== Bus: determinism across identical webs ==")


def bus_outcome_snapshot(web, sid):
    """Full normalized record — IDs alone can match while content differs."""
    transfers = [(r.receipt_id, round(r.magnitude, 12), r.sign,
                  round(r.discount, 12), r.provenance, r.source_ledger,
                  r.family_id, tuple(r.channel_indices),
                  tuple(r.parent_receipt_ids))
                 for r in web.slots[sid].ledger.receipts
                 if r.kind == 'transfer']
    slot = web.slots[sid]
    return (transfers, slot.state, slot.ledger.receipt_count,
            round(slot.ledger.rent_balance, 12),
            tuple(np.round(slot.geometry.centroid, 12)),
            dict(web.attest_reliability))


w0e, w1e, a0e, a1e = build_pair_of_webs()
bus_e = PoseAttestBus(blind_families={0: set(), 1: set()}, q=3)
bus_e.episode_boundary({0: w0e, 1: w1e}, generation=0, episode=0)
w0f, w1f, a0f, a1f = build_pair_of_webs()
bus_f = PoseAttestBus(blind_families={0: set(), 1: set()}, q=3)
bus_f.episode_boundary({0: w0f, 1: w1f}, generation=0, episode=0)
check("identical webs -> identical bus outcome (full records)",
      bus_outcome_snapshot(w0e, a0e) == bus_outcome_snapshot(w0f, a0f))

print("== Bus: blind-slot labeling + masks ==")
masks, blind_fams = build_masks(6, blind_w=8, stride=4)
check("masks cover distinct family windows",
      blind_fams[0] != blind_fams[5]
      and len(blind_fams[0] & blind_fams[1]) > 0)
check("mask indices are receptor indices", all(
    m.max() < 86 for m in masks.values() if len(m)))
# organism blind to family 0: its fam0 slot is labeled blind
w0g, w1g, a0g, a1g = build_pair_of_webs()
bus_g = PoseAttestBus(blind_families={0: {0}, 1: set()}, q=3)
check("blind slot detection",
      a0g in bus_g._blind_slot_ids(w0g, 0))
bus_g.episode_boundary({0: w0g, 1: w1g}, generation=0, episode=0)
check("blind imports counted as coverage, not yield",
      bus_g.blind_imports.get(0, 0) > 0
      and bus_g.blind_slots_touched.get(0)
      and (0, 1) not in bus_g.corroborated)

print("== CoFitTracker (P75 add-on: resonance precursor) ==")
from social_ledger import CoFitTracker

_blind = {0: set(range(33)) - {3, 7}, 1: set(range(33)) - {3, 7}}


def _feed(tracker):
    for step in range(50):
        f0 = ([3] if step < 10 else []) + ([7] if 20 <= step < 30 else [])
        f1 = ([3] if step < 10 else []) + ([7] if step >= 40 else [])
        tracker.record(0, 0, step, f0)
        tracker.record(1, 0, step, f1)
    return tracker.end_generation()


tr_stats = _feed(CoFitTracker(2, 50, _blind))
check("matched pairs measured", tr_stats['n_matched_series'] == 2
      and tr_stats['n_shuffled_series'] == 2)
check("common-cause pattern: matched > shuffled",
      tr_stats['matched_mean'] > tr_stats['shuffled_mean'])
check("identical world-driven series correlate fully",
      tr_stats['matched_mean'] == round((1.0 + -0.25) / 2, 5))
check("tracker is deterministic",
      _feed(CoFitTracker(2, 50, _blind)) == tr_stats)

print("== F16 exports: pose floor / earned prior / fresh tightness ==")
# Pose floor: a starved blind slot must win a reserved pose slot.
wf0 = fresh_web(); wf0.ledger_id = 'ORG_0'
wf1 = fresh_web(); wf1.ledger_id = 'ORG_1'
s_a = make_slot(wf0, [0], name='struggleA')
s_b = make_slot(wf0, [1], name='struggleB')
s_c = make_slot(wf0, [2], name='struggleC')
hole = make_slot(wf0, [5], name='blind_hole')     # never fits, rent deficit
partner = make_slot(wf1, [5], name='knows_fam5')
do_fit(wf1, partner, [5], e1 * 1.0, 100, support=np.ones(96))
for i in range(NEAR_MISS_STRIDE * 2):             # struggles renew
    for sid, fam in ((s_a, 0), (s_b, 1), (s_c, 2)):
        do_fit(wf0, sid, [fam], e1, 100 + i, val=0.45)
wf0.anneal_all(600)                                # hole accrues rent deficit
webs_f = {0: wf0, 1: wf1}
bus_noq = PoseAttestBus(blind_families={0: {5}, 1: set()}, q=3,
                        curiosity_quota=0)
bus_noq.episode_boundary(webs_f, 0, 0)
posed_noq = {p['slot'] for p in bus_noq.pose_log if p['org'] == 0}
check("without the floor, the hole is never posed (E1 replication)",
      hole not in posed_noq)
wf0b = wf0  # same webs, fresh bus with the floor
bus_q = PoseAttestBus(blind_families={0: {5}, 1: set()}, q=3,
                      curiosity_quota=1)
bus_q.episode_boundary(webs_f, 0, 1)
posed_q = {p['slot'] for p in bus_q.pose_log if p['org'] == 0}
check("the pose floor reserves a slot for the hole",
      hole in posed_q and len(posed_q) <= 3)
check("floor pose reaches the partner and imports",
      bus_q.blind_imports.get(0, 0) > 0
      and hole in bus_q.blind_slots_touched.get(0, set()))

# Earned prior: calibrate_priors maps measured co-fit to pair discounts.
bus_p = PoseAttestBus(blind_families={0: set(), 1: set()}, q=3)
priors = bus_p.calibrate_priors({'0-1': 0.13}, shuffled_baseline=0.007)
check("earned prior set symmetrically from co-fit (ceiling-clipped)",
      bus_p.pair_priors[(0, 1)] == 0.9   # 0.13/0.137≈0.949 -> ceil 0.9
      and bus_p.pair_priors[(0, 1)] == bus_p.pair_priors[(1, 0)])
check("earned prior clipped to registered bounds",
      bus_p.calibrate_priors({'0-1': 0.0}, 0.007)[(0, 1)] == 0.1)

# Per-pair co-fit retention in the tracker.
tr_stats2 = _feed(CoFitTracker(2, 50, _blind))
check("tracker retains per-pair co-fit",
      '0-1' in tr_stats2.get('pair_cofit', {})
      and tr_stats2['pair_cofit']['0-1'] == round((1.0 + -0.25) / 2, 5))

# Fresh-fit tightness: pops the window mean, then resets.
wt = fresh_web()
st = make_slot(wt, [0])
do_fit(wt, st, [0], e1 * 1.0, 700)          # first fit: no dist yet
do_fit(wt, st, [0], e1 * 1.0, 701)          # tight
do_fit(wt, st, [0], e1 * 1.0, 702)
m1, n1 = wt.pop_fresh_tightness()
check("fresh tightness measures the window", n1 == 2 and m1 is not None)
m2, n2 = wt.pop_fresh_tightness()
check("fresh tightness resets on pop", n2 == 0 and m2 is None)
do_fit(wt, st, [0], e1 * 6.0, 703)          # loose fit in new window
m3, n3 = wt.pop_fresh_tightness()
check("fresh tightness is window-sensitive (unlike the frozen EMA)",
      n3 == 1 and m3 > (m1 or 0))

print("== Dormancy (F23): orphaned K loses assertion rights ==")
from sov import DORMANCY_WINDOW
wd = fresh_web()
sd = make_slot(wd, [0])
for i in range(25):
    do_fit(wd, sd, [0], e1.copy(), 800 + i, support=np.ones(96))
check("K closed and assertable while in contact",
      wd.slots[sd].state == 'closed' and not wd.slots[sd].dormant
      and wd.get_stats()['assertable'] == 1)
wd.anneal_all(825 + DORMANCY_WINDOW + 1)      # economy sweep, far future
check("orphaned K demoted to dormant by the audit",
      wd.slots[sd].dormant and wd.get_stats()['dormant'] == 1
      and wd.get_stats()['assertable'] == 0)
check("dormancy demotes, never reopens (history stays citable)",
      wd.slots[sd].state == 'closed'
      and any(ev.event_type == 'dormant' for ev in wd.etymology))
wd._global_step = 826 + DORMANCY_WINDOW
do_fit(wd, sd, [0], e1.copy(), 826 + DORMANCY_WINDOW)
check("contact reawakens the K (assertion = funding + recency)",
      not wd.slots[sd].dormant and wd.get_stats()['assertable'] == 1
      and any(ev.event_type == 'reawakened' for ev in wd.etymology))
check("dormancy conserves the books", wd.check_conservation_laws() == [])

print("== Information-priced rent (2026-08-12): selectivity pays ==")
from sov import INFO_RATE_ALPHA
wi = fresh_web()
s_on = make_slot(wi, [0])    # will fire every step (always-on)
s_sel = make_slot(wi, [1])   # will fire rarely (selective)
rv_on = np.zeros(400); rv_on[0:4] = 0.9          # family 0 hot
rv_both = rv_on.copy(); rv_both[4:8] = 0.9       # families 0+1 hot
for i in range(200):
    rv = rv_both if i % 20 == 0 else rv_on       # slot 1 fires 5%
    wi.fit_all(rv, e1.copy(), None, None, 0.0, 3000 + i, 0, 3000 + i)
fr_on = wi.slots[s_on].ledger.fire_rate
fr_sel = wi.slots[s_sel].ledger.fire_rate
check("fire rate tracks firing base rate",
      fr_on > 0.9 and fr_sel < 0.2, f"on={fr_on:.3f} sel={fr_sel:.3f}")
# net-per-fire economics: past breakeven (p* = 0.75) the always-on
# slot LOSES funding per fire — volume accelerates its starvation —
# while the selective slot earns. Survival tracks selectivity at any
# volume (the check-6 volume corollary, closed same-day).
bal_on = wi.slots[s_on].ledger.rent_balance
bal_sel = wi.slots[s_sel].ledger.rent_balance
fits_on = wi.slots[s_on].ledger.fit_count
fits_sel = wi.slots[s_sel].ledger.fit_count
check("always-on slot runs NET NEGATIVE (volume cannot save it)",
      fits_on > fits_sel * 5 and bal_on < 0,
      f"on: fits={fits_on} bal={bal_on:.4f}")
check("selective slot earns net positive",
      bal_sel > 0, f"sel: fits={fits_sel} bal={bal_sel:.4f}")
check("info pricing conserves the books (mass/Law 1 untouched)",
      wi.check_conservation_laws() == [])

print("== Expectation receipts (P76's transmission) ==")
from sov import EXPECT_ALPHA
we = fresh_web()
se = make_slot(we, [0])
for i in range(30):    # varied embeddings: slot stays OPEN (radius up)
    do_fit(we, se, [0], e1 * (1.0 + 0.5 * (i % 3)), 7000 + i,
           support=np.ones(96))
assert we.slots[se].state == 'open'
c0 = we.slots[se].geometry.centroid.copy()
target = e1 * 3.0
rec = we.expectation_receipt(se, target, err=0.05)
c1 = we.slots[se].geometry.centroid
moved = float(np.linalg.norm(c1 - c0))
check("confirmed expectation bills a LIVED transfer receipt",
      len(rec) == 1 and rec[0].kind == 'transfer'
      and rec[0].provenance == 'LIVED' and rec[0].sign == 1)
check("expectation moves geometry at EXPECT_ALPHA (beats EMA inertia)",
      abs(moved - EXPECT_ALPHA * float(np.linalg.norm(target - c0)))
      < 1e-9 and moved > 0)
check("expectation receipts conserve the books (mass/Law 1 exempt)",
      we.check_conservation_laws() == [])

print("== Occlude/Enumerate (the Omission Cycle, §5a) ==")
wo = fresh_web()
oa = make_slot(wo, [0])
ob = make_slot(wo, [1])
for i in range(120):
    do_fit(wo, oa, [0], e1.copy(), 5000 + i, support=np.ones(96))
    do_fit(wo, ob, [1], e1 * 2.0, 5200 + i, support=np.ones(96))
cid, _ = wo.compose(oa, ob)
receipts_before = sum(s.ledger.receipt_count for s in wo.slots.values())
occ = wo.occlude(oa)
check("occlude seals truth without touching the funded ledger (C4)",
      occ is not None and occ['sealed_truth']['dominant_family'] == 0
      and sum(s.ledger.receipt_count
              for s in wo.slots.values()) == receipts_before)
starved = make_slot(wo, [2])
check("Case-3 guard: occluding an unfunded region is refused",
      wo.occlude(starved) is None)
ranked, funding = wo.enumerate_gap(oa)
check("enumerate returns receipt-funded ranked candidates",
      len(ranked) > 0 and funding > 0)
check("enumeration is register-level (imagination events, state clean)",
      any(ev['op'] == 'occlude' for ev in wo.imagination_log)
      and any(ev['op'] == 'enumerate' for ev in wo.imagination_log)
      and wo.slots[oa].state in ('open', 'closed'))
check("omission cycle conserves the books",
      wo.check_conservation_laws() == [])

print("== Conservation after bus traffic ==")
check("importer web conservation PASS", w0.check_conservation_laws() == [],
      str(w0.check_conservation_laws()))
check("exporter web conservation PASS", w1.check_conservation_laws() == [])

print("== Threshold rebase (F41 impl. 10 — the autopsy's mechanism) ==")
wt = ConstraintWeb(eigen_coder=None, debug_level=0, ledger_id='RETH')
rt = make_slot(wt, [0], thr=0.5)
rng_t = np.random.RandomState(7)
# OLD world: family-0 activations ~ U(0, 1); threshold 0.5 = median.
for _ in range(sov.RETHRESH_MIN_SAMPLES + 50):
    rv = make_rv([0], val=float(rng_t.uniform(0.0, 1.0)))
    wt.observe_activations(rv)
snap = wt.snapshot_activation_dist()
check("scout path fills the ring without receipts",
      snap is not None
      and sum(s.ledger.receipt_count for s in wt.slots.values()) == 0)
# NEW world: activations ~ U(0, 2) — everything clears the old 0.5.
for _ in range(sov.RETHRESH_MIN_SAMPLES + 50):
    rv = make_rv([0], val=float(rng_t.uniform(0.0, 2.0)))
    wt.observe_activations(rv)
old_thr = float(wt.slots[rt].geometry.family_thresholds[0])
n_adj = wt.rethreshold(snap)
new_thr = float(wt.slots[rt].geometry.family_thresholds[0])
check("quantile preserved: median threshold maps toward new median",
      n_adj == 1 and 0.8 < new_thr < 1.2,
      f"old={old_thr} new={new_thr}")
check("rethreshold mints no receipts and conserves the books",
      sum(s.ledger.receipt_count for s in wt.slots.values()) == 0
      and wt.check_conservation_laws() == [])
# identical distribution -> no-op within tolerance (snapshot resets
# the ring, so refill it from the same distribution first)
snap2 = wt.snapshot_activation_dist()
for _ in range(sov.RETHRESH_MIN_SAMPLES + 50):
    rv = make_rv([0], val=float(rng_t.uniform(0.0, 2.0)))
    wt.observe_activations(rv)
thr_before = float(wt.slots[rt].geometry.family_thresholds[0])
wt.rethreshold(snap2)
check("identical distribution is a near-no-op",
      abs(float(wt.slots[rt].geometry.family_thresholds[0])
          - thr_before) < 0.15)
check("threshold floor holds (never reaches zero)",
      float(wt.slots[rt].geometry.family_thresholds[0])
      >= sov.RETHRESH_FLOOR)
wc = ConstraintWeb(eigen_coder=None, debug_level=0, ledger_id='RETHC')
kc = make_slot(wc, [1], thr=0.5)
for _ in range(sov.RETHRESH_MIN_SAMPLES + 10):
    wc.observe_activations(make_rv([1],
                                   val=float(rng_t.uniform(0.0, 1.0))))
snapc = wc.snapshot_activation_dist()   # resets the ring
wc.slots[kc].state = 'closed'
for _ in range(sov.RETHRESH_MIN_SAMPLES + 10):
    wc.observe_activations(make_rv([1],
                                   val=float(rng_t.uniform(0.0, 2.0))))
thr_k = float(wc.slots[kc].geometry.family_thresholds[1])
wc.rethreshold(snapc)
check("closed K's boundary untouched by default (commitment identity)",
      float(wc.slots[kc].geometry.family_thresholds[1]) == thr_k)
wc.rethreshold(snapc, include_closed=True)
check("closed K rethresholds only on explicit opt-in",
      float(wc.slots[kc].geometry.family_thresholds[1]) != thr_k)

print(f"\n{'=' * 50}\nRESULT: {PASS} passed, {FAIL} failed")
if FAIL:
    raise SystemExit(1)
