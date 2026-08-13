# -*- coding: utf-8 -*-
"""
condensation_bound.py -- Phase 2, Step 2.4

A convergence-independent test of whether solute condensation is an equilibrium
property or a sampling artefact.

The composition scan showed complete condensation at 0.10 and 0.25 at.%, but
with zero replica round trips, so by the criterion set in Step 2.2 those runs do
not demonstrate convergence. Rather than spend more compute chasing round trips
at 573 K, this test moves the question to temperatures where equilibration is
easy and acceptance is high, and uses a bounding argument:

  if the solutes condense at a temperature where the chain demonstrably
  equilibrates, then at any lower temperature - where the ordered state is more
  strongly favoured relative to k_B T - they must condense as well.

Each state point is run twice, from a random (dispersed) start and from a
pre-condensed start in which all solutes are placed in one compact region. If
the two agree in cluster count AND in energy, and the acceptance ratio is high
enough that the chain is genuinely mobile, the result is an equilibrium one.
"""
import json, time
import numpy as np
from sphalerite_mc import SphaleriteLattice, HamiltonianParams
from samplers import MixedMoveMC

L, SEED = 20, 33113
SWEEPS = 4000
lat = SphaleriteLattice(L); lat._lut = None
par = HamiltonianParams()

def condensed_start(mc, lat):
    """Move every solute into a compact ball around the box centre."""
    n_cu = int(np.count_nonzero(mc.spec == 1))
    n_in = int(np.count_nonzero(mc.spec == 2))
    c = lat.pos.mean(axis=0)
    d = np.linalg.norm(lat.pos - c, axis=1)
    core = np.argsort(d)[: n_cu + n_in]
    mc.spec[:] = 0
    mc.spec[core[0::2][:n_cu]] = 1
    mc.spec[core[1::2][:n_in]] = 2
    # top up in case of parity shortfall
    for s, n in ((1, n_cu), (2, n_in)):
        have = int(np.count_nonzero(mc.spec == s))
        if have < n:
            free = [i for i in core if mc.spec[i] == 0][: n - have]
            mc.spec[free] = s
    mc._refresh()

print(f"L={L} (N={lat.N}), {SWEEPS} sweeps per point, two starts each\n")
print(f"{'x (at.%)':>9} {'T (K)':>7} {'start':>11} {'acc':>8} {'clusters':>9} "
      f"{'largest%':>9} {'E/N (eV)':>11}")
rows = []
for x in (0.0010, 0.0025):
    for T in (1000.0, 1400.0, 1800.0):
        out = {}
        for tag in ("random", "condensed"):
            mc = MixedMoveMC(lat, par, x_cu=x, x_in=x, seed=SEED, p_ss=0.5)
            if tag == "condensed":
                condensed_start(mc, lat)
            acc = mc.run(T, SWEEPS, seed_offset=7, validate=True)
            cs = mc.cluster_sizes()
            out[tag] = dict(acc=float(acc), n_clusters=int(cs.size),
                            largest=float(cs[0]/cs.sum()),
                            E=float(mc.E/lat.N))
            print(f"{x*100:9.2f} {T:7.0f} {tag:>11} {acc:8.4f} {cs.size:9d} "
                  f"{100*cs[0]/cs.sum():9.1f} {mc.E/lat.N:+11.6f}")
        dE = abs(out['random']['E'] - out['condensed']['E'])
        agree = dE < 2e-5 and out['random']['n_clusters'] == out['condensed']['n_clusters']
        print(f"{'':9} {'':7} {'agreement':>11} "
              f"dE/N = {dE:.2e} eV  -> {'CONVERGED' if agree else 'not converged'}\n")
        rows.append(dict(x=x, T=T, dE=dE, agree=bool(agree), **{k: v for k, v in out.items()}))

json.dump(rows, open("../data/condensation_bound.json","w"), indent=2)
print("Written to ../data/condensation_bound.json")
