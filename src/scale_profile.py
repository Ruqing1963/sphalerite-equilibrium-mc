# -*- coding: utf-8 -*-
"""
scale_profile.py  --  Phase 2, Step 2.3

Scale-up of the sphalerite lattice model towards 50^3 cells (N = 5e5 cation
sites). Three questions must be answered before any production campaign is
committed:

  1. Does the implementation remain CORRECT at large N? The integer adjacency
     construction, the incremental energy bookkeeping and the order-parameter
     statistics were all verified at N = 256 to 6912. None of that guarantees
     correctness at N = 5e5, where index arithmetic could silently overflow and
     where the sharing Lemma is asserted only for a single reference site.

  2. How does THROUGHPUT scale? The kernel is O(1) per trial move, so naively
     the cost per sweep should be linear in N. It will not be: the move set
     draws the partner site uniformly from all N sites, so every trial move
     makes a random access into arrays that stop fitting in cache somewhere
     between N = 1e4 and 1e5. The measured deviation from linearity is what
     sets the real production cost.

  3. How does the LADDER LENGTH scale? Replica-exchange acceptance depends on
     the energy gap between adjacent replicas, which is extensive, while the
     acceptance exponent is not. The number of replicas M should therefore grow
     roughly as sqrt(N), and the total cost of a production run as N^1.5, not N.

The environment here has 1 core and 3 GB of RAM, so this script profiles and
extrapolates rather than attempting production.
"""

import gc
import json
import time

import numpy as np

from sphalerite_mc import (SphaleriteLattice, HamiltonianParams, SphaleriteMC,
                           _anion_charges, _total_energy, _pair_counts,
                           DELTA_Q, KB_EV)

SIZES = [10, 20, 30, 40, 50]
X = 0.02
T = 573.0
MOVES = 20_000_000          # fixed number of trial moves per throughput point

out = {"sizes": [], "env": {"cores": 1, "ram_gb": 3}}

print("=" * 78)
print("Step 2.3  scale profile")
print("=" * 78)
print(f"{'L':>3} {'N':>8} {'build':>7} {'adj MB':>8} {'rep MB':>8} "
      f"{'moves/s':>10} {'s/sweep':>9} {'verify':>8}")
print("-" * 78)

for L in SIZES:
    rec = {"L": L}
    t0 = time.time()
    lat = SphaleriteLattice(L)
    rec["build_s"] = time.time() - t0
    rec["N"] = int(lat.N)

    adj_mb = sum(a.nbytes for a in (lat.pos, lat.nn1, lat.nn2,
                                    lat.an_of_cat, lat.cat_of_an)) / 1e6
    lut_mb = lat._lut.nbytes / 1e6
    lat._lut = None                      # the lookup table is build-time only
    gc.collect()
    rec["adjacency_mb"] = adj_mb
    rec["lut_mb"] = lut_mb

    par = HamiltonianParams()
    mc = SphaleriteMC(lat, par, x_cu=X, x_in=X, seed=1234 + L)
    rep_mb = (mc.spec.nbytes + mc.Q.nbytes + mc.solutes.nbytes) / 1e6
    rec["replica_mb"] = rep_mb

    # ---- correctness at this N ------------------------------------
    ok = []
    # (a) sharing Lemma, on a random sample of sites rather than site 0 only
    rng = np.random.default_rng(7)
    lemma_ok = True
    for i in rng.choice(lat.N, size=25, replace=False):
        for j in lat.nn1[i]:
            if np.intersect1d(lat.an_of_cat[i], lat.an_of_cat[j]).size != 1:
                lemma_ok = False
        for j in lat.nn2[i]:
            if np.intersect1d(lat.an_of_cat[i], lat.an_of_cat[j]).size != 0:
                lemma_ok = False
    ok.append(("lemma", lemma_ok))
    # (b) adjacency reciprocity on the same sample
    recip = all(i in lat.nn1[j] for i in rng.choice(lat.N, 25, replace=False)
                for j in lat.nn1[i])
    ok.append(("reciprocity", recip))
    # (c) random-solution limit of the order parameters
    R0 = mc.pair_enrichment(1, 2)
    a0 = mc.warren_cowley(2, 1)
    q0 = mc.mean_dq2()
    ok.append(("R->1", abs(R0 - 1.0) < 0.3))
    ok.append(("identity", abs(R0 - (1 - a0)) < 1e-9))
    ok.append(("dq2=8x", abs(q0 - 8 * X) < 0.01))

    # ---- throughput ------------------------------------------------
    n_sweeps = max(1, MOVES // lat.N)
    t0 = time.time()
    mc.run(T, n_sweeps, seed_offset=1, validate=False)
    dt = time.time() - t0
    rec["moves_per_s"] = n_sweeps * lat.N / dt
    rec["s_per_sweep"] = dt / n_sweeps

    # (d) energy bookkeeping after a real run, at this N
    E_ref = _total_energy(mc.spec, lat.nn1, lat.nn2,
                          _anion_charges(mc.spec, lat.cat_of_an, DELTA_Q),
                          mc.P1, mc.P2, mc.SELF, par.lam)
    drift = abs(mc.E - E_ref) / max(1.0, abs(E_ref))
    ok.append(("energy", drift < 1e-9))
    rec["energy_drift_rel"] = float(drift)
    # (e) composition conservation
    ok.append(("composition", mc.stoichiometry_check()[:2] == mc.counts0[:2]))
    # (f) solute list integrity
    ok.append(("solute list",
               np.array_equal(np.sort(mc.solutes), np.where(mc.spec != 0)[0])))

    rec["checks"] = {k: bool(v) for k, v in ok}
    verdict = "PASS" if all(v for _, v in ok) else "FAIL"

    print(f"{L:3d} {lat.N:8d} {rec['build_s']:6.2f}s {adj_mb:7.1f} {rep_mb:7.1f} "
          f"{rec['moves_per_s']:10,.0f} {rec['s_per_sweep']:9.3f} {verdict:>8}")
    if verdict == "FAIL":
        for k, v in ok:
            if not v:
                print(f"      FAILED CHECK: {k}")

    out["sizes"].append(rec)
    del mc, lat
    gc.collect()

# ---------------- scaling analysis ----------------
print("\n" + "=" * 78)
Ns = np.array([r["N"] for r in out["sizes"]], dtype=float)
mps = np.array([r["moves_per_s"] for r in out["sizes"]], dtype=float)
print("Throughput degradation with box size (cache behaviour):")
base = mps[0]
for r, m in zip(out["sizes"], mps):
    print(f"  L={r['L']:3d}  N={r['N']:7d}  {m:10,.0f} moves/s  "
          f"({m/base*100:5.1f}% of the L=10 rate)")
# power-law fit of moves/s versus N
sl, ic = np.polyfit(np.log(Ns), np.log(mps), 1)
print(f"  fit: moves/s ~ N^{sl:+.3f}")
out["throughput_exponent"] = float(sl)

print("\nExtrapolated cost of one production sweep at 50^3:")
r50 = [r for r in out["sizes"] if r["L"] == 50]
if r50:
    s = r50[0]["s_per_sweep"]
    print(f"  {s:.3f} s per sweep, single core")
    for M in (20, 40):
        for nsw in (1e5, 1e6):
            hours = s * M * nsw / 3600
            print(f"  M={M:2d} replicas x {nsw:.0e} sweeps each: "
                  f"{hours:9,.0f} core-hours ({hours/24:7,.0f} core-days)")
    out["cost_50"] = {"s_per_sweep": s}

with open("../data/scale_profile.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWritten to ../data/scale_profile.json")
