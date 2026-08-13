# -*- coding: utf-8 -*-
"""
a3_production.py  --  Phase 2, round 2 of the A3 re-measurement

Long replica-exchange production runs at 573 K for both

  A3  electrostatics only  (lambda = 0.30 eV, J = 0)   -- the target of v3
  A1  full model           (lambda = 0.30 eV, J on)    -- re-run under the same
                                                          protocol so that the
                                                          two rows of Table 3
                                                          are directly comparable

Round 1 (a3_ladder_scan.py) selected T_hi = 2000 K for A3; the Step 2.2 addendum
selected T_hi = 2200 K for A1. Both use one sweep per exchange attempt and
p_ss = 0.5.

Uncertainties are block averages: the production trace is split into 20 blocks
and the standard error taken across block means, which is valid provided the
block length exceeds the correlation time. Replica round-trip count is reported
alongside, since it is the diagnostic that says whether that proviso holds.
"""

import json
import time

import numpy as np

from sphalerite_mc import (SphaleriteLattice, HamiltonianParams,
                           _pair_counts)
from samplers import ReplicaExchange
from calibrate_ladder import build_ladder

L, X, SEED = 10, 0.02, 828282
T_LO, SPC, N_BLOCKS = 573.0, 1, 20
TOTAL_SWEEPS = 400_000

lat = SphaleriteLattice(L)

par_a3 = HamiltonianParams(); par_a3.J = np.zeros((4, 4))
CASES = [("A3_electrostatic", par_a3, 2000.0),
         ("A1_full", HamiltonianParams(), 2200.0)]

# Phase 1 published annealing values at 573 K
P1 = {"A3_electrostatic": dict(alpha=-14.99, R=15.99, dq2=0.0065, E=+0.00262),
      "A1_full":          dict(alpha=-23.48, R=24.48, dq2=0.0120, E=-0.00528)}

out = {"config": dict(L=L, N=int(lat.N), x=X, T=T_LO, spc=SPC,
                      total_sweeps=TOTAL_SWEEPS, n_blocks=N_BLOCKS),
       "cases": {}}

for name, par, T_hi in CASES:
    print("=" * 78)
    print(f"{name}:  lambda={par.lam}, J_CuIn={par.J[1,2]:+.2f} eV, T_hi={T_hi:.0f} K")
    print("=" * 78)

    temps, _ = build_ladder(lat, par, T_LO, T_hi, X, X, M0=8, target=0.20,
                            max_M=48, max_rounds=6, n_cycles=120,
                            sweeps_per_cycle=4, seed=SEED, p_ss=0.5,
                            verbose=False)
    M = len(temps)
    n_cycles = TOTAL_SWEEPS // (M * SPC)
    print(f"  ladder M={M}, {temps[0]:.0f}-{temps[-1]:.0f} K; "
          f"{n_cycles:,} cycles")

    rex = ReplicaExchange(lat, par, temps, X, X, seed=SEED, p_ss=0.5)
    n_burn = n_cycles // 4
    rex.run(n_burn, sweeps_per_cycle=SPC, record_every=10**9)
    rex.swap_att[:] = 0; rex.swap_acc[:] = 0; rex.round_trips = 0

    t0 = time.time()
    tr = rex.run(n_cycles - n_burn, sweeps_per_cycle=SPC, record_every=5)
    rex.validate_all()
    wall = time.time() - t0

    # block averages over the cold slot
    def blocked(x):
        x = np.asarray(x)
        k = len(x) // N_BLOCKS
        bm = np.array([x[b * k:(b + 1) * k].mean() for b in range(N_BLOCKS)])
        return float(bm.mean()), float(bm.std(ddof=1) / np.sqrt(N_BLOCKS))

    E_m, E_e = blocked(tr["E_per_site"])
    a_m, a_e = blocked(tr["alpha"])
    q_m, q_e = blocked(tr["mean_dq2"])

    cold = rex.reps[0]
    R = float(cold.pair_enrichment(1, 2))
    cs = cold.cluster_sizes()
    C = _pair_counts(cold.spec, lat.nn1)
    n_in = int(np.count_nonzero(cold.spec == 2)); z = lat.nn1.shape[1]
    shell = (float(C[2, 1] / (n_in * z)), float(2 * C[2, 2] / (n_in * z)),
             float(C[2, 0] / (n_in * z)))
    rates = rex.swap_rates()

    p = P1[name]
    print(f"  production {wall:.0f} s; round trips = {rex.round_trips}; "
          f"swap median = {np.median(rates):.3f}")
    print(f"  {'observable':14s} {'replica exchange':>24s} {'Phase 1':>10s} {'shift':>9s}")
    print("  " + "-" * 60)
    for lbl, m, e, ref, fmt in (("H/N (eV)", E_m, E_e, p["E"], "{:+.5f}"),
                                ("alpha", a_m, a_e, p["alpha"], "{:+.2f}"),
                                ("<dQ^2>", q_m, q_e, p["dq2"], "{:.4f}")):
        val = (fmt + " +/- " + fmt.replace("+", "")).format(m, e)
        print(f"  {lbl:14s} {val:>24s} {fmt.format(ref):>10s} "
              f"{100*(m-ref)/abs(ref):>8.1f}%")
    print(f"  {'R_Cu-In':14s} {R:>24.2f} {p['R']:>10.2f} "
          f"{100*(R-p['R'])/p['R']:>8.1f}%")
    print(f"  clusters = {cs.size} (largest {cs[0]}); "
          f"In shell Cu/In/Zn = {shell[0]:.3f}/{shell[1]:.3f}/{shell[2]:.3f}")
    print(f"  charge imbalance removed = {100*(1-q_m/0.1585):.1f}% "
          f"(Phase 1 {100*(1-p['dq2']/0.1585):.1f}%)\n")

    out["cases"][name] = dict(T_hi=T_hi, M=M, cycles=n_cycles,
                              round_trips=int(rex.round_trips),
                              swap_median=float(np.median(rates)),
                              E_per_site=E_m, E_sem=E_e,
                              alpha=a_m, alpha_sem=a_e,
                              dq2=q_m, dq2_sem=q_e, R=R,
                              clusters=int(cs.size), max_cluster=int(cs[0]),
                              shell_Cu=shell[0], shell_In=shell[1], shell_Zn=shell[2],
                              wall_s=float(wall), phase1=p,
                              temps=[float(x) for x in temps])

with open("../data/a3_production.json", "w") as f:
    json.dump(out, f, indent=2)
print("Written to ../data/a3_production.json")
