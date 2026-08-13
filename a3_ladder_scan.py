# -*- coding: utf-8 -*-
"""
a3_ladder_scan.py  --  Phase 2, round 1 of the A3 re-measurement

Experiment A3 (electrostatics only, lambda = 0.30 eV, J = 0) was left
un-re-measured in version 2 of the Phase 1 paper. Converging it requires a
ladder tuned to A3 rather than to A1, because the two Hamiltonians order at
different temperatures: Phase 1 located the A3 crossover near 2000 K and the A1
crossover 400-600 K higher.

This round scans the ladder top at fixed compute and reports replica round
trips, which the Step 2.2 addendum established as the correct convergence
diagnostic. T_lo is held at 573 K: the previous attempt extended down to 373 K
and collapsed to a single round trip, so the cold end is deliberately not
pushed here.

Production settings inherited from Step 2.2: one sweep per exchange attempt,
p_ss = 0.5, ladder built adaptively.
"""

import json
import time

import numpy as np

from sphalerite_mc import SphaleriteLattice, HamiltonianParams
from samplers import ReplicaExchange
from calibrate_ladder import build_ladder

L, X, SEED = 10, 0.02, 515151
T_LO = 573.0
TOTAL_SWEEPS = 200_000          # per configuration, all replicas counted
SPC = 1

lat = SphaleriteLattice(L)
par = HamiltonianParams()
par.J = np.zeros((4, 4))        # A3: electrostatics only

print("A3 (electrostatics only, J = 0): ladder-top scan at fixed compute")
print(f"L={L} (N={lat.N}), T_lo={T_LO:.0f} K, {TOTAL_SWEEPS:,} sweeps per run\n")

runs = []
for T_hi in (2000.0, 2400.0, 2800.0):
    t0 = time.time()
    temps, _ = build_ladder(lat, par, T_LO, T_hi, X, X, M0=8, target=0.20,
                            max_M=48, max_rounds=6, n_cycles=120,
                            sweeps_per_cycle=4, seed=SEED, p_ss=0.5,
                            verbose=False)
    M = len(temps)
    t_cal = time.time() - t0

    n_cycles = TOTAL_SWEEPS // (M * SPC)
    t0 = time.time()
    rex = ReplicaExchange(lat, par, temps, X, X, seed=SEED, p_ss=0.5)
    rex.run(n_cycles // 4, sweeps_per_cycle=SPC, record_every=10**9)  # burn-in
    rex.swap_att[:] = 0; rex.swap_acc[:] = 0; rex.round_trips = 0
    tr = rex.run(n_cycles - n_cycles // 4, sweeps_per_cycle=SPC, record_every=5)
    rex.validate_all()
    wall = time.time() - t0

    half = slice(len(tr["E_per_site"]) // 2, None)
    E = float(np.mean(tr["E_per_site"][half]))
    a = float(np.mean(tr["alpha"][half]))
    q = float(np.mean(tr["mean_dq2"][half]))
    rates = rex.swap_rates()
    print(f"T_hi={T_hi:.0f} K: M={M:3d}  cycles={n_cycles:6d}  "
          f"round trips={rex.round_trips:4d}  swap med={np.median(rates):.3f}  "
          f"E/N={E:+.5f}  alpha={a:+.2f}  <dQ2>={q:.4f}  "
          f"({t_cal:.0f}+{wall:.0f}s)")
    runs.append(dict(T_hi=T_hi, M=M, cycles=n_cycles,
                     round_trips=int(rex.round_trips),
                     swap_median=float(np.median(rates)),
                     swap_min=float(rates.min()),
                     E_per_site=E, alpha=a, dq2=q,
                     temps=[float(x) for x in temps],
                     wall_s=wall + t_cal))

best = max(runs, key=lambda r: r["round_trips"])
print(f"\nBest: T_hi={best['T_hi']:.0f} K, M={best['M']}, "
      f"{best['round_trips']} round trips")
spread = max(r["E_per_site"] for r in runs) - min(r["E_per_site"] for r in runs)
print(f"Energy spread across the three ladders: {spread*1e3:.3f} meV per site")

with open("../data/a3_ladder_scan.json", "w") as f:
    json.dump(dict(L=L, N=int(lat.N), T_lo=T_LO, total_sweeps=TOTAL_SWEEPS,
                   spc=SPC, runs=runs, best_T_hi=best["T_hi"]), f, indent=2)
print("\nWritten to ../data/a3_ladder_scan.json")
