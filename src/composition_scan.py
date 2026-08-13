# -*- coding: utf-8 -*-
"""
composition_scan.py  --  Phase 2, Step 2.4

The question Phase 1 could not answer: does indium in sphalerite sit in true
solid solution, or does it condense into nanoscale inclusions?

Phase 1 failed on this because at N = 4000 and 4 at.% total solute the 160
solute atoms were exactly enough to build one domain. Once nucleation happened
the entire solute budget was consumed and the distinction between "dispersed
pairs", "several competing nuclei" and "a single inclusion" was lost.

This scan attacks the same question in a box eight times larger and at
compositions down to twenty times more dilute, which is where the solute budget
stops being the limiting factor. The observable that decides it is the cluster
size distribution, not the pair statistics.

Ladder length is set from the scaling law measured in this step:

    sigma_E ~ x^0.85 * N^0.51

so M is scaled from the calibrated reference (M = 23 at N = 4000, x = 2 at.%)
rather than from the sqrt(N) rule of Step 2.3, which was measured at fixed
composition and overestimates M badly in the dilute limit.

Results are checkpointed after every composition so that an interrupted run
loses at most one point.
"""

import json
import os
import time

import numpy as np

from sphalerite_mc import (SphaleriteLattice, HamiltonianParams, KB_EV,
                           _pair_counts)
from samplers import ReplicaExchange, geometric_ladder
from calibrate_ladder import build_ladder

# ---------------- configuration ----------------
L = 20
T_LO, T_HI = 573.0, 2200.0
SPC = 1                       # one sweep per exchange attempt (Step 2.2 setting)
SWEEPS_PER_REPLICA = 12000
SEED = 2024
COMPOSITIONS = [0.0010, 0.0025, 0.0050, 0.0200]      # x_Cu = x_In

# reference ladder: M = 23 at N = 4000, x = 2 at.%  (Step 2.2)
M_REF, N_REF, X_REF = 23, 4000, 0.02
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, os.pardir, "data")
os.makedirs(_DATA, exist_ok=True)
OUT = os.path.join(_DATA, "composition_scan.json")


def predicted_M(N, x):
    """M ~ sigma_E ~ x^0.85 N^0.51, normalised to the calibrated reference."""
    m = M_REF * (x / X_REF) ** 0.85 * (N / N_REF) ** 0.51
    return int(max(8, round(m)))


def analyse(mc, lat):
    cs = mc.cluster_sizes()
    C = _pair_counts(mc.spec, lat.nn1)
    n_in = int(np.count_nonzero(mc.spec == 2))
    z = lat.nn1.shape[1]
    shell = (float(C[2, 1] / (n_in * z)), float(2 * C[2, 2] / (n_in * z)),
             float(C[2, 0] / (n_in * z))) if n_in else (0., 0., 0.)
    n_sol = int(cs.sum())
    return dict(n_clusters=int(cs.size), max_cluster=int(cs[0]) if cs.size else 0,
                mean_cluster=float(cs.mean()) if cs.size else 0.0,
                monomer_fraction=float(np.count_nonzero(cs == 1) / cs.size) if cs.size else 0.0,
                largest_fraction=float(cs[0] / n_sol) if n_sol else 0.0,
                n_solute=n_sol,
                sizes_hist={str(int(s)): int(c) for s, c in
                            zip(*np.unique(cs, return_counts=True))},
                shell_Cu=shell[0], shell_In=shell[1], shell_Zn=shell[2])


CONFIG = dict(L=L, N=4 * L ** 3, T=T_LO, spc=SPC,
              sweeps_per_replica=SWEEPS_PER_REPLICA)

results = {}
if os.path.exists(OUT):
    old = json.load(open(OUT))
    if old.get("config") == CONFIG:
        results = old.get("results", {})
        print(f"resuming; {len(results)} composition(s) already done\n")
    else:
        bak = OUT.replace(".json", ".superseded.json")
        os.replace(OUT, bak)
        print("existing checkpoint was written with different settings "
              f"(e.g. a different sweep count); moved to\n  {bak}\n"
              "starting fresh.\n")

lat = SphaleriteLattice(L)
lat._lut = None
par = HamiltonianParams()
print(f"Composition scan: L={L} (N={lat.N}), T={T_LO:.0f} K, "
      f"{SWEEPS_PER_REPLICA:,} sweeps per replica")
_rate = 4.1e6 if L <= 20 else 2.0e6          # measured trial moves per second
_est = sum(predicted_M(lat.N, x) * 3 * SWEEPS_PER_REPLICA * lat.N / _rate
           for x in COMPOSITIONS if f"{x:.4f}" not in results)
print(f"Rough estimate for the remaining points: {_est/3600:.1f} h single-core. "
      f"Interrupting is safe; each composition is checkpointed.\n")

for x in COMPOSITIONS:
    key = f"{x:.4f}"
    if key in results:
        print(f"x = {x*100:.2f}%: already done, skipping")
        continue
    M0 = predicted_M(lat.N, x)
    n_sol = int(round(2 * x * lat.N))
    print(f"x_Cu = x_In = {x*100:.2f}%  ({n_sol} solutes)  predicted M = {M0}")

    t0 = time.time()
    temps, _ = build_ladder(lat, par, T_LO, T_HI, x, x, M0=M0, target=0.20,
                            max_M=max(M0 * 3, 40), max_rounds=4,
                            n_cycles=40, sweeps_per_cycle=2,
                            seed=SEED, p_ss=0.5, verbose=False)
    M = len(temps)
    t_cal = time.time() - t0

    n_cycles = SWEEPS_PER_REPLICA // SPC
    rex = ReplicaExchange(lat, par, temps, x, x, seed=SEED, p_ss=0.5)
    rex.run(n_cycles // 4, sweeps_per_cycle=SPC, record_every=10**9)
    rex.swap_att[:] = 0; rex.swap_acc[:] = 0; rex.round_trips = 0
    t1 = time.time()
    tr = rex.run(n_cycles - n_cycles // 4, sweeps_per_cycle=SPC, record_every=5)
    rex.validate_all()
    wall = time.time() - t1

    cold = rex.reps[0]
    half = slice(len(tr["E_per_site"]) // 2, None)
    rec = dict(x=x, M=M, M_predicted=M0, n_cycles=n_cycles,
               round_trips=int(rex.round_trips),
               swap_median=float(np.median(rex.swap_rates())),
               E_per_site=float(np.mean(tr["E_per_site"][half])),
               alpha=float(np.mean(tr["alpha"][half])),
               dq2=float(np.mean(tr["mean_dq2"][half])),
               R=float(cold.pair_enrichment(1, 2)),
               wall_s=float(wall + t_cal))
    rec.update(analyse(cold, lat))

    print(f"  M={M} (calibrated {t_cal:.0f}s), production {wall:.0f}s, "
          f"round trips={rec['round_trips']}, swap med={rec['swap_median']:.2f}")
    print(f"  alpha={rec['alpha']:+.2f}  R={rec['R']:.2f}  "
          f"<dQ2>={rec['dq2']:.4f}")
    print(f"  clusters={rec['n_clusters']}, largest={rec['max_cluster']} "
          f"({100*rec['largest_fraction']:.0f}% of all solute), "
          f"monomers={100*rec['monomer_fraction']:.0f}%\n")

    results[key] = rec
    json.dump(dict(config=CONFIG, results=results), open(OUT, "w"), indent=2)

# ---------------- summary ----------------
if len(results) > 1:
    print("=" * 74)
    print(f"{'x (at.%)':>9} {'N_sol':>7} {'M':>4} {'RT':>4} {'alpha':>8} {'R':>7} "
          f"{'clusters':>9} {'largest %':>10} {'monomer %':>10}")
    for k in sorted(results, key=float):
        r = results[k]
        print(f"{r['x']*100:9.2f} {r['n_solute']:7d} {r['M']:4d} "
              f"{r['round_trips']:4d} {r['alpha']:+8.2f} {r['R']:7.2f} "
              f"{r['n_clusters']:9d} {100*r['largest_fraction']:10.1f} "
              f"{100*r['monomer_fraction']:10.1f}")
print(f"\nWritten to {OUT}")
