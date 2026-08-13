# -*- coding: utf-8 -*-
"""
validate_vs_phase1.py  --  Phase 2, Step 2.2

Regression check: the new sampler must reproduce the Phase 1 equilibrium
observables at 573 K, or else explain the discrepancy.

The benchmark showed replica exchange reaching a state 3.04 eV below the lowest
state Phase 1 ever attained. This script runs it longer, collects the full set
of Phase 1 observables, and compares them item by item with the published
Table 3, so that we know exactly which published numbers survive better
sampling and which do not.

Phase 1 values at 573 K, full model (lambda = 0.30 eV, J on):
    E/N              = -0.00528 eV      (pre-annealed trajectory)
    alpha_In-Cu      = -23.48
    R_Cu-In          =  24.48
    <dQ^2>           =   0.0120
    clusters         =  1 (all 160 solutes)
    In first shell   =  0.490 Cu / 0.183 In / 0.327 Zn
"""

import json
import time

import numpy as np

from sphalerite_mc import (SphaleriteLattice, HamiltonianParams,
                           _pair_counts)
from samplers import ReplicaExchange, integrated_autocorr_time

L, X, T = 10, 0.02, 573.0
SEED = 20260806
SPC = 5
N_CYCLES = 5000            # 29 replicas x 5 sweeps x 5000 = 725,000 sweeps

P1 = dict(E_per_site=-0.00528, alpha=-23.48, R=24.48, dq2=0.0120,
          clusters=1, shell=(0.490, 0.183, 0.327))

lat = SphaleriteLattice(L)
par = HamiltonianParams()
TEMPS = np.asarray(json.load(open("../data/ladder_L10.json"))["temps_K"])

print(f"Replica exchange, M={len(TEMPS)}, {N_CYCLES} cycles x {SPC} sweeps")
print(f"Total {len(TEMPS)*SPC*N_CYCLES:,} sweeps; cold slot at {TEMPS[0]:.0f} K\n")

t0 = time.time()
rex = ReplicaExchange(lat, par, TEMPS, X, X, seed=SEED, p_ss=0.5, mixed=True)
tr = rex.run(N_CYCLES, sweeps_per_cycle=SPC, record_every=1)
rex.validate_all()
wall = time.time() - t0

half = slice(N_CYCLES // 2, None)
cold = rex.reps[0]

E_mean = float(np.mean(tr["E_per_site"][half]))
E_sem = float(np.std(tr["E_per_site"][half]) / np.sqrt(len(tr["E_per_site"][half])))
a_mean = float(np.mean(tr["alpha"][half]))
a_sem = float(np.std(tr["alpha"][half]) / np.sqrt(len(tr["alpha"][half])))
q_mean = float(np.mean(tr["mean_dq2"][half]))
q_sem = float(np.std(tr["mean_dq2"][half]) / np.sqrt(len(tr["mean_dq2"][half])))
tau, _ = integrated_autocorr_time(tr["E_per_site"][half])

# inflate the naive standard error by the autocorrelation factor
infl = np.sqrt(max(2.0 * tau, 1.0))
E_sem *= infl; a_sem *= infl; q_sem *= infl

C = _pair_counts(cold.spec, lat.nn1)
n_in = int(np.count_nonzero(cold.spec == 2))
z = lat.nn1.shape[1]
shell = (C[2, 1] / (n_in * z), 2 * C[2, 2] / (n_in * z), C[2, 0] / (n_in * z))
R = float(cold.pair_enrichment(1, 2))
cs = cold.cluster_sizes()

print(f"wall clock {wall:.0f} s, tau_int = {tau:.1f} cycles, "
      f"round trips = {rex.round_trips}")
print(f"swap acceptance: min {rex.swap_rates().min():.3f}, "
      f"median {np.median(rex.swap_rates()):.3f}\n")

rows = [
    ("E/N (eV)",        f"{E_mean:+.5f} +/- {E_sem:.5f}", f"{P1['E_per_site']:+.5f}"),
    ("alpha_In-Cu",     f"{a_mean:+.2f} +/- {a_sem:.2f}",  f"{P1['alpha']:+.2f}"),
    ("R_Cu-In",         f"{R:.2f}",                        f"{P1['R']:.2f}"),
    ("<dQ^2>",          f"{q_mean:.4f} +/- {q_sem:.4f}",   f"{P1['dq2']:.4f}"),
    ("clusters",        f"{cs.size} (max {cs[0]})",        f"{P1['clusters']}"),
    ("In shell Cu",     f"{shell[0]:.3f}",                 f"{P1['shell'][0]:.3f}"),
    ("In shell In",     f"{shell[1]:.3f}",                 f"{P1['shell'][1]:.3f}"),
    ("In shell Zn",     f"{shell[2]:.3f}",                 f"{P1['shell'][2]:.3f}"),
]
print(f"{'observable':16s} {'replica exchange':>26s} {'Phase 1':>12s}")
print("-" * 58)
for name, new, old in rows:
    print(f"{name:16s} {new:>26s} {old:>12s}")

print()
print(f"charge imbalance removed: {100*(1 - q_mean/0.1585):.1f}%  "
      f"(Phase 1: {100*(1 - P1['dq2']/0.1585):.1f}%)")
print(f"ideal roquesite In shell: 0.667 Cu / 0.333 In / 0 Zn")
print(f"Cu:In in solute shell = {shell[0]/shell[1]:.2f} "
      f"(Phase 1 {P1['shell'][0]/P1['shell'][1]:.2f}, ideal 2.00)")

out = dict(config=dict(L=L, N=int(lat.N), x=X, T=T, M=len(TEMPS),
                       cycles=N_CYCLES, sweeps_per_cycle=SPC,
                       total_sweeps=int(len(TEMPS)*SPC*N_CYCLES)),
           wall_s=wall, tau_cycles=float(tau),
           round_trips=int(rex.round_trips),
           swap_min=float(rex.swap_rates().min()),
           swap_median=float(np.median(rex.swap_rates())),
           replica_exchange=dict(E_per_site=E_mean, E_sem=E_sem,
                                 alpha=a_mean, alpha_sem=a_sem,
                                 R=R, dq2=q_mean, dq2_sem=q_sem,
                                 clusters=int(cs.size), max_cluster=int(cs[0]),
                                 shell_Cu=float(shell[0]), shell_In=float(shell[1]),
                                 shell_Zn=float(shell[2])),
           phase1=P1)
with open("../data/validate_vs_phase1.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
np.savez_compressed("../data/rex_trace_573K.npz", **tr)
print("\nWritten to ../data/validate_vs_phase1.json and rex_trace_573K.npz")
