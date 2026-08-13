# -*- coding: utf-8 -*-
"""
benchmark_samplers.py  --  Phase 2, Step 2.2

Head-to-head comparison of three samplers at 573 K (300 C), the centre of the
ore-forming window, at the Phase 1 box size so that results are directly
comparable with the published values.

  A  plain Kawasaki                 (Phase 1 baseline)
  B  mixed move set, p_ss = 0.5     (boosted solute-solute exchange)
  C  replica exchange               (calibrated ladder, 29 replicas)
  D  replica exchange + mixed moves

All four start from the same quenched (random) configuration and are given the
same total sweep budget, counting every replica's sweeps against the budget, so
the comparison is on equal computational footing rather than equal wall clock
per replica.

Reference values from Phase 1 at 573 K:
  quenched trajectory   E/N = -0.00211 eV,  alpha = -23.17
  pre-annealed          E/N = -0.00528 eV,  alpha = -23.48
The pre-annealed state is the lowest previously reached. A sampler that gets
below it from a quenched start, and stays there, has solved the equilibration
problem that Phase 1 left open.
"""

import json
import time

import numpy as np

from sphalerite_mc import SphaleriteLattice, HamiltonianParams, SphaleriteMC
from samplers import (MixedMoveMC, ReplicaExchange,
                      integrated_autocorr_time)

# ---------------- configuration ----------------
L = 10
X = 0.02
T_TARGET = 573.0
TOTAL_SWEEPS = 150_000          # per method, counting all replicas
SWEEPS_PER_BLOCK = 100          # recording granularity for A and B
SEED = 4242

PHASE1_QUENCHED = -0.00211
PHASE1_ANNEALED = -0.00528
PHASE1_ALPHA = -23.48

lat = SphaleriteLattice(L)
par = HamiltonianParams()
with open("../data/ladder_L10.json") as f:
    TEMPS = np.asarray(json.load(f)["temps_K"])

print(f"Box L={L} (N={lat.N}), x_Cu = x_In = {X}, target T = {T_TARGET:.0f} K")
print(f"Budget: {TOTAL_SWEEPS:,} sweeps per method (all replicas counted)")
print(f"Replica ladder: M = {len(TEMPS)}, {TEMPS[0]:.0f}-{TEMPS[-1]:.0f} K\n")

results = {}


def summarise(name, tr_E, tr_alpha, wall, n_sweeps_total, extra=""):
    """Report equilibration quality and sampling efficiency."""
    n = len(tr_E)
    tail = slice(n // 2, None)                       # second half = production
    E_mean = float(np.mean(tr_E[tail]))
    E_final = float(tr_E[-1])
    a_mean = float(np.mean(tr_alpha[tail]))
    tau, W = integrated_autocorr_time(tr_E[tail])
    block = n_sweeps_total / n                       # sweeps per recorded point
    tau_sweeps = tau * block
    print(f"{name}")
    print(f"  E/N  mean(2nd half) = {E_mean:+.5f} eV   final = {E_final:+.5f} eV")
    print(f"  alpha mean          = {a_mean:+.2f}")
    print(f"  tau_int(E)          = {tau:.1f} blocks = {tau_sweeps:,.0f} sweeps")
    print(f"  wall clock          = {wall:.1f} s   ({n_sweeps_total/wall:,.0f} sweeps/s)")
    print(f"  cost per independent sample = {tau_sweeps * wall / n_sweeps_total:,.1f} s")
    if extra:
        print(f"  {extra}")
    print()
    results[name] = dict(E_mean=E_mean, E_final=E_final, alpha_mean=a_mean,
                         tau_blocks=float(tau), tau_sweeps=float(tau_sweeps),
                         wall_s=float(wall), n_sweeps=int(n_sweeps_total),
                         note=extra)


# ============================================================
# A. plain Kawasaki
# ============================================================
t0 = time.time()
mc = SphaleriteMC(lat, par, x_cu=X, x_in=X, seed=SEED)
nb = TOTAL_SWEEPS // SWEEPS_PER_BLOCK
trE, trA = [], []
acc_last = 0.0
for b in range(nb):
    acc_last = mc.run(T_TARGET, SWEEPS_PER_BLOCK, seed_offset=b, validate=False)
    trE.append(mc.E / lat.N)
    trA.append(mc.warren_cowley(2, 1))
mc.validate_state()
summarise("A  plain Kawasaki", np.array(trE), np.array(trA),
          time.time() - t0, TOTAL_SWEEPS,
          extra=f"acceptance = {acc_last:.2e}")

# ============================================================
# B. mixed move set
# ============================================================
t0 = time.time()
mcB = MixedMoveMC(lat, par, x_cu=X, x_in=X, seed=SEED, p_ss=0.5)
trE, trA = [], []
for b in range(nb):
    acc_last = mcB.run(T_TARGET, SWEEPS_PER_BLOCK, seed_offset=b, validate=False)
    trE.append(mcB.E / lat.N)
    trA.append(mcB.warren_cowley(2, 1))
mcB.validate_state()
summarise("B  mixed move set (p_ss = 0.5)", np.array(trE), np.array(trA),
          time.time() - t0, TOTAL_SWEEPS,
          extra=f"acceptance = {acc_last:.2e}, solute-solute acceptance = {mcB.last_acc_ss:.2e}")

# ============================================================
# C, D. replica exchange
# ============================================================
for tag, p_ss in (("C  replica exchange", 0.0),
                  ("D  replica exchange + mixed moves", 0.5)):
    t0 = time.time()
    M = len(TEMPS)
    spc = 5
    n_cycles = TOTAL_SWEEPS // (M * spc)
    rex = ReplicaExchange(lat, par, TEMPS, X, X, seed=SEED, p_ss=p_ss,
                          mixed=(p_ss > 0))
    tr = rex.run(n_cycles, sweeps_per_cycle=spc, record_every=1)
    rex.validate_all()
    rates = rex.swap_rates()
    summarise(tag, tr["E_per_site"], tr["alpha"], time.time() - t0,
              n_cycles * M * spc,
              extra=(f"swap acc min={rates.min():.3f} median={np.median(rates):.3f}; "
                     f"round trips = {rex.round_trips}; cycles = {n_cycles}"))

# ============================================================
# verdict
# ============================================================
print("=" * 72)
print("Reference (Phase 1, 573 K):")
print(f"  quenched start    E/N = {PHASE1_QUENCHED:+.5f} eV")
print(f"  pre-annealed      E/N = {PHASE1_ANNEALED:+.5f} eV,  alpha = {PHASE1_ALPHA:+.2f}")
print()
best = min(results.items(), key=lambda kv: kv[1]["E_mean"])
print(f"Lowest mean energy: {best[0]}  ({best[1]['E_mean']:+.5f} eV)")
gain = PHASE1_ANNEALED - best[1]["E_mean"]
print(f"  versus the Phase 1 pre-annealed state: {gain:+.5f} eV per site "
      f"({gain * lat.N:+.2f} eV over the box)")
if gain > 1e-5:
    print("  -> reaches a lower state than Phase 1 ever did, from a quenched start")
elif abs(gain) <= 1e-5:
    print("  -> matches the Phase 1 annealed state; the plateau is confirmed as equilibrium")
else:
    print("  -> does NOT reach the Phase 1 annealed state; equilibration still unsolved")

with open("../data/benchmark_samplers.json", "w") as f:
    json.dump({"config": dict(L=L, N=int(lat.N), x=X, T=T_TARGET,
                              total_sweeps=TOTAL_SWEEPS, M=len(TEMPS)),
               "phase1_reference": dict(quenched=PHASE1_QUENCHED,
                                        annealed=PHASE1_ANNEALED,
                                        alpha=PHASE1_ALPHA),
               "results": results}, f, indent=2)
print("\nWritten to ../data/benchmark_samplers.json")
