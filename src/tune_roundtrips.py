# -*- coding: utf-8 -*-
"""
tune_roundtrips.py  --  Phase 2, Step 2.2 addendum

Step 2.2 left one defect: only 5 replica round trips in 5000 cycles, which means
the cold slot holds essentially one configuration and its error bars are not
meaningful. Round-trip count, not swap acceptance, is the diagnostic that
matters for replica exchange, and it must be fixed before scaling up.

Two cheap knobs are tested at fixed total compute:

  (1) sweeps per exchange attempt. Fewer sweeps between attempts means more
      attempts per unit compute, hence faster diffusion along the ladder, at the
      cost of each replica decorrelating less between attempts.

  (2) top of the ladder. The Phase 1 crossover for the full model lies at
      2400-3000 K, so T_hi = 3000 K may sit further above it than necessary,
      spending replicas where they buy nothing.

Compute is held fixed by scaling the cycle count inversely with sweeps per
cycle, so every configuration below performs the same number of single-site
trial moves.
"""

import json
import time

import numpy as np

from sphalerite_mc import SphaleriteLattice, HamiltonianParams, KB_EV
from samplers import ReplicaExchange
from calibrate_ladder import build_ladder

L, X, SEED = 10, 0.02, 31337
TOTAL_SWEEPS = 145_000          # per configuration, all replicas counted

lat = SphaleriteLattice(L)
par = HamiltonianParams()

full_ladder = np.asarray(json.load(open("../data/ladder_L10.json"))["temps_K"])

print(f"Round-trip tuning at fixed compute ({TOTAL_SWEEPS:,} sweeps per run)\n")

runs = []

# ---- (1) exchange frequency, on the existing 29-replica ladder ----
print("A. sweeps per exchange attempt (29-replica ladder, 573-3000 K)")
for spc in (1, 2, 5):
    M = len(full_ladder)
    n_cycles = TOTAL_SWEEPS // (M * spc)
    t0 = time.time()
    rex = ReplicaExchange(lat, par, full_ladder, X, X, seed=SEED, p_ss=0.5)
    tr = rex.run(n_cycles, sweeps_per_cycle=spc, record_every=10)
    wall = time.time() - t0
    rt = rex.round_trips
    E = float(np.mean(tr["E_per_site"][len(tr["E_per_site"]) // 2:]))
    print(f"  spc={spc}: cycles={n_cycles:6d}  round trips={rt:4d}  "
          f"swap median={np.median(rex.swap_rates()):.3f}  "
          f"E/N={E:+.5f}  wall={wall:.0f}s")
    runs.append(dict(kind="spc", spc=spc, M=M, T_hi=float(full_ladder[-1]),
                     cycles=n_cycles, round_trips=int(rt), E_per_site=E,
                     swap_median=float(np.median(rex.swap_rates())), wall_s=wall))

# ---- (2) lower the top of the ladder, rebuilt from scratch ----
print("\nB. ladder top temperature (rebuilt ladder, spc = best from A)")
best_spc = min(runs, key=lambda r: -r["round_trips"])["spc"]
print(f"   using spc = {best_spc}")
for T_hi in (2200.0, 2600.0):
    temps, _ = build_ladder(lat, par, 573.0, T_hi, X, X,
                            M0=8, target=0.20, max_M=48, max_rounds=6,
                            n_cycles=100, sweeps_per_cycle=4,
                            seed=SEED, verbose=False)
    M = len(temps)
    n_cycles = TOTAL_SWEEPS // (M * best_spc)
    t0 = time.time()
    rex = ReplicaExchange(lat, par, temps, X, X, seed=SEED, p_ss=0.5)
    tr = rex.run(n_cycles, sweeps_per_cycle=best_spc, record_every=10)
    wall = time.time() - t0
    E = float(np.mean(tr["E_per_site"][len(tr["E_per_site"]) // 2:]))
    print(f"  T_hi={T_hi:.0f} K: M={M:3d}  cycles={n_cycles:6d}  "
          f"round trips={rex.round_trips:4d}  "
          f"swap median={np.median(rex.swap_rates()):.3f}  "
          f"E/N={E:+.5f}  wall={wall:.0f}s")
    runs.append(dict(kind="T_hi", spc=best_spc, M=M, T_hi=T_hi,
                     cycles=n_cycles, round_trips=int(rex.round_trips),
                     E_per_site=E, temps=[float(t) for t in temps],
                     swap_median=float(np.median(rex.swap_rates())), wall_s=wall))

# ---- verdict ----
print("\n" + "=" * 70)
best = max(runs, key=lambda r: r["round_trips"])
base = runs[2]      # spc = 5, full ladder: the Step 2.2 configuration
print(f"Step 2.2 configuration : {base['round_trips']} round trips "
       f"(spc=5, M=29, T_hi=3000)")
print(f"Best configuration     : {best['round_trips']} round trips "
      f"(spc={best['spc']}, M={best['M']}, T_hi={best['T_hi']:.0f})")
if base["round_trips"] > 0:
    print(f"Improvement            : {best['round_trips']/base['round_trips']:.1f}x "
          f"at identical compute")
print(f"Energies agree to      : "
      f"{abs(best['E_per_site'] - base['E_per_site'])*1e3:.3f} meV per site")

with open("../data/roundtrip_tuning.json", "w") as f:
    json.dump(dict(total_sweeps=TOTAL_SWEEPS, L=L, N=int(lat.N), runs=runs), f, indent=2)
print("\nWritten to ../data/roundtrip_tuning.json")
