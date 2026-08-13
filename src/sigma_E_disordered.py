# -*- coding: utf-8 -*-
"""
sigma_E_disordered.py  --  Phase 2, Step 2.3 (corrected measurement)

Establishes how the replica-exchange ladder length must grow with box size.

Replica-exchange acceptance between adjacent replicas is governed by the
dimensionless product (delta beta) * sigma_E, where sigma_E is the width of the
energy distribution. Since sigma_E^2 = k_B T^2 C_v is extensive, sigma_E must
grow as sqrt(N), and holding acceptance fixed over a given temperature range
therefore requires M ~ sqrt(N) replicas.

Two earlier attempts to measure this failed, and the second failure was
informative rather than merely a bug:

  (i)  the first used integer division to set sweeps per block, which collapsed
       to one sweep at L >= 20 and two sweeps of burn-in at L = 30, so it
       measured relaxation drift and returned the impossible exponent N^1.34;

  (ii) the second fixed the sweep counts but sampled at 1800-2200 K, which lies
       BELOW the ordering crossover of the full model. Large boxes cannot
       equilibrate there in a few hundred sweeps, so the measurement was again
       contaminated by drift. That is not a coincidental difficulty: calibrating
       a ladder near the crossover requires equilibrating near the crossover,
       which is the very problem replica exchange exists to solve.

The resolution is to measure in the disordered phase, well above the crossover,
where equilibration takes a handful of sweeps. The prefactor there is smaller
than at the crossover, but the exponent is the same, because C_v is extensive at
every temperature. The exponent is what sets the scaling of M with N.
"""

import json
import time

import numpy as np

from sphalerite_mc import SphaleriteLattice, HamiltonianParams
from samplers import MixedMoveMC

X, SEED = 0.02, 90210
BURN, NBLOCK, PER_BLOCK = 200, 60, 10
TEMPS = (3500.0, 5000.0)          # disordered phase: crossover is near 2400-3000 K

res = {}
print("Equilibrium energy fluctuations in the disordered phase")
print(f"{'L':>3} {'N':>8} {'T':>6} {'sigma_E (eV)':>14} {'sigma/sqrt(N)':>15} "
      f"{'drift/sigma':>12} {'s':>6}")
print("-" * 74)

for L, temps in ((10, TEMPS), (20, TEMPS), (30, TEMPS), (40, TEMPS), (50, (3500.0,))):
    lat = SphaleriteLattice(L)
    lat._lut = None
    par = HamiltonianParams()
    for T in temps:
        t0 = time.time()
        mc = MixedMoveMC(lat, par, x_cu=X, x_in=X, seed=SEED + L, p_ss=0.5)
        mc.run(T, BURN, seed_offset=1, validate=False)
        Es = []
        for b in range(NBLOCK):
            mc.run(T, PER_BLOCK, seed_offset=100 + b, validate=False)
            Es.append(mc.E)
        Es = np.asarray(Es)
        s = float(Es.std(ddof=1))
        # residual drift: slope of a linear fit across the series, in units of sigma.
        # If this is not small, the series is still relaxing and sigma is not an
        # equilibrium fluctuation.
        sl = np.polyfit(np.arange(len(Es)), Es, 1)[0] * len(Es)
        drift = abs(sl) / s if s > 0 else np.nan
        res[f"L{L}_T{int(T)}"] = dict(N=int(lat.N), T=T, sigma=s,
                                      drift_over_sigma=float(drift))
        flag = "" if drift < 1.0 else "  <-- still relaxing"
        print(f"{L:3d} {lat.N:8d} {T:6.0f} {s:14.3f} {s/np.sqrt(lat.N):15.4f} "
              f"{drift:12.2f} {time.time()-t0:6.0f}{flag}")
    del lat

print()
M10 = 23      # calibrated in Step 2.2 for 573-2200 K at L = 10
for T in (3500, 5000):
    ks = [k for k in res if k.endswith(f"T{T}") and res[k]["drift_over_sigma"] < 1.0]
    if len(ks) < 3:
        continue
    N = np.array([res[k]["N"] for k in ks], float)
    S = np.array([res[k]["sigma"] for k in ks])
    o = np.argsort(N); N, S = N[o], S[o]
    p, c = np.polyfit(np.log(N), np.log(S), 1)
    print(f"T = {T} K, {len(ks)} clean points:  sigma_E ~ N^{p:+.3f}  "
          f"(theory +0.500)")
    res[f"exponent_T{T}"] = float(p)
    print(f"   implied M(50^3) = {M10 * (500000/4000)**p:.0f} replicas "
          f"(sqrt(N) rule gives {M10*np.sqrt(125):.0f})")

json.dump(res, open("../data/sigma_E_disordered.json", "w"), indent=2)
print("\nWritten to ../data/sigma_E_disordered.json")
