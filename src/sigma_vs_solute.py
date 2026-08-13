# -*- coding: utf-8 -*-
"""
sigma_vs_solute.py -- Phase 2, Step 2.4 (feasibility check)

Step 2.3 measured sigma_E ~ sqrt(N) at fixed composition x = 2 at.%, and
concluded M ~ sqrt(N). But at fixed x the solute count is proportional to N, so
that measurement cannot distinguish sqrt(N) from sqrt(N_solute). The distinction
matters enormously for a dilute composition scan: if the controlling variable is
the solute count, then a large box at low concentration needs far fewer replicas
than sqrt(N) would suggest.

Physically the solute count should be the right variable, because Zn-Zn contacts
carry no configurational energy: the entire configurational heat capacity comes
from the solute subsystem.

Test: hold the box fixed and vary x.
"""
import json, time
import numpy as np
from sphalerite_mc import SphaleriteLattice, HamiltonianParams
from samplers import MixedMoveMC

SEED, BURN, NBLOCK, PER = 4711, 200, 60, 10
T = 3500.0

print("Fixed box, varying composition (T = 3500 K, disordered phase)")
print(f"{'L':>3} {'N':>7} {'x':>7} {'N_sol':>7} {'sigma_E':>10} "
      f"{'sig/sqrt(N)':>12} {'sig/sqrt(Nsol)':>15}")
rows = []
for L in (20,):
    lat = SphaleriteLattice(L); lat._lut = None
    par = HamiltonianParams()
    for x in (0.0025, 0.005, 0.01, 0.02):
        mc = MixedMoveMC(lat, par, x_cu=x, x_in=x, seed=SEED, p_ss=0.5)
        n_sol = int(np.count_nonzero(mc.spec != 0))
        mc.run(T, BURN, seed_offset=1, validate=False)
        Es = []
        for b in range(NBLOCK):
            mc.run(T, PER, seed_offset=100+b, validate=False)
            Es.append(mc.E)
        s = float(np.std(Es, ddof=1))
        print(f"{L:3d} {lat.N:7d} {x:7.4f} {n_sol:7d} {s:10.3f} "
              f"{s/np.sqrt(lat.N):12.4f} {s/np.sqrt(n_sol):15.4f}")
        rows.append(dict(L=L, N=int(lat.N), x=x, n_sol=n_sol, sigma=s))

ns = np.array([r["n_sol"] for r in rows], float)
ss = np.array([r["sigma"] for r in rows])
p = np.polyfit(np.log(ns), np.log(ss), 1)[0]
print(f"\nAt fixed N = {rows[0]['N']}:  sigma_E ~ N_solute^{p:+.3f}")
print("  -> the controlling variable is the solute count" if abs(p-0.5) < 0.15
      else "  -> NOT simply the solute count; investigate")

# implied replica counts
M_ref, nsol_ref = 23, 160          # calibrated at L=10, x=2%
print(f"\nImplied ladder length M = {M_ref} * sqrt(N_sol / {nsol_ref}):")
for L in (20, 30, 50):
    N = 4*L**3
    for x in (0.001, 0.005, 0.02):
        nsol = int(2*x*N)
        print(f"  L={L:2d} (N={N:6d}), x_In={x*100:4.1f}%: "
              f"N_sol={nsol:6d} -> M ~ {M_ref*np.sqrt(nsol/nsol_ref):5.0f}")
json.dump(rows, open("../data/sigma_vs_solute.json","w"), indent=2)
