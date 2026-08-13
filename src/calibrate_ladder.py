# -*- coding: utf-8 -*-
"""
calibrate_ladder.py  --  Phase 2, Step 2.2

Builds a replica-exchange temperature ladder by adaptive refinement.

Starting from a coarse geometric ladder, a short exchange run measures the
acceptance of every adjacent pair. Wherever acceptance falls below the target,
a new temperature is inserted at the midpoint in beta, and the procedure
repeats. This concentrates replicas where the energy histograms are widest -
in practice around the ordering crossover, where the specific heat peaks -
rather than spreading them uniformly, which a fixed geometric ladder does badly.

The ladder is size-dependent, because the energy difference between adjacent
replicas is extensive while the acceptance exponent is not. It must therefore
be recalibrated for every box size, and this script is the tool for doing so.
"""

import json
import sys
import time

import numpy as np

from sphalerite_mc import SphaleriteLattice, HamiltonianParams, KB_EV
from samplers import ReplicaExchange, geometric_ladder


def measure_swaps(lat, par, temps, x_cu, x_in, n_cycles, sweeps_per_cycle,
                  seed, p_ss, burn_frac=0.4):
    """Run a short replica exchange and return per-pair swap acceptance."""
    rex = ReplicaExchange(lat, par, temps, x_cu, x_in, seed=seed, p_ss=p_ss)
    n_burn = max(1, int(burn_frac * n_cycles))
    rex.run(n_burn, sweeps_per_cycle=sweeps_per_cycle, record_every=10**9)
    rex.swap_att[:] = 0                       # discard burn-in statistics
    rex.swap_acc[:] = 0
    rex.run(n_cycles - n_burn, sweeps_per_cycle=sweeps_per_cycle,
            record_every=10**9)
    return rex.swap_rates(), rex


def build_ladder(lat, par, T_lo, T_hi, x_cu, x_in,
                 M0=8, target=0.20, max_M=48, max_rounds=6,
                 n_cycles=120, sweeps_per_cycle=4, seed=99, p_ss=0.0,
                 verbose=True):
    """
    Iteratively refine a ladder until every adjacent pair exchanges at least
    `target` of the time, or the replica cap is reached.
    """
    temps = geometric_ladder(T_lo, T_hi, M0)
    history = []
    for rnd in range(max_rounds):
        rates, _ = measure_swaps(lat, par, temps, x_cu, x_in,
                                 n_cycles, sweeps_per_cycle, seed + rnd, p_ss)
        history.append({"round": rnd, "M": len(temps),
                        "temps": [float(t) for t in temps],
                        "rates": [float(r) for r in rates]})
        if verbose:
            print(f"  round {rnd}: M={len(temps):3d}  "
                  f"swap acc min={rates.min():.3f} median={np.median(rates):.3f} "
                  f"max={rates.max():.3f}")
        weak = np.where(rates < target)[0]
        if weak.size == 0:
            if verbose:
                print("  all pairs above target; ladder converged")
            break
        if len(temps) + weak.size > max_M:
            # insert only where it is worst, up to the cap
            order = np.argsort(rates[weak])
            weak = weak[order][:max(0, max_M - len(temps))]
            if weak.size == 0:
                if verbose:
                    print(f"  replica cap {max_M} reached; stopping")
                break
        # insert midpoints in beta (equivalently, harmonic mean in T)
        betas = 1.0 / (KB_EV * temps)
        new_b = []
        for k in range(len(temps) - 1):
            new_b.append(betas[k])
            if k in weak:
                new_b.append(0.5 * (betas[k] + betas[k + 1]))
        new_b.append(betas[-1])
        temps = 1.0 / (KB_EV * np.asarray(new_b))
    return temps, history


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    T_LO, T_HI = 573.0, 3000.0
    X = 0.02

    t0 = time.time()
    lat = SphaleriteLattice(L)
    par = HamiltonianParams()
    print(f"Calibrating replica-exchange ladder for L={L} (N={lat.N}), "
          f"T = {T_LO:.0f}-{T_HI:.0f} K, x_Cu = x_In = {X}")

    temps, history = build_ladder(lat, par, T_LO, T_HI, X, X,
                                  M0=8, target=0.20, max_M=48, max_rounds=6,
                                  n_cycles=120, sweeps_per_cycle=4, seed=99)

    print(f"\nFinal ladder: M = {len(temps)}")
    print("  " + "  ".join(f"{t:.0f}" for t in temps))

    out = {"L": L, "N": int(lat.N), "T_lo": T_LO, "T_hi": T_HI,
           "x_cu": X, "x_in": X, "lambda_eV": par.lam,
           "temps_K": [float(t) for t in temps], "history": history}
    with open(f"../data/ladder_L{L}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to ../data/ladder_L{L}.json")
    print(f"Elapsed {time.time() - t0:.1f} s")
