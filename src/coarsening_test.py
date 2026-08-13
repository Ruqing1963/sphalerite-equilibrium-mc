# -*- coding: utf-8 -*-
"""
coarsening_test.py  --  Phase 2, Step 2.4

Decides one question: at x = 2.0 at.% the composition scan returned TWO solute
clusters (1090 + 190 atoms) where every more dilute composition returned one.
Is that incomplete coarsening, or a genuine characteristic domain size?

The distinction matters. In a canonical ensemble at fixed composition, a phase
separating system should end with a single domain, because that minimises
interfacial area. If two domains survive at equilibrium, something is stabilising
a finite size - and "nanoinclusion" would then be a statement about equilibrium
structure rather than about kinetics.

The test uses the bracketing logic that worked in condensation_bound.py, and
adds a time trace so that a partial run is still informative:

  RUN A  starts from a random configuration, which the scan showed reaches two
         clusters. If the cluster count trends downwards, that is coarsening.

  RUN B  starts from a SINGLE compact ball containing all solutes. If a single
         domain is unstable, it must split; if it does not split, the two-cluster
         state cannot be the equilibrium one.

Cluster count is recorded throughout both runs, so the trend answers the question
even if neither run reaches full convergence. A monotone decrease in run A, with
run B staying at one cluster, means coarsening. Run A flattening at two while
run B splits to two means a characteristic size.

Cost: about 5.5 h per run single-core at the default budget. Both runs are
checkpointed every few hundred cycles and can be interrupted safely; rerunning
resumes from the checkpoint.
"""

import json
import os
import time

import numpy as np

# The three modules below must sit in the SAME FOLDER as this script. Python
# puts a script's own directory on the import path, so nothing else is needed -
# but copying this file somewhere on its own will fail with a bare
# ModuleNotFoundError that does not say why.
try:
    from sphalerite_mc import SphaleriteLattice, HamiltonianParams
    from samplers import ReplicaExchange
    from calibrate_ladder import build_ladder
except ModuleNotFoundError as err:
    _here = os.path.dirname(os.path.abspath(__file__))
    _need = ["sphalerite_mc.py", "samplers.py", "calibrate_ladder.py"]
    _missing = [f for f in _need if not os.path.exists(os.path.join(_here, f))]
    print("=" * 70)
    print(f"Cannot import '{err.name}'.")
    print(f"\nThis script lives in:\n  {_here}")
    if _missing:
        print("\nMissing from that folder:")
        for f in _missing:
            print(f"  {f}")
        print("\nFix: put this script in the same folder as the other Phase 2")
        print("source files (phase2/src), or copy the missing files next to it.")
    else:
        print("\nThe required files are present, so this is an environment")
        print("problem rather than a layout one. Check that numpy is installed.")
    print("=" * 70)
    raise SystemExit(1)

# ---------------- configuration ----------------
L = 20
X = 0.0200                      # the composition where two clusters appeared
T_LO, T_HI = 573.0, 2200.0
SPC = 1
SWEEPS = 36_000                 # three times the composition scan
RECORD_EVERY = 200              # cycles between cluster-count measurements
SEED = 5150

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, os.pardir, "data")
os.makedirs(_DATA, exist_ok=True)
OUT = os.path.join(_DATA, "coarsening_test.json")

CONFIG = dict(L=L, x=X, T=T_LO, spc=SPC, sweeps=SWEEPS,
              record_every=RECORD_EVERY, seed=SEED)


def single_ball_start(mc, lat):
    """
    Place every solute in ONE connected domain, alternating Cu and In so that
    the starting cluster is already charge-compensated.

    Connectivity matters here: this run exists to test whether a single domain
    is stable, so the starting configuration must actually BE a single cluster.
    Taking the n_sol sites nearest the box centre does not guarantee that, since
    the fcc sublattice is not filled contiguously by a distance cut. The domain
    is therefore grown by breadth-first search over the nearest-neighbour graph
    from a central seed, which is connected by construction and is asserted so.
    """
    n_cu = int(np.count_nonzero(mc.spec == 1))
    n_in = int(np.count_nonzero(mc.spec == 2))
    n_tot = n_cu + n_in

    centre = lat.pos.mean(axis=0)
    seed = int(np.argmin(np.linalg.norm(lat.pos - centre, axis=1)))

    seen = np.zeros(lat.N, dtype=bool)
    seen[seed] = True
    order = [seed]
    head = 0
    while len(order) < n_tot and head < len(order):
        for nb in lat.nn1[order[head]]:
            if not seen[nb]:
                seen[nb] = True
                order.append(int(nb))
                if len(order) == n_tot:
                    break
        head += 1
    assert len(order) == n_tot, "BFS could not reach enough sites"

    core = np.asarray(order)
    mc.spec[:] = 0
    # Species assignment must produce a CHARGE-COMPENSATED domain, or the start
    # is high in energy and fragments before it can test anything. Alternating
    # along the BFS order does not do this: BFS order has no relation to the
    # lattice, so it leaves many homovalent contacts. Use instead the layered
    # CuAu-I rule verified in Phase 1 to give every S tetrahedron 2 Cu + 2 In,
    # hence E_c = 0 in the domain interior.
    layer = (lat.pos[core, 2] // 2) % 2
    mc.spec[core] = np.where(layer == 0, 1, 2)
    # correct the counts exactly, preferring to reassign sites at the domain edge
    for a, b, target in ((1, 2, n_cu), (2, 1, n_in)):
        have = int(np.count_nonzero(mc.spec[core] == a))
        if have > target:
            cand = core[mc.spec[core] == a][::-1]      # last added = outermost
            mc.spec[cand[: have - target]] = b
    mc._refresh()
    cs = mc.cluster_sizes()
    assert cs.size == 1, f"single-ball start is not connected: {cs.size} clusters"


def cluster_state(mc):
    cs = mc.cluster_sizes()
    return dict(n=int(cs.size), largest=int(cs[0]),
                largest_frac=float(cs[0] / cs.sum()),
                top5=[int(v) for v in cs[:5]])


# ---------------- resume logic ----------------
state = {}
if os.path.exists(OUT):
    old = json.load(open(OUT))
    if old.get("config") == CONFIG:
        state = old.get("runs", {})
        print(f"resuming; {len(state)} run(s) already complete\n")
    else:
        bak = OUT.replace(".json", ".superseded.json")
        os.replace(OUT, bak)
        print(f"settings changed; previous results moved to\n  {bak}\n")

lat = SphaleriteLattice(L)
lat._lut = None
par = HamiltonianParams()
n_sol = int(round(2 * X * lat.N))
print(f"Coarsening test: L={L} (N={lat.N}), x={X*100:.2f}% ({n_sol} solutes), "
      f"T={T_LO:.0f} K")
print(f"{SWEEPS:,} sweeps per replica, cluster count recorded every "
      f"{RECORD_EVERY} cycles\n")

print("calibrating ladder ...", end=" ", flush=True)
t0 = time.time()
temps, _ = build_ladder(lat, par, T_LO, T_HI, X, X, M0=66, target=0.20,
                        max_M=110, max_rounds=4, n_cycles=40,
                        sweeps_per_cycle=2, seed=SEED, p_ss=0.5, verbose=False)
M = len(temps)
print(f"M = {M}  ({time.time()-t0:.0f} s)")
est = M * SWEEPS * lat.N / 4.1e6 / 3600
print(f"estimated {est:.1f} h per run, {2*est:.1f} h for both\n")

for tag in ("random", "single_ball"):
    if tag in state:
        print(f"[{tag}] already done, skipping")
        continue
    print(f"[{tag}] starting")
    rex = ReplicaExchange(lat, par, temps, X, X, seed=SEED, p_ss=0.5)
    if tag == "single_ball":
        for r in rex.reps:
            single_ball_start(r, lat)

    trace = []
    t0 = time.time()
    n_cycles = SWEEPS // SPC
    for c in range(n_cycles):
        rex._cycle = c
        rex._sweep_all(SPC)
        rex._exchange(c % 2)
        rex._track_round_trips()
        if c % RECORD_EVERY == 0:
            cold = rex.reps[0]
            cl = cluster_state(cold)
            cl.update(cycle=c, E_per_site=float(cold.E / lat.N),
                      alpha=float(cold.warren_cowley(2, 1)))
            trace.append(cl)
            if c % (RECORD_EVERY * 10) == 0:
                print(f"   cycle {c:6d}/{n_cycles}  clusters={cl['n']:3d}  "
                      f"largest={cl['largest']:5d} ({100*cl['largest_frac']:.0f}%)  "
                      f"E/N={cl['E_per_site']:+.6f}  "
                      f"[{(time.time()-t0)/60:.0f} min]")
                # checkpoint mid-run so an interrupt keeps the trace
                json.dump(dict(config=CONFIG, M=M, runs={**state,
                               tag + "_partial": dict(trace=trace)}),
                          open(OUT, "w"), indent=2)
    rex.validate_all()
    final = cluster_state(rex.reps[0])
    state[tag] = dict(trace=trace, final=final, M=M,
                      round_trips=int(rex.round_trips),
                      swap_median=float(np.median(rex.swap_rates())),
                      wall_s=time.time() - t0)
    state.pop(tag + "_partial", None)
    print(f"[{tag}] done: {final['n']} cluster(s), largest {final['largest']} "
          f"({100*final['largest_frac']:.0f}%), round trips "
          f"{state[tag]['round_trips']}, {(time.time()-t0)/60:.0f} min\n")
    json.dump(dict(config=CONFIG, M=M, runs=state), open(OUT, "w"), indent=2)

# ---------------- verdict ----------------
if "random" in state and "single_ball" in state:
    a, b = state["random"], state["single_ball"]
    na = [p["n"] for p in a["trace"]]
    nb = [p["n"] for p in b["trace"]]
    print("=" * 70)
    print(f"random start      : {na[0]} -> {na[-1]} clusters "
          f"(min {min(na)}, last quarter mean {np.mean(na[3*len(na)//4:]):.2f})")
    print(f"single-ball start : {nb[0]} -> {nb[-1]} clusters "
          f"(max {max(nb)}, last quarter mean {np.mean(nb[3*len(nb)//4:]):.2f})")
    ea = np.mean([p["E_per_site"] for p in a["trace"][3*len(na)//4:]])
    eb = np.mean([p["E_per_site"] for p in b["trace"][3*len(nb)//4:]])
    print(f"energies: random {ea:+.6f}, single ball {eb:+.6f} eV per site "
          f"(difference {abs(ea-eb)*lat.N:.2f} eV over the box)")
    if b["final"]["n"] == 1 and na[-1] <= na[0]:
        print("\nVERDICT: the single domain is stable and the random start is "
              "coarsening towards it.\n         The two-cluster state is kinetic, "
              "not equilibrium.")
    elif b["final"]["n"] > 1 and na[-1] > 1:
        print("\nVERDICT: a single domain SPLITS and the random start does not "
              "merge.\n         There is a characteristic domain size - this is "
              "a real result and\n         needs following up with a size-vs-N "
              "study.")
    else:
        print("\nVERDICT: inconclusive at this budget; the two runs have not "
              "bracketed.\n         Increase SWEEPS and rerun.")
print(f"\nWritten to {OUT}")
