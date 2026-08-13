# -*- coding: utf-8 -*-
"""
================================================================================
 samplers.py  --  Phase 2, Step 2.2

 Enhanced sampling for the sphalerite Cu-In lattice model.

 Phase 1 established that plain Kawasaki exchange dynamics collapses to an
 acceptance ratio of about 1e-5 below 1300 K, which makes a 50^3 production run
 in the ore-forming window impossible. Two independent remedies are implemented
 and benchmarked here.

 (1) MIXED MOVE SET.
     In the Phase 1 proposal, site i is drawn from the solute list and site j
     uniformly from all N sites, so the probability that j is also a solute is
     only x_solute (4 per cent at the composition studied). Solute-solute
     exchanges - Cu <-> In swaps - are therefore rare, yet they are exactly the
     moves that reorder the interior of a condensed domain without breaking any
     Cu-In bond. Phase 1 Experiment D identified precisely that interior
     ordering as the slow mode: quenched and annealed states were
     indistinguishable in pair statistics but differed by 79 meV per solute atom
     in energy. Boosting solute-solute exchange to a fixed fraction p_ss of all
     proposals targets that mode directly and costs nothing.

     Detailed balance: each move type is internally symmetric (both draws
     uniform over a set whose size is conserved), and the type is chosen with a
     fixed probability independent of the configuration, so the composite
     proposal is symmetric and the Metropolis criterion is unchanged.

 (2) REPLICA EXCHANGE (PARALLEL TEMPERING).
     M replicas are propagated at temperatures T_1 < ... < T_M. Configurations
     are periodically exchanged between adjacent replicas with probability
     min(1, exp[(beta_i - beta_j)(E_i - E_j)]). High-temperature replicas melt
     and re-form domains freely; low-temperature replicas inherit those
     decorrelated configurations. Composition is identical across replicas, so
     the exchange is valid without reweighting.

 ON CLUSTER MOVES.
     A Wolff or Swendsen-Wang construction is NOT directly applicable here, for
     two independent reasons. First, the charge-compensation coupling
     2*lambda*dq_i*dq_j changes sign with the species pair - attractive for
     heterovalent contacts, repulsive for homovalent ones - so the model is
     sign-frustrated and the Fortuin-Kasteleyn mapping that underpins those
     algorithms does not hold. Second, cluster flips do not conserve
     composition, whereas the canonical ensemble used here requires it. The
     correct generalisation is the geometric cluster algorithm of Dress-Krauth
     and Liu-Luijten, which conserves composition by construction; it is
     deferred rather than attempted, and replica exchange is used instead.

 Requires: sphalerite_mc.py (Phase 1 core module)
================================================================================
"""

from __future__ import annotations

import time

import numpy as np

from sphalerite_mc import (
    HAS_NUMBA, KB_EV, njit,
    SphaleriteLattice, HamiltonianParams, SphaleriteMC,
    _anion_charges, _total_energy,
)


# ==============================================================================
# 1. Mixed-move-set kernel
# ==============================================================================
@njit(cache=True, fastmath=True)
def _mc_run_mixed(spec, Q, nn1, nn2, an_of_cat, dq, P1, P2, lam,
                  beta, n_steps, solutes, seed, p_ss):
    """
    Metropolis sampling with a two-component move set.

    With probability p_ss both sites are drawn from the solute list (a
    solute-solute exchange, e.g. Cu <-> In); otherwise site j is drawn from all
    N sites, reproducing the Phase 1 move. Both components are symmetric
    proposals, and the mixing probability is configuration-independent, so
    detailed balance holds under the usual Metropolis criterion.

    Returns (cumulative energy change, overall acceptance, solute-solute
    acceptance, fraction of proposals that were solute-solute).
    """
    np.random.seed(seed)
    N = spec.shape[0]
    n_sol = solutes.shape[0]
    n_acc = 0
    n_acc_ss = 0
    n_ss = 0
    dE_total = 0.0

    a_list = np.empty(8, dtype=np.int64)
    d_list = np.empty(8, dtype=np.float64)

    for _ in range(n_steps):
        i = solutes[np.random.randint(n_sol)]
        is_ss = np.random.random() < p_ss
        if is_ss:
            j = solutes[np.random.randint(n_sol)]
            n_ss += 1
        else:
            j = np.random.randint(N)
        si = spec[i]
        sj = spec[j]
        if si == sj:
            continue

        # ---- E_c increment over the affected S tetrahedra ----
        d = dq[sj] - dq[si]
        n_aff = 0
        for k in range(4):
            a_list[n_aff] = an_of_cat[i, k]
            d_list[n_aff] = d
            n_aff += 1
        for k in range(4):
            a = an_of_cat[j, k]
            found = False
            for m in range(n_aff):
                if a_list[m] == a:
                    d_list[m] -= d
                    found = True
                    break
            if not found:
                a_list[n_aff] = a
                d_list[n_aff] = -d
                n_aff += 1

        dE = 0.0
        for m in range(n_aff):
            q0 = Q[a_list[m]]
            q1 = q0 + d_list[m]
            dE += lam * (q1 * q1 - q0 * q0)

        # ---- pair increments ----
        for k in range(nn1.shape[1]):
            nb = nn1[i, k]
            if nb != j:
                dE += P1[sj, spec[nb]] - P1[si, spec[nb]]
        for k in range(nn1.shape[1]):
            nb = nn1[j, k]
            if nb != i:
                dE += P1[si, spec[nb]] - P1[sj, spec[nb]]
        for k in range(nn2.shape[1]):
            nb = nn2[i, k]
            if nb != j:
                dE += P2[sj, spec[nb]] - P2[si, spec[nb]]
        for k in range(nn2.shape[1]):
            nb = nn2[j, k]
            if nb != i:
                dE += P2[si, spec[nb]] - P2[sj, spec[nb]]

        # ---- Metropolis ----
        acc = False
        if dE <= 0.0:
            acc = True
        elif np.random.random() < np.exp(-beta * dE):
            acc = True

        if acc:
            spec[i] = sj
            spec[j] = si
            for m in range(n_aff):
                Q[a_list[m]] += d_list[m]
            # Solute list: only changes when a solute moves onto a former host
            # site. In a solute-solute exchange both sites stay solutes.
            if sj == 0:
                for t in range(n_sol):
                    if solutes[t] == i:
                        solutes[t] = j
                        break
            dE_total += dE
            n_acc += 1
            if is_ss:
                n_acc_ss += 1

    return (dE_total,
            n_acc / max(n_steps, 1),
            n_acc_ss / max(n_ss, 1),
            n_ss / max(n_steps, 1))


class MixedMoveMC(SphaleriteMC):
    """SphaleriteMC with a tunable fraction of solute-solute exchange moves."""

    def __init__(self, *args, p_ss: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_ss = float(p_ss)

    def run(self, T_kelvin: float, n_sweeps: int, seed_offset: int = 0,
            validate: bool = False):
        beta = 1.0 / (KB_EV * T_kelvin)
        n_steps = int(n_sweeps) * self.lat.N
        dE, acc, acc_ss, frac_ss = _mc_run_mixed(
            self.spec, self.Q, self.lat.nn1, self.lat.nn2, self.lat.an_of_cat,
            self.dq, self.P1, self.P2, self.par.lam, beta, n_steps,
            self.solutes, self.seed + seed_offset, self.p_ss)
        self.E += dE
        self.last_acc_ss = acc_ss
        self.last_frac_ss = frac_ss
        if validate:
            self.validate_state()
        return acc


# ==============================================================================
# 2. Replica exchange
# ==============================================================================
class ReplicaExchange:
    """
    Parallel tempering over a fixed temperature ladder.

    Slot k is permanently associated with temperature temps[k]; configurations
    move between slots on exchange, so observables at a given temperature are
    read directly from the corresponding slot.
    """

    def __init__(self, lattice, params, temps_K, x_cu, x_in,
                 seed=20260803, p_ss=0.0, mixed=True):
        self.lat = lattice
        self.par = params
        self.temps = np.asarray(temps_K, dtype=np.float64)
        self.M = len(self.temps)
        self.betas = 1.0 / (KB_EV * self.temps)
        cls = MixedMoveMC if mixed else SphaleriteMC
        kw = dict(p_ss=p_ss) if mixed else {}
        self.reps = [cls(lattice, params, x_cu=x_cu, x_in=x_in,
                         seed=seed + 7919 * k, **kw) for k in range(self.M)]
        self.rng = np.random.default_rng(seed)
        # exchange bookkeeping
        self.swap_att = np.zeros(self.M - 1, dtype=np.int64)
        self.swap_acc = np.zeros(self.M - 1, dtype=np.int64)
        # replica-identity tracking, for round-trip diagnostics
        self.label = np.arange(self.M)      # label[k] = identity now in slot k
        self.rt_hits_bottom = np.zeros(self.M, dtype=np.int64)
        self.rt_hits_top = np.zeros(self.M, dtype=np.int64)
        self.round_trips = 0
        self._last_end = np.full(self.M, -1, dtype=np.int64)   # -1 none, 0 bottom, 1 top
        self._cycle = 0

    # ------------------------------------------------------------------
    def _sweep_all(self, sweeps_per_cycle):
        accs = np.empty(self.M)
        for k in range(self.M):
            accs[k] = self.reps[k].run(self.temps[k], sweeps_per_cycle,
                                       seed_offset=self._cycle * 131 + k,
                                       validate=False)
        return accs

    def _exchange(self, parity):
        """Attempt swaps on adjacent pairs of the given parity."""
        for k in range(parity, self.M - 1, 2):
            a, b = self.reps[k], self.reps[k + 1]
            delta = (self.betas[k] - self.betas[k + 1]) * (a.E - b.E)
            self.swap_att[k] += 1
            if delta >= 0.0 or self.rng.random() < np.exp(delta):
                self.swap_acc[k] += 1
                # exchange full configuration state between the two slots
                a.spec, b.spec = b.spec, a.spec
                a.Q, b.Q = b.Q, a.Q
                a.E, b.E = b.E, a.E
                a.solutes, b.solutes = b.solutes, a.solutes
                self.label[k], self.label[k + 1] = self.label[k + 1], self.label[k]

    def _track_round_trips(self):
        """Count how many replica identities have travelled bottom -> top -> bottom."""
        bottom_id = self.label[0]
        top_id = self.label[-1]
        for ident, end in ((bottom_id, 0), (top_id, 1)):
            prev = self._last_end[ident]
            if prev == -1:
                self._last_end[ident] = end
            elif prev != end:
                self._last_end[ident] = end
                if end == 0:
                    self.round_trips += 1

    # ------------------------------------------------------------------
    def run(self, n_cycles, sweeps_per_cycle=5, record_every=1,
            observables=("E_per_site", "alpha", "mean_dq2")):
        """
        Propagate all replicas and record observables of the coldest slot.
        Returns a dict of traces.
        """
        tr = {k: [] for k in observables}
        tr["cycle"] = []
        tr["acc_cold"] = []
        for c in range(n_cycles):
            self._cycle = c
            accs = self._sweep_all(sweeps_per_cycle)
            self._exchange(c % 2)
            self._track_round_trips()
            if c % record_every == 0:
                cold = self.reps[0]
                tr["cycle"].append(c)
                tr["acc_cold"].append(accs[0])
                if "E_per_site" in tr:
                    tr["E_per_site"].append(cold.E / self.lat.N)
                if "alpha" in tr:
                    tr["alpha"].append(cold.warren_cowley(2, 1))
                if "mean_dq2" in tr:
                    tr["mean_dq2"].append(cold.mean_dq2())
        return {k: np.asarray(v) for k, v in tr.items()}

    # ------------------------------------------------------------------
    def swap_rates(self):
        return self.swap_acc / np.maximum(self.swap_att, 1)

    def validate_all(self):
        for r in self.reps:
            r.validate_state()
        return True

    def report(self):
        r = self.swap_rates()
        lines = [f"replicas: {self.M}   T = {self.temps[0]:.0f} .. {self.temps[-1]:.0f} K",
                 f"swap acceptance: min {r.min():.3f}  median {np.median(r):.3f}  max {r.max():.3f}",
                 f"round trips completed: {self.round_trips}"]
        return "\n".join(lines)


# ==============================================================================
# 3. Diagnostics
# ==============================================================================
def integrated_autocorr_time(x, c=6.0, max_lag=None):
    """
    Integrated autocorrelation time by the automatic-windowing procedure of
    Sokal: tau_int = 1 + 2 * sum_{t=1}^{W} rho(t), with the window W the
    smallest integer satisfying W >= c * tau_int(W).

    Returns (tau_int, window). Units are the spacing of the input series.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 16:
        return np.nan, 0
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var <= 0:
        return np.nan, 0
    if max_lag is None:
        max_lag = n // 4
    # FFT-based autocovariance
    nf = 1
    while nf < 2 * n:
        nf *= 2
    f = np.fft.rfft(x, nf)
    acov = np.fft.irfft(f * np.conjugate(f), nf)[:n].real / n
    rho = acov / acov[0]
    tau = 1.0
    W = 0
    for t in range(1, min(max_lag, n - 1)):
        tau += 2.0 * rho[t]
        W = t
        if t >= c * tau:
            break
    return max(tau, 0.5), W


def geometric_ladder(T_lo, T_hi, M):
    """Geometric temperature ladder, which spaces beta roughly uniformly in
    log and is the standard first choice for replica exchange."""
    return T_lo * (T_hi / T_lo) ** (np.arange(M) / (M - 1))


if __name__ == "__main__":
    t0 = time.time()
    print("samplers.py smoke test")
    lat = SphaleriteLattice(6)
    par = HamiltonianParams()

    m = MixedMoveMC(lat, par, x_cu=0.02, x_in=0.02, seed=1, p_ss=0.5)
    acc = m.run(1200.0, 200, validate=True)
    print(f"  mixed move set: acc={acc:.4f}  frac_ss={m.last_frac_ss:.3f}  "
          f"acc_ss={m.last_acc_ss:.4f}  -> state valid")

    temps = geometric_ladder(573.0, 3000.0, 6)
    rex = ReplicaExchange(lat, par, temps, 0.02, 0.02, seed=2, p_ss=0.3)
    tr = rex.run(60, sweeps_per_cycle=5)
    rex.validate_all()
    print("  replica exchange:")
    for line in rex.report().split("\n"):
        print("   ", line)
    print(f"    cold-slot alpha = {tr['alpha'][-1]:+.3f}, "
          f"E/site = {tr['E_per_site'][-1]:+.5f} eV  -> all states valid")

    tau, W = integrated_autocorr_time(np.random.randn(2000))
    print(f"  autocorrelation on white noise: tau={tau:.2f} (expect ~1), window={W}")
    print(f"\nElapsed {time.time() - t0:.1f} s ; Numba = {HAS_NUMBA}")
