# sphalerite-equilibrium-mc

Replica-exchange Monte Carlo evidence that indium in sphalerite does not form a
dilute solid solution. Across every composition examined, from 0.10 to
2.00 at.%, the equilibrium state is condensed.

Code, data and manuscript for:

> **He Yu**, **Ruqing Chen**, **Huanzhang Lu**, *The Equilibrium Fate of Indium
> in Sphalerite: Large-Scale Replica-Exchange Simulation and the Absence of a
> Dilute Solid Solution.*

- He Yu — Geological Laboratory, Hezhou University, Hezhou, Guangxi 542899, China (`yuhe@hzxy.edu.cn`)
- Ruqing Chen — GUT Geoservice Inc., Montreal, Quebec, Canada (`ruqing@hotmail.com`)
- Huanzhang Lu — Université du Québec à Chicoutimi, Quebec, Canada (`hzlu@uqac.uquebec.ca`)

This is the second paper of the *Computable Mineral Deposit Chemistry* series.
The Hamiltonian, its exact reduction and the analytic ground state are
established in Paper 1, archived at
[10.5281/zenodo.21880502](https://doi.org/10.5281/zenodo.21880502) with code at
[Ruqing1963/sphalerite-lattice-mc](https://github.com/Ruqing1963/sphalerite-lattice-mc).

---

## The question, and why simulation can answer it

Whether indium in sphalerite sits in true solid solution or in nanoscale
inclusions has resisted empirical resolution because the two states differ at
length scales near the limit of detection. Paper 1 could not settle it either:
at 4000 cation sites and 4 at.% solute, the solute budget is exhausted by a
single domain, so complete condensation was consistent with — but not diagnostic
of — a thermodynamic preference.

This work removes that limitation in both directions: a box eight times larger,
and compositions twenty times more dilute.

## Headline results

| Quantity | Value |
|---|---|
| Compositions examined | 0.10 – 2.00 at.% (≈1000 – 20 000 ppm In) |
| Box | *L* = 20, *N* = 32 000 cation sites, 10.8 × 10.8 × 10.8 nm |
| Solute in largest cluster | 100 % at every composition |
| Monomer fraction | 0 %, against 98 % expected for a random solid solution at 0.10 at.% |
| Discrepancy from random solution | four orders of magnitude |
| Equilibrium bracket at 1800 K | 64–88 % (0.10 at.%), 95–97 % (0.25 at.%) |
| First-shell Cu around In | 0.443 – 0.545, essentially composition-independent |
| Ladder scaling | σ<sub>E</sub> ∝ *x*<sup>0.85</sup> *N*<sup>0.51</sup> |

## The argument in five steps

1. **Condensation is not a finite-size artefact.** The same 160 solute atoms
   that exhausted the Paper 1 box condense completely in eight times the volume;
   halving the concentration again, so each solute has 500 sites of room, does
   not change the outcome.
2. **It is an equilibrium property, and the proof does not need convergence.**
   Each state point is run from a dispersed *and* from a pre-condensed
   configuration. These approach equilibrium from opposite sides, so it lies
   between them whether or not either run has converged. At 1800 K, where the
   acceptance ratio is near 10 %, dispersed solutes spontaneously aggregate
   while condensed solutes do not disperse.
3. **The transfer to ore-forming temperature is monotone.** Ordering strengthens
   as *T* falls, so condensation at 573 K can be no weaker than the bracket
   measured at 1000–1800 K.
4. **The enrichment factor is the wrong observable in the dilute limit.**
   *R*<sub>Cu–In</sub> rises sixteenfold across the scan while the quantity it
   describes — the Cu content of the In first shell — falls slightly. Its ceiling
   is 1/*x*<sub>Cu</sub>; the apparent variation is almost entirely in the
   normalisation. Report cluster-size distributions instead.
5. **The conclusion does not depend on the unconstrained chemical term.**
   Electrostatics alone gives a sixteenfold enrichment, and removing *J*
   *strengthens* the claim, because the resulting energy landscape is smoother
   and the system condenses more readily.

## Repository layout

```
src/
  sphalerite_mc.py          core module, unchanged from Paper 1: lattice,
                            Hamiltonian, MC engine, verification suite
  samplers.py               replica exchange, mixed move set, autocorrelation
  calibrate_ladder.py       adaptive ladder construction
  composition_scan.py       the composition scan (Section 5)
  condensation_bound.py     the bracketing test (Section 5.3)
  coarsening_test.py        single-domain vs multi-domain, 36 000 sweeps
  scale_profile.py          verification and profiling to N = 5e5
  sigma_E_disordered.py     the sigma_E ~ N^0.51 measurement
  sigma_vs_solute.py        the sigma_E ~ N_sol^0.85 measurement
  tune_roundtrips.py        round trips vs exchange interval and ladder top
  benchmark_samplers.py     Kawasaki vs mixed moves vs replica exchange
  validate_vs_phase1.py     regression against the published Paper 1 values
  a3_*.py                   re-measurement of the electrostatics-only model
  dft_inputs.py             structure generation and least-squares fitting
  collect_energies.py       VASP energy collection
  qe_inputs.py              Quantum ESPRESSO conversion
data/     numerical results underlying every figure and table, as JSON
          (machine-readable, full precision) and as CSV (one file per
          manuscript table, openable in any spreadsheet)
figures/  Figures 1-2 in vector PDF and raster PNG, plus the script that makes them
paper/    manuscript in plain article format, in Elsevier elsarticle format
          for GCA submission, and in Chinese translation; LaTeX source and
          compiled PDF for each
dft/      12 POSCAR files, INCAR, KPOINTS and manifest for the deferred
          parameterisation campaign
```

## Requirements

```bash
pip install numpy matplotlib numba
```

Numba is optional but gives roughly a fiftyfold speed-up; without it the
composition scan takes days rather than hours.

## Reproducing the results

```bash
cd src

# 1. Verification suite (~5 s). Checks lattice topology, energy bookkeeping,
#    every analytic limit to machine precision, and the random-solution limit.
python sphalerite_mc.py

# 2. The composition scan (~12 h single core, checkpointed per composition).
python composition_scan.py

# 3. The bracketing test (~30 min).
python condensation_bound.py

# 4. Single domain versus multiple domains (~11 h, two starting configurations).
python coarsening_test.py

# 5. Scaling and verification up to N = 5e5 (~40 min).
python scale_profile.py
python sigma_E_disordered.py
python sigma_vs_solute.py
```

Scripts that write to `data/` resolve the path relative to the script, so they
can be run from any working directory. Long runs checkpoint and resume; deleting
the output JSON forces a fresh start.

Figures:

```bash
cd figures && python mkfig.py
```

Manuscript:

```bash
cd paper
pdflatex paper2.tex          # plain article format; run twice
pdflatex paper2_gca.tex      # Elsevier elsarticle; needs texlive-publishers
```

## What this work does not establish

Stated plainly, because these bound what the results support.

- **Where the behaviour ceases.** The compositions examined run an order of
  magnitude above natural tenors. An entropic estimate places the composition
  boundary between roughly 3 ppm and 10⁻⁵ ppm — below crustal abundance — but
  locating it requires boxes of order 5×10⁵ sites. The implementation is
  verified correct at that scale and the scaling laws are established; the
  obstacle is computational, not conceptual.
- **The chemical term J.** The parameterisation campaign is specified and its
  structure generation and fitting harness are validated against synthetic data,
  but not run: a 216-atom cell needs 18.7 GB by the code's own estimate, against
  ~12 GB available on a 16 GB workstation. Everything needed to run it is in
  `dft/` and `src/dft_inputs.py`.
- **Whether natural sphalerite attains equilibrium.** Monte Carlo sweeps are not
  physical time. Multi-domain configurations in natural samples are expected to
  reflect kinetic arrest during cooling, which is the subject of the next paper
  in the series.
- **Iron.** Absent from the model. Fe²⁺ is isovalent with Zn²⁺ so does not enter
  the charge-compensation term, but it modifies the dielectric response that
  fixes λ.

## Notes for anyone building on this

**Report replica round trips, not swap acceptance.** Reducing the interval
between exchange attempts from five sweeps to one takes the round-trip count
from 0 to 7 at identical cost, while the median swap acceptance moves by less
than 0.01. A study reporting acceptance alone would have concluded its ladder
was performing well while no replica traversed it.

**Two agreeing trajectories are not an equilibration test.** In a system with an
acceptance ratio of 10⁻⁵, two chains launched into the same basin agree with
each other whether or not that basin is the equilibrium one. Paper 1 drew a
stronger conclusion from such agreement than it could support; the correction is
in Appendix A of the manuscript, and it is why the central claim here rests on
bracketing rather than on convergence.

**Boosted solute–solute exchange fails on its own.** Its acceptance at 573 K in
a single replica is exactly zero — swapping Cu for In inside an ordered domain
costs of order 4λ against k<sub>B</sub>T = 0.049 eV. It is useful only inside
replica exchange, where hot replicas accept it and pass decorrelated
configurations down the ladder.

**Cluster algorithms do not apply.** The coupling 2λ δq<sub>i</sub> δq<sub>j</sub>
is sign-frustrated, so the Fortuin–Kasteleyn mapping fails; and cluster flips do
not conserve composition, which the canonical ensemble requires.

**Ladders cannot be pre-calibrated near the crossover.** Doing so requires
equilibrating near the crossover, which is the problem replica exchange exists
to solve. Measure the scaling in the disordered phase and grow the ladder
downwards during production.

## Citation

> Yu, H., Chen, R., Lu, H. (2026) *The Equilibrium Fate of Indium in Sphalerite:
> Large-Scale Replica-Exchange Simulation and the Absence of a Dilute Solid
> Solution.* https://github.com/Ruqing1963/sphalerite-equilibrium-mc
> [Zenodo DOI to be minted on release]

See `CITATION.cff` for machine-readable metadata.

## Licence

- **Code** (`src/`, `figures/mkfig.py`): MIT — see `LICENSE`.
- **Data and figures** (`data/`, `figures/`): CC BY 4.0 — see `LICENSE-DATA`.
- **Manuscript** (`paper/`): © the authors.
