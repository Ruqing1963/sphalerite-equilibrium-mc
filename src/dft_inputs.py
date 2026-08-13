# -*- coding: utf-8 -*-
"""
dft_inputs.py  --  Phase 2, Step 2.1 implementation

Generates the supercell structures specified in docs/step2.1_dft_design.md, as
VASP POSCAR files, and provides the least-squares harness that turns the
resulting total energies back into Hamiltonian parameters.

Only configuration sets A, B and C are generated here. They are the ones that
matter most and the ones that are cheap, because every structure in them is
CHARGE NEUTRAL and COMPOSITION MATCHED within its set: no chemical potentials
and no charged-defect corrections are needed, and within-set energy differences
are the most robust quantity a plane-wave code produces.

  SET A   one Cu and one In at increasing separation. Tests the two structural
          claims of Phase 1 at once: whether the fitted lambda matches the
          dielectric estimate 0.227-0.368 eV, and whether the binding energy
          really vanishes beyond first neighbours as the sharing Lemma requires.

  SET B   two Cu and two In in different topologies, which separates
          J(Cu-Cu) and J(In-In) from J(Cu-In).

  SET C   a Zn vacancy with two In, giving the configurational part of the
          competition between the two compensation routes at fixed In content -
          far more robust than computing the absolute vacancy formation energy.

Set D (CuInS2 cation orderings) is not generated: those are small end-member
cells that are easier to build in a structure editor than to script, and the
orderings are standard.

USAGE

    python dft_inputs.py                  # writes ../dft/POSCAR files + manifest
    python dft_inputs.py --fit energies.csv

The CSV for the fit needs two columns, `name` and `energy_eV`, one row per
structure, using the names in the generated manifest.
"""

import json
import os
import sys

import numpy as np

try:
    from sphalerite_mc import SphaleriteLattice, HamiltonianParams, DELTA_Q, _anion_charges, _pair_counts
except ModuleNotFoundError as err:
    print(f"Cannot import '{err.name}'. Run this from the folder containing "
          f"sphalerite_mc.py.")
    raise SystemExit(1)

A0 = 5.4093                # ZnS cubic cell parameter, Angstrom
NCELL = 3                  # 3x3x3 conventional cells = 216 atoms, 108 cation sites
_HERE = os.path.dirname(os.path.abspath(__file__))
_DFT = os.path.join(_HERE, os.pardir, "dft")


# ----------------------------------------------------------------------
# structure generation
# ----------------------------------------------------------------------
def build_cells():
    """Cation and anion fractional coordinates of an NCELL^3 supercell."""
    lat = SphaleriteLattice(NCELL)
    lat._lut = None
    cat = lat.pos / (4.0 * NCELL)                     # fractional, in [0,1)
    an = ((lat.pos + 1) % (4 * NCELL)) / (4.0 * NCELL)
    return lat, cat, an


def write_poscar(path, title, cat_frac, species, an_frac):
    """
    VASP POSCAR, species ordered Zn, Cu, In, S. Vacancies are written by simply
    omitting the site, which is what a vacancy is.
    """
    order = [("Zn", 0), ("Cu", 1), ("In", 2)]
    blocks, names, counts = [], [], []
    for name, code in order:
        sel = cat_frac[species == code]
        if len(sel):
            names.append(name); counts.append(len(sel)); blocks.append(sel)
    names.append("S"); counts.append(len(an_frac)); blocks.append(an_frac)

    with open(path, "w") as f:
        f.write(f"{title}\n1.0\n")
        for i in range(3):
            v = [0.0, 0.0, 0.0]
            v[i] = A0 * NCELL
            f.write(f"  {v[0]:.8f}  {v[1]:.8f}  {v[2]:.8f}\n")
        f.write("  " + "  ".join(names) + "\n")
        f.write("  " + "  ".join(str(c) for c in counts) + "\n")
        f.write("Direct\n")
        for b in blocks:
            for p in b:
                f.write(f"  {p[0]:.8f}  {p[1]:.8f}  {p[2]:.8f}\n")


def _mic_dist(lat, i):
    """Minimum-image distances from site i to every site, in units of a0/4."""
    G = 4 * NCELL
    d = (lat.pos - lat.pos[i]) % G
    d = np.minimum(d, G - d)
    return np.linalg.norm(d, axis=1)


def shell_of(lat, i, k):
    """
    A representative site in the k-th coordination shell of i.

    Distances are computed by minimum image in units of a0/4, in which the
    shells fall at 2*sqrt(2), 4, 2*sqrt(6), 4*sqrt(2), ... The largest possible
    separation in a 3x3x3 cell is half the body diagonal, about 10.4 of these
    units, or 14 Angstrom. Passing k = 0 returns the most distant site, which is
    what the "separated" reference structures need.
    """
    d = _mic_dist(lat, i)
    if k == 0:
        return int(np.argmax(d))
    shells = np.unique(np.round(d[d > 1e-9], 4))
    if k > len(shells):
        raise RuntimeError(f"shell {k} does not exist in a {NCELL}^3 cell "
                           f"(only {len(shells)} shells)")
    target = shells[k - 1]
    return int(np.flatnonzero(np.isclose(d, target))[0])


def far_sites(lat, i, n):
    """The n sites most distant from i and from each other, for reference cells."""
    d = _mic_dist(lat, i)
    order = np.argsort(-d)
    picked = []
    for c in order:
        c = int(c)
        if c == i:
            continue
        if all(_mic_dist(lat, c)[p] > 6.0 for p in picked):
            picked.append(c)
            if len(picked) == n:
                return picked
    raise RuntimeError("could not find enough mutually distant sites")


def generate():
    lat, cat, an = build_cells()
    os.makedirs(_DFT, exist_ok=True)
    manifest = []
    i0 = 0

    def emit(name, spec, note):
        write_poscar(os.path.join(_DFT, f"POSCAR_{name}"), f"{name}  {note}",
                     cat, spec, an)
        q = _anion_charges(np.ascontiguousarray(spec.astype(np.int8)),
                           lat.cat_of_an, DELTA_Q)
        C = _pair_counts(np.ascontiguousarray(spec.astype(np.int8)), lat.nn1)
        manifest.append(dict(
            name=name, note=note,
            n_Cu=int(np.count_nonzero(spec == 1)),
            n_In=int(np.count_nonzero(spec == 2)),
            n_vac=int(np.count_nonzero(spec == 3)),
            sum_dQ2=float(np.sum(q ** 2)),                 # the E_c descriptor
            n_CuIn=float(C[1, 2]), n_CuCu=float(C[1, 1]),
            n_InIn=float(C[2, 2]), n_InVac=float(C[2, 3])))

    # ---- pure reference ----
    emit("ref_pure", np.zeros(lat.N, dtype=int), "pure ZnS supercell")

    # ---- SET A: one Cu, one In, increasing separation ----
    for k, label in ((1, "1NN"), (2, "2NN"), (3, "3NN"), (4, "4NN")):
        s = np.zeros(lat.N, dtype=int)
        s[i0] = 1
        s[shell_of(lat, i0, k)] = 2
        emit(f"A_{label}", s, f"Cu and In at {label} separation")
    s = np.zeros(lat.N, dtype=int)
    s[i0] = 1
    s[shell_of(lat, i0, 0)] = 2
    emit("A_far", s, "Cu and In maximally separated (reference for set A)")

    # ---- SET B: two Cu, two In ----
    nb = lat.nn1[i0]
    # B2 "square": the predicted ground state of 2Cu + 2In, in which each Cu is a
    # first neighbour of each In while the like pairs are NOT first neighbours.
    # That gives four Cu-In bonds and no homovalent bonds, hence
    # E_c = 4*lambda*sum(dq^2) + 2*lambda*sum(dq_i dq_j) = 16*lambda - 8*lambda
    #     = 8*lambda, the value quoted in the paper. Found by search rather than
    # by hand, because an arbitrary choice of neighbours gives 14*lambda instead.
    square = None
    for b in lat.nn2[i0]:                       # b at second-neighbour separation
        common = [c for c in lat.nn1[i0] if c in list(lat.nn1[b])]
        for ci in range(len(common)):
            for cj in range(ci + 1, len(common)):
                c, d = int(common[ci]), int(common[cj])
                if d in list(lat.nn1[c]):       # the two In must not be neighbours
                    continue
                square = (i0, int(b), c, d)
                break
            if square:
                break
        if square:
            break
    assert square is not None, "no square configuration found"
    s = np.zeros(lat.N, dtype=int)
    s[square[0]] = 1; s[square[1]] = 1        # the two Cu, second neighbours
    s[square[2]] = 2; s[square[3]] = 2        # the two In, second neighbours
    emit("B_square", s, "two Cu each 1NN of two In, no like-pair contacts "
                        "(predicted ground state, 8*lambda)")

    # B1: two well-separated 1NN Cu-In pairs
    s = np.zeros(lat.N, dtype=int)
    s[i0] = 1; s[lat.nn1[i0, 0]] = 2
    j0 = shell_of(lat, i0, 0)
    s[j0] = 1; s[lat.nn1[j0, 0]] = 2
    emit("B_two_pairs", s, "two isolated 1NN Cu-In pairs")
    # B4: a Cu-Cu pair and an In-In pair, mutually distant
    s = np.zeros(lat.N, dtype=int)
    s[i0] = 1; s[lat.nn1[i0, 0]] = 1
    s[j0] = 2; s[lat.nn1[j0, 0]] = 2
    emit("B_like_pairs", s, "Cu-Cu and In-In first-neighbour pairs, separated")
    # B6: all four maximally separated
    s = np.zeros(lat.N, dtype=int)
    picks = [i0] + far_sites(lat, i0, 3)
    for p, sp in zip(picks, (1, 1, 2, 2)):
        s[p] = sp
    emit("B_all_separated", s, "two Cu and two In all separated (reference for set B)")

    # ---- SET C: vacancy plus two In ----
    v = i0
    an_v = lat.an_of_cat[v]
    # C1: the two In share different anion tetrahedra with the vacancy and are
    #     not first neighbours of each other
    cands = [j for j in lat.nn1[v]]
    pick = []
    used_an = set()
    for j in cands:
        a = int(np.intersect1d(lat.an_of_cat[v], lat.an_of_cat[j])[0])
        if a in used_an:
            continue
        if any(j in lat.nn1[p] for p in pick):
            continue
        pick.append(j); used_an.add(a)
        if len(pick) == 2:
            break
    s = np.zeros(lat.N, dtype=int); s[v] = 3
    for j in pick:
        s[j] = 2
    emit("C_split", s, "V_Zn with two In on different S tetrahedra (predicted GS)")
    # C4: all three separated
    s = np.zeros(lat.N, dtype=int)
    f2 = far_sites(lat, i0, 2)
    s[i0] = 3; s[f2[0]] = 2; s[f2[1]] = 2
    emit("C_separated", s, "V_Zn and two In all separated (reference for set C)")

    # ---- INCAR and KPOINTS, tuned for a memory-limited workstation ----
    incar = """SYSTEM = ZnS supercell, Cu-In substitution

# --- parallelisation and memory ---
# On 8 physical cores with ~13 GB usable, the binding constraint is memory,
# not speed. Three settings matter:
#   KPAR = 1     KPAR replicates the wavefunction per k-point group and would
#                double the resident set. Do not raise it on this machine.
#   LREAL = Auto real-space projectors; at 216 atoms reciprocal-space
#                projection is prohibitive in both memory and time.
#   LWAVE/LCHARG single WAVECAR here is several GB and is never read back,
#                since only total energies are needed.
NCORE  = 4
KPAR   = 1
LREAL  = Auto
LWAVE  = .FALSE.
LCHARG = .FALSE.

# --- electronic ---
ENCUT  = 500
PREC   = Normal
EDIFF  = 1E-6
ALGO   = Normal
ISMEAR = 0
SIGMA  = 0.05
NELM   = 120

# --- ionic relaxation at fixed cell (protocol R1) ---
IBRION = 2
ISIF   = 2
NSW    = 60
EDIFFG = -1E-2

# --- Hubbard U.  Starting values only; see the campaign design.  Order of
# LDAUL/LDAUU/LDAUJ must match the species order of the POSCAR, which this
# generator writes as Zn Cu In S (species absent from a given structure are
# omitted from that POSCAR, so these lines must be trimmed to match).
LDAU     = .TRUE.
LDAUTYPE = 2
LDAUL    =    2    2    2   -1
LDAUU    =  8.0  5.0  6.0  0.0
LDAUJ    =  0.0  0.0  0.0  0.0
LMAXMIX  = 4
"""
    open(os.path.join(_DFT, "INCAR"), "w").write(incar)
    open(os.path.join(_DFT, "KPOINTS"), "w").write(
        "Gamma-centred 2x2x2\n0\nGamma\n2 2 2\n0 0 0\n")

    with open(os.path.join(_DFT, "manifest.json"), "w") as f:
        json.dump(dict(a0=A0, ncell=NCELL, n_cation_sites=int(lat.N),
                       structures=manifest), f, indent=2)

    print(f"wrote {len(manifest)} POSCAR files to\n  {_DFT}")
    print(f"{'name':<18} {'Cu':>3} {'In':>3} {'vac':>4} {'sum dQ^2':>10} "
          f"{'CuIn':>5} {'CuCu':>5} {'InIn':>5}")
    for m in manifest:
        print(f"{m['name']:<18} {m['n_Cu']:3d} {m['n_In']:3d} {m['n_vac']:4d} "
              f"{m['sum_dQ2']:10.1f} {m['n_CuIn']:5.0f} {m['n_CuCu']:5.0f} "
              f"{m['n_InIn']:5.0f}")
    # ---- crude memory estimate, so the queue is not started blind ----
    # Number of plane waves inside the cutoff sphere:
    #   E_cut = hbar^2 k_c^2 / 2m  ->  k_c = sqrt(2 m E_cut)/hbar
    #   in atomic-friendly units, k_c [1/A] = sqrt(E_cut[eV]) / 3.81, since
    #   hbar^2/2m = 3.81 eV A^2.  N_pw = V k_c^3 / (6 pi^2).
    n_at = 2 * lat.N
    n_val = lat.N * 12 + lat.N * 6                    # Zn 3d10 4s2, S 3s2 3p4
    nbands = int(0.65 * n_val)
    vol = (A0 * NCELL) ** 3                            # Angstrom^3
    k_c = np.sqrt(500.0 / 3.81)                        # 1/Angstrom
    npw = int(vol * k_c ** 3 / (6 * np.pi ** 2))
    nk = 4                                             # irreducible, 2x2x2 Gamma-centred
    gb = nbands * npw * 16 * nk / 1e9                  # complex double
    gb_gamma = nbands * npw * 8 / 1e9      # Gamma-only: real wavefunction
    print(f"\nWrote INCAR and KPOINTS alongside the structures.")
    print(f"\nMEMORY ESTIMATE for one job ({n_at} atoms, NBANDS ~ {nbands}, "
          f"{npw:,} plane waves):")
    print(f"  2x2x2 Gamma-centred ({nk} irreducible k-points): "
          f"wavefunction ~ {gb:.1f} GB, peak {1.5*gb:.0f}-{2*gb:.0f} GB")
    print(f"  Gamma-only (vasp_gam, real arithmetic):          "
          f"wavefunction ~ {gb_gamma:.1f} GB, peak {1.5*gb_gamma:.0f}-"
          f"{2*gb_gamma:.0f} GB")
    print("""
  On a workstation with about 13 GB usable the 2x2x2 mesh is marginal at best.
  Two ways out, in order of preference:

  (a) Gamma-only with vasp_gam. The wavefunction is real rather than complex
      and there is one k-point instead of four, which is roughly an eightfold
      saving. At a 16.2 Angstrom cell the Brillouin zone is small and Gamma-only
      is defensible for total-energy DIFFERENCES between configurations that
      share a cell, which is all this campaign needs; the absolute energies are
      less well converged but they cancel. Set KPOINTS to a 1x1x1 Gamma mesh
      and check on ONE structure that the 1NN binding energy agrees with the
      2x2x2 value to better than about 10 meV before adopting it.

  (b) Keep 2x2x2 and halve the ranks: mpirun -np 4 with NCORE = 2. Slower, and
      it does not reduce the wavefunction itself, only the per-rank duplication
      of work arrays, so it may still not fit.

  Measure the first job before queueing the rest.
""")

    print("\nRun each with identical settings (see docs/step2.1_dft_design.md "
          "section 2):\n  PBEsol+U, 500 eV, 2x2x2 Gamma-centred, ions relaxed at "
          "fixed cell.\nThen: python dft_inputs.py --fit energies.csv")


# ----------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------
def fit(csv_path):
    import csv as _csv
    man = json.load(open(os.path.join(_DFT, "manifest.json")))["structures"]
    by = {m["name"]: m for m in man}
    E = {}
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            E[row["name"].strip()] = float(row["energy_eV"])
    missing = [n for n in by if n not in E]
    if missing:
        print("missing energies for:", ", ".join(missing))
        raise SystemExit(1)

    Eref = E["ref_pure"]
    rows, targets, names = [], [], []
    for m in man:
        if m["name"] == "ref_pure":
            continue
        # descriptors: sum dQ^2 (gives lambda), and the three bond counts (give J)
        rows.append([m["sum_dQ2"], m["n_CuIn"], m["n_CuCu"] + m["n_InIn"]])
        targets.append(E[m["name"]] - Eref)
        names.append(m["name"])
    Amat = np.asarray(rows, float)
    b = np.asarray(targets, float)
    # composition differs between sets, so fit within-set differences only:
    # subtract each set's own reference row
    refs = {"A_": "A_far", "B_": "B_all_separated", "C_": "C_separated"}
    keep, A2, b2 = [], [], []
    for i, n in enumerate(names):
        for pre, r in refs.items():
            if n.startswith(pre) and n != r:
                j = names.index(r)
                A2.append(Amat[i] - Amat[j]); b2.append(b[i] - b[j]); keep.append(n)
    A2, b2 = np.asarray(A2), np.asarray(b2)
    coef, *_ = np.linalg.lstsq(A2, b2, rcond=None)
    lam, j_cuin, j_like = coef
    pred = A2 @ coef
    rms = float(np.sqrt(np.mean((b2 - pred) ** 2)))

    print("least-squares fit of the lattice Hamiltonian to DFT energies")
    print(f"  lambda        = {lam:+.4f} eV     "
          f"(dielectric estimate 0.227-0.368 eV)")
    print(f"  J(Cu-In)      = {j_cuin:+.4f} eV     (Phase 1 assumed -0.10)")
    print(f"  J(like pairs) = {j_like:+.4f} eV     (Phase 1 assumed +0.05)")
    print(f"  RMS residual  = {1000*rms:.1f} meV over {len(b2)} differences")
    if rms > 0.030:
        print("  WARNING: residual exceeds 30 meV, comparable to k_B T at "
              "ore-forming\n           temperatures. The four-term Hamiltonian "
              "may be missing a term.")
    print(f"\n{'structure':<18} {'DFT (eV)':>10} {'model (eV)':>11} {'resid (meV)':>12}")
    for n, y, p in zip(keep, b2, pred):
        print(f"{n:<18} {y:10.4f} {p:11.4f} {1000*(y-p):12.1f}")
    lo, hi = 0.227, 0.368
    print(f"\nVERDICT on the central claim of Phase 1:")
    if lo <= lam <= hi:
        print(f"  lambda = {lam:.3f} eV lies inside the parameter-free dielectric "
              f"interval.\n  The charge-compensation term is confirmed as derived, "
              f"not fitted.")
    elif 0.15 <= lam <= 0.45:
        print(f"  lambda = {lam:.3f} eV is outside [{lo}, {hi}] but within the "
              f"range over which\n  Phase 1 showed the conclusions to be robust.")
    else:
        print(f"  lambda = {lam:.3f} eV falls outside the range Phase 1 tested. "
              f"The central\n  claim does not survive and the Hamiltonian needs "
              f"revision before use.")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--fit":
        fit(sys.argv[2])
    else:
        generate()
