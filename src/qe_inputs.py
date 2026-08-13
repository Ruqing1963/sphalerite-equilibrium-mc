# -*- coding: utf-8 -*-
"""
qe_inputs.py  --  Phase 2, Step 2.1 (Quantum ESPRESSO route)

Converts the twelve POSCAR structures written by dft_inputs.py into Quantum
ESPRESSO input files, and collects the resulting total energies into the CSV
that `dft_inputs.py --fit` consumes.

WHY QE RATHER THAN VASP

VASP is licensed software requiring a paid licence and compilation from source.
Quantum ESPRESSO is free, installs from conda in one command, and is entirely
adequate here. What this campaign needs is total-energy DIFFERENCES between
configurations sharing one cell and one composition; those differences are far
less sensitive to the choice of code, pseudopotentials or exchange-correlation
implementation than absolute energies are, because the systematic errors cancel
between structures. The absolute formation energies would be another matter,
but they are not what the lattice Hamiltonian is fitted to.

INSTALLATION

    conda install -c conda-forge qe

Pseudopotentials: download the SSSP efficiency library (free, curated) from
https://www.materialscloud.org/discover/sssp and place the UPF files in a
directory, or use PSlibrary. The files needed are for Zn, Cu, In and S. Set
PSEUDO_DIR below to that directory.

MEMORY

At 216 atoms the wavefunction is the binding constraint on a 16 GB machine, as
it would be with VASP. The settings below use a Gamma-only k-point sampling,
for which QE stores real rather than complex wavefunctions, giving roughly an
eightfold saving over a 2x2x2 mesh. That choice must be VERIFIED, not assumed:
run the two-structure test described in `check_gamma.md` before committing to
the full set.

USAGE

    python qe_inputs.py                    # write qe/*.in
    python qe_inputs.py --collect          # gather energies into energies.csv
"""

import os
import re
import sys
import glob

try:
    from ase.io import read, write
except ModuleNotFoundError:
    print("This script needs ASE:  pip install ase")
    raise SystemExit(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DFT = os.path.join(_HERE, os.pardir, "dft")
_QE = os.path.join(_HERE, os.pardir, "qe")

# ---------------------------------------------------------------- settings
#
# QE VERSION. Ubuntu 24.04 ships Quantum ESPRESSO 6.7, which uses the OLD
# Hubbard syntax: lda_plus_u and Hubbard_U(i) inside &SYSTEM, indexed by the
# position of the species in ATOMIC_SPECIES. Version 7.x replaced this with a
# separate HUBBARD card indexed by species NAME. The two are not compatible and
# a 7.x input fails on 6.7. Set this to match your installation; check with
#     pw.x -h | head -3
QE_VERSION = 6.7

PSEUDO_DIR = "./pseudo"
# SSSP v2.0 efficiency, PBE. Filenames as downloaded; confirm with `ls pseudo/`.
PSEUDOPOTENTIALS = {
    "Zn": "Zn.paw.z_12.atompaw.jth.v1.1-std.upf",
    "Cu": "Cu.paw.pbe.z_11.ld1.psl.v1.0.0-low.upf",
    "In": "In.us.pbe.z_13.ld1.psl.v0.2.2.upf",
    "S":  "S.nc.pbe.z_6.oncvpsp4.spms.v1.upf",
}

# Functional. SSSP supplies PBE and PBEsol sets, but the library's own download
# page states that the PBEsol pseudopotentials were never tested under the SSSP
# protocol and their correctness is not guaranteed. We therefore use the tested
# PBE set. The campaign measures total-energy DIFFERENCES between structures
# sharing one cell and one composition, for which the choice of functional is a
# second-order effect: the systematic error largely cancels.
FUNCTIONAL = "pbe"

# Cutoffs. SSSP publishes a recommended pair per element; for a multi-element
# cell the MAXIMUM over the elements present must be used, not an average.
#   Zn 30/240   Cu 55/165   In 50/200   S 30/120   (SSSP v2.0 efficiency, PBE)
# Note that Zn's PAW dataset demands a high charge-density cutoff even though
# its wavefunction cutoff is modest, and In's ultrasoft dataset likewise; the
# binding pair is therefore Cu for ecutwfc and Zn for ecutrho.
ECUTWFC = 55.0                   # Ry
ECUTRHO = 240.0                  # Ry
KPTS = (1, 1, 1)                 # Gamma-only; see the memory note above

HUBBARD_U = {"Zn": 8.0, "Cu": 5.0, "In": 6.0}
HUBBARD_ORBITAL = {"Zn": "3d", "Cu": "3d", "In": "4d"}   # 7.x only

NAMES = ["ref_pure", "A_1NN", "A_2NN", "A_3NN", "A_4NN", "A_far",
         "B_square", "B_two_pairs", "B_like_pairs", "B_all_separated",
         "C_split", "C_separated"]


def hubbard_lines_67(species):
    """
    QE 6.7: Hubbard U goes inside &SYSTEM, indexed by the ORDER of the species
    in ATOMIC_SPECIES. Because ASE writes that list in the order the species
    first appear, the index must be computed from the same ordering rather than
    assumed - which is the same trap as trimming LDAUU by hand in VASP, and the
    reason this is generated rather than written out.
    """
    out = []
    for i, s in enumerate(species, start=1):
        if s in HUBBARD_U:
            out.append(f"   Hubbard_U({i})   = {HUBBARD_U[s]:.4f}")
    if not out:
        return ""
    return "   lda_plus_u       = .true.\n   lda_plus_u_kind  = 0\n" + "\n".join(out)


def hubbard_card_7x(species):
    """QE 7.x: a separate HUBBARD card, indexed by species name."""
    lines = ["HUBBARD (ortho-atomic)"]
    for s in species:
        if s in HUBBARD_U:
            lines.append(f"U {s}-{HUBBARD_ORBITAL[s]} {HUBBARD_U[s]:.4f}")
    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


def generate():
    os.makedirs(_QE, exist_ok=True)
    made = []
    for name in NAMES:
        src = os.path.join(_DFT, f"POSCAR_{name}")
        if not os.path.exists(src):
            print(f"  missing {src}; run dft_inputs.py first")
            continue
        atoms = read(src, format="vasp")
        species = sorted(set(atoms.get_chemical_symbols()))
        pseudos = {s: PSEUDOPOTENTIALS[s] for s in species}

        path = os.path.join(_QE, f"{name}.in")
        write(path, atoms, format="espresso-in",
              pseudopotentials=pseudos, kpts=KPTS,
              input_data={
                  "control": {
                      "calculation": "relax",
                      "prefix": name,
                      "outdir": f"./tmp_{name}",
                      "pseudo_dir": PSEUDO_DIR,
                      "tprnfor": True,
                      "etot_conv_thr": 1.0e-5,
                      "forc_conv_thr": 1.0e-3,
                      "disk_io": "none",      # do not write the wavefunction
                  },
                  "system": {
                      "ecutwfc": ECUTWFC,
                      "ecutrho": ECUTRHO,
                      "occupations": "smearing",
                      "smearing": "gaussian",
                      "degauss": 0.005,
                      "input_dft": FUNCTIONAL,
                  },
                  "electrons": {
                      "conv_thr": 1.0e-8,
                      "mixing_beta": 0.3,     # conservative; 216-atom cells
                      "electron_maxstep": 200,
                  },
                  "ions": {"ion_dynamics": "bfgs"},
              })
        # Three fixes to what ASE writes.
        #
        # (1) ASE emits empty &FCP and &RISM namelists, which QE 6.7 rejects.
        # (2) For Gamma-only sampling ASE writes "K_POINTS automatic / 1 1 1
        #     0 0 0". That is NOT the same as "K_POINTS gamma": the automatic
        #     form makes QE store complex wavefunctions, while the gamma form
        #     uses real arithmetic and halves both memory and time. On a 12 GB
        #     machine that difference decides whether the job runs at all.
        # (3) The Hubbard specification, whose syntax depends on the QE version.
        txt = open(path).read()
        for dead in ("&FCP\n/\n", "&RISM\n/\n"):
            txt = txt.replace(dead, "")
        if tuple(KPTS) == (1, 1, 1):
            txt = re.sub(r"K_POINTS automatic\n[^\n]*\n", "K_POINTS gamma\n", txt)

        # species order exactly as written in ATOMIC_SPECIES
        block = txt.split("ATOMIC_SPECIES")[1].split("\n\n")[0]
        order = [ln.split()[0] for ln in block.strip().split("\n") if ln.split()]

        if QE_VERSION < 7.0:
            hub = hubbard_lines_67(order)
            if hub:
                txt = txt.replace("&SYSTEM\n", "&SYSTEM\n" + hub + "\n", 1)
        else:
            card = hubbard_card_7x(order)
            if card:
                txt = txt.rstrip() + "\n\n" + card
        open(path, "w").write(txt)

        made.append((name, order))
        print(f"  {name:<18} species {' '.join(order)}  "
              f"(Hubbard on {', '.join(s for s in order if s in HUBBARD_U)})")
    print(f"\nwrote {len(made)} input files to\n  {_QE}")
    print(f"""
BEFORE RUNNING THE FULL SET

  1. Set PSEUDO_DIR at the top of this script and download the four UPF files.
  2. Verify that Gamma-only sampling is adequate, using the two structures that
     carry the sharpest signal:

       mpirun -np 8 pw.x -nk 1 -in A_1NN.in  > A_1NN.out
       mpirun -np 8 pw.x -nk 1 -in A_far.in  > A_far.out

     then repeat both with KPTS = (2, 2, 2) in this script. The binding energy
     E(1NN) - E(far) should agree between the two settings to about 10 meV. If
     it does, use Gamma-only for the remaining ten structures; if not, the
     2x2x2 mesh is required and the memory ceiling must be respected by running
     on four ranks instead of eight.

  3. Queue the rest. Use -nk 1: k-point parallelisation replicates the
     wavefunction across pools and is the wrong knob on a memory-limited
     machine, exactly as KPAR is in VASP.
""")


def collect():
    rows, bad = [], []
    print(f"{'structure':<18} {'E (eV)':>14}   status")
    for name in NAMES:
        out = os.path.join(_QE, f"{name}.out")
        if not os.path.exists(out):
            print(f"{name:<18} {'--':>14}   no output file")
            bad.append(name)
            continue
        txt = open(out, errors="ignore").read()
        # final total energy, in Ry
        e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)\s+Ry", txt, re.M)
        if not e:
            print(f"{name:<18} {'--':>14}   no total energy (job may have died)")
            bad.append(name)
            continue
        ev = float(e[-1]) * 13.605693122994
        conv = ("End of BFGS Geometry Optimization" in txt
                or "Final energy" in txt
                or "convergence has been achieved" in txt.split("Final")[-1])
        note = "OK" if conv else "ionic relaxation NOT converged"
        print(f"{name:<18} {ev:14.6f}   {note}")
        if conv:
            rows.append((name, ev))
        else:
            bad.append(name)

    if bad:
        print(f"\n{len(bad)} job(s) need attention: {', '.join(bad)}")
        print("Not writing energies.csv; an unconverged energy in a least-squares")
        print("fit biases every parameter, which is worse than a missing point.")
        return
    dest = os.path.join(_HERE, "energies.csv")
    with open(dest, "w") as f:
        f.write("name,energy_eV\n")
        for n, e in rows:
            f.write(f"{n},{e:.6f}\n")
    print(f"\nWrote {dest} with {len(rows)} entries.")
    print("Next: python dft_inputs.py --fit energies.csv")


if __name__ == "__main__":
    if "--collect" in sys.argv:
        collect()
    else:
        generate()
