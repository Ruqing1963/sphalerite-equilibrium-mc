# -*- coding: utf-8 -*-
"""
collect_energies.py -- gather VASP total energies into the CSV that
dft_inputs.py --fit expects.

Run from the directory holding the run_* folders:

    python collect_energies.py            # writes energies.csv
    python collect_energies.py --check    # report convergence only

The last "free  energy   TOTEN" in each OUTCAR is taken, which is the value
after the final ionic step. Jobs whose ionic relaxation did not converge, or
whose electronic loop hit NELM, are flagged rather than silently included: an
unconverged energy in a least-squares fit is worse than a missing one, because
it biases every parameter instead of merely widening the error.
"""
import os, re, sys, glob

NAMES = ["ref_pure", "A_1NN", "A_2NN", "A_3NN", "A_4NN", "A_far",
         "B_square", "B_two_pairs", "B_like_pairs", "B_all_separated",
         "C_split", "C_separated"]

def scan(d):
    out = os.path.join(d, "OUTCAR")
    if not os.path.exists(out):
        return None, "no OUTCAR"
    txt = open(out, errors="ignore").read()
    e = re.findall(r"free  energy   TOTEN\s*=\s*([-\d.]+)", txt)
    if not e:
        return None, "no TOTEN (job may have died)"
    nsteps = txt.count("aborting loop because EDIFF is reached")
    nelm_hit = "aborting loop EDIFF was not reached" in txt
    reached = "reached required accuracy" in txt
    note = []
    if nelm_hit:
        note.append("NELM hit in at least one step")
    if not reached:
        note.append("ionic relaxation NOT converged")
    return float(e[-1]), "; ".join(note) or f"OK ({nsteps} ionic steps)"

check = "--check" in sys.argv
rows, bad = [], []
print(f"{'structure':<18} {'E (eV)':>14}   status")
for n in NAMES:
    d = f"run_{n}"
    if not os.path.isdir(d):
        print(f"{n:<18} {'--':>14}   directory {d} not found"); bad.append(n); continue
    e, note = scan(d)
    print(f"{n:<18} {e if e is None else f'{e:14.6f}'}   {note}")
    if e is None or "NOT converged" in note:
        bad.append(n)
    else:
        rows.append((n, e))

if bad:
    print(f"\n{len(bad)} job(s) need attention: {', '.join(bad)}")
if not check:
    if len(rows) < len(NAMES):
        print("\nNot writing energies.csv while jobs are missing or unconverged.")
        print("Fix those first, or pass --check to inspect without writing.")
    else:
        with open("energies.csv", "w") as f:
            f.write("name,energy_eV\n")
            for n, e in rows:
                f.write(f"{n},{e:.6f}\n")
        print(f"\nWrote energies.csv with {len(rows)} entries.")
        print("Next: python dft_inputs.py --fit energies.csv")
