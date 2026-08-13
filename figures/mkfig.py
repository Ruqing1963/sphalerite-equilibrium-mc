import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
D = "/home/claude/phase2/data/"
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))

# (a) calibrated ladder at L=10
t = np.asarray(json.load(open(D+"ladder_L10.json"))["temps_K"])
ax[0].plot(np.arange(len(t)), t, "o-", ms=3.2, lw=1.1, color="#c0392b")
ax[0].axhspan(573, 811, color="#f1c40f", alpha=0.20, lw=0)
ax[0].axhspan(2400, 3000, color="#f1c40f", alpha=0.20, lw=0)
ax[0].set_xlabel("replica index"); ax[0].set_ylabel("temperature (K)")
ax[0].set_title("(a) adaptively calibrated ladder", fontsize=8.5, loc="left")
ax[0].grid(alpha=0.25, lw=0.5)
ax[0].text(0.97, 0.06, f"$M$ = {len(t)}", transform=ax[0].transAxes,
           ha="right", fontsize=8)

# (b) sigma_E vs N  (disordered phase)
d = json.load(open(D+"sigma_E_disordered.json"))
for T, c, m in ((3500, "#2980b9", "o"), (5000, "#7f8c8d", "s")):
    ks = [k for k in d if k.endswith(f"T{T}") and isinstance(d[k], dict)]
    N = np.array([d[k]["N"] for k in ks], float)
    S = np.array([d[k]["sigma"] for k in ks]); o = np.argsort(N)
    p = np.polyfit(np.log(N[o]), np.log(S[o]), 1)[0]
    ax[1].loglog(N[o], S[o], m+"-", ms=3.5, lw=1.1, color=c,
                 label=f"{T} K:  $N^{{{p:+.3f}}}$")
Nr = np.array([4e3, 5e5])
ax[1].loglog(Nr, 0.083*np.sqrt(Nr), "k--", lw=0.9, label=r"$N^{1/2}$ (theory)")
ax[1].set_xlabel("$N$ (cation sites)"); ax[1].set_ylabel(r"$\sigma_E$ (eV)")
ax[1].set_title(r"(b) energy fluctuation vs box size", fontsize=8.5, loc="left")
ax[1].legend(fontsize=7); ax[1].grid(alpha=0.25, lw=0.5, which="both")

# (c) round trips vs exchange interval / ladder top
r = json.load(open(D+"roundtrip_tuning.json"))["runs"]
items = sorted(r, key=lambda x: x["round_trips"])
lbl = [(f"spc={x['spc']}" if x["kind"] == "spc"
        else f"$T_{{hi}}${x['T_hi']:.0f}") + f"\n$M$={x['M']}" for x in items]
val = [x["round_trips"] for x in items]
col = ["#7f8c8d" if x["kind"] == "spc" else "#c0392b" for x in items]
bars = ax[2].bar(range(len(val)), val, color=col, width=0.62)
for i, v in enumerate(val):
    ax[2].text(i, v + 0.35, "0" if v == 0 else str(v), ha="center", fontsize=7.5)
ax[2].set_xticks(range(len(val))); ax[2].set_xticklabels(lbl, fontsize=6.8)
ax[2].set_ylabel("replica round trips"); ax[2].set_ylim(0, max(val) * 1.22)
ax[2].set_title("(c) round trips at fixed compute", fontsize=8.5, loc="left")
ax[2].grid(alpha=0.25, lw=0.5, axis="y")
from matplotlib.patches import Patch
ax[2].legend(handles=[Patch(color="#7f8c8d", label="exchange interval"),
                      Patch(color="#c0392b", label="ladder top")],
             fontsize=6.8, loc="upper left", framealpha=0.9)

fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"fig_methods.{e}", bbox_inches="tight")
print("wrote fig_methods.pdf / .png")

# ---------------- Figure: equilibrium condensation ----------------
fig2, ax2 = plt.subplots(1, 3, figsize=(9.6, 2.9))
c = json.load(open(D+"composition_scan.json"))["results"]
ks = sorted(c, key=float)
x = np.array([c[k]["x"] for k in ks]) * 100
R = np.array([c[k]["R"] for k in ks])
P = np.array([c[k]["P_Cu_given_In"] for k in ks])
frac = np.array([c[k]["largest_fraction"] for k in ks])

# (a) R explodes but P(Cu|In) is flat -- the normalisation artefact
a = ax2[0]
a.semilogx(x, R, "o-", ms=4, lw=1.2, color="#7f8c8d", label=r"$R_{\rm Cu-In}$")
a.semilogx(x, 1/(x/100), "k:", lw=0.9, label=r"ceiling $1/x_{\rm Cu}$")
a.set_yscale("log"); a.set_xlabel("$x_{\\rm Cu}=x_{\\rm In}$ (at.%)")
a.set_ylabel(r"$R_{\rm Cu-In}$", color="#7f8c8d")
a2 = a.twinx()
a2.semilogx(x, P, "s-", ms=4, lw=1.2, color="#c0392b")
a2.axhline(2/3, color="#c0392b", ls="--", lw=0.9)
a2.set_ylabel(r"$P({\rm Cu}\,|\,{\rm In})$", color="#c0392b")
a2.set_ylim(0, 0.75)
a2.text(0.42, 0.667, "ideal ordering", fontsize=6.5, color="#c0392b", va="bottom")
a.legend(fontsize=6.8, loc="lower left")
a.set_title("(a) enrichment factor vs shell composition", fontsize=8.5, loc="left")
a.grid(alpha=0.25, lw=0.5, which="both")

# (b) largest cluster vs the random-solution expectation, on a log axis so that
#     the four orders of magnitude between them are visible
b_ = ax2[1]
Nsol = np.array([c[k]["n_solute"] for k in ks], float)
b_.loglog(x, 100*frac, "o-", ms=5, lw=1.4, color="#c0392b", label="simulated")
xr = np.logspace(-1.1, 0.4, 60)
# largest cluster expected for a random solute distribution: at these dilutions
# essentially every solute is isolated, so the largest cluster is a few atoms
rand_frac = np.maximum(2.0 / (2*xr/100 * 32000), 1e-4)
b_.loglog(xr, 100*rand_frac, "k--", lw=1.0, label="random solid solution")
b_.set_xlabel("$x_{\\rm Cu}=x_{\\rm In}$ (at.%)")
b_.set_ylabel("solute in largest cluster (%)")
b_.set_ylim(0.05, 300); b_.legend(fontsize=6.8, loc="lower left")
b_.set_title("(b) condensation across the scan", fontsize=8.5, loc="left")
b_.grid(alpha=0.25, lw=0.5, which="both")
b_.text(0.13, 3, "four orders of\nmagnitude", fontsize=6.5, color="0.35")

# (c) the bracketing argument: show BOTH endpoints, not a midline, since the
#     equilibrium value lies between them and the midline is not meaningful
br = json.load(open(D+"condensation_bound.json"))
c_ = ax2[2]
for xv, col, off in ((0.0010, "#c0392b", -12), (0.0025, "#2980b9", +12)):
    rows = sorted([r for r in br if r["x"] == xv], key=lambda r: r["T"])
    T = np.array([r["T"] for r in rows], float) + off
    d_ = np.array([r["random"]["largest"] for r in rows]) * 100
    s_ = np.array([r["condensed"]["largest"] for r in rows]) * 100
    for i in range(len(T)):
        c_.plot([T[i], T[i]], [min(d_[i], s_[i]), max(d_[i], s_[i])],
                "-", lw=5, color=col, alpha=0.30, solid_capstyle="butt")
    c_.plot(T, d_, "v", ms=5, color=col, label=f"$x$ = {xv*100:.2f}%, from dispersed")
    c_.plot(T, s_, "^", ms=5, mfc="none", mew=1.2, color=col,
            label=f"$x$ = {xv*100:.2f}%, from condensed")
c_.axhline(100, color="0.4", lw=0.8, ls=":")
c_.set_xlabel("temperature (K)"); c_.set_ylabel("solute in largest cluster (%)")
c_.set_xlim(900, 1900); c_.set_ylim(50, 106)
c_.legend(fontsize=6.0, loc="lower left", ncol=1, framealpha=0.92)
c_.set_title("(c) equilibrium bracketed from both sides", fontsize=8.5, loc="left")
c_.grid(alpha=0.25, lw=0.5)

fig2.tight_layout()
for e in ("pdf", "png"):
    fig2.savefig(f"fig_condensation.{e}", bbox_inches="tight")
print("wrote fig_condensation.pdf / .png")
