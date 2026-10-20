"""Figures 1 and 2 of the note (written to ../figures/). Usage: python make_figures.py [note2_ring note2_why]"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.5, "savefig.bbox": "tight", "font.family": "serif",
})


def note2_ring():
    """Note on Problem 2, Figure 1: the counterexample ring, and how each kind of driver reacts."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.transforms import Affine2D
    from scipy.signal import lti
    N = 22
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.55), gridspec_kw={"width_ratios": [1, 1.35]})
    ax1.set_aspect("equal"); ax1.axis("off")
    t = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(t), np.sin(t), color=GRID, lw=12, solid_capstyle="butt", zorder=0)
    for i in range(N):                       # car i follows car i - 1; travel is anticlockwise
        th = np.pi / 2 - 2 * np.pi * i / N
        car = FancyBboxPatch((-0.08, -0.045), 0.16, 0.09, boxstyle="round,pad=0,rounding_size=0.03",
                             fc=ORANGE if i == 0 else BLUE, ec="white", lw=0.6, zorder=2)
        car.set_transform(Affine2D().rotate(th + np.pi / 2).translate(np.cos(th), np.sin(th)) + ax1.transData)
        ax1.add_patch(car)
    r, a0, a1 = 0.68, np.radians(200), np.radians(250)
    ax1.add_patch(FancyArrowPatch((r * np.cos(a0), r * np.sin(a0)), (r * np.cos(a1), r * np.sin(a1)),
                                  connectionstyle="arc3,rad=0.2", arrowstyle="-|>", mutation_scale=9,
                                  color=INK2, lw=1.0))
    ax1.text(-0.27, -0.3, "travel", fontsize=7, color=INK2, ha="center")
    ax1.text(0, 1.2, r"sluggish driver, $a=1/8$", color=ORANGE, ha="center", fontsize=8.5)
    ax1.text(0, 0.16, "21 alert drivers", color=BLUE, ha="center", fontsize=8.5)
    ax1.text(0, -0.04, r"$a=20$", color=BLUE, ha="center", fontsize=8.5)
    ax1.text(0, -1.38, r"$N=22$ cars, headway $h=2$", color=INK, ha="center", fontsize=8)
    ax1.set_xlim(-1.3, 1.3); ax1.set_ylim(-1.5, 1.35)
    ax1.set_title("(a) the ring", loc="left")
    tt = np.linspace(0, 60, 1201)
    for a, col, lab in [(1 / 8, ORANGE, r"sluggish, $a=1/8$"), (20.0, BLUE, r"alert, $a=20$")]:
        _, y = lti([a], [1.0, a, a]).step(T=tt)        # V'(h) = 1: G(s) = a / (s^2 + a s + a)
        ax2.plot(tt, y, color=col, label=lab)
    ax2.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax2.set_xlabel("time since the car ahead sped up"); ax2.set_ylabel("change of speed")
    ax2.set_ylim(0, 1.7)
    ax2.legend(frameon=False, loc="lower right")
    ax2.set_title("(b) the car ahead speeds up by 1", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "note2_ring.pdf"))


def note2_why():
    """Note on Problem 2, Figure 2: the gain of a wave going once around the ring, and the
    instability in the nonlinear model (results/counterexample_sim.npz)."""
    a_s, a_f, N = 1 / 8, 20.0, 22
    rate = 0.0015209528                              # Re(lambda), certify_counterexample.py
    w = np.linspace(1e-4, 0.8, 8001)
    gain = lambda a: a / np.sqrt((a - w ** 2) ** 2 + (a * w) ** 2)   # |G_a(i w)| for V'(h) = 1
    gs, gf = gain(a_s), gain(a_f) ** (N - 1)
    loop = gs * gf
    inv_mean = (1 / a_s + (N - 1) / a_f) / N
    quad = np.exp(-0.5 * N * w ** 2 * (1 - 2 * inv_mean))           # what the harmonic-mean test sees
    print(f"max round-trip gain {loop.max():.4f} at w = {w[loop.argmax()]:.4f}; "
          f"gain > 1 for {w[loop > 1].min():.3f} < w < {w[loop > 1].max():.3f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax1.plot(w, gs, color=ORANGE, label="sluggish driver")
    ax1.plot(w, gf, color=BLUE, label="21 alert drivers")
    ax1.plot(w, loop, color=INK, label="round trip")
    ax1.plot(w, quad, color=INK2, ls=(0, (4, 2)), lw=1.2, label="harmonic-mean view")
    ax1.axhline(1, color=INK2, lw=0.8)
    ax1.fill_between(w, 1, loop, where=loop > 1, color=ORANGE, alpha=0.35, lw=0)
    ax1.set_yscale("log"); ax1.set_ylim(0.05, 5)
    ax1.set_xlabel(r"frequency $\omega$"); ax1.set_ylabel("amplitude gain")
    ax1.legend(frameon=False, loc="lower left", fontsize=7)
    ax1.set_title("(a) gain of a speed wave", loc="left")
    ins = ax1.inset_axes([0.6, 0.6, 0.38, 0.37])
    z = (w > 0.15) & (w < 0.45)
    ins.plot(w[z], loop[z], color=INK, lw=1.2)
    ins.plot(w[z], quad[z], color=INK2, ls=(0, (4, 2)), lw=1.0)
    ins.axhline(1, color=INK2, lw=0.7)
    ins.fill_between(w[z], 1, loop[z], where=loop[z] > 1, color=ORANGE, alpha=0.45, lw=0)
    ins.set_xlim(0.15, 0.45); ins.set_ylim(0.88, 1.12)
    ins.set_xticks([0.3, 0.4]); ins.set_yticks([0.9, 1.0, 1.1])
    ins.tick_params(labelsize=6, length=2)
    ins.grid(False)
    ins.text(0.312, 1.092, r"$+8\%$", fontsize=6.5, ha="center", color=INK)
    d = np.load(os.path.join(RES, "counterexample_sim.npz"))
    t, s = d["t"], d["spread"]
    m = s > 0
    ax2.plot(t[m], s[m], color=INK, lw=0.7, label="simulation")
    t0 = 2000.0
    s0 = s[(t > t0 - 12) & (t < t0 + 12)].max()                       # envelope over one period
    tl = np.linspace(700, 5300, 50)
    ax2.plot(tl, s0 * np.exp(rate * (tl - t0)), color=ORANGE, ls=(0, (4, 2)),
             label=r"$e^{0.00152\,t}$ (eigenvalue)")
    ax2.set_yscale("log"); ax2.set_ylim(1e-4, 3)
    ax2.text(7200, 1.5, "persistent wave", fontsize=7.5, ha="center", color=INK)
    ax2.set_xlabel("time"); ax2.set_ylabel("max speed $-$ min speed")
    ax2.legend(frameon=False, loc="lower right")
    ax2.set_title("(b) the full nonlinear model", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "note2_why.pdf"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["note2_ring", "note2_why"]
    for name in which:
        globals()[name]()
        print("wrote", name)
