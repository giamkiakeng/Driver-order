"""Problem 2(b): does the order of drivers matter once the dynamics are nonlinear?

N = 30 drivers at mean headway h = 1.3 (dense side, where the OVM ring is bistable):
9 sluggish drivers (a_s) and 21 alert ones (a_f). a_f is tuned so that the ring is linearly
stable with spectral abscissa -2e-4 -- the same for every order (Corollary 1). For two orders,
'clustered' (the 9 sluggish drivers in a row) and 'spread' (as evenly spaced as possible), find
by bisection the smallest kick at a given vehicle that ends in a persistent jam.
"""
import numpy as np
from ring import ring_matrix
from ovm_sim import simulate, Vprime, kicked_start, spread

N, h, ns = 30, 1.3, 9
Vp = Vprime(h)


def abscissa(a):
    return np.linalg.eigvals(ring_matrix(a * Vp, a, np.zeros(N))).real.max()


a_s = 1.6 * Vp                            # sluggish: well below the homogeneous threshold 2V'
lo, hi = 2 * Vp, 20 * Vp
for _ in range(60):                       # alert sensitivity with abscissa -2e-4
    mid = 0.5 * (lo + hi)
    lo, hi = (lo, mid) if abscissa(np.array([a_s] * ns + [mid] * (N - ns))) < -2e-4 else (mid, hi)
a_f = hi
base = np.array([a_s] * ns + [a_f] * (N - ns))
orders = {"clustered": base,
          "spread": np.array([a_s if (i * ns) % N < ns else a_f for i in range(N)])}
print(f"h={h}, V'={Vp:.4f}, a_s={a_s:.4f}, a_f={a_f:.4f}; harmonic mean {1 / np.mean(1 / base):.4f}, "
      f"arithmetic mean {base.mean():.4f}, 2V'={2 * Vp:.4f}")


def jams(a, pos, kick):
    _, v = simulate(a, N * h, T=5000, x0=kicked_start(N, h, kick, pos))
    return spread(v) > 1.0


for name, a in orders.items():
    out = []
    for pos in [0, 4, 8, 15, 22]:
        lo, hi = 0.3, 1.3
        for _ in range(7):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if jams(a, pos, mid) else (mid, hi)
        kind = "sluggish" if a[pos] == a_s else "alert"
        out.append(f"vehicle {pos} ({kind}): {0.5 * (lo + hi):.3f}")
    print(f"{name:9s} abscissa {abscissa(a):+.2e}; critical kick -> " + "; ".join(out), flush=True)
