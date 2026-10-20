"""Note on Problem 2: the spectrum ignores the order of the drivers, transient growth does not.

Same ring as order_matters.py (N = 30, h = 1.3, 9 sluggish drivers a_s = 1.6 V'(h), 21 alert drivers
tuned to spectral abscissa -2e-4). Full 2N-dimensional linearisation restricted to the invariant
hyperplane sum_i ds_i = 0 (fixed ring length) with an orthonormal basis, so that ||exp(A t)||_2 is the
Euclidean amplification of (ds, dv). Prints the peak transient growth for the clustered and the
spread order (identical spectra by Lemma 1 of the note).
"""
import numpy as np
from scipy.linalg import expm, null_space
from ovm_sim import Vprime

N, h, ns = 30, 1.3, 9
Vp = Vprime(h)
a_s = 1.6 * Vp


def restricted_matrix(a):
    A = np.zeros((2 * N, 2 * N))
    for i in range(N):                     # vehicle i follows vehicle i - 1
        A[i, N + (i - 1) % N] += 1.0
        A[i, N + i] -= 1.0
        A[N + i, i] = a[i] * Vp
        A[N + i, N + i] = -a[i]
    U = null_space(np.r_[np.ones(N), np.zeros(N)][None, :])
    return U.T @ A @ U


lo, hi = 2 * Vp, 20 * Vp
for _ in range(60):                        # alert sensitivity with abscissa -2e-4, as in order_matters.py
    mid = 0.5 * (lo + hi)
    a = np.array([a_s] * ns + [mid] * (N - ns))
    lo, hi = (lo, mid) if np.linalg.eigvals(restricted_matrix(a)).real.max() < -2e-4 else (mid, hi)
a_f = hi
orders = {"clustered": np.array([a_s] * ns + [a_f] * (N - ns)),
          "spread": np.array([a_s if (i * ns) % N < ns else a_f for i in range(N)])}
print(f"N={N}, h={h}, V'={Vp:.4f}, a_s={a_s:.4f}, a_f={a_f:.4f}")
for name, a in orders.items():
    M = restricted_matrix(a)
    ts = np.linspace(0, 60, 1201)
    g = np.array([np.linalg.norm(expm(M * t), 2) for t in ts])
    k = g.argmax()
    tf = np.linspace(max(ts[k] - 0.1, 0), ts[k] + 0.1, 201)
    gf = np.array([np.linalg.norm(expm(M * t), 2) for t in tf])
    late = max(np.linalg.norm(expm(M * t), 2) for t in np.linspace(60, 3000, 99))
    print(f"{name:9s} spectral abscissa {np.linalg.eigvals(M).real.max():+.3e}; "
          f"max_t ||exp(At)||_2 = {gf.max():.4f} at t = {tf[gf.argmax()]:.2f} (max over 60 <= t <= 3000: {late:.4f})")
