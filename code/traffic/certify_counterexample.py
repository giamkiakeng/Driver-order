"""Note on Problem 2: an exact certificate that the long-wave (harmonic-mean) condition is not sufficient.

Ring of N = 22 OVM drivers at headway h = 2, where V(h) = tanh(h - 2) + tanh 2 has V'(h) = 1.
One sluggish driver (a = 1/8) and 21 alert drivers (a = 20). The long-wave condition
<1/a> V'(h) <= 1/2 (harmonic mean H >= 2V') holds: <1/a> = (8 + 21/20)/22 = 181/440 < 1/2.

Linearised characteristic polynomial (any order of the drivers):
    P(lambda) = d_s d_f^21 - n_s n_f^21,  d = lambda^2 + a lambda + a V',  n = a V'.
P(0) = 0 (conserved total headway); Q = P / lambda has degree 2N - 1 = 43.
We count the roots of Q in the open right half-plane with the Routh array in exact rational
arithmetic (Fractions): the number of sign changes in the first column equals that count,
provided no first-column entry vanishes (checked). The unstable root is then refined with
mpmath.findroot in 50-digit arithmetic, starting from the float64 eigenvalue of the ring matrix.
"""
from fractions import Fraction as Fr
import numpy as np
import mpmath as mp
from ring import ring_matrix

N, Vp = 22, Fr(1)
a_s, a_f = Fr(1, 8), Fr(20)
a = [a_s] + [a_f] * (N - 1)


def polymul(p, q):                       # coefficient lists, highest degree first
    r = [Fr(0)] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            r[i + j] += x * y
    return r


d, n = [Fr(1)], [Fr(1)]
for ai in a:
    d = polymul(d, [Fr(1), ai, ai * Vp])
    n = polymul(n, [ai * Vp])
P = [x for x in d]
for k, c in enumerate(n):                  # P = d - n (n is a constant here since alpha_3 = 0)
    P[len(P) - len(n) + k] -= c
assert P[-1] == 0, "lambda = 0 must be a root"
Q = P[:-1]                                 # P / lambda
assert len(Q) - 1 == 2 * N - 1


def routh_sign_changes(c):
    rows = [c[0::2], c[1::2]]
    width = len(rows[0])
    rows = [r + [Fr(0)] * (width - len(r)) for r in rows]
    for _ in range(len(c) - 2):
        r1, r2 = rows[-2], rows[-1]
        if r2[0] == 0:
            raise RuntimeError("zero pivot in the Routh array (singular case)")
        new = [(r2[0] * r1[j + 1] - r1[0] * r2[j + 1]) / r2[0] for j in range(width - 1)] + [Fr(0)]
        rows.append(new)
    first = [r[0] for r in rows]
    if any(x == 0 for x in first):
        raise RuntimeError("zero in the first column")
    return sum(1 for x, y in zip(first, first[1:]) if (x > 0) != (y > 0)), first


inv_mean = sum(1 / ai for ai in a) / N
print(f"N = {N}, a_s = {a_s}, a_f = {a_f}, V'(h) = {Vp}")
print(f"<1/a> = {inv_mean} = {float(inv_mean):.6f};  long-wave condition <1/a> V' <= 1/2: {inv_mean * Vp <= Fr(1, 2)}")
print(f"harmonic mean H = {1 / inv_mean} = {float(1 / inv_mean):.6f}  vs  2V' = {2 * Vp}")
changes, first = routh_sign_changes(Q)
print(f"Routh array of P/lambda (degree {len(Q) - 1}): {changes} sign changes in the first column "
      f"-> {changes} roots with Re > 0 (exact rational arithmetic)")

# the unstable root, refined in 50-digit arithmetic
mp.mp.dps = 50
A = [mp.mpf(ai.numerator) / ai.denominator for ai in a]
ev = np.linalg.eigvals(ring_matrix(np.array([float(x) for x in a]), np.array([float(x) for x in a]), np.zeros(N)))
z0 = ev[np.argmax(ev.real)]
Qm = [mp.mpf(c.numerator) / c.denominator for c in Q]
f = lambda z: mp.polyval(Qm, z)
z = mp.findroot(f, mp.mpc(z0.real, abs(z0.imag)))
print(f"unstable root of P/lambda (50 digits, mpmath.findroot): {mp.nstr(z, 20)}")
print(f"residual |P(z)/z| = {mp.nstr(abs(f(z)), 3)};  eigensolver (float64) spectral abscissa {z0.real:+.6e}")
G = lambda ai, zz: ai * 1 / (zz ** 2 + ai * zz + ai)
loop = G(A[0], z) * G(A[1], z) ** (N - 1)
print(f"check |G_s(z) G_f(z)^21 - 1| = {mp.nstr(abs(loop - 1), 3)}")

# Large-N view of the same two driver types (fraction f of sluggish drivers), HPX-type criterion:
# the ring is stable for every N if sup_w Phi_f(w) <= 0 and unstable for large N if sup_w Phi_f(w) > 0,
# where Phi_f(w) = f log|G_s(iw)| + (1 - f) log|G_f(iw)|.
w = np.linspace(1e-5, 3, 600001)
lg = lambda ai, ww: np.log(ai) - 0.5 * np.log((ai - ww ** 2) ** 2 + (ai * ww) ** 2)   # V' = 1
ls, lf = lg(float(a_s), w), lg(float(a_f), w)
phi22 = (ls + (N - 1) * lf) / N
k = phi22.argmax()
print(f"N = 22 ring: sup_w Phi_22(w) = {phi22[k]:+.4e} at w = {w[k]:.4f}")
lo, hi = 0.0, 1.0
for _ in range(50):
    mid = 0.5 * (lo + hi)
    lo, hi = (lo, mid) if (mid * ls + (1 - mid) * lf).max() > 1e-13 else (mid, hi)
f_harm = (0.5 - 1 / float(a_f)) / (1 / float(a_s) - 1 / float(a_f))
print(f"critical sluggish fraction: exact large-N criterion f* = {0.5 * (lo + hi):.5f}; "
      f"harmonic-mean law f_H = {f_harm:.5f}; this ring has f = 1/22 = {1 / 22:.5f}")
