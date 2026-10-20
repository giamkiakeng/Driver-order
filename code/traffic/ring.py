"""Linearised single-lane ring road (Problems 2 and 3).

N vehicles, vehicle i follows vehicle i-1 (vehicle 0 follows vehicle N-1). Linearised about
uniform flow:  ds_i/dt = dv_{i-1} - dv_i,  dv_i/dt = A1_i ds_i - A2_i dv_i + A3_i dv_{i-1}.
The conserved total headway (sum ds_i = 0) is removed by ds_{N-1} = -sum_{i<N-1} ds_i,
leaving 2N - 1 states whose characteristic polynomial is
    P(lambda) / lambda,   P(lambda) = prod_i (lambda^2 + A2_i lambda + A1_i) - prod_i (A3_i lambda + A1_i).
"""
import numpy as np


def ring_matrix(A1, A2, A3):
    A1, A2, A3 = (np.asarray(x, float) for x in (A1, A2, A3))
    N = len(A1)
    M = np.zeros((2 * N - 1, 2 * N - 1))
    col_v = lambda i: (N - 1) + (i % N)
    for i in range(N - 1):
        M[i, col_v(i - 1)] += 1.0
        M[i, col_v(i)] -= 1.0
    for i in range(N):
        r = col_v(i)
        if i < N - 1:
            M[r, i] += A1[i]
        else:
            M[r, :N - 1] -= A1[i]
        M[r, col_v(i)] -= A2[i]
        M[r, col_v(i - 1)] += A3[i]
    return M


def char_poly_roots(A1, A2, A3):
    """Roots of P(lambda)/lambda from the product formula (for cross-checks, small N only)."""
    d = np.poly1d([1.0])
    n = np.poly1d([1.0])
    for a1, a2, a3 in zip(A1, A2, A3):
        d = d * np.poly1d([1.0, a2, a1])
        n = n * np.poly1d([a3, a1])
    P = d - n
    q, r = np.polydiv(P.coeffs, [1.0, 0.0])
    return np.roots(q)


def ovm_coeffs(a, Vp):
    """Bando et al. (1995) optimal velocity model dv/dt = a (V(s) - v): A1 = a V'(h), A2 = a, A3 = 0."""
    return a * Vp, a, 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 1) the product formula equals the matrix spectrum
    N = 7
    A1, A2, A3 = rng.uniform(0.5, 2, N), rng.uniform(0.5, 3, N), rng.uniform(0, 0.4, N)
    ev = np.sort_complex(np.linalg.eigvals(ring_matrix(A1, A2, A3)))
    pr = np.sort_complex(char_poly_roots(A1, A2, A3))
    print("max |matrix eigenvalues - roots of P(lambda)/lambda| =", np.abs(ev - pr).max())
    # 2) classical homogeneous OVM threshold a_c(N) = V'(h) (1 + cos(2 pi / N))
    Vp = 1.0
    for N in (5, 10, 22, 50):
        ac = Vp * (1 + np.cos(2 * np.pi / N))
        out = []
        for a in (0.99 * ac, 1.01 * ac):
            c = ovm_coeffs(a, Vp)
            out.append(np.linalg.eigvals(ring_matrix([c[0]] * N, [c[1]] * N, [c[2]] * N)).real.max())
        print(f"N={N:3d}: a_c={ac:.4f}; spectral abscissa at 0.99 a_c = {out[0]:+.2e}, at 1.01 a_c = {out[1]:+.2e}")
