"""Nonlinear ring-road simulator for the optimal velocity model (Bando et al. 1995).

dx_i/dt = v_i,  dv_i/dt = a_i (V(x_{i-1} - x_i) - v_i),  V(h) = tanh(h - 2) + tanh(2),
vehicle i follows vehicle i-1 and vehicle 0 follows vehicle N-1 across the seam (ring length L).
Arrays may carry a leading batch axis (independent replicas). Classical RK4; optional additive
velocity noise sigma dW (stochastic Heun).
"""
import numpy as np


def V(h):
    return np.tanh(h - 2.0) + np.tanh(2.0)


def Vprime(h):
    return 1.0 / np.cosh(h - 2.0) ** 2


def headways(x, L):
    h = np.roll(x, 1, axis=-1) - x
    h[..., 0] += L
    return h


def accel(x, v, a, L):
    return a * (V(headways(x, L)) - v)


def simulate(a, L, T, dt=0.02, x0=None, v0=None):
    """Deterministic RK4 from (x0, v0); default start is uniform flow."""
    a = np.asarray(a, float)
    N = a.shape[-1]
    h0 = L / N
    x = -h0 * np.arange(N) if x0 is None else np.array(x0, float)
    v = np.full(N, V(h0)) if v0 is None else np.array(v0, float)
    for _ in range(int(T / dt)):
        k1x, k1v = v, accel(x, v, a, L)
        k2x, k2v = v + 0.5 * dt * k1v, accel(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, a, L)
        k3x, k3v = v + 0.5 * dt * k2v, accel(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, a, L)
        k4x, k4v = v + dt * k3v, accel(x + dt * k3x, v + dt * k3v, a, L)
        x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x, v


def kicked_start(N, h, kick, pos=0):
    """Uniform flow with vehicle `pos` moved forward by kick * 0.9 * h (a large headway disturbance)."""
    x0 = -h * np.arange(N)
    x0[pos] += 0.9 * kick * h
    return x0


def spread(v):
    return v.max(-1) - v.min(-1)
