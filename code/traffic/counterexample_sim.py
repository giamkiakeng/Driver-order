"""Note on Problem 2: the counterexample ring in the full nonlinear OVM.

N = 22 vehicles at headway h = 2 (V'(h) = 1); vehicle 0 is sluggish (a = 1/8), the other 21 are
alert (a = 20). The harmonic-mean condition holds, yet the uniform flow is linearly unstable with
growth rate 0.0015209528 (certify_counterexample.py). Starting from uniform flow with vehicle 0
moved forward by 1e-3, we integrate the nonlinear model, check that the velocity spread grows at
the predicted rate, and describe the state it saturates to. Output: the printed summary and
results/counterexample_sim.npz (used by make_figures.py).
"""
import os
import numpy as np
from ovm_sim import V, accel, spread

N, h = 22, 2.0
L = N * h
a = np.array([1 / 8] + [20.0] * (N - 1))
RATE = 0.0015209528                      # Re(lambda) from certify_counterexample.py
dt, T, every = 0.02, 9000.0, 25          # RK4 step; record every 0.5 time units

x = -h * np.arange(N)
x[0] += 1e-3
v = np.full(N, V(h))
ts, spreads, vs = [], [], []
for n in range(int(T / dt) + 1):
    if n % every == 0:
        ts.append(n * dt)
        spreads.append(spread(v))
        vs.append(v.copy())
    k1x, k1v = v, accel(x, v, a, L)
    k2x, k2v = v + 0.5 * dt * k1v, accel(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, a, L)
    k3x, k3v = v + 0.5 * dt * k2v, accel(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, a, L)
    k4x, k4v = v + dt * k3v, accel(x + dt * k3x, v + dt * k3v, a, L)
    x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
ts, spreads, vs = np.array(ts), np.array(spreads), np.array(vs)

# growth rate of the envelope in the linear regime (spread between 1e-3 and 1e-1)
sel = (spreads > 1e-3) & (spreads < 1e-1) & (ts > 200)
env_t, env_s = [], []
period = 2 * np.pi / 0.27563151
for t0 in np.arange(ts[sel].min(), ts[sel].max() - period, period):
    w = (ts >= t0) & (ts < t0 + period)
    env_t.append(ts[w].mean()); env_s.append(spreads[w].max())
slope, icpt = np.polyfit(env_t, np.log(env_s), 1)
print(f"linear regime {ts[sel].min():.0f} <= t <= {ts[sel].max():.0f}: fitted growth rate {slope:.6f} "
      f"(eigenvalue {RATE:.6f}, ratio {slope / RATE:.4f})")

late = ts >= T - 400
vl = vs[late]
print(f"final state (t in [{T - 400:.0f}, {T:.0f}]): velocity spread {spread(vs[-1]):.4f}; "
      f"min velocity {vl.min():.4f}, max velocity {vl.max():.4f} (uniform-flow velocity V(h) = {V(h):.4f})")
print(f"sluggish car: min {vl[:, 0].min():.4f}, max {vl[:, 0].max():.4f}; "
      f"alert cars: min {vl[:, 1:].min():.4f}, max {vl[:, 1:].max():.4f}")
s0 = vl[:, 0] - vl[:, 0].mean()
crossings = ts[late][1:][(s0[:-1] < 0) & (s0[1:] >= 0)]
if len(crossings) > 2:
    print(f"oscillation period of the sluggish car's velocity: {np.diff(crossings).mean():.2f} "
          f"(linear prediction 2 pi / 0.2756 = {period:.2f})")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "counterexample_sim.npz")
np.savez_compressed(out, t=ts, spread=spreads, v_late=vl, t_late=ts[late], rate=slope)
print("wrote results/counterexample_sim.npz")
