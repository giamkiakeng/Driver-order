# Supplementary material

**Does the order of drivers matter? A counterexample and an open problem for mixed drivers on a
ring road.** Kia Keng Giam, Temasek Junior College, Singapore.

This supplementary material contains the Python code that produces every number and both figures
in the note, and the output it produced. It is also available at
https://github.com/giamkiakeng/Driver-order. The code is released under the MIT license (see
`LICENSE`).

## Contents

| File | What it computes | Where it is used in the note | Time |
|---|---|---|---|
| `code/traffic/certify_counterexample.py` | Exact Routh–Hurwitz count of the unstable eigenvalues of the 22-car ring (rational arithmetic); the unstable eigenvalue to 50 digits; the exact large-ring threshold f* against the harmonic-mean threshold | Proposition 3, Remark 4, Section 3 | 1 s |
| `code/traffic/counterexample_sim.py` | The 22-car ring in the full nonlinear model: growth rate of a small disturbance and the traffic wave it saturates to; saves `results/counterexample_sim.npz` | Section 3, Figure 2(b) | 30 s |
| `code/traffic/order_transients.py` | Peak transient growth max_t ‖exp(At)‖₂ for the clustered and the spread order | Section 4 | 2 s |
| `code/traffic/order_matters.py` | Critical kick κ_c for the clustered and the spread order, by bisection | Table in Section 4 | 20–60 min |
| `code/make_figures.py` | Figures 1 and 2 (written to `figures/`) | Figures 1–2 | 5 s |
| `code/traffic/ring.py` | The linearized ring matrix (helper module) | | |
| `code/traffic/ovm_sim.py` | Optimal velocity model simulator, fourth-order Runge–Kutta (helper module) | | |
| `results/` | The output of the scripts above, as quoted in the note | | |
| `figures/` | Figures 1 and 2 as they appear in the note | | |

Conventions: car i follows car i − 1 (car 0 follows car N − 1); V(s) = tanh(s − 2) + tanh 2
(Bando et al. 1995); time and length are dimensionless.

## How to reproduce

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
PYTHON=venv/bin/python bash run_all.sh
```

`run_all.sh` first copies the shipped outputs to `results_shipped/`, then re-runs every script
(writing to `results/` and `figures/`), and finally reports, for each text output, whether the new
file is identical to the shipped one. The pinned packages need Python 3.12 or later (we used 3.13).
Every script is deterministic (it uses no random numbers); on another machine, floating-point
rounding can change the last printed digits.
