# Physics-Informed Neural Networks (PINNs)

A PyTorch implementation of Physics-Informed Neural Networks for solving partial
differential equations using automatic differentiation — no labelled interior
data required, only the PDE itself and boundary/initial conditions.

This project is built around a reusable, equation-agnostic core: the network
architecture, derivative computation, and training loop are all independent of
which PDE is being solved. New equations are added by writing a small residual
function, not by modifying the model.

**Repo:** [GitHub link]

---

## Overview

| Equation | Relative L2 Error | Notes |
|---|---|---|
| Heat equation (with source) | [X]% | Baseline case |
| Heat equation (no source) | [X]% | Known PINN failure mode — see below |
| Burgers' equation (shock) | [X]% ± [X]% | Causal training applied |

Three PDEs were solved, each introducing a new challenge:
1. A forced diffusion equation, as a baseline sanity check.
2. An unforced diffusion equation with fast exponential decay — a known
   PINN difficulty, investigated via causal loss-weighting.
3. The nonlinear, shock-forming Burgers' equation — validated against the
   benchmark reference from Raissi et al.'s original PINN paper, and improved
   significantly using causal training.

### Project structure
```
src/
  model.py          # PINN architecture, derivative computation, loss functions
  model_causal.py   # PINN variant with causal loss-weighting
  data.py           # boundary/initial/collocation point generation
  train.py           # training loop (Adam + L-BFGS)
  train_causal.py    # training loop with causal weighting
  evaluate.py        # exact solutions, relative error metric
  visualize.py        # animation utilities
notebooks/
  diffusion_with_source.ipynb
  diffusion_no_source.ipynb
  burgers_equation.ipynb
```

---

## 1. Diffusion equation (with source)

$$u_t - u_{xx} = e^{-t}\left(\sin(\pi x) - \pi^2 \sin(\pi x)\right), \quad x \in [-1,1],\ t \in [0,1]$$
$$u(-1,t) = u(1,t) = 0, \qquad u(x,0) = \sin(\pi x)$$

Exact solution: $u(x,t) = e^{-t}\sin(\pi x)$.

This PDE served as the initial baseline. The manufactured source term gives the
network a strong, spatially-varying training signal throughout the domain, and
training converges reliably.

**Result:** [X]% relative L2 error (Adam + L-BFGS, [N] collocation points).

![Diffusion with source result](images/diffusion_source_heatmap.png)

---

## 2. Diffusion equation (no source)

$$u_t = u_{xx}, \quad x \in [-1,1],\ t \in [0,1]$$
$$u(-1,t) = u(1,t) = 0, \qquad u(x,0) = \sin(\pi x)$$

Exact solution: $u(x,t) = e^{-\pi^2 t}\sin(\pi x)$ — decaying roughly 10x faster
than the source case.

This case exposed a known PINN failure mode: because the solution decays so
quickly, most of the time domain has near-zero target magnitude, giving the
optimizer little incentive to fit the late-time region accurately. The error
map below shows the network satisfying the loss well on average while failing
to respect the boundary condition and propagate the solution correctly at
later times.

![Diffusion no-source error map](images/diffusion_no_source_error_map.png)

### Causal training investigation

**Causal training** addresses this by weighting the PDE loss so that later
time-regions only contribute once earlier regions have converged — forcing
training to proceed sequentially through time, following Wang et al.,
*"Respecting Causality for Training Physics-Informed Neural Networks."*
Implemented here by bucketing collocation points by time and scaling each
bucket's loss by `exp(-ε · cumulative loss of earlier buckets)`.

**Finding:** causal training did not produce a reliable improvement over the
uniform-weighted baseline for this equation ([X]% vs [X]%). Diffusion is a
naturally smoothing PDE, and the baseline was already achieving low error by
the time causal training was tested — likely leaving little headroom for a
causality fix to improve on. A fixed (non-annealed) ε and hard bucket
boundaries, rather than the smoothly-varying weighting used in the original
method, may also have limited its effectiveness. Full investigation (N-
collocation, bucket-count, and ε sweeps) is in `notebooks/diffusion_no_source.ipynb`.

---

## 3. Burgers' equation

$$u_t + u \cdot u_x = \nu\, u_{xx}, \quad x \in [-1,1],\ t \in [0,1]$$
$$u(-1,t) = u(1,t) = 0, \qquad u(x,0) = -\sin(\pi x)$$

($\nu = 0.01/\pi$ — the standard benchmark viscosity, producing a steep moving
shock.)

Burgers' equation has no closed-form solution; results are validated against
the benchmark reference solution from Raissi et al.'s original PINN paper
(`burgers_shock.mat`), spectrally computed.

### Shock formation

The initial condition's odd symmetry about $x=0$ forces the shock to form
exactly at $x=0$, with a predicted formation time of $t \approx 1/\pi \approx 0.32$
for the inviscid case. The error map below shows both effects clearly: a
sharp vertical error band at $x=0$, and a wedge-shaped pattern below $t
\approx 0.32$ tracing the characteristic curves converging toward the shock.

**Baseline result:** [X]% relative L2 error (uniform collocation sampling,
[N] points).

### Causal training

Given Burgers' equation is the original benchmark case for causal training in
the literature — and, unlike diffusion, involves genuine nonlinear advection —
it was a stronger candidate for the method. Applying causal loss-weighting
improved the result substantially:

**Result with causal training:** [X]% ± [X]% relative L2 error (mean ± std
over [N] seeds, ε=[X], [N] buckets).

![Burgers' heatmap comparison](images/burgers_heatmap.png)

![Burgers' shock animation](images/burgers_shock_animation.gif)

### Other approaches tried

- **Truncated-normal collocation biasing** (concentrating points near $x=0$):
  did not improve results — the shock's true physical width (~$\nu$) is far
  narrower than any tested distribution spread.
- **Residual-adaptive refinement (RAR):** did not improve results, likely due
  to early-training residuals being an unreliable signal for where to add
  points before the network has learned anything useful to refine around.

---

## Limitations & future work

- Burgers' results show non-trivial run-to-run variance; reported figures are
  averaged over [N] seeds, but a more principled treatment (e.g. annealed ε,
  smooth time-weighting) would likely reduce this further.
- Extending to a 2D spatial domain (e.g. 2D heat equation) would better
  demonstrate PINNs' mesh-free advantage over classical solvers.
- A natural next step beyond this project is the Navier-Stokes equations,
  for which Burgers' equation (sharing the same nonlinear advection +
  diffusion structure) serves as a useful 1D stepping stone.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.* Journal of
  Computational Physics.
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality for
  training physics-informed neural networks.*
