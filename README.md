# Physics-Informed Neural Networks (PINNs)

A PyTorch implementation of Physics-Informed Neural Networks for solving partial
differential equations using automatic differentiation — no labelled interior
data required, only the PDE itself and boundary/initial conditions.

This project is built around a reusable, equation-agnostic core: the network
architecture, derivative computation, and training loop are all independent of
which PDE is being solved. New equations are added by writing a small residual
function, not by modifying the model.

---

## Overview

| Equation | Relative L2 Error Achieved | Notes |
|---|---|---|
| Heat equation (with source) | [0.258]% | Baseline case |
| Heat equation (no source) | [0.618]% | Known PINN difficulty — see below |
| Burgers' equation (shock) | [2.37]% | Causal training applied |

Three PDEs were solved, each introducing a new challenge:
1. Heat equation with source, as a first test.
2. Heat equation no source - fast exponential decay — a known
   PINN difficulty, investigated via causal loss-weighting.
3. The nonlinear, shock-forming Burgers' equation — validated against the
   benchmark reference from Raissi et al.'s original PINN paper. Relative error improved
   significantly using causal training.

### Project structure
```
src/
  model.py          # PINN architecture, derivative calculation, total loss function
  model_causal.py   # model.py variant with causal loss-weighting
  data.py           # boundary/initial/collocation point generating functions
  train.py           # training loop (Adam + L-BFGS)
  train_causal.py    # train.py variant with causal loss-weighting
  evaluate.py        # exact solutions, relative error function
  visualize.py        # animation utilities (created using Claude AI)
notebooks/
  heat_with_source.ipynb
  heat_no_source.ipynb
  Burger's_equation.ipynb
data/
   burgers_shock.mat   # benchmark reference solution from Raissi et al.'s original PINN paper
```

---

## 1. Heat equation (with source)

$$u_t - u_{xx} = e^{-t}\left(\sin(\pi x) - \pi^2 \sin(\pi x)\right), \quad x \in [-1,1],\ t \in [0,1]$$
$$u(-1,t) = u(1,t) = 0, \qquad u(x,0) = \sin(\pi x)$$

Exact solution: $u(x,t) = e^{-t}\sin(\pi x)$.

This PDE served as the initial baseline. The function is smooth. The source term gives the solution a slower exponential decay.
Training converges reliably.

**Result:** 0.258% relative L2 error (Adam + L-BFGS, 10000 collocation points).

![Diffusion with source result](images/heat_source.png)

---

## 2. Diffusion equation (no source)

$$u_t = u_{xx}, \quad x \in [-1,1],\ t \in [0,1]$$
$$u(-1,t) = u(1,t) = 0, \qquad u(x,0) = \sin(\pi x)$$

Exact solution: $u(x,t) = e^{-\pi^2 t}\sin(\pi x)$ — decaying roughly 10x faster
than the source case.

This case exposed a known PINN difficulty: due to the fast exponential decay the optimiser
has little incentive to fit the later time points due their small magnitude. 
The error map below shows the relative error between the PINN and the exact solution. 

![Diffusion no-source error map](images/heat_no_source_error.png)

### Causal training investigation

**Causal training** following Wang et al.,
*"Respecting Causality for Training Physics-Informed Neural Networks."* the PDE loss
becomes weighted so that later time regions only contribute to the loss once earlier regions
have converged. Implemented by splitting the collocation points into bucketed time regions and multiplying 
each buckets loss by `exp(-ε · cumulative loss of earlier buckets)`.

**Finding:** causal training did not result in noticeable improvement for this equation
(0.718% vs 0.622%). This is most likely due to the heat equation solution being smooth with no discontinuities, 
as well as the already low error before testing.

---

## 3. Burgers' equation

$$u_t + u \cdot u_x = \nu\, u_{xx}, \quad x \in [-1,1],\ t \in [0,1]$$
$$u(-1,t) = u(1,t) = 0, \qquad u(x,0) = -\sin(\pi x)$$

($\nu = 0.01/\pi$ — the standard benchmark viscosity, producing a steep moving
shock.)

Burgers' equation has no closed-form solution; results are validated against
the computed benchmark reference solution from Raissi et al.'s original PINN paper
(`burgers_shock.mat`).

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

![Burgers' heatmap comparison](images/burgers.png)

![Burgers' shock animation](images/burgers_exact.gif)
![Burgers' shock animation](images/burgers_predicted.gif)
---

## Future work

- For each equation, testing over more seeds using different starting parameters,
and investigating standard deviation. 
- Extending to a 2D spatial domain (e.g. 2D heat equation)
- Navier-Stokes equations would be interesting to study using this PINN architecture

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.* Journal of
  Computational Physics.
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality for
  training physics-informed neural networks.*
