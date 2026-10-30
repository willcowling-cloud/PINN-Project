# Physics-Informed Neural Networks (PINNs)

A PyTorch implementation of a Physics-Informed Neural Network for solving partial
differential equations using automatic differentiation and no labelled interior
data required. Only the PDE itself and boundary/initial conditions are used to train
the Neural Network. 

The network architecture, derivative computation, and training loop are all independent of
which PDE is being solved. A new PDE can be added by writing a residual function as well
as any boundary/initial conditions. 

---

## Overview

| Equation | Relative L2 Error Achieved | Notes |
|---|---|---|
| Heat equation (with source) | 0.258% | Baseline case |
| Heat equation (no source) | 0.618% | Known PINN difficulty — see below |
| Burgers' equation (shock) | 5.65% | Causal training applied |

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

## 2. Heat equation (no source)

$$u_t = u_{xx}, \quad x \in [-1,1],\ t \in [0,1]$$
$$u(-1,t) = u(1,t) = 0, \qquad u(x,0) = \sin(\pi x)$$

Exact solution: $u(x,t) = e^{-\pi^2 t}\sin(\pi x)$ — decaying roughly 10x faster
than the source case.

This case exposed a known PINN difficulty: due to the fast exponential decay the optimiser
has little incentive to fit the later time points due their small magnitude. 
The error map below shows the relative error between the PINN and the exact solution. 

<p align="center">
  <img src="images/heat_no_source.png" height="300">
  <img src="images/heat_no_source_error.png" height="300">
</p>



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

($\nu = 0.01/\pi$ — the standard benchmark viscosity, producing a steep moving shock.)

Burgers' equation has no closed-form solution; results are validated against the computed benchmark reference solution from Raissi et al.'s original PINN paper (`burgers_shock.mat`).

### Shock formation

A steep gradient develops around $x=0$, at approximately $t=0.3$. This discontinuity is particularly difficult for PINNs to reproduce accurately, resulting in a large vertical band of prediction relative error.

### Baseline performance

Initial experiments using uniform random collocation point sampling produced a relative L2 error of 22.976%.

Further testing across five different collocation point seeds revealed large variations in the PINNs performance, with a mean relative L2 error of **27.48% ± 26.64%** (10,000 collocation points).

### Improving training performance

Given the difficulty of resolving the steep shock, I investigated two approaches to improving PINN performance:

**1. Latin Hypercube Sampling (LHS)**

I implemented LHS to even out the distribution of collocation points across the x-t domain. This reduced the mean relative L2 error from 27.48% to **17.14% ± 12.53%** across five seeds.

**2. Causal training**

I then implemented causal loss-weighting, using the same approach that I had attempted for the heat equation. This method shows more promise for fixing this PINN failure mode as it should force the PINN to accurately capture the smooth part of the solution before the shock wave forms.

Using causal training (ε = 0.2, 10 time buckets) reduced the mean relative L2 error to **5.65% ± 3.52%** across five seeds. The best result in this experiment was 2.26%.

### Comparing sampling strategies

Finally, I investigated whether combining LHS sampling with causal training would produce further improvements.

Interestingly, the combination increased the mean relative L2 error to **8.67% ± 4.17%**, compared with 5.65% using causal training with uniform random sampling alone.

These experiments demonstrate that collocation point placement has a substantial influence on PINN performance, particularly for solutions containing sharp gradients. While LHS improved standard PINN performance, causal training provided the largest overall improvement and combining the two approaches did not produce any additional benefit.

![Burgers' heatmap comparison](images/burgers.png)
![Burgers' heatmap comparison](images/burgers_error.png)

<p align="center">
  <img src="images/Burgers_exact.gif" width="45%">
  <img src="images/Burgers_predicted.gif" width="45%">
</p>

---

## Future work

- For each equation, investigating how to reduce training randomness between different seeds
  with different starting parameters. 
- Extending to a 2D spatial domain (e.g. 2D heat equation).
- Navier-Stokes equations would be interesting to study using this PINN architecture.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.* Journal of
  Computational Physics.
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality for
  training physics-informed neural networks.*
