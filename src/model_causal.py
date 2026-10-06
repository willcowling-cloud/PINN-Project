import torch
import torch.nn as nn
import torch.autograd as autograd

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = nn.MSELoss(reduction = 'mean')
        self.layer1 = nn.Linear(2,32)
        self.layer2 = nn.Linear(32,32)
        self.layer3 = nn.Linear(32,32)
        self.layer4 = nn.Linear(32,32)
        self.layer5 = nn.Linear(32,1)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        x = self.layer4(x)
        x = torch.tanh(x)
        x = self.layer5(x)
        return x

    def BCloss(self,x_bc, y_bc):
        BC_prediction = self.forward(x_bc)
        BC_loss = self.loss_function(BC_prediction,y_bc)
        return BC_loss

    def ICloss(self,x_ic, y_ic):
        IC_prediction = self.forward(x_ic)
        IC_loss = self.loss_function(IC_prediction,y_ic)
        return IC_loss

    def Calc_derivatives(self, u, g, var_names):
        derivatives = {}

        for index, name in enumerate(var_names):
            # Step 1: get the first derivative w.r.t. chosen column
            first = autograd.grad(u, g, torch.ones_like(u), retain_graph=True, create_graph=True)[0][:, [index]]
            derivatives[f'u_{name}'] = first

            # Step 2: differentiate the same column again, for 2nd derivative
            second = autograd.grad(first, g, torch.ones_like(first), retain_graph=True, create_graph=True)[0][:, [index]]
            derivatives[f'u_{name}{name}'] = second

        return derivatives

    def causal_PDEloss(self, col_points, residual_fn, var_names, n_buckets=10, eps=1.0):
        g = col_points.clone().requires_grad_(True)
        u = self.forward(g)
        derivs = self.Calc_derivatives(u, g, var_names)
        residual = residual_fn(g, u, derivs)
        # Square residual, to create loss at each point, don't need to track gradient for this
        pointwise_loss = residual**2

        t_vals = g[:, -1]
        bucket_edges = torch.linspace(0, 1, n_buckets + 1)
        bucket_losses = []

        for i in range(n_buckets):
            if i == n_buckets - 1:
                mask = (t_vals >= bucket_edges[i]) & (t_vals <= bucket_edges[i + 1])
            else:
                mask = (t_vals >= bucket_edges[i]) & (t_vals < bucket_edges[i + 1])

            if mask.sum() > 0:
                bucket_losses.append(torch.mean(pointwise_loss[mask]))
            else:
                bucket_losses.append(torch.tensor(0.0))

        bucket_losses = torch.stack(bucket_losses)

        # zero_shift so that bucket 0 always gets cumulative_loss = 0
        zero_shift = torch.zeros(1)
        # how much unresolved error exists in all the buckets BEFORE this one?
        cumulative_loss = torch.cat([zero_shift, torch.cumsum(bucket_losses[:-1], dim=0)])
        # weights 
        weights = torch.exp(-eps * cumulative_loss).detach()
        # Final loss: a weighted average of all bucket losses, normalized by the total weight

        return torch.sum(weights * bucket_losses) / torch.sum(weights)


    def Totalloss(self, x_bc, y_bc, x_ic, y_ic, x_collocation_points, PDE_residual_fn, var_names, n_buckets=10, eps=1.0):
        return (self.BCloss(x_bc, y_bc)
                + self.ICloss(x_ic, y_ic)
                + self.causal_PDEloss(x_collocation_points, PDE_residual_fn, var_names, n_buckets, eps))
