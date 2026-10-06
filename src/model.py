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

    def PDEloss(self, x_collocation_points, var_names, PDE_residual_fn):
        g = x_collocation_points.clone()
        g.requires_grad = True
        u = self.forward(g)

        derivs = self.Calc_derivatives(u,g,var_names)

        PDE_prediction = PDE_residual_fn(g,u,derivs)
        PDE_loss = self.loss_function(PDE_prediction, torch.zeros_like(PDE_prediction))

        return PDE_loss


    def Totalloss(self, x_bc,y_bc,x_ic,y_ic,x_collocation_points, var_names, PDE_residual_fn):
        return self.BCloss(x_bc,y_bc) + self.ICloss(x_ic,y_ic) + self.PDEloss(x_collocation_points, var_names, PDE_residual_fn)
