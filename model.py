import torch
import torch.nn as nn
import torch.autograd as autograd

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = nn.MSELoss(reduction = 'mean')
        self.layer1 = nn.Linear(2,32)
        self.layer2 = nn.Linear(32,32)
        self.layer3 = nn.Linear(32,1)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        return x

    def BCloss(self,x_bc, y_bc):
        BC_prediction = self.forward(x_bc)
        BC_loss = self.loss_function(BC_prediction,y_bc)
        return BC_loss

    def ICloss(self,x_ic, y_ic):
        IC_prediction = self.forward(x_ic)
        IC_loss = self.loss_function(IC_prediction,y_ic)
        return IC_loss

    def PDEloss(self, x_collocation_points):
        g = x_collocation_points.clone()
        g.requires_grad = True
        u = self.forward(g)

        u_x = autograd.grad(u,g,torch.ones(g.shape[0],1), retain_graph= True, create_graph= True)[0][:,[0]]
        u_t = autograd.grad(u,g,torch.ones(g.shape[0],1), retain_graph= True, create_graph= True)[0][:,[1]]
        u_xx = autograd.grad(u_x,g,torch.ones(g.shape[0],1), retain_graph= True, create_graph= True)[0][:,[0]]

        PDE_prediction = (u_t- u_xx + torch.exp(-g[:, [1]]) * (torch.sin(torch.pi * g[:, [0]])- torch.pi**2 * torch.sin(torch.pi * g[:, [0]])))
        PDE_loss = self.loss_function(PDE_prediction, torch.zeros_like(PDE_prediction))

        return PDE_loss


    def Totalloss(self, x_bc,y_bc,x_ic,y_ic,x_collocation_points):
        return self.BCloss(x_bc,y_bc) + self.ICloss(x_ic,y_ic) + self.PDEloss(x_collocation_points)
