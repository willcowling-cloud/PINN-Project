import torch
import torch.nn as nn
import torch.autograd as autograd

def train_pinn(pinn,x_bc, y_bc, x_ic, y_ic, col_points, learning_rate = 0.01, epochs = 1000, use_LBFGS = True):
    optimiser = torch.optim.Adam(pinn.parameters(),lr = learning_rate)
    history = []

    for i in range(epochs):
        loss = pinn.Totalloss(x_bc, y_bc, x_ic, y_ic, col_points)

    #Clear Old Gradient
        optimiser.zero_grad()
    #Calculate New Gradient
        loss.backward()
    #Step
        optimiser.step()
        

        if i % 100 == 0:
                history.append(
                f"Iteration {i} | "
                f"Total: {loss.item():.6f}")

    # Refine optimisation using LBFGS
    if use_LBFGS == True:
        optimiser = torch.optim.LBFGS(pinn.parameters(),lr = 1.0)

        def closure():
            #Clear Old Gradient
            optimiser.zero_grad()
            #Calculate Loss
            loss = pinn.Totalloss(x_bc, y_bc, x_ic, y_ic, col_points)
            #Calculate New Gradient
            loss.backward()
            return loss

        #Step
        optimiser.step(closure)
    final_loss = pinn.Totalloss(x_bc, y_bc, x_ic, y_ic, col_points)
    history.append(f"Final Total Loss = {final_loss}")

    return history, final_loss
