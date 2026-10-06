import torch



# How accurate is it numerically?
def relativeERROR(predicted_values,exact_values):
    relative_error = (torch.sqrt(torch.sum((predicted_values - exact_values)**2))/ torch.sqrt(torch.sum(exact_values**2)))
    return relative_error

