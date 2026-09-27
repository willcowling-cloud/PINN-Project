import torch
import torch.nn as nn
import torch.autograd as autograd


def exact_solution(x,t):
    exact_solution = torch.exp(-t) * torch.sin(torch.pi * x)
    return exact_solution

# How accurate is it numerically?
def relativeERROR(predicted_values,exact_values):
    relative_error = (torch.sqrt(torch.sum((predicted_values - exact_values)**2))/ torch.sqrt(torch.sum(exact_values**2)))
    return relative_error
