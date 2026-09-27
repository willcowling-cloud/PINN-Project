import torch
import torch.nn as nn
import torch.autograd as autograd

def generate_boundary_points(n_points):
    bp_left_column = (torch.randint(0,2,(n_points,1))*2 -1).float()
    bp_right_column = torch.rand((n_points,1))
    boundary_points = torch.cat((bp_left_column, bp_right_column), dim=1)
    boundary_target = torch.zeros_like(bp_left_column).float()
    return boundary_points, boundary_target

def generate_initial_points(n_points):
    ip_left_column = torch.rand((n_points,1))*2-1
    ip_right_column = torch.zeros_like(ip_left_column)
    initial_points = torch.cat((ip_left_column,ip_right_column),dim=1)
    initial_target = torch.sin(torch.pi * ip_left_column)
    return initial_points, initial_target

def generate_collocation_points(n_points):
    col_left_column = torch.rand(((n_points,1)))*2-1
    col_right_column = torch.rand((n_points,1))
    col_points = torch.cat((col_left_column,col_right_column),dim=1)
    return col_points
