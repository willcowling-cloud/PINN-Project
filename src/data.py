import torch
import torch.nn as nn

def generate_boundary_points(n_points):
    bp_left_column = (torch.randint(0,2,(n_points,1))*2 -1).float()
    bp_right_column = torch.rand((n_points,1))
    boundary_points = torch.cat((bp_left_column, bp_right_column), dim=1)
    boundary_target = torch.zeros_like(bp_left_column).float()
    return boundary_points, boundary_target

def generate_initial_points(n_points, initial_function):
    ip_left_column = torch.rand((n_points,1))*2-1
    ip_right_column = torch.zeros_like(ip_left_column)
    initial_points = torch.cat((ip_left_column,ip_right_column),dim=1)
    initial_target = initial_function(ip_left_column)
    return initial_points, initial_target

def generate_collocation_points(n_points):
    col_left_column = torch.rand(((n_points,1)))*2-1
    col_right_column = torch.rand((n_points,1))
    col_points = torch.cat((col_left_column,col_right_column),dim=1)
    return col_points

def generate_collocation_points_shock(n_points, n_points_centered, std):
        x_uncentered = torch.rand(((n_points-n_points_centered,1)))*2-1
        x_centered = torch.linspace(-0.05,0.05, 2000).reshape(2000,1)
        col_left_column = torch.cat((x_centered, x_uncentered))
        #Fill with a truncated normal distribution between -1 and 1
        torch.nn.init.trunc_normal_(col_left_column, mean=0.0, std= std, a=-1, b=1)
        col_right_column = torch.rand((n_points,1))
        col_points = torch.cat((col_left_column,col_right_column),dim=1)
        return col_points

