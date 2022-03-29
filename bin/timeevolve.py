import torch
from torch.autograd.functional import jacobian, hessian

def step(peps, H, dt):
    """Perform one step in the direction of the gradient"""
    A = smatrix(peps) 
    B = fvec(peps, H)
    new_vars = torch.linalg.solve(A, B)
    peps.load_variables(dt * new_vars)
    peps.normalize()

def loss1(peps, variables):
    pl = peps.copy()
    pl.load_variables(variables)
    pl.normalize()
    pr = peps
    pr.load_variables(variables)
    pr.normalize()
    return torch.abs(pr.overlap_with(pl))

def smatrix(peps):
    variables = peps.output_variables()
    return hessian(lambda x: loss1(peps, x), variables)

def loss2(H, peps, variables):
    pl = peps.copy()
    pl.load_variables(variables)
    pl.normalize()
    pr = peps
    pr.normalize()
    pr.apply_Hamiltonian(H)
    return torch.abs(pr.overlap_with(pl))

def fvec(peps, H):
    variables = peps.output_variables()
    return -1j * jacobian(lambda x: loss2(H, peps, x), variables)