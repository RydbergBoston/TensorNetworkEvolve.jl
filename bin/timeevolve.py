import torch
from torch.autograd.functional import jacobian, hessian
import warnings

def step(peps, H, dt):
    """Perform one step in the direction of the gradient"""
    A = smatrix(peps) 
    B = fvec(peps, H)
    dvars = torch.linalg.solve(A, B)
    old_vars = peps.output_variables()
    new_vars = dvars * dt
    peps.load_variables(new_vars)
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

if __name__ == "__main__":
    warnings.warn("Not tested")
    L = 2
    labelsNtensors, graph = example_rand_peps(L, D=3, d=2)
    P = PEPS(*labelsNtensors)
    H = IsingModel(graph, J=-1., h=0.001).H
    for i in range(20):
        step(P, H, 0.1)
    P.get_statetensor()