import numpy as np
import torch
from torch.autograd import Variable

#TODO how to constrain the norm

def init_spinup_MPS(N, D=1):
    """MPS: all spins up
    Args
    ----
    D : physical dimension, product state for D=1
    """
    Bs = np.zeros([5, D, 2, D], dtype=np.csingle)
    Bs[:, 0, 0, 0] = 1. # spinup
    Bs = torch.from_numpy(Bs).requires_grad_(True)
    Ss = torch.ones([5, 1], dtype=torch.cfloat, requires_grad=True)
    return MPS(Bs, Ss)

class MPS:
    """Class for handling matrix product states assuming a Vidal form"""
    def __init__(self, Bs, Ss):
        assert len(Bs) == len(Ss)
        self.Bs = Bs
        self.Ss = Ss
        self.N = len(Bs)

    def get_theta1(self, i):
        """Effective single-site wave function on site i"""
        return torch.tensordot(torch.diag(self.Ss[i]), self.Bs[i], [[1], [0]])  

    def get_theta2(self, i):
        """Effective two-site wave function on i,(i+1)"""
        j = i + 1
        return torch.tensordot(self.get_theta1(i), self.Bs[j], [[2], [0]])  

    def bond_expectation_value(self, op):
        """Compute exp. values of local operator 'op' at bonds"""
        result = []
        for i in range(self.N - 1):
            theta = self.get_theta2(i) 
            op_theta = torch.tensordot(op[i], theta, [[2, 3], [1, 2]])
            result.append(torch.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
        return torch.stack(result)

    def norm(self):
        """ <psi|psi> """
        N = self.N
        overlap = torch.tensordot(self.Bs[N-1], self.Bs[N-1].conj(), axes=([2, 1], [2, 1]))
        for i in reversed(range(N-1)):
            B = self.Bs[i]
            Bc = self.Bs[i].conj()
            overlap = torch.tensordot(B, overlap, ([2], [0])) 
            overlap = torch.tensordot(overlap, Bc, ([2, 1], [2, 1])) 
        return overlap.item()
        
    def MPO_expectation_value(self, mpo):
        """Expectation value of an MPO"""
        N = self.N
        assert(N == len(mpo))
        # Close right
        DR = mpo[N-1].shape[1]
        chi = self.Bs[-1].shape[0]
        RP = torch.zeros([chi, DR, chi], dtype=torch.cfloat) 
        RP[:, -1, :] = torch.eye(chi, dtype=torch.cfloat)
        # Iterate from right to left
        for i in reversed(range(N)): 
            B = self.Bs[i] 
            Bc = B.conj() 
            RP = torch.tensordot(B, RP, [[2], [0]]) 
            RP = torch.tensordot(RP, mpo[i], [[1, 2], [3, 1]]) 
            RP = torch.tensordot(RP, Bc, [[1, 3], [2, 1]])  
        # Close left
        DL = mpo[0].shape[0]
        LP = torch.zeros([chi, DL, chi], dtype=torch.cfloat)  
        LP[:, 0, :] = torch.eye(chi)
        result = torch.tensordot(LP, RP, ([0, 1, 2], [0, 1, 2])) 
        return result.item()


class IsingModel:
    """Ising Hamiltonian with transverse field for a spin-1/2 system
    with open boundary conditions
    """
    def __init__(self, N, J=1., h=1.):
        self.N = N
        self.J = J
        self.h = h
        self.d = 2 # physical dimension
        self.sigma_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.cfloat)
        self.sigma_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.cfloat)
        self.ident = torch.eye(2, dtype=torch.cfloat)
        self.init_H_bonds()
        self.init_H_mpo()

    def init_H_bonds(self):
        """Decompose local Hamiltonian into energy bonds"""
        sx, sz, ident = self.sigma_x, self.sigma_z, self.ident
        d = self.d
        H_list = []
        for i in range(self.N - 1):
            hL = hR = .5 * self.h
            if i == 0: # first bond
                hL = self.h
            elif i == self.N - 2: # last bond
                hL = self.h
            H_bond = self.J * torch.kron(sz, sz)
            H_bond += hL * torch.kron(sx, ident) + hR * torch.kron(ident, sx)
            H_list.append(H_bond.reshape([d, d, d, d]))
        self.H_bonds = H_list

    def init_H_mpo(self):
        """Hamiltonian MPO"""
        Ws = []
        d = self.d
        for i in range(self.N):
            W = torch.zeros((3, 3, d, d), dtype=torch.cfloat)
            W[0, 0,:, :] = W[2, 2, :, :] = self.ident
            W[0, 1, :, :] = self.sigma_z
            W[0, 2, :, :] = self.h * self.sigma_x
            W[1, 2, :, :] = self.J * self.sigma_z   
            Ws.append(W)
        self.H_mpo = Ws

    def energy_bonds(self, psi):
        """Compute energy E = <psi|H|psi> for MPS."""
        assert psi.N == self.N
        E = torch.sum(psi.bond_expectation_value(self.H_bonds))
        return E


        