import numpy as np
import torch
from torch.autograd import Variable

def init_spinup_MPS(N, D=1):
    """MPS: all spins up
    Args
    ----
    D : physical dimension, product state for D=1
    """
    Ms = np.zeros([N, D, 2, D], dtype=np.csingle)
    Ms[:, 0, 0, 0] = 1. # spinup
    Ms = torch.from_numpy(Ms).requires_grad_(True)
    return MPS(Ms)

class MPS:
    """Class for handling matrix product states assuming a Vidal form"""
    def __init__(self, Ms):
        self.Ms = Ms
        self.N = len(Ms)

    def get_theta1(self, i):
        """Effective single-site wave function on site i"""
        return torch.tensordot(torch.diag(self.Ss[i]), self.Ms[i], [[1], [0]])  

    def get_theta2(self, i):
        """Effective two-site wave function on i,(i+1)"""
        j = i + 1
        return torch.tensordot(self.get_theta1(i), self.Ms[j], [[2], [0]])  

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
        overlap = torch.tensordot(self.Ms[N-1], self.Ms[N-1].conj(), ([2, 1], [2, 1]))
        for i in reversed(range(N-1)):
            B = self.Ms[i]
            Bc = self.Ms[i].conj()
            overlap = torch.tensordot(B, overlap, ([2], [0])) 
            overlap = torch.tensordot(overlap, Bc, ([2, 1], [2, 1])) 
        return overlap
        
    def MPO_expectation_value(self, mpo):
        """Expectation value of an MPO"""
        N = self.N
        assert(N == len(mpo))
        # Close right
        DR = mpo[N-1].shape[1]
        chi = self.Ms[-1].shape[0]
        RP = torch.zeros([chi, DR, chi], dtype=torch.cfloat) 
        RP[:, -1, :] = torch.eye(chi, dtype=torch.cfloat)
        # Iterate from right to left
        for i in reversed(range(N)): 
            B = self.Ms[i] 
            Bc = B.conj() 
            RP = torch.tensordot(B, RP, [[2], [0]]) 
            RP = torch.tensordot(RP, mpo[i], [[1, 2], [3, 1]]) 
            RP = torch.tensordot(RP, Bc, [[1, 3], [2, 1]])  
        # Close left
        DL = mpo[0].shape[0]
        LP = torch.zeros([chi, DL, chi], dtype=torch.cfloat)  
        LP[:, 0, :] = torch.eye(chi)
        result = torch.tensordot(LP, RP, ([0, 1, 2], [0, 1, 2])) 
        return result


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
        self.init_H_mpo()

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

    def energy(self, psi):
        """Compute energy E = <psi|H|psi>/<psi|psi> for MPS."""
        try:
             psi.N == self.N
        except:
            print(f'Length need to match: MPS length={psi.N}, MPO length={self.N}')
            raise ValueError
        E = psi.MPO_expectation_value(self.H_mpo) / psi.norm()
        return E


if __name__ == "__main__":
    N = 5
    D = 1
    print(f"+++ Ising chain with N={N}, MPS bond dim D={D} +++")
    print(f"Finding the ground state with PyTorch autograd...\n")
    psi = init_spinup_MPS(N, D)
    model = IsingModel(N)
    loss = model.energy(psi)
    loss.backward()
    optim = torch.optim.SGD([psi.Ms], lr=1e-2, momentum=0.9)

    num_iter = 500
    for i in range(num_iter):
        optim.zero_grad()
        loss = model.energy(psi)
        loss.backward()
        optim.step()
        if i%(num_iter/10) == 0:
            print(f"Iteration {i:03d}/{num_iter} - Current GS energy: {loss.item()}")
