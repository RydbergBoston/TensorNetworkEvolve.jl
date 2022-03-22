import numpy as np


def init_spinup_MPS(N):
    """MPS: all spins up"""
    B = np.zeros([1, 2, 1], dtype=complex)
    B[0, 0, 0] = 1.
    S = np.ones([1], float)
    Bs = [B.copy() for i in range(N)]
    Ss = [S.copy() for i in range(N)]
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
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  

    def get_theta2(self, i):
        """Effective two-site wave function on i,(i+1)"""
        j = i + 1
        return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  

    def bond_expectation_value(self, op):
        """Compute exp. values of local operator 'op' at bonds"""
        result = []
        for i in range(self.N - 1):
            theta = self.get_theta2(i) 
            op_theta = np.tensordot(op[i], theta, axes=[[2, 3], [1, 2]])
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
        return result

class IsingModel:
    """Ising Hamiltonian with transverse field for a spin-1/2 system
    with open boundary conditions
    """
    def __init__(self, N, J=1., h=1.):
        self.N = N
        self.J, self.h = J, h
        self.sigma_x = np.array([[0., 1.], [1., 0.]])
        self.sigma_z = np.array([[1., 0.], [0., -1.]])
        self.ident = np.eye(2)
        self.init_H_bonds()

    def init_H_bonds(self):
        """Decompose local Hamiltonian into energy bonds"""
        sx, sz, ident = self.sigma_x, self.sigma_z, self.ident
        d = 2 # physical dimension
        H_list = []
        for i in range(self.N - 1):
            hL = hR = .5 * self.h
            if i == 0: # first bond
                hL = self.h
            elif i == self.N - 2: # last bond
                hL = self.h
            H_bond = self.J * np.kron(sz, sz)
            H_bond += hL * np.kron(sx, ident) + hR * np.kron(ident, sx)
            H_list.append(H_bond.reshape([d, d, d, d]))
        self.H_bonds = np.array(H_list)

    def energy(self, psi):
        """Compute energy E = <psi|H|psi> for MPS."""
        assert psi.N == self.N
        E = np.sum(psi.bond_expectation_value(self.H_bonds))
        return E

        