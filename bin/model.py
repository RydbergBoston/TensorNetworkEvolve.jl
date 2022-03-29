import torch
import networkx as nx

def example_rand_peps(L, D=5, d=2):
    G = nx.grid_2d_graph(L, L)
    N = len(G)
    G = nx.relabel_nodes(G, dict(zip(G, range(0, N))))
    vert_dict = dict(zip(G.nodes, range(N)))
    edge_dict = dict(zip(G.edges, range(N, N + len(G.edges))))
    physical_labels = list(range(len(G.nodes)))
    virtual_labels =  list(range(N, N + len(G.edges)))
    # Make random tensors
    vertex_tensors = []
    for i in range(N):
        shape = ([D] * len(G.edges(i))) + [d]
        t = torch.rand(shape, dtype=torch.cdouble) # tensors have phys. dim last
        vertex_tensors.append(t)
    # Vertex labels
    vertex_labels = []
    for i in range(N):
        v_els = []
        for edge in G.edges(i):
            try:
                el = edge_dict[edge]
            except:
                el = edge_dict[edge[::-1]]
            v_els.append(el)
        v_els.append(i)
        vertex_labels.append(v_els)
    return (physical_labels, virtual_labels, vertex_labels, vertex_tensors), G

class IsingModel:
    """Ising Hamiltonian with transverse field for a spin-1/2 system
    with open boundary conditions
    """
    def __init__(self, graph, J=1., h=1.):
        self.graph = graph
        self.J = J
        self.h = h
        self.d = 2 # physical dimension
        self.sigma_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.cdouble)
        self.sigma_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.cdouble)
        self.init_H()

    def init_H(self):
        H_list = []
        # interaction
        for i, j in self.graph.edges:
            h = (self.J * self.sigma_z, i)
            H_list.append(h)
            h = (self.sigma_z, j)
            H_list.append(h)
        # local term
        for i in self.graph.nodes:
            h = (self.h * self.sigma_x, i)
            H_list.append(h)
        self.H = H_list