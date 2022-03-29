import torch
import numpy as np
import opt_einsum as oe


class PEPS():
    def __init__(self, physical_labels, virtual_lables, vertex_labels,
                 vertex_tensors):
        self.physical_labels = physical_labels
        self.virtual_lables = virtual_lables
        self.vertex_labels = vertex_labels
        self.vertex_tensors = vertex_tensors

    def get_einsum_order(self):
        """ There seems to be a #BUG in oe.contract_expression, it throws
        an error while oe.contract works fine. Workaround with on the fly contraction
        order for now"""
        # Statetensor
        sublist = []
        for i,t in enumerate(self.vertex_tensors):
            sublist.append(t)
            sublist.append(self.vertex_labels[i])
        self.code_statetensor = oe.contract_expression(*sublist)
        # Inner product
        largest_label = self.virtual_lables.max()
        for i,t in enumerate(self.vertex_tensors):
            sublist.append(t)
            sublist.append(self.vertex_labels[i])
            sublist.append(t.conj())
            t_c_labels = self.vertex_labels[i] + largest_label
            t_c_labels[-1] = self.vertex_labels[i][-1]
            sublist.append(t_c_labels)
        self.code_inner_product = oe.contract_expression(*sublist)

    def load_variables(self, variables):
        new_tensors = []
        index = 0
        for i, t in enumerate(self.vertex_tensors):
            num_elements = np.prod(t.shape)
            new_tensor_vars = variables[index : index + num_elements]
            index += num_elements
            new_tensors.append(new_tensor_vars.reshape(t.shape))
        self.vertex_tensors = new_tensors

    def inner_product(self):
        largest_label = np.max(self.virtual_lables)
        sublist = []
        for i,t in enumerate(self.vertex_tensors):
            sublist.append(t)
            sublist.append(self.vertex_labels[i])
            sublist.append(t.conj())
            t_c_labels = self.vertex_labels[i].copy() + largest_label
            t_c_labels[-1] = self.vertex_labels[i][-1]
            sublist.append(t_c_labels)
        return oe.contract(*sublist)
        # expr = self.code_inner_product
        # inner_product = expr(tensor)
        # return inner_product
    
    def apply_Hamiltonian(self, H):
        for h in H:
            t, i = h
            self.vertex_tensors[i][:] = oe.contract('...i,ij->...j', self.vertex_tensors[i], t)

    def get_statetensor(self):
        sublist = []
        for i,t in enumerate(self.vertex_tensors):
            sublist.append(t)
            t_c_labels = self.vertex_labels[i].copy()
            sublist.append(t_c_labels)
        return oe.contract(*sublist).flatten()
        # expr = self.code_inner_product
        # statetensor = expr(tensor)
        # return statetensor

    def normalize(self):
        inner_product = self.inner_product()
        N = len(self.vertex_tensors)
        for t in self.vertex_tensors:
            t[:] = torch.div(t, torch.sqrt(torch.abs(inner_product))**(1/N))