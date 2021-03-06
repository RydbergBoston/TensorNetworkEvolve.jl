module TensorNetworkEvolve

using Yao
using Yao.ConstGate: P1
using Yao: Add
using Graphs
using Graphs: SimpleEdge
using Random
using LinearAlgebra
using OMEinsum
using KrylovKit
#using Zygote: gradient, @non_differentiable

export PEPS, VidalPEPS, SimplePEPS, zero_vidalpeps, zero_simplepeps, rand_simplepeps, rand_vidalpeps
export state, statevec, getvlabel, getphysicallabel, newlabel, findbondtensor, virtualbonds
export apply_onbond!, apply_onsite!, inner_product, norm, normalize!
export variables, load_variables!, load_variables

# load AD submodule
include("TensorAD/TensorAD.jl")
using .TensorAD: gradient, DiffTensor

include("peps.jl")
include("tebd.jl")
include("timeevolve.jl")
#include("cracker.jl")

end
