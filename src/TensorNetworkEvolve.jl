module TensorNetworkEvolve

using Yao
using Yao.ConstGate: P1
using Yao: Add
using Graphs
using Graphs: SimpleEdge
using Random
using LinearAlgebra
using OMEinsumContractionOrders
using OMEinsumContractionOrders: CodeOptimizer, CodeSimplifier, SlicedEinsum
using OMEinsum
using OMEinsum: DynamicEinCode, NestedEinsum
using KrylovKit
#using IterativeSolvers
#using Zygote: gradient, @non_differentiable

export PEPS, VidalPEPS, SimplePEPS, zero_vidalpeps, zero_simplepeps, rand_simplepeps, rand_vidalpeps
export state, statevec, getvlabel, getphysicallabel, newlabel, findbondtensor, virtualbonds
export apply_onbond!, apply_onsite!, inner_product, norm, normalize!
export variables, load_variables!, load_variables
export time_evolve_Euclidean!

# load AD submodule
include("TensorAD/TensorAD.jl")
using .TensorAD: gradient, DiffTensor

include("peps.jl")
include("tebd.jl")
include("timeevolve.jl")
#include("cracker.jl")

end
