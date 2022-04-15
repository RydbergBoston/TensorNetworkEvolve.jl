export VectorPEPS, QubitVectorPEPS, zero_vectorpeps, rand_vectorpeps, zero_qubitvectorpeps, rand_qubitvectorpeps
export state, statevec, getvlabel, getphysicallabel, newlabel, findbondtensor, virtualbonds
export apply_onbond!, apply_onsite!, inner_product, norm, normalize!, apply_onsite, apply_on!, apply_onbond
export variables, load_variables!, load_variables 
using LinearAlgebra

"""
VectorPEPS, and all its overloads.
"""

# VectorPEPS
struct VectorPEPS{T,NFLAVOR} <: PEPS{T,Int}
    vec::Vector{T}
    nsite::Int
    ϵ::Float64
end

VectorPEPS{NFLAVOR}(vec::Vector{T}, nsite, ϵ) where {T, NFLAVOR} = VectorPEPS{T,NFLAVOR}(vec, nsite, ϵ)

const QubitVectorPEPS{T} = VectorPEPS{T,2}

QubitVectorPEPS(vec::Vector{T}, nsite, ϵ) where T = QubitVectorPEPS{T}(vec, nsite, ϵ)

nsite(peps::VectorPEPS) = peps.nsite
nflavor(peps::VectorPEPS{T,NFLAVOR}) where {T,NFLAVOR} = NFLAVOR
Yao.nqubits(peps::VectorPEPS) = nsite(peps)
Yao.nactive(peps::VectorPEPS) = nsite(peps)
Yao.statevec(peps::VectorPEPS) = vec(peps)

function Random.randn!(peps::VectorPEPS)
    randn!(peps.vec)
    return peps
end

function zero_vectorpeps(::Type{T}, nsite::Int; nflavor::Int=2, ϵ=1e-12) where T
    vec = zeros(T, nflavor^nsite)
    vec[1] = 1
    VectorPEPS{nflavor}(vec, nsite, ϵ)
end

function zero_qubitvectorpeps(::Type{T}, nsite::Int; ϵ=1e-12) where T
    vec = zeros(T, 2^nsite)
    vec[1] = 1
    QubitVectorPEPS(vec, nsite, ϵ) 
end

function rand_vectorpeps(::Type{T}, nsite::Int; nflavor::Int=2, ϵ=1e-12) where T
    randn!(zero_vectorpeps(T, nsite; nflavor=nflavor, ϵ=ϵ))
end

function rand_qubitvectorpeps(::Type{T}, nsite::Int; ϵ=1e-12) where T
    randn!(zero_qubitvectorpeps(T, nsite; ϵ=ϵ))
end

function Base.vec(peps::VectorPEPS)
    return peps.vec
end

function Base.conj(peps::VectorPEPS)
    return VectorPEPS{nflavor(peps)}(conj.(peps.vec), peps.nsite, peps.ϵ)
end

# variables correspond to the already-flattened vector
variables(peps::VectorPEPS) = peps.vec

# load all variables to vector
function load_variables!(peps::VectorPEPS, variables)
    peps.vec .= variables
    return peps
end

function load_variables(peps::VectorPEPS{T}, variables::Vector{T}) where T  # for AD
    VectorPEPS{nflavor(peps)}(variables, peps.nsite, peps.ϵ)
end

LinearAlgebra.norm(peps::VectorPEPS) = norm(vec(peps))
function inner_product(p1::VectorPEPS{T}, p2::VectorPEPS{T}) where T
    # We assume `p1` and `p2` have the same structure
    return dot(p1.vec, p2.vec)
end


function LinearAlgebra.rmul!(peps::VectorPEPS, c::Number)
    # Scalar multiplication of vector
    peps.vec .*= c
    return peps
end

function LinearAlgebra.normalize!(peps::VectorPEPS)  # !!!
    nm = sqrt(abs(inner_product(peps, peps))) 
    return rmul!(peps, 1/nm)
end

function Base.:*(peps::VectorPEPS{T,NFLAVOR}, c::Number) where {T,NFLAVOR}
    VectorPEPS{NFLAVOR}(peps.vec .* c)
end


function apply_onsite!(peps::VectorPEPS{T}, i, mat::AbstractMatrix) where {T}
    # Apply operator onto single site, in place
    @assert size(mat, 1) == size(mat, 2)
    peps.vec .= state(ArrayReg(peps.vec) |> chain(peps.nsite, put(i=>matblock(mat))))
    return peps
end

function apply_onsite(peps::VectorPEPS{T}, i, mat::AbstractMatrix) where {T}
    # Apply operator onto single site, not in place
    @assert size(mat, 1) == size(mat, 2)
    return VectorPEPS{nflavor(peps)}(vec(state(ArrayReg(peps.vec) |> chain(peps.nsite, put(i=>matblock(mat))))), peps.nsite, peps.ϵ)
end


function apply_onbond!(peps::QubitVectorPEPS{T}, i, j, mat::AbstractArray{T, 4}) where T
    # Apply operator onto bond (pair of sites), in place
    peps.vec .= state(apply!(ArrayReg(peps.vec), put(nqubits(peps), (i,j)=>matblock(reshape(mat, 4, 4)))))
end

function apply_onbond(peps::QubitVectorPEPS{T}, i, j, mat::AbstractArray{T, 4}) where T
    # Apply operator onto bond (pair of sites), not in place
    return QubitVectorPEPS(vec(state(apply!(ArrayReg(peps.vec), put(nqubits(peps), (i,j)=>matblock(reshape(mat, 4, 4)))))), peps.nsite, peps.ϵ)
end