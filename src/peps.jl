export PEPS, VidalPEPS, SimplePEPS, zero_vidalpeps, zero_simplepeps, rand_simplepeps, rand_vidalpeps
export state, statevec, getvlabel, getphysicallabel, newlabel, findbondtensor, virtualbonds
export apply_onbond!, apply_onsite!, inner_product, norm, normalize!
export variables, load_variables!, load_variables
using LinearAlgebra
using OMEinsumContractionOrders: CodeOptimizer, CodeSimplifier
using OMEinsum: DynamicEinCode, NestedEinsum

# we implement the register interface because we want to use the operator system in Yao.
abstract type PEPS{T,LT} <:AbstractRegister{1} end

"""
    SimplePEPS{T,LT}

 a     b     c     d
 ┆     ┆     ┆     ┆ ← size = nflavor
 ●--α--●--β--●--γ--● 
                ↑
            size ≤ Dmax

`α`, `β` and `γ` are virtual labels.
`a`, `b`, `c` and `d` are physical labels.
`aα`, `αbβ`, `βcγ` and `γd` are vertex (tensor) labels.

It has fields

* `physical_labels` is a vector of unique physical labels, should be a vector of integers.
* `virtual_labels` is a vector if unique virtual labels, should be a vector of integer 2-tuples.

* `vertex_labels` is a vector of vectors, i-th vector is the labels for i-th vertex tensor.
* `vertex_tensors` is a vector of tensors defined on vertices.
* `max_index` is the maximum index, used for creating new labels.

* `code_statetensor` is the optimized contraction code for obtaining state vector.
* `code_inner_product` is the optimized contraction code for obtaining the overlap.

* `nflavor` is the size of physical dimension. For spins, it is 2.
* `Dmax` is the maximum virtual bond dimension.
* `ϵ` is useful in compression (e.g. with SVD), to determine the cutoff precision.
"""
struct SimplePEPS{T, LT<:Union{Int,Char}} <: PEPS{T,LT}
    physical_labels::Vector{LT}
    virtual_labels::Vector{LT}

    vertex_labels::Vector{Vector{LT}}
    vertex_tensors::Vector{<:AbstractArray{T}}
    max_index::LT

    # optimized contraction codes
    code_statetensor::NestedEinsum{DynamicEinCode{LT}}
    code_inner_product::NestedEinsum{DynamicEinCode{LT}}

    nflavor::Int
    Dmax::Int
    ϵ::Float64
end

function SimplePEPS(vertex_labels::AbstractVector{<:AbstractVector{LT}}, vertex_tensors::Vector{<:AbstractArray{T}},
        virtual_labels::AbstractVector{LT}, nflavor::Int, Dmax::Int, ϵ::Real, optimizer::CodeOptimizer, simplifier::CodeSimplifier) where {LT,T}
    physical_labels = [vl[findall(∉(virtual_labels), vl)[]] for vl in vertex_labels]
    max_ind = max(maximum(physical_labels), maximum(virtual_labels))
    # optimal contraction orders
    optcode_statetensor, optcode_inner_product = _optimized_code(vertex_labels, physical_labels, virtual_labels,
        max_ind, nflavor, Dmax, optimizer, simplifier)
    SimplePEPS(physical_labels, virtual_labels, vertex_labels, vertex_tensors, max_ind,
        optcode_statetensor, optcode_inner_product, nflavor, Dmax, ϵ)
end

function _optimized_code(alllabels, physical_labels::AbstractVector{LT}, virtual_labels, max_ind, nflavor, D, optimizer, simplifier) where LT
    code_statetensor = EinCode(alllabels, physical_labels)
    size_dict = Dict([[l=>nflavor for l in physical_labels]..., [l=>D for l in virtual_labels]...])
    optcode_statetensor = optimize_code(code_statetensor, size_dict, optimizer, simplifier)
    rep = [l=>max_ind+i for (i, l) in enumerate(virtual_labels)]
    merge!(size_dict, Dict([l.second=>D for l in rep]))
    code_inner_product = EinCode([alllabels..., [replace(l, rep...) for l in alllabels]...], LT[])
    optcode_inner_product = optimize_code(code_inner_product, size_dict, optimizer, simplifier)
    return optcode_statetensor, optcode_inner_product
end

# all labels for vertex tensors and bond tensors (if any)
alllabels(s::SimplePEPS) = s.vertex_labels
# all vertex tensors and bond tensors (if any)
alltensors(s::SimplePEPS) = s.vertex_tensors
Base.copy(peps::SimplePEPS) = SimplePEPS(copy(peps.physical_labels), copy(peps.virtual_labels),
    copy(peps.vertex_labels), copy(peps.vertex_tensors), peps.max_index, 
    peps.code_statetensor, peps.code_inner_product, peps.nflavor, peps.Dmax, peps.ϵ)

# ●----●----●----●   ← ⟨peps1|
# ┆    ┆    ┆    ┆
# ●----●----●----●   ← |peps2⟩
function inner_product(p1::PEPS, p2::PEPS)
    p1c = conj(p1)
    # we assume `p1` and `p2` have the same structure and virtual bond dimension.
    p1.code_inner_product(alltensors(p1c)..., alltensors(p2)...)[]
end

"""
    VidalPEPS

Similar to `SimplePEPS` except it contains one extra field

 a     b     c     d
 ┆  α  ┆  β  ┆  γ  ┆ ← size = nflavor
 ●--◆--●--◆--●--◆--● 
                ↑
            bond tensor

* `bond_tensors` is a vector of tensors on bonds.
"""
struct VidalPEPS{T, LT<:Union{Int,Char}} <: PEPS{T,LT}
    physical_labels::Vector{LT}
    virtual_labels::Vector{LT}

    vertex_labels::Vector{Vector{LT}}
    vertex_tensors::Vector{<:AbstractArray{T}}
    # bond labels are same as virtual_labels
    bond_tensors::Vector{<:AbstractVector{T}}
    max_index::LT

    # optimized contraction codes
    code_statetensor::NestedEinsum{DynamicEinCode{LT}}
    code_inner_product::NestedEinsum{DynamicEinCode{LT}}

    nflavor::Int
    Dmax::Int
    ϵ::Float64
end
function VidalPEPS(vertex_labels::AbstractVector{<:AbstractVector{LT}}, vertex_tensors::Vector{<:AbstractArray{T}},
        virtual_labels::AbstractVector{LT}, bond_tensors::Vector{<:AbstractVector}, nflavor::Int, Dmax::Int, ϵ::Real,
        optimizer::CodeOptimizer, simplifier::CodeSimplifier) where {LT,T}
    physical_labels = [vl[findall(∉(virtual_labels), vl)[]] for vl in vertex_labels]
    max_ind = max(maximum(physical_labels), maximum(virtual_labels))
    optcode_statetensor, optcode_inner_product = _optimized_code([vertex_labels..., [[l] for l in virtual_labels]...],
        physical_labels, virtual_labels, max_ind, nflavor, Dmax, optimizer, simplifier)
    VidalPEPS(physical_labels, virtual_labels, vertex_labels, vertex_tensors, bond_tensors, max_ind,
        optcode_statetensor, optcode_inner_product, nflavor, Dmax, ϵ)
end

alllabels(peps::VidalPEPS) = [peps.vertex_labels..., [[l] for l in peps.virtual_labels]...]
alltensors(peps::VidalPEPS) = [peps.vertex_tensors..., peps.bond_tensors...]
# find bond tensor by virtual label
findbondtensor(peps::VidalPEPS, b) = peps.bond_tensors[findall(==(b), peps.virtual_labels)[]]
Base.copy(peps::VidalPEPS) = VidalPEPS(copy(peps.physical_labels), copy(peps.virtual_labels),
    copy(peps.vertex_labels), copy(peps.virtual_tensors), copy(peps.bond_tensors), peps.max_index,
    peps.code_statetensor, peps.code_inner_product, peps.nflavor, peps.Dmax, peps.ϵ)
function Base.conj(peps::PEPS)
    replace_tensors(peps, conj.(alltensors(peps)))
end

# label APIs
getvlabel(peps::PEPS, i::Int) = peps.vertex_labels[i]  # vertex tensor labels
getphysicallabel(peps::PEPS, i::Int) = peps.physical_labels[i]  # physical label
newlabel(peps::PEPS, offset) = peps.max_index + offset  # create a new label
function virtualbonds(peps::PEPS)  # list all virtual bonds (a bond is a 2-tuple)
    bs = Tuple{Int,Int}[]
    for b in peps.virtual_labels
        i,j = findall(l->b∈l, peps.vertex_labels)
        push!(bs, (i, j))
    end
    return bs
end
nsite(peps::PEPS) = length(peps.physical_labels)
nflavor(peps::PEPS) = peps.nflavor
Dmax(peps::PEPS) = peps.Dmax

# all variables by flattening the tensors
variables(peps::PEPS) = vcat(vec.(alltensors(peps))...)
# load all variables to tensors
function load_variables!(peps::PEPS, variables)
    k = 0
    for t in alltensors(peps)
        t .= reshape(variables[k+1:k+length(t)], size(t))
        k += length(t)
    end
    return peps
end
function load_variables(peps::SimplePEPS, variables)  # for AD
    ats = alltensors(peps)
    ks = cumsum(length.(ats))
    tensors = map(1:length(ats)) do i
        t = ats[i]
        reshape(variables[(i>1 ? ks[i-1] : 0)+1:ks[i]], size(t))
    end
    return replace_tensors(peps, tensors)
end

function replace_tensors(peps::SimplePEPS, tensors)
    SimplePEPS(peps.physical_labels, peps.virtual_labels,
        peps.vertex_labels, tensors, peps.max_index,
        peps.code_statetensor, peps.code_inner_product, peps.nflavor, peps.Dmax, peps.ϵ
    )
end
function replace_tensors(peps::VidalPEPS, tensors)
    nv = nsite(peps)
    VidalPEPS(peps.physical_labels, peps.virtual_labels,
        peps.vertex_labels, tensors[1:nv], typeof(tensors[nv+1])[tensors[nv+1:end]...], peps.max_index,
        peps.code_statetensor, peps.code_inner_product, peps.nflavor, peps.Dmax, peps.ϵ
    )
end

function Base.show(io::IO, ::MIME"text/plain", peps::PEPS)
    println(io, typeof(peps), " (Dmax = $(peps.Dmax), ϵ = $(peps.ϵ))")
    println(io, " # of spins = $(nsite(peps))")
    print(io, " # of virtual degrees = $(length(peps.virtual_labels))")
end
Base.show(io::IO, peps::PEPS) = show(io, MIME"text/plain"(), peps)

# random and zero PEPSs
zero_vidalpeps(::Type{T}, g::SimpleGraph, D::Int; nflavor::Int=2, optimizer=GreedyMethod(), simplifier=MergeGreedy(), Dmax=D, ϵ=1e-12) where T = _peps_zero_state(Val(:Vidal), T, g, D, nflavor, Dmax, ϵ, optimizer, simplifier)
zero_simplepeps(::Type{T}, g::SimpleGraph, D::Int; nflavor::Int=2, optimizer=GreedyMethod(), simplifier=MergeGreedy(), Dmax=D, ϵ=1e-12) where T = _peps_zero_state(Val(:Simple), T, g, D, nflavor, Dmax, ϵ, optimizer, simplifier)
function _peps_zero_state(::Val{TYPE}, ::Type{T}, g::SimpleGraph, D::Int, nflavor::Int, Dmax::Int, ϵ::Real,
        optimizer::CodeOptimizer, simplifier::CodeSimplifier) where {TYPE, T}
    virtual_labels = collect(nv(g)+1:nv(g)+ne(g))
    vertex_labels = Vector{Int}[]
    vertex_tensors = Array{T}[]
    edge_map = Dict(zip(edges(g), virtual_labels))
    for i=1:nv(g)
        push!(vertex_labels, [i,[get(edge_map, SimpleEdge(i,nb), get(edge_map,SimpleEdge(nb,i),0)) for nb in neighbors(g, i)]...])
        t = zeros(T, nflavor, fill(D, degree(g, i))...)
        t[1] = 1
        push!(vertex_tensors, t)
    end
    if any(vl->any(iszero, vl), vertex_labels)
        error("incorrect input labels1")
    end
    if TYPE === :Vidal
        bond_tensors = [ones(T, D) for _=1:ne(g)]
        return VidalPEPS(vertex_labels, vertex_tensors, virtual_labels, bond_tensors, nflavor, Dmax, ϵ, optimizer, simplifier)
    else
        return SimplePEPS(vertex_labels, vertex_tensors, virtual_labels, nflavor, Dmax, ϵ, optimizer, simplifier)
    end
end

function rand_vidalpeps(::Type{T}, g::SimpleGraph, D::Int; nflavor::Int=2, optimizer=GreedyMethod(), simplifier=MergeGreedy(), Dmax::Int=D, ϵ=1e-12) where T
    randn!(zero_vidalpeps(T, g, D; nflavor=nflavor, Dmax=Dmax, ϵ=ϵ, optimizer=optimizer, simplifier=simplifier))
end
function rand_simplepeps(::Type{T}, g::SimpleGraph, D::Int; nflavor::Int=2, optimizer=GreedyMethod(), simplifier=MergeGreedy(), Dmax::Int=D, ϵ=1e-12) where T
    randn!(zero_simplepeps(T, g, D; nflavor=nflavor, Dmax=Dmax, ϵ=ϵ, optimizer=optimizer, simplifier=simplifier))
end
function Random.randn!(peps::PEPS)
    for t in alltensors(peps)
        randn!(t)
    end
    return peps
end

# multiply a peps with a constant
function LinearAlgebra.rmul!(peps::PEPS, c::Number)
    peps.vertex_tensors .*= c^(1/nsite(peps))
    return peps
end
Base.:*(c::Number, peps::PEPS) = peps * c
function Base.:*(peps::PEPS, c::Number)   # to support AD
    tensors = peps.vertex_tensors .* c^(1/nsite(peps))
    return replace_tensors(peps, tensors)
end

# norm of a peps
#
# ●----●----●----●   ← ⟨peps|
# ┆    ┆    ┆    ┆
# ●----●----●----●   ← |peps⟩
LinearAlgebra.norm(peps::PEPS) = sqrt(abs(inner_product(peps, peps)))
function LinearAlgebra.normalize!(peps::PEPS)
    nm = sqrt(abs(inner_product(peps, peps)))
    return rmul!(peps, 1/nm)
end

# contractor, the not cached version.
function direct_contract(code::EinCode, tensors)
    size_dict = OMEinsum.get_size_dict(OMEinsum.getixs(code), tensors)
    optcode = optimize_code(code, size_dict, GreedyMethod())
    optcode(tensors...)
end

# contract the peps and obtain the state vector
#
# ┆    ┆    ┆    ┆
# ●----●----●----●   ← |peps⟩
Base.vec(peps::PEPS) = vec(statetensor(peps))
function statetensor(peps::PEPS)
    peps.code_statetensor(alltensors(peps)...)
end

# apply a single site operator
#      ┆
#      ■  ← (operator)
# ┆    ┆    ┆    ┆
# ●----●----●----●   ← |peps⟩
function apply_onsite!(peps::PEPS{T,LT}, i, mat::AbstractMatrix) where {T,LT}
    @assert size(mat, 1) == size(mat, 2)
    ti = peps.vertex_tensors[i]
    old = getvlabel(peps, i)
    mlabel = [newlabel(peps, 1), getphysicallabel(peps, i)]
    peps.vertex_tensors[i] = EinCode([old, mlabel], replace(old, mlabel[2]=>mlabel[1]))(ti, mat)
    return peps
end

function apply_onsite(peps::PEPS{T,LT}, i, mat::AbstractMatrix) where {T,LT}  # to support AD
    @assert size(mat, 1) == size(mat, 2)
    ti = peps.vertex_tensors[i]
    old = getvlabel(peps, i)
    mlabel = [newlabel(peps, 1), getphysicallabel(peps, i)]
    tensors = map(1:length(peps.vertex_tensors)) do j
        tj = peps.vertex_tensors[j]
        j == i ? EinCode([old, mlabel], replace(old, mlabel[2]=>mlabel[1]))(tj, mat) : tj
    end
    return replace_tensors(peps, tensors)
end
# apply a two site operator
#      ┆    ┆
#      ■----■  ← (operator)
# ┆    ┆    ┆    ┆
# ●----●----●----●    ← |peps⟩
function apply_onbond!(peps::PEPS, i, j, mat::AbstractArray{T,4}) where T
    ti, tj = peps.vertex_tensors[i], peps.vertex_tensors[j]
    li, lj = getvlabel(peps, i), getvlabel(peps, j)
    shared_label = li ∩ lj; @assert length(shared_label) == 1
    only_left, only_right = setdiff(li, lj), setdiff(lj, li)
    lij = only_left ∪ only_right

    if peps isa VidalPEPS
        # multiple S tensors
        for b in li
            b == getphysicallabel(peps, i) && continue
            ti = _apply_vec(ti, li, sqrt.(findbondtensor(peps, b)), b)
        end
        for b in lj
            b == getphysicallabel(peps, j) && continue
            tj = _apply_vec(tj, lj, sqrt.(findbondtensor(peps, b)), b)
        end
    end

    # contract
    tij = EinCode([li, lj], lij)(ti, tj)
    lijkl = [newlabel(peps, 1), newlabel(peps, 2), getphysicallabel(peps, i), getphysicallabel(peps, j)]
    lkl = replace(lij, lijkl[3]=>lijkl[1], lijkl[4]=>lijkl[2])
    tkl = EinCode([lij, lijkl], lkl)(tij, mat)

    # SVD and truncate
    sl = [size(ti, findall(==(l), li)[]) for l in only_left]
    sr = [size(tj, findall(==(l), lj)[]) for l in only_right]
    U, S, Vt = svd(reshape(tkl, prod(sl), prod(sr)))
    Vt = conj.(Vt)
    D0 = findfirst(<(peps.ϵ), S)
    D = D0 === nothing ? min(peps.Dmax, size(U,2)) : min(D0-1, peps.Dmax)
    if D < size(U, 2)
        @info "truncation error is $(sum(S[D+1:end]))"
        S = S[1:D]
        U = U[:,1:D]
        Vt = Vt[:,1:D]
    end
    # reshape back
    ti_ = EinCode([[only_left..., shared_label[]]], li)(reshape(U, (sl..., size(U,2))))
    tj_ = EinCode([[only_right..., shared_label[]]], lj)(reshape(Vt, (sr..., size(Vt,2))))
    
    if peps isa VidalPEPS
        # devide S tensors
        for b in only_left
            b == getphysicallabel(peps, i) && continue
            ti_ = _apply_vec(ti_, li, safe_inv.(sqrt.(findbondtensor(peps, b))), b)  # assume S being positive
        end
        for b in only_right
            b == getphysicallabel(peps, j) && continue
            tj_ = _apply_vec(tj_, lj, safe_inv.(sqrt.(findbondtensor(peps, b))), b)
        end
        peps.bond_tensors[findall(==(shared_label[]), peps.virtual_labels)[]] = S
    else
        b = shared_label[]
        sqrtS = sqrt.(S)
        ti_ = _apply_vec(ti_, li, sqrtS, b)  # assume S being positive
        tj_ = _apply_vec(tj_, lj, sqrtS, b)
    end

    # update tensors
    peps.vertex_tensors[i] = ti_
    peps.vertex_tensors[j] = tj_
    return peps
end

function safe_inv(y::T) where T
    ϵ = 1e-10
    if abs(y) < ϵ
        return inv(real(y) >= 0 ? T(ϵ) : T(-ϵ))
    else
        return inv(y)
    end
end

function _apply_vec(t, l, v, b)
    return EinCode([l, [b]], l)(t, v)
end



"""
VectorPEPS, and all its overloads.
"""
#=

# VectorPEPS
struct VectorPEPS{T, LT<:Union{Int,Char}} <: PEPS{T,LT}
    vec::Vector{T}
    nsite::Int
    nflavor::Int
    ϵ::Float64
end

function nsite(peps::VectorPEPS)
    return peps.nsite
end

function nflavor(peps::VectorPEPS)
    return peps.nflavor
end

function vec(peps::VectorPEPS)
    return peps.vec
end

function conj(peps::VectorPEPS)
    p1c = VectorPEPS(peps.nsite, peps.nflavor, conj.(peps.vec), peps.ϵ)
    return p1c
end

function inner_product(p1::VectorPEPS, p2::VectorPEPS)
    # We assume `p1` and `p2` have the same structure
    p1c = conj(p1)
    return dot(p1c.vec, p2.vec)
end


function apply_onbond!(peps::VectorPEPS, i, j, mat::AbstractArray{T,4}) where T
    # Apply operator onto bond (pair of sites), not in place
    # modify peps.vec as it would be modified if you applied an operator T to i,j?
    # think this can almost be handled as a YaoBlocks.KronBlock or PutBlock plus an application of T
    # as a primitive Block. only uncertainty because of possible 'non-contiguous qubits'
    return peps
end

function apply_onbond(peps::VectorPEPS, i, j)
    # Apply operator onto bond (pair of sites), not in place
    return peps
end
function apply_onsite(peps::VectorPEPS, i)
    # Apply operator onto single site, not in place
    return
end
function apply_onsite!(peps::VectorPEPS, i)
    # Apply operator onto single site, in place
    return
end

function rmul!(peps::VectorPEPS, x)
    # Scalar multiplication of vector
    peps.vec .*= x
    return peps
end

function Base.:*(peps::VectorPEPS, c::Number)
    peps.vec .*= c
    return peps
end

=#



#### Circuit interfaces ####
# NOTE: we should have a register type.
Yao.nqubits(peps::PEPS) = nsite(peps)
Yao.nactive(peps::PEPS) = nsite(peps)
Yao.statevec(peps::PEPS) = vec(peps)
for (APPLY, APPLY_ONSITE, MUL) in [(:_apply!, :apply_onsite!, :mul!),
        (:apply, :apply_onsite, :*)]
    @eval function YaoBlocks.$APPLY(peps::PEPS{T}, block::PutBlock{N,1}) where {T,N}
        $APPLY_ONSITE(peps, block.locs[1], Matrix{T}(block.content))
    end
    @eval function YaoBlocks.$APPLY(peps::PEPS{T}, block::KronBlock{N,M,BT}) where {T,N,M,BT<:NTuple{M,AbstractBlock{1}}}
        for (loc, g) in zip(block.locs, subblocks(block))
            peps = $APPLY_ONSITE(peps, loc[1], Matrix{T}(g))
        end
        return peps
    end
    @eval function YaoBlocks.$APPLY(peps::PEPS{T}, block::ControlBlock{N,BT,1,1}) where {T,N,BT}
        # forward to PutBlock.
        YaoBlocks.$APPLY(peps, put(N, (block.ctrl_locs[1], block.locs[1])=>control(2,1,2=>block.content)))
    end
    @eval function YaoBlocks.$APPLY(peps::PEPS{T}, block::Scale) where T
        $MUL(YaoBlocks.$APPLY(peps, content(block)), (1.0+0im)*Yao.factor(block))
    end
end
# no non-inplace version defined.
function YaoBlocks._apply!(peps::PEPS{T}, block::PutBlock{N,2}) where {T,N}
    apply_onbond!(peps, block.locs..., reshape(Matrix{T}(block.content), 2, 2, 2, 2))
end

# compute the expectation value of a Hamiltonian
#
# ●----●----●----●   ← ⟨peps|
# ┆    ┆    ┆    ┆
# ■    ■    ■    ■   ← (product operator)
# ┆    ┆    ┆    ┆
# ●----●----●----●   ← |peps⟩
function Yao.expect(operator::Add, pa::PEPS, pb::PEPS)
    res = 0.0im
    for term in Yao.subblocks(operator)
        res += expect(term, pa, pb)
    end
    return res
end
function Yao.expect(operator::AbstractBlock, pa::PEPS, pb::PEPS)
    inner_product(pa, apply(pb, operator))
end