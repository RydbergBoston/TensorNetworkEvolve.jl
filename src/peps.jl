export PEPS, VidalPEPS, SimplePEPS, zero_vidalpeps, zero_simplepeps, rand_simplepeps, rand_vidalpeps
export state, statevec, getvlabel, getphysicallabel, newlabel, findbondtensor, virtualbonds
export apply_onbond!, apply_onsite!, inner_product, norm, normalize!
export variables, load_variables!
using LinearAlgebra

# we implement the register interface because we want to use the operator system in Yao.
abstract type PEPS{T,LT} <:AbstractRegister{1} end

"""
    SimplePEPS

* `physical_labels` is a vector of unique physical labels
* `virtual_labels` is a vector if unique virtual labels

* `vertex_labels` is a vector of vectors, each vector is a labels for tensors.
* `vertex_tensors` is a vector of tensors on vertices.
* `max_index` is the maximum index, used for creating new labels.

* `Dmax` is the maximum virtual bond dimension.
* `ϵ` is useful in compression (e.g. with SVD), to determine the cutoff precision.
"""
struct SimplePEPS{T, LT<:Union{Int,Char}} <: PEPS{T,LT}
    physical_labels::Vector{LT}
    virtual_labels::Vector{LT}

    vertex_labels::Vector{Vector{LT}}
    vertex_tensors::Vector{<:AbstractArray{T}}
    max_index::LT

    Dmax::Int
    ϵ::Float64
end

function SimplePEPS(vertex_labels::AbstractVector{<:AbstractVector{LT}}, vertex_tensors::Vector{<:AbstractArray{T}},
        virtual_labels::AbstractVector{LT}, Dmax::Int, ϵ::Real) where {LT,T}
    physical_labels = [vl[findall(∉(virtual_labels), vl)[]] for vl in vertex_labels]
    max_ind = max(maximum(physical_labels), maximum(virtual_labels))
    SimplePEPS(physical_labels, virtual_labels, vertex_labels, vertex_tensors, max_ind, Dmax, ϵ)
end

# all labels for vertex tensors and bond tensors (if any)
alllabels(s::SimplePEPS) = s.vertex_labels
# all vertex tensors and bond tensors (if any)
alltensors(s::SimplePEPS) = s.vertex_tensors
Base.copy(peps::SimplePEPS) = SimplePEPS(copy(peps.physical_labels), copy(peps.virtual_labels),
    copy(peps.vertex_labels), copy(peps.vertex_tensors), peps.max_index, peps.Dmax, peps.ϵ)
function Base.conj(peps::SimplePEPS)
    SimplePEPS(peps.physical_labels, peps.virtual_labels,
        peps.vertex_labels, conj.(peps.vertex_tensors), peps.max_index, peps.Dmax, peps.ϵ
    )
end

# ●----●----●----●   ← ⟨peps1|
# ┆    ┆    ┆    ┆
# ●----●----●----●   ← |peps2⟩
function inner_product(p1::PEPS{T,LT}, p2::PEPS{T,LT}) where {T,LT}
    p1c = conj(p1)
    rep = [l=>newlabel(p1, i) for (i, l) in enumerate(p2.virtual_labels)]
    code = EinCode([alllabels(p1c)..., [replace(l, rep...) for l in alllabels(p2)]...], LT[])
    direct_contract(code, (alltensors(p1c)..., alltensors(p2)...))[]
end

"""
    VidalPEPS

Similar to `SimplePEPS` except it contains one extra field

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

    Dmax::Int
    ϵ::Float64
end
function VidalPEPS(vertex_labels::AbstractVector{<:AbstractVector{LT}}, vertex_tensors::Vector{<:AbstractArray{T}},
        virtual_labels::AbstractVector{LT}, bond_tensors::Vector{<:AbstractVector}, Dmax::Int, ϵ::Real) where {LT,T}
    physical_labels = [vl[findall(∉(virtual_labels), vl)[]] for vl in vertex_labels]
    max_ind = max(maximum(physical_labels), maximum(virtual_labels))
    VidalPEPS(physical_labels, virtual_labels, vertex_labels, vertex_tensors, bond_tensors, max_ind, Dmax, ϵ)
end

alllabels(peps::VidalPEPS) = [peps.vertex_labels..., [[l] for l in peps.virtual_labels]...]
alltensors(peps::VidalPEPS) = [peps.vertex_tensors..., peps.bond_tensors...]
# find bond tensor by virtual label
findbondtensor(peps::VidalPEPS, b) = peps.bond_tensors[findall(==(b), peps.virtual_labels)[]]
Base.copy(peps::VidalPEPS) = VidalPEPS(copy(peps.physical_labels), copy(peps.virtual_labels),
    copy(peps.vertex_labels), copy(peps.virtual_tensors), copy(peps.bond_tensors), peps.max_index, peps.Dmax, peps.ϵ)
function Base.conj(peps::VidalPEPS)
    VidalPEPS(peps.physical_labels, peps.virtual_labels,
        peps.vertex_labels, conj.(peps.vertex_tensors), conj.(peps.bond_tensors), peps.max_index, peps.Dmax, peps.ϵ
    )
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

function Base.show(io::IO, ::MIME"text/plain", peps::PEPS)
    println(io, typeof(peps), " (Dmax = $(peps.Dmax), ϵ = $(peps.ϵ))")
    println(io, " # of spins = $(nsite(peps))")
    print(io, " # of virtual degrees = $(length(peps.virtual_labels))")
end
Base.show(io::IO, peps::PEPS) = show(io, MIME"text/plain"(), peps)

# random and zero PEPSs
zero_vidalpeps(::Type{T}, g::SimpleGraph, D::Int; Dmax=D, ϵ=1e-12) where T = _peps_zero_state(Val(:Vidal), T, g, D, Dmax, ϵ)
zero_simplepeps(::Type{T}, g::SimpleGraph, D::Int; Dmax=D, ϵ=1e-12) where T = _peps_zero_state(Val(:Simple), T, g, D, Dmax, ϵ)
function _peps_zero_state(::Val{TYPE}, ::Type{T}, g::SimpleGraph, D::Int, Dmax::Int, ϵ::Real) where {TYPE, T}
    virtual_labels = collect(nv(g)+1:nv(g)+ne(g))
    vertex_labels = Vector{Int}[]
    vertex_tensors = Array{T}[]
    edge_map = Dict(zip(edges(g), virtual_labels))
    for i=1:nv(g)
        push!(vertex_labels, [i,[get(edge_map, SimpleEdge(i,nb), get(edge_map,SimpleEdge(nb,i),0)) for nb in neighbors(g, i)]...])
        t = zeros(T, 2, fill(D, degree(g, i))...)
        t[1] = 1
        push!(vertex_tensors, t)
    end
    if any(vl->any(iszero, vl), vertex_labels)
        error("incorrect input labels1")
    end
    if TYPE === :Vidal
        bond_tensors = [ones(T, D) for _=1:ne(g)]
        return VidalPEPS(vertex_labels, vertex_tensors, virtual_labels, bond_tensors, Dmax, ϵ)
    else
        return SimplePEPS(vertex_labels, vertex_tensors, virtual_labels, Dmax, ϵ)
    end
end

function rand_vidalpeps(::Type{T}, g::SimpleGraph, D::Int; Dmax::Int=D, ϵ=1e-12) where T
    randn!(zero_vidalpeps(T, g, D; Dmax=Dmax, ϵ=ϵ))
end
function rand_simplepeps(::Type{T}, g::SimpleGraph, D::Int; Dmax::Int=D, ϵ=1e-12) where T
    randn!(zero_simplepeps(T, g, D; Dmax=Dmax, ϵ=ϵ))
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
function direct_contract(code::EinCode, tensors; kwargs...)
    size_dict = OMEinsum.get_size_dict(OMEinsum.getixs(code), tensors)
    optcode = optimize_code(code, size_dict, GreedyMethod())
    optcode(tensors...)
end

# contract the peps and obtain the state vector
#
# ┆    ┆    ┆    ┆
# ●----●----●----●   ← |peps⟩
Base.vec(peps::PEPS) = vec(statetensor(peps))
function statetensor(peps::PEPS; kwargs...)
    code = EinCode(alllabels(peps), peps.physical_labels)
    direct_contract(code, alltensors(peps); kwargs...)
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

#### Circuit interfaces ####
# NOTE: we should have a register type.
Yao.nqubits(peps::PEPS) = nsite(peps)
Yao.nactive(peps::PEPS) = nsite(peps)
Yao.statevec(peps::PEPS) = vec(peps)
function YaoBlocks._apply!(peps::PEPS{T}, block::PutBlock{N,1}) where {T,N}
    apply_onsite!(peps, block.locs[1], Matrix{T}(block.content))
end
function YaoBlocks._apply!(peps::PEPS{T}, block::PutBlock{N,2}) where {T,N}
    apply_onbond!(peps, block.locs..., reshape(Matrix{T}(block.content), 2, 2, 2, 2))
end
function YaoBlocks._apply!(peps::PEPS{T}, block::KronBlock{N,M,BT}) where {T,N,M,BT<:NTuple{M,AbstractBlock{1}}}
    for (loc, g) in zip(block.locs, subblocks(block))
        apply_onsite!(peps, loc[1], Matrix{T}(g))
    end
    return peps
end
function YaoBlocks._apply!(peps::PEPS{T}, block::ControlBlock{N,BT,1,1}) where {T,N,BT}
    # forward to PutBlock.
    YaoBlocks._apply!(peps, put(N, (block.ctrl_locs[1], block.locs[1])=>control(2,1,2=>block.content)))
end
function YaoBlocks._apply!(peps::PEPS{T}, block::Scale) where T
    rmul!(YaoBlocks._apply!(peps, content(block)), Yao.factor(block))
end
# compute the expectation value of a Hamiltonian
#
# ●----●----●----●   ← ⟨peps|
# ┆    ┆    ┆    ┆
# ■    ■    ■    ■   ← (product operator)
# ┆    ┆    ┆    ┆
# ●----●----●----●   ← |peps⟩
function Yao.expect(operator::Add, pa::PEPS{T}, pb::PEPS{T}) where T
    res = 0.0im
    for term in Yao.subblocks(operator)
        res += expect(term, pa, pb)
    end
    return res
end
function Yao.expect(operator::AbstractBlock, pa::PEPS{T}, pb::PEPS{T}) where {T}
    inner_product(pa, apply(pb, operator))
end