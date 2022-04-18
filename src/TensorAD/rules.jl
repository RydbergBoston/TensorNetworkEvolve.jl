#_rrule(f, args...; kwargs...) = ChainRules.rrule(f, args...; kwargs...)

function difftensor(data::AbstractArray, debug_info, backs::Pair...)
    DiffTensor(data, any(pair->any(t->t.tracker.requires_grad, pair.first), backs), BackInfo(debug_info, backs...))
end

for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, @nospecialize(xs::NTuple{N,DiffTensor} where N), size_dict::Dict)
        y = einsum(code, getdata.(xs), size_dict)
        return difftensor(y, debug_info(einsum, code, xs, size_dict), ntuple(i->((xs[i],)=>
            dy->(OMEinsum.einsum_grad(OMEinsum.getixs(code), xs, OMEinsum.getiy(code), size_dict, conj(dy), i),)
            ), length(xs))...)
    end
end

function Base.:(+)(x::DiffTensor, y::DiffTensor)
    difftensor(x.data + y.data, debug_info(+, x, y), (x,)=>dy->(dy,), (y,)=>dy->(dy,))
end
function Base.copy(x::DiffTensor)
    difftensor(copy(x.data), debug_info(copy, x), (x,)=>dy->(dy,))
end
function Base.conj(x::DiffTensor)
    difftensor(conj(x.data), debug_info(conj, x), (x,)=>dy->(conj(dy),))
end
function Base.getindex(x::DiffTensor, index1::Int, indices::Int...)
    error("get element is forbidden!")
end
function Base.getindex(x::DiffTensor, indices::Union{Int,AbstractRange,Colon}...)
    difftensor(Base.getindex(x.data, indices...), debug_info(getindex, x, indices...),
        (x,)=>dy->(accum(zero(x), indices, dy),))
end

function accum(x::AbstractArray, indices, y::AbstractArray)
    z = copy(x)
    z[indices...] .+= y
    return z
end
function accum(x::DiffTensor, indices, y::DiffTensor)
    difftensor(accum(x.data, indices, y.data),
        debug_info(accum, x, indices, y), (x,)=>dz->(dz,), (y,)=>dz->(dz[indices...],))
end
function Base.transpose(x::DiffTensor)
    difftensor(transpose(x.data),
        debug_info(transpose, x), (x,)=>dy->(transpose(dy),))
end

function Base.reshape(x::DiffTensor, sz::Int...)
    size0 = size(x)
    difftensor(reshape(x.data, sz...), debug_info(reshape, x, sz...),
        (x,)=>dy->(reshape(dy, size0...),))
end

function Base.broadcast(::typeof(*), a::DiffTensor{T}, b::DiffTensor{T}) where T
    difftensor(sin.(a.data), debug_info(broadcast, sin, a), (a,)=>dy->dy .* cos.(b), (b,)=>dy .* a)
end
function Base.broadcast(::typeof(sin), a::DiffTensor{T}) where T
    difftensor(sin.(a.data), debug_info(broadcast, sin, a), (a,)=>dy .* cos.(a))
end
function Base.broadcast(::typeof(cos), a::DiffTensor{T}) where T
    difftensor(sin.(a.data), debug_info(broadcast, sin, a), (a,)=>-dy .* sin.(a))
end
function Base.cat(a::DiffTensor, B::DiffTensor...; dims)
    A = (a, B...)
    ends = ntuple(i->cumsum(size.(A, i)), ndims(a))
    difftensor(cat(getdata.(A)...; dims=dims), 
        debug_info(cat, a, B...; dims),
        ntuple(i->(A[i],)=>dy->(getindex(dy, ntuple(idim->idim ∈ dims ? ((i==1 ? 1 : ends[idim][i-1]+1):ends[idim][i]) : (:), ndims(dy))...),), length(A))...
    )
end
function Base.vcat(a::DiffTensor{T, 1}...) where T
    cat(a...; dims=1)
end
function Base.hcat(a::DiffTensor{T, 1}...) where T
    cat(a...; dims=2)
end