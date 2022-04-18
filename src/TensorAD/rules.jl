#_rrule(f, args...; kwargs...) = ChainRules.rrule(f, args...; kwargs...)

for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, @nospecialize(xs::NTuple{N,DiffTensor} where N), size_dict::Dict)
        y = einsum(code, getdata.(xs), size_dict)
        requires_grad = any(x->x.tracker.requires_grad, xs)
        return DiffTensor(y; requires_grad, info=BackInfo(debug_info(einsum, code, xs, size_dict), ntuple(i->((xs[i],)=>
            dy->(OMEinsum.einsum_grad(OMEinsum.getixs(code), xs, OMEinsum.getiy(code), size_dict, conj(dy), i),)
            ), length(xs))...))
    end
end

function Base.:(+)(x::DiffTensor, y::DiffTensor)
    DiffTensor(x.data + y.data, x.tracker.requires_grad || y.tracker.requires_grad, BackInfo(debug_info(+, x, y), (x,)=>dy->(dy,), (y,)=>dy->(dy,)))
end
function Base.copy(x::DiffTensor)
    DiffTensor(copy(x.data), x.tracker.requires_grad, BackInfo(debug_info(copy, x), (x,)=>dy->(dy,)))
end
function Base.conj(x::DiffTensor)
    DiffTensor(conj(x.data), x.tracker.requires_grad, BackInfo(debug_info(conj, x), (x,)=>dy->(conj(dy),)))
end
function Base.getindex(x::DiffTensor, index1::Int, indices::Int...)
    error("get element is forbidden!")
end
function Base.getindex(x::DiffTensor, indices::Union{Int,AbstractRange,Colon}...)
    DiffTensor(Base.getindex(x.data, indices...),
        x.tracker.requires_grad, BackInfo(debug_info(getindex, x, indices...),
        (x,)=>dy->(accum(zero(x), indices, dy),)))
end

function accum(x::AbstractArray, indices, y::AbstractArray)
    z = copy(x)
    z[indices...] .+= y
    return z
end
function accum(x::DiffTensor, indices, y::DiffTensor)
    DiffTensor(accum(x.data, indices, y.data), x.tracker.requires_grad,
        BackInfo(debug_info(accum, x, indices, y), (x,)=>dz->(dz,), (y,)=>dz->(dz[indices...],)))
end
function Base.transpose(x::DiffTensor)
    DiffTensor(transpose(x.data), x.tracker.requires_grad,
        BackInfo(debug_info(transpose, x), (x,)=>dy->(transpose(dy),)))
end

function Base.reshape(x::DiffTensor, sz::Int...)
    size0 = size(x)
    DiffTensor(reshape(x.data, sz...), x.tracker.requires_grad, BackInfo(debug_info(reshape, x, sz...),
        (x,)=>dy->(reshape(dy, size0...),)))
end

function Base.broadcast(::typeof(*), a::DiffTensor{T}, b::DiffTensor{T}) where T
    DiffTensor(sin.(a.data), requires_grad=a.tracker.requires_grad || b.tracker.requires_grad, BackInfo(debug_info(broadcast, sin, a), (a,)=>dy->dy .* cos.(b), (b,)=>dy .* a))
end
function Base.broadcast(::typeof(sin), a::DiffTensor{T}) where T
    DiffTensor(sin.(a.data), requires_grad=a.tracker.requires_grad, BackInfo(debug_info(broadcast, sin, a), (a,)=>dy .* cos.(a)))
end
function Base.broadcast(::typeof(cos), a::DiffTensor{T}) where T
    DiffTensor(sin.(a.data), requires_grad=a.tracker.requires_grad, BackInfo(debug_info(broadcast, sin, a), (a,)=>-dy .* sin.(a)))
end
function Base.cat(a::DiffTensor, B::DiffTensor...; dims)
    A = (a, B...)
    ends = ntuple(i->cumsum(size.(A, i)), ndims(a))
    DiffTensor(cat(getdata.(A)...; dims=dims), any(t->t.tracker.requires_grad, A), BackInfo(
        debug_info(cat, a, B...; dims),
        ntuple(i->(A[i],)=>dy->(getindex(dy, ntuple(idim->idim ∈ dims ? ((i==1 ? 1 : ends[idim][i-1]+1):ends[idim][i]) : (:), ndims(dy))...),), length(A))...
    ))
end
function Base.vcat(a::DiffTensor{T, 1}...) where T
    cat(a...; dims=1)
end
function Base.hcat(a::DiffTensor{T, 1}...) where T
    cat(a...; dims=2)
end