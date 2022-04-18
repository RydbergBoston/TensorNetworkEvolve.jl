#_rrule(f, args...; kwargs...) = ChainRules.rrule(f, args...; kwargs...)

function difftensor(data::AbstractArray, debug_info, backs::Pair...)
    mask = [any(t->t.requires_grad, pair.first) for pair in backs]
    t = DiffTensor(data, any(mask))
    push!(GLOBAL_TAPE.instructs, Instruction(debug_info, t,backs[mask]...))
    return t
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
    difftensor(x.data + y.data, debug_info(+, x, y), (x,)=>dz->(dz,), (y,)=>dz->(dz,))
end
function Base.:(-)(x::DiffTensor, y::DiffTensor)
    difftensor(x.data - y.data, debug_info(-, x, y), (x,)=>dz->(dz,), (y,)=>dz->(-dz,))
end
function Base.:(-)(x::DiffTensor)
    difftensor(-x.data, debug_info(-, x), (x,)=>dz->(-dz,))
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

function Base.broadcasted(::typeof(*), a::DiffTensor{T,N}, b::DiffTensor{T,N}) where {T,N}
    res = N == 0 ? fill(a.data[] * b.data[]) : a.data .* b.data
    difftensor(res, debug_info(".*", a, b), (a,)=>dy->(dy .* b,), (b,)=>dy->(dy .* a,))
end
function Base.broadcasted(::typeof(sin), a::DiffTensor{T}) where T
    res = ndims(a) == 0 ? fill(sin(a.data[])) : sin.(a.data)
    difftensor(res, debug_info("sin.", a), (a,)=>dy -> (dy .* cos.(a),))
end
function Base.broadcasted(::typeof(cos), a::DiffTensor{T}) where T
    res = ndims(a) == 0 ? fill(cos(a.data[])) : cos.(a.data)
    difftensor(res, debug_info("cos.", a), (a,)=>dy -> (-dy .* sin.(a),))
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