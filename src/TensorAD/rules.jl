function difftensor(data::AbstractArray, debug_info, backs::Pair...)
    # filter out not required AD rules.
    mask = [requires_grad(pair.first) for pair in backs]
    t = DiffTensor(data, any(mask))
    push!(GLOBAL_TAPE.instructs, Instruction(debug_info, t,backs[mask]...))
    return t
end
requires_grad(t::DiffTensor) = t.requires_grad
requires_grad(t::Tuple) = any(x->requires_grad(x), t)

for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, @nospecialize(xs::NTuple{N,DiffTensor} where N), size_dict::Dict)
        y = einsum(code, getdata.(xs), size_dict)
        return difftensor(y, debug_info(einsum, code, xs, size_dict), ntuple(i->(xs[i]=>
            dy->OMEinsum.einsum_grad(OMEinsum.getixs(code), xs, OMEinsum.getiy(code), size_dict, conj(dy), i)
            ), length(xs))...)
    end
end

function Base.:(+)(x::DiffTensor, y::DiffTensor)
    difftensor(x.data + y.data, debug_info(+, x, y), x=>projectto(x), y=>projectto(y))
end
function Base.:(-)(x::DiffTensor, y::DiffTensor)
    difftensor(x.data - y.data, debug_info(-, x, y), x=>projectto(x), y=>dz->projectto(y,-dz))
end
function Base.:(-)(x::DiffTensor)
    difftensor(-x.data, debug_info(-, x), x=>(-))
end
function Base.broadcasted(::typeof(/), x::DiffTensor, y::DiffTensor)
    difftensor(x.data ./ y.data, debug_info("./", x, y), x=>dz->projectto(x, dz ./ conj(y)), y=>dz->projectto(y, -dz .* conj(x ./ Base.broadcasted(^, y, 2))))
end
function Base.broadcasted(::typeof(^), x::DiffTensor, y::Number)
    difftensor(x.data .^ y, debug_info(".^", x, y), x=>dz-> dz .* conj(y * x .^ (y-1)))
end
function Base.broadcasted(::typeof(sqrt), x::DiffTensor)
    difftensor(sqrt.(x.data), debug_info("sqrt.", x), x=>dz-> 0.5 * dz ./ conj(sqrt.(x)))
end
function Base.copy(x::DiffTensor)
    difftensor(copy(x.data), debug_info(copy, x), x=>identity)
end
function Base.getindex(x::DiffTensor, index1::Int, indices::Int...)
    error("get element is forbidden!")
end
function Base.getindex(x::DiffTensor, indices::Union{Int,AbstractRange,Colon}...)
    difftensor(Base.getindex(x.data, indices...), debug_info(getindex, x, indices...),
        x=>dy->accum(zero(x), indices, dy))
end

function accum(x::AbstractArray, indices, y::AbstractArray)
    z = copy(x)
    z[indices...] .+= y
    return z
end
function accum(x::DiffTensor, indices, y::DiffTensor)
    difftensor(accum(x.data, indices, y.data),
        debug_info(accum, x, indices, y), x=>projectto(x), y=>dz->projectto(y, dz[indices...]))
end
function Base.transpose(x::DiffTensor)
    difftensor(transpose(x.data),
        debug_info(transpose, x), x=>transpose)
end

function Base.reshape(x::DiffTensor, sz::Int...)
    size0 = size(x)
    difftensor(reshape(x.data, sz...), debug_info(reshape, x, sz...),
        x=>dy->reshape(dy, size0...))
end

function Base.:(*)(a::Number, b::DiffTensor{T,N}) where {T,N}
    res = N == 0 ? fill(a * b.data[]) : a * b.data
    difftensor(res, debug_info("*", a, b), b=>dy->projectto(b, dy * conj(a)))
end
Base.:(*)(a::DiffTensor{T,N}, b::Number) where {T,N} = b * a
Base.:(*)(a::DiffTensor{T1,2}, b::DiffTensor{T2,2}) where {T1,T2} = ein"ij,jk->ik"(a, b)

function Base.broadcasted(::typeof(*), a::DiffTensor{T1,N}, b::DiffTensor{T2,N}) where {T1,T2,N}
    res = N == 0 ? fill(a.data[] * b.data[]) : a.data .* b.data
    difftensor(res, debug_info(".*", a, b), a=>dy->dy .* conj(b), b=>dy->dy .* conj(a))
end
function Base.broadcasted(::typeof(sin), a::DiffTensor{T}) where T
    res = ndims(a) == 0 ? fill(sin(a.data[])) : sin.(a.data)
    difftensor(res, debug_info("sin.", a), a=>dy -> dy .* conj(cos.(a)))
end
function Base.broadcasted(::typeof(cos), a::DiffTensor{T}) where T
    res = ndims(a) == 0 ? fill(cos(a.data[])) : cos.(a.data)
    difftensor(res, debug_info("cos.", a), a=>dy -> -dy .* conj(sin.(a)))
end
function Base.cat(a::DiffTensor{T}, B::DiffTensor{T}...; dims) where T
    A = (a, B...)
    ends = ntuple(i->cumsum(size.(A, i)), ndims(a))
    difftensor(cat(getdata.(A)...; dims=dims), 
        debug_info(cat, a, B...; dims),
        ntuple(i->A[i]=>dy->getindex(dy, ntuple(idim->idim ∈ dims ? ((i==1 ? 1 : ends[idim][i-1]+1):ends[idim][i]) : (:), ndims(dy))...), length(A))...
    )
end
function Base.vcat(a::DiffTensor{T, 1}...) where T
    cat(a...; dims=1)
end
function Base.hcat(a::DiffTensor{T, 1}...) where T
    cat(a...; dims=2)
end
function projectto(::DiffTensor{<:Real}, x::DiffTensor{<:Complex})
    return real(x)
end
function projectto(::DiffTensor{<:Real}, x::DiffTensor{<:Real})
    return x
end
function projectto(::DiffTensor{<:Complex}, x::DiffTensor{<:Real})
    return x + im * zero(x)
end
function projectto(::DiffTensor{<:Complex}, x::DiffTensor{<:Complex})
    return x
end
projectto(x::DiffTensor) = y->projectto(x, y)

### COMPLEX ###
function Base.real(x::DiffTensor)
    difftensor(real(x.data), debug_info(real, x), x=>projectto(x))
end
function Base.imag(x::DiffTensor{T}) where T<:Complex
    difftensor(imag(x.data), debug_info(imag, x), x=>x->(im)*x)
end
function Base.conj(x::DiffTensor)
    difftensor(conj(x.data), debug_info(conj, x), x=>conj)
end
function Base.broadcasted(::typeof(abs), x::DiffTensor)
    difftensor(abs.(x.data), debug_info("abs.", x), x=>dy->sign.(x) .* dy)
end
function Base.broadcasted(::typeof(sign), x::DiffTensor)
    x ./ abs.(x)
end

# TODO
# complex.
# accum!
# inv