for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, xs::Union{Tuple{<:DiffTensor}, Tuple{<:DiffTensor,<:DiffTensor}, Tuple{<:DiffTensor,<:AbstractArray}, Tuple{<:AbstractArray,<:DiffTensor}}, size_dict::Dict)
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
function Base.broadcasted(::typeof(/), x::DiffTensor{T1,N}, y::DiffTensor{T2,N}) where {T1,T2,N}
    res = asarray(x.data ./ y.data, x.data)
    difftensor(res, debug_info("./", x, y), x=>dz->projectto(x, dz ./ conj(y)), y=>dz->projectto(y, -dz .* conj(x ./ y .^ 2)))
end
function Base.broadcasted(::typeof(inv), x::DiffTensor{T1,N}) where {T1,N}
    res = asarray(inv.(x.data), x.data)
    difftensor(res, debug_info("inv.", x), x=>dz->projectto(x, -dz .* conj(inv.(x) .^ 2)))
end
function Base.broadcasted(::typeof(^), x::DiffTensor, y::Number)
    res = asarray(x.data .^ y, x.data)
    difftensor(res, debug_info(".^", x, y), x=>dz-> dz .* conj(y * x .^ (y-1)))
end
# literal and variables are called into different functions.
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::DiffTensor, ::Val{y}) where y
    res = asarray(Base.broadcast(Base.literal_pow, ^, x.data, Val(y)), x.data)
    difftensor(res, debug_info(".^", x, y), x=>dz-> dz .* conj(y * x .^ (y-1)))
end
function Base.broadcasted(::typeof(sqrt), x::DiffTensor)
    res = asarray(sqrt.(x.data), x.data)
    difftensor(res, debug_info("sqrt.", x), x=>dz-> 0.5 * dz ./ conj(sqrt.(x)))
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
    difftensor(projectto(x.data, transpose(x.data)),
        debug_info(transpose, x), x=>transpose)
end

function Base.reshape(x::DiffTensor, sz::NTuple{N,Int}) where N
    size0 = size(x)
    difftensor(reshape(x.data, sz), debug_info(reshape, x, sz),
        x=>dy->reshape(dy, size0))
end

function Base.:(*)(a::Number, b::DiffTensor{T,N}) where {T,N}
    res = asarray(a * b.data, b.data)
    difftensor(res, debug_info("*", a, b), b=>dy->projectto(b, dy * conj(a)))
end
Base.:(*)(a::DiffTensor{T,N}, b::Number) where {T,N} = b * a
Base.:(*)(a::DiffTensor{T1,2}, b::DiffTensor{T2,2}) where {T1,T2} = ein"ij,jk->ik"(a, b)

# scalar-array multiply
function Base.:(*)(a::DiffTensor{T1,0}, b::DiffTensor{T2,N}) where {T1,T2,N}
    res = asarray(a.data[] * b.data, b.data)
    difftensor(res, debug_info("*", a, b), a=>dy->projectto(a, sum(dy .* conj(b))), b=>dy->projectto(b, dy * conj(a)))
end
# array-scalar multiply
Base.:(*)(a::DiffTensor{T1,N}, b::DiffTensor{T2,0}) where {T1,T2,N} = b * a
# scalar multiply
function Base.:(*)(a::DiffTensor{T1,0}, b::DiffTensor{T2,0}) where {T1,T2}
    difftensor(asarray(a.data[] * b.data[], a.data), debug_info("*", a, b), a=>dz->projectto(a, conj(b) * dz), b=>dz->projectto(b, conj(a) * dz))
end

# sum and fill
function Base.sum(a::DiffTensor{T,N}) where {T,N}
    difftensor(asarray(sum(a.data), a.data), debug_info("sum", a), a=>dz->fill(dz, size(a)...))
end
function Base.fill(a::DiffTensor{T,0}, size::Union{Integer, AbstractUnitRange}...) where {T}
    difftensor(fill(a.data[], size...), debug_info("fill", a), a=>dz->sum(dz))
end

function Base.broadcasted(::typeof(*), a::DiffTensor{T1,N}, b::DiffTensor{T2,N}) where {T1,T2,N}
    res = asarray(a.data .* b.data, a.data)
    difftensor(res, debug_info(".*", a, b), a=>dy->dy .* conj(b), b=>dy->dy .* conj(a))
end
function Base.broadcasted(::typeof(sin), a::DiffTensor{T}) where T
    res = asarray(sin.(a.data), a.data)
    difftensor(res, debug_info("sin.", a), a=>dy -> dy .* conj(cos.(a)))
end
function Base.broadcasted(::typeof(cos), a::DiffTensor{T}) where T
    res = asarray(cos.(a.data), a.data)
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
    return Complex.(x)
end
function projectto(::DiffTensor{<:Complex}, x::DiffTensor{<:Complex})
    return x
end
projectto(x::DiffTensor) = y->projectto(x, y)
function projectto(x::AbstractArray, y::LinearAlgebra.Transpose)
    return typeof(x)(y)
end

### COMPLEX ###
function Base.real(x::DiffTensor)
    difftensor(real(x.data), debug_info(real, x), x=>projectto(x))
end
function Base.imag(x::DiffTensor{T}) where T<:Complex
    difftensor(imag(x.data), debug_info(imag, x), x=>dz->(im)*dz)
end
function Base.conj(x::DiffTensor)
    difftensor(conj(x.data), debug_info(conj, x), x=>conj)
end
function Base.broadcasted(::typeof(abs), x::DiffTensor)
    res = asarray(abs.(x.data), x.data)
    difftensor(res, debug_info("abs.", x), x=>dy->sign.(x) .* dy)
end
function Base.broadcasted(::typeof(sign), x::DiffTensor)
    x ./ abs.(x)
end
Base.broadcasted(::Type{T}, x::DiffTensor{<:Complex}) where T<:Complex = x
function Base.broadcasted(::Type{T}, x::DiffTensor{<:Real}) where T<:Complex
    res = asarray(T.(x.data), x.data)
    difftensor(res, debug_info("Complex.", x), x=>dy->real(dy))
end

##### maps
Base.map(::OMEinsum.ProjectTo{T}, x::DiffTensor) where T<:Complex = T.(x)
Base.map(::OMEinsum.ProjectTo{T}, x::DiffTensor) where T<:Real = real(x)


# TODO
# complex.
# accum!