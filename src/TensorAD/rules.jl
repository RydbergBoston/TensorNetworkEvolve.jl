_rrule(f, args...; kwargs...) = ChainRules.rrule(f, args...; kwargs...)

for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, @nospecialize(xs::NTuple{N,DiffTensor} where N), size_dict::Dict)
        y = einsum(code, getdata.(xs), size_dict)
        function einsum_pullback(dy)
            dxs = ntuple(i -> ChainRules.@thunk(OMEinsum.einsum_grad(OMEinsum.getixs(code), xs, OMEinsum.getiy(code), size_dict, conj(dy), i)), length(xs))
            return (NoTangent(), dxs, NoTangent())
        end
        requires_grad = any(x->x.requires_grad, xs)
        return DiffTensor(y; requires_grad, info=BackInfo(einsum, (code, xs, size_dict), einsum_pullback))
    end
end

function Base.:(+)(x::DiffTensor, y::DiffTensor)
    DiffTensor(x.data + y.data, x.requires_grad || y.requires_grad, BackInfo(+, (x, y), dy->(dy, dy)))
end

function Base.conj(x::DiffTensor)
    DiffTensor(conj(x.data), x.requires_grad, BackInfo(conj, (x,), dy->(conj(dy),)))
end
function Base.getindex(x::DiffTensor, indices::AbstractRange)
    DiffTensor(DiffTensor(Base.getindex(x.data, indices); requires_grad=x.requires_grad),
        x.requires_grad, BackInfo(getindex, (x, indices), function (dy)
        dx = zero(x)
        dx_ = accum(dx, indices, dy)
        (dx_, NoTangent())
    end))
end
function accum(x::DiffTensor, indices, y::DiffTensor)
    z = copy(x)
    z.data[indices] .+= y.data
    DiffTensor(z, x.requires_grad,
        BackInfo(accum, (x, indices..., y),
        function (dz)
            dz, NoTangent(), dz[indices]
        end
        ))
end

function reshape(x::DiffTensor, sz...)
    size0 = size(x)
    DiffTensor(reshape(x.data, sz...), x.requires_grad, BackInfo(reshape, (x, sz...), dy->(reshape(dy, size0...), ntuple(i->NoTangent(), length(sz))...)))
end

