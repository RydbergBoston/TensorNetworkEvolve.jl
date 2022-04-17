_rrule(f, args...; kwargs...) = ChainRules.rrule(f, args...; kwargs...)

for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, @nospecialize(xs::NTuple{N,DiffTensor} where N), size_dict::Dict)
        y = einsum(code, getdata.(xs), size_dict)
        function einsum_pullback(dy)
            dxs = ntuple(i -> ChainRules.@thunk(OMEinsum.einsum_grad(OMEinsum.getixs(code), xs, OMEinsum.getiy(code), size_dict, conj(dy), i)), length(xs))
            return (dxs,)
        end
        requires_grad = any(x->x.requires_grad, xs)
        return DiffTensor(y; requires_grad, info=BackInfo(einsum, (xs,), einsum_pullback))
    end
end

function Base.:(+)(x::DiffTensor, y::DiffTensor)
    DiffTensor(x.data + y.data, x.requires_grad || y.requires_grad, BackInfo(+, (x, y), dy->(dy, dy)))
end

function Base.conj(x::DiffTensor)
    DiffTensor(conj(x.data), x.requires_grad, BackInfo(conj, (x,), dy->(conj(dy),)))
end
function Base.getindex(x::DiffTensor, indices::AbstractRange...)
    DiffTensor(DiffTensor(Base.getindex(x.data, indices...); requires_grad=x.requires_grad),
        x.requires_grad, BackInfo(getindex, (x,), function (dy)
        dx = zero(x)
        dx_ = accum(dx, indices, dy)
        (dx_,)
    end))
end
function accum(x::DiffTensor, indices, y::DiffTensor)
    z = copy(x)
    z.data[indices...] .+= y.data
    DiffTensor(z, x.requires_grad,
        BackInfo(accum, (x, y),
        function (dz)
            dz, dz[indices...]
        end
        ))
end

function Base.reshape(x::DiffTensor, sz...)
    size0 = size(x)
    DiffTensor(reshape(x.data, sz...), x.requires_grad, BackInfo(:reshape, (x,),
        dy->(reshape(dy, size0...),)))
end

function Base.cat(a::DiffTensor, B::DiffTensor...; dims)
    A = (a, A...)
    ends = ntuple(i->cumsum(size.(A, dim)), ndims(a))
    DiffTensor(cat(getdata.(A)...; dims=dims), any(t->t.requires_grad, A), BackInfo(
        cat, A, function (dy)
            #return ntuple(i=>dy[ntuple(i->:, dim-1)..., (i==1 ? 1 : ends[i-1]+1):ends[i], ntuple(i->:, ndims(dy)-dim)...], length(A))
            return ntuple(i->getindex(dy, ntuple(idim->begin
                idim ∈ dims ? ((i==1 ? 1 : ends[idim][i-1]+1):ends[idim][i]) : 2
                end,
                 ndims(dy))...), length(A))
        end
    ))
end
