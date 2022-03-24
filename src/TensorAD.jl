mutable struct DiffTensor{T,N,AT<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::AT
    requires_grad::Bool
    grad::DiffTensor{TT}
    parent::Tuple
    function DiffTensor(data::AT, requires_grad::Bool, grad::DiffTensor{T,N,AT}, parent::Tuple) where {T,N,AT<:AbstractArray{T,N}}
        new{T,N,AT}(data, requires_grad, grad, parent)
    end
    function DiffTensor(data::AT; requires_grad=false) where {T,N,AT<:AbstractArray{T,N}}
        if requires_grad
            new{T,N,AT}(data, requires_grad, zero(data))
        else
            new{T,N,AT}(data, requires_grad)
        end
    end
end

Base.zero(x::Diffractor)
getdata(x::DiffTensor) = x.data
getgrad(x::DiffTensor) = x.grad
getgrad(::Any) = nothing  # return nothing for non-DiffTensor

# does not return gradients in kwargs
# always return a tuple
function gradient(f, args...; kwargs...)
    y = f(args...; kwargs...)
    if ndims(y) != 0 || !(eltype(y) <: Real)
        @warn "differentiating a tensor not a scalar real number, got eltype `$(eltype(y))` and rank `$(ndims(y))`"
    end
    y.grad .= fill(one(eltype(y)), size(y.data)...)
    back!(y)
    return getgrad.(args)
end

function back!(y::DiffTensor)
    @debug "Differentiating $(y.parent)"
    grads = ChainRules.rrule(y.parent...)
    for (t, g) in zip(y.parent, grads)
        if t isa DiffTensor && t.requires_grad
            t.grad .+= g
            back!(t)
        end
    end
end


for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, @nospecialize(xs::NTuple{N,DiffTensor} where N), size_dict::Dict)
        einsum(code, xs, size_dict)
    end
end

using Test, OMEinsum

@testset "diff tensor" begin
    function f(x, y)
        return ein"ij,jk->ik"(x, y)
    end
    gs = gradient(f, x, y)
end