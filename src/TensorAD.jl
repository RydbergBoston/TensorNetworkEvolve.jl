using OMEinsum
using ChainRules

mutable struct DiffTensor{T,N,AT<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::AT
    requires_grad::Bool
    grad_initialized::Bool
    parent::Union{Nothing, Tuple}
    grad::DiffTensor{T,N,AT}
    function DiffTensor(data::AT, requires_grad::Bool, grad::DiffTensor{T,N,AT}, parent::Tuple) where {T,N,AT<:AbstractArray{T,N}}
        @assert length(parent) == 3
        new{T,N,AT}(data, requires_grad, false, grad, parent)
    end
    function DiffTensor(data::AT; requires_grad::Bool=false, parent=nothing) where {T,N,AT<:AbstractArray{T,N}}
        if parent !== nothing
            @assert length(parent) == 3
        end
        new{T,N,AT}(data, requires_grad, false, parent)
    end
end

#Base.zero(x::Diffractor) =
Base.size(x::DiffTensor, indices...) = Base.size(x.data, indices...)
getdata(x::DiffTensor) = x.data
getgrad(x::DiffTensor) = x.grad
getgrad(::Any) = nothing  # return nothing for non-DiffTensor

# does not return gradients in kwargs
# always return a tuple
function gradient(f, args...; kwargs...)
    args = DiffTensor.(args)
    y = f(args...; kwargs...)
    if ndims(y) != 0 || !(eltype(y) <: Real)
        @warn "differentiating a tensor not a scalar real number, got eltype `$(eltype(y))` and rank `$(ndims(y))`"
    end
    y.grad = DiffTensor(fill(one(eltype(y)), size(y.data)...); requires_grad=false)
    back!(y)
    return getgrad.(args)
end

function back!(y::DiffTensor)
    @debug "Differentiating $(y.parent)"
    grads = _rrule(y.parent[1], y.parent[2]...; y.parent[3]...)
    for (t, g) in zip(y.parent, grads)
        if t isa DiffTensor && t.requires_grad
            if !isdefined(t, :grad)
                t.grad = g
            else
                t.grad = t.grad + g
            end
            back!(t)
        end
    end
end

_rrule(f, args...; kwargs...) = ChainRules.rrule(f, args...; kwargs...)

for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, @nospecialize(xs::NTuple{N,DiffTensor} where N), size_dict::Dict)
        y = einsum(code, getdata.(xs), size_dict)
        requires_grad = any(x->x.requires_grad, xs)
        parent = (einsum, (code, xs, size_dict), ())
        return DiffTensor(y; requires_grad, parent)
    end
end
function Base:(+)(x::DiffTensor, y::AbstractArray)
    DiffTensor(x.data + y, )
end

using Test, OMEinsum

@testset "diff tensor" begin
    function f(x, y)
        return ein"ij,jk->ik"(x, y)
    end
    x = randn(10, 10)
    y = randn(10, 10)
    gs = gradient(f, x, y)
    @show gs
end