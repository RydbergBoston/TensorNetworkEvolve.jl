using OMEinsum
using ChainRules

struct BackInfo
    f
    args
    back
end

mutable struct DiffTensor{T,N,AT<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::AT
    requires_grad::Bool
    info::BackInfo
    grad::DiffTensor{T,N,AT}
    function DiffTensor(data::AT, requires_grad::Bool, info, grad::DiffTensor{T,N,AT}) where {T,N,AT<:AbstractArray{T,N}}
        new{T,N,AT}(data, requires_grad, info, grad)
    end
    function DiffTensor(data::AT; requires_grad::Bool, info=nothing) where {T,N,AT<:AbstractArray{T,N}}
        if info === nothing
            return new{T,N,AT}(data, requires_grad)
        else
            return new{T,N,AT}(data, requires_grad, info)
        end
    end
end

#Base.zero(x::Diffractor) =
Base.getindex(x::DiffTensor, indices...) = Base.getindex(x.data, indices...)
Base.size(x::DiffTensor, indices...) = Base.size(x.data, indices...)
getdata(x::DiffTensor) = x.data
getgrad(x::DiffTensor) = x.grad
getgrad(::Any) = nothing  # return nothing for non-DiffTensor
function Base.show(io::IO, x::DiffTensor)
    s = "$(typeof(x))"
    if x.requires_grad
        s *= "(gradient required)"
    end
    if isdefined(x, :grad)
        s *= " (gradient computed)"
    end
    print(io, s)
end

# does not return gradients in kwargs
# always return a tuple
function gradient(f, args...; kwargs...)
    args = DiffTensor.(args; requires_grad=true)
    y = f(args...; kwargs...)
    if ndims(y) != 0 || !(eltype(y) <: Real)
        @warn "differentiating a tensor not a scalar real number, got eltype `$(eltype(y))` and rank `$(ndims(y))`"
    end
    y.grad = DiffTensor(fill(one(eltype(y)), size(y.data)...); requires_grad=false)
    back!(y)
    return getdata.(getgrad.(args))
end

function back!(y::DiffTensor)
    f, args, back = y.info.f, y.info.args, y.info.back
    @debug "Differentiating $(f) on argument types $(typeof(args))"
    grads = back(y.grad)
    _extract_grads!(args, grads)
end

function _extract_grads!(args::Tuple, grads::Tuple)
    for (t, g) in zip(args, grads)
        if t isa DiffTensor && t.requires_grad
            # accumulate or create
            if !isdefined(t, :grad)
                t.grad = DiffTensor(g; requires_grad=true)
            else
                t.grad = t.grad + DiffTensor(g; requires_grad=true)
            end
            # has parents
            if isdefined(t, :info)
                back!(t)
            end
        elseif t isa Tuple && g isa Tuple
            # recurse on tuples
            _extract_grads!(t, g)
        end
    end
end

_rrule(f, args...; kwargs...) = ChainRules.rrule(f, args...; kwargs...)

for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, @nospecialize(xs::NTuple{N,DiffTensor} where N), size_dict::Dict)
        y, back = _rrule(einsum, code, getdata.(xs), size_dict)
        requires_grad = any(x->x.requires_grad, xs)
        return DiffTensor(y; requires_grad, info=BackInfo(einsum, (code, xs, size_dict), (args...; kwargs...)->ChainRules.unthunk.(back(args...; kwargs...)[2:end])))
    end
end

function Base.:(+)(x::DiffTensor, y::DiffTensor)
    DiffTensor(x.data + y.data; requires_grad=x.requires_grad || y.requires_grad, back)
end

using Test, OMEinsum
using ForwardDiff

@testset "diff tensor" begin
    function f(x, y)
        z = ein"ij,jk->ik"(x, y)
        return ein"ii->"(z)
    end
    x = randn(10, 10)
    y = randn(10, 10)
    gs = gradient(f, x, y)
    hs = ForwardDiff.gradient(x->f(reshape(x[1:100], 10, 10), reshape(x[101:200], 10, 10))[], vcat(vec(x), vec(y)))
    @test vcat(vec(gs[1]), vec(gs[2])) ≈ hs
end