using OMEinsum
using ChainRules
using ChainRules: NoTangent, unthunk

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
    y = f(args...; kwargs...)
    if ndims(y) != 0 || !(eltype(y) <: Real)
        @warn "differentiating a tensor not a scalar real number, got eltype `$(eltype(y))` and rank `$(ndims(y))`"
    end
    y.grad = DiffTensor(fill(one(eltype(y)), size(y.data)...); requires_grad=true)
    back!(y)
    return getgrad.(args)
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
                t.grad = unthunk(g)
            else
                t.grad = t.grad + unthunk(g)
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
    DiffTensor(x.data + y.data; requires_grad=x.requires_grad || y.requires_grad, info=BackInfo(+, (x, y), dy->(dy, dy)))
end

function Base.conj(x::DiffTensor)
    DiffTensor(conj(x.data); requires_grad=x.requires_grad, info=BackInfo(conj, (x,), dy->(conj(dy),)))
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
    gs = getdata.(gradient(f, x, y))
    hs = ForwardDiff.gradient(x->f(reshape(x[1:100], 10, 10), reshape(x[101:200], 10, 10))[], vcat(vec(x), vec(y)))
    @test vcat(vec(gs[1]), vec(gs[2])) ≈ hs
end

@testset "hessian" begin
    function f(x, y)
        z = ein"ij,jk->ik"(x, y)
        return ein"ii->"(z)
    end
    x = DiffTensor(randn(10, 10); requires_grad=true)
    y = DiffTensor(randn(10, 10); requires_grad=true)
    gs = gradient(f, x, y)
    gs[1].grad = DiffTensor(fill(one(eltype(gs[1])), size(gs[1].data)...); requires_grad=true)
    back!(gs[1])
    @show x.grad
    #hs = ForwardDiff.hessian(x->f(reshape(x[1:100], 10, 10), reshape(x[101:200], 10, 10))[], vcat(vec(x), vec(y)))
    #@show hs
    #@test vcat(vec(gs[1]), vec(gs[2])) ≈ hs
end