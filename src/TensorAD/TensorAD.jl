module TensorAD
export DiffTensor

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
    function DiffTensor(data::AT, requires_grad::Bool, info) where {T,N,AT<:AbstractArray{T,N}}
        new{T,N,AT}(data, requires_grad, info)
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
Base.size(x::DiffTensor, indices...) = Base.size(x.data, indices...)
getdata(x::DiffTensor) = x.data
function getgrad(d::AbstractDict, x::DiffTensor)
    key = objectid(x)
    if haskey(d, key)
        return d[key]
    else
        return zero(x)
    end
end
getgrad(d::AbstractDict, ::Any) = nothing  # return nothing for non-DiffTensor
function accumulate_gradient!(grad_storage, t, g)
    key = objectid(t)
    # accumulate or create
    if !haskey(grad_storage, key)
        grad_storage[key] = unthunk(g)
    else
        grad_storage[key] = grad_storage[key] + unthunk(g)
    end
end

function Base.show(io::IO, x::DiffTensor)
    sz = join(string.(size(x)), "×")
    s = "$(typeof(x))[$sz]"
    if x.requires_grad
        s *= "(gradient required)"
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
    grad_storage = Dict{UInt,Any}()
    accumulate_gradient!(grad_storage, y, DiffTensor(fill(one(eltype(y)), size(y.data)...); requires_grad=true))
    back!(y, grad_storage)
    return getgrad.(Ref(grad_storage), args)
end

function back!(y::DiffTensor, grad_storage::AbstractDict)
    f, args, back = y.info.f, y.info.args, y.info.back
    @debug "Differentiating $(f) on argument types $(typeof(args))"
    grads = back(getgrad(grad_storage, y))
    _extract_grads!(grad_storage, args, grads)
end

function _extract_grads!(grad_storage, args::Tuple, grads::Tuple)
    for (t, g) in zip(args, grads)
        if t isa DiffTensor && t.requires_grad
            accumulate_gradient!(grad_storage, t, g)
            # has parents
            if isdefined(t, :info)
                back!(t, grad_storage)
            end
        elseif t isa Tuple && g isa Tuple
            # recurse on tuples
            _extract_grads!(grad_storage, t, g)
        end
    end
end

function hessian(f, x::AbstractVector{T}) where T
    gx, = gradient(f, x)

    slices = typeof(x)[]
    for i=1:length(x)
        grad_storage = Dict{UInt,Any}()
        hx = DiffTensor(fill(one(eltype(gx)), size(gx.data)...); requires_grad=true)
        accumulate_gradient!(grad_storage, gx, hx)
        back!(gx, grad_storage)
        getgrad(grad_storage, x)
    end
end

include("rules.jl")
end