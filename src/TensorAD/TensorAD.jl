module TensorAD
export DiffTensor

using OMEinsum
using ChainRules
using ChainRules: NoTangent, unthunk
const ADTypes = Union{Float32, Float64, ComplexF64, ComplexF32}

struct BackInfo
    info::String
    backs::NTuple{N,Pair} where N
end
BackInfo(info::String, backs::Pair...) = BackInfo(info, backs)

function debug_info(f, args...; kwargs...)
    "∂"*string(:($f($(map(arg->Expr(:(::), typeof(arg)), args)...); $(kwargs...))))
end

struct DiffTensor{T<:ADTypes,N,AT<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::AT
    requires_grad::Bool
    info::BackInfo
    function DiffTensor(data::AT, requires_grad::Bool, info) where {T,N,AT<:AbstractArray{T,N}}
        if AT <: DiffTensor
            error("DiffTensor in DiffTensor is forbidden to prevent errors.")
        end
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

Base.zero(x::DiffTensor) = DiffTensor(zero(x.data); requires_grad=false)
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

Base.show(io::IO, ::MIME"text/plain", x::DiffTensor) = Base.show(io, x)
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
    debug_info, backs = y.info.info, y.info.backs
    @debug debug_info
    for (args, back) in backs
        if any(arg->arg.requires_grad, args)
            grads = back(getgrad(grad_storage, y))
            _extract_grads!(grad_storage, args, grads)
        end
    end
end

function _extract_grads!(grad_storage, args::Tuple, grads::Tuple)
    for (t, g) in zip(args, grads)
        if t isa DiffTensor && g isa DiffTensor
            t.requires_grad || continue
            accumulate_gradient!(grad_storage, t, g)
            # has parents
            if isdefined(t, :info)
                back!(t, grad_storage)
            end
        elseif t isa Tuple && g isa Tuple
            # recurse on tuples
            _extract_grads!(grad_storage, t, g)
        else
            error("can not extract gradients from input argument types: $(typeof(t)) and $(typeof(g))")
        end
    end
end

function hessian(f, x::AbstractVector{T}) where T
    gx, = gradient(f, x)

    slices = typeof(x)[]
    for i=1:length(x)
        grad_storage = Dict{UInt,Any}()
        hx = zero(gx)
        hx.data[i] = one(T)
        accumulate_gradient!(grad_storage, gx, hx)
        back!(gx, grad_storage)
        push!(slices, getgrad(grad_storage, x))
    end
    return cat(slices...; dims=2)
end

function jacobian(f, x::AbstractVector{T}) where T
    slices = typeof(x)[]
    y = f(x)
    for i=1:length(y)
        grad_storage = Dict{UInt,Any}()
        gy = zero(y)
        gy.data[i] = one(T)
        accumulate_gradient!(grad_storage, y, gy)
        back!(y, grad_storage)
        push!(slices, getgrad(grad_storage, x))
    end
    return transpose(cat(slices...; dims=2))
end

include("rules.jl")
end