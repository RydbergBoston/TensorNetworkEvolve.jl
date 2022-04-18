module TensorAD
export DiffTensor

using OMEinsum
using ChainRules
import AbstractTrees

const ADTypes = Union{Float32, Float64, ComplexF64, ComplexF32}

struct PartialBack
    trackers::Tuple  # contents are Trackers
    back
end
struct BackInfo
    info::String
    backs::NTuple{N,PartialBack} where N
end
struct Tracker
    id::UInt
    requires_grad::Bool
    info::BackInfo
end
struct DiffTensor{T<:ADTypes,N,AT<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::AT
    tracker::Tracker
end
function DiffTensor(data::AT, requires_grad::Bool, info) where {T,N,AT<:AbstractArray{T,N}}
    if AT <: DiffTensor
        error("DiffTensor in DiffTensor is forbidden to prevent errors.")
    end
    #DiffTensor(data, Tracker(objectid((data, info)), requires_grad, info))
    DiffTensor(data, Tracker(rand(UInt), requires_grad, info))
end
function DiffTensor(data::AT; requires_grad::Bool, info=BackInfo("INPUT", ())) where {T,N,AT<:AbstractArray{T,N}}
    #return DiffTensor(data, Tracker(objectid((data, info)), requires_grad, info))
    return DiffTensor(data, Tracker(rand(UInt), requires_grad, info))
end

function debug_info(f, args...; kwargs...)
    args = join(map(arg->"::$(typeof(arg))", args), ", ")
    if length(kwargs) != 0
        kwargs = join(["$k=$v" for (k,v) in kwargs], "")
        "∂$f($args; $kwargs)"
    else
        "∂$f($args)"
    end
end
BackInfo(info::String, backs::Pair{<:NTuple{N,DiffTensor} where N}...) = BackInfo(info, PartialBack.(backs))
function PartialBack(x::Pair)
    PartialBack(getfield.(x.first, :tracker), x.second)
end

Base.zero(x::DiffTensor) = DiffTensor(zero(x.data); requires_grad=false)
Base.size(x::DiffTensor, indices...) = Base.size(x.data, indices...)
getdata(x::DiffTensor) = x.data
getgrad(d::AbstractDict, x::DiffTensor) = getgrad(d, x.tracker.id, ()->zero(x))
function getgrad(d::AbstractDict, key::UInt, default)
    if haskey(d, key)
        return d[key]
    else
        return default()
    end
end
#getgrad(d::AbstractDict, ::Any) = nothing  # return nothing for non-DiffTensor
function accumulate_gradient!(grad_storage, key::UInt, g)
    # accumulate or create
    if !haskey(grad_storage, key)
        grad_storage[key] = g
    else
        grad_storage[key] = grad_storage[key] + g
    end
end

Base.show(io::IO, ::MIME"text/plain", x::DiffTensor) = Base.show(io, x)
function Base.show(io::IO, x::DiffTensor)
    sz = join(string.(size(x)), "×")
    s = "$(typeof(x))[$sz]"
    if x.tracker.requires_grad
        s *= " ✓"
    end
    print(io, s)
end

Base.show(io::IO, ::MIME"text/plain", tracker::Tracker) = Base.show(io, tracker)
Base.show(io::IO, tracker::Tracker) = AbstractTrees.print_tree(io, tracker; maxdepth=10)
AbstractTrees.children(tracker::Tracker) = vcat([collect(Tracker, pb.trackers) for pb in tracker.info.backs]...)
function AbstractTrees.printnode(io::IO, tracker::Tracker)
    s = tracker.info.info
    if tracker.requires_grad
        s *= " ✓"
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
    accumulate_gradient!(grad_storage, y.tracker.id, DiffTensor(ones(eltype(y), size(y.data)...); requires_grad=false))
    back!(y.tracker, grad_storage)
    return getgrad.(Ref(grad_storage), args)
end

function back!(y::Tracker, grad_storage::AbstractDict)
    debug_info, backs = y.info.info, y.info.backs
    @debug "$debug_info"
    for pb in backs
        if any(t->t.requires_grad, pb.trackers)
            grads = pb.back(grad_storage[y.id])
            _extract_grads!(grad_storage, pb.trackers, grads)
        end
    end
    # NOTE: backward must go breadth first search?
    for pb in backs
        for t in pb.trackers
            if t.requires_grad
                # has parents
                back!(t, grad_storage)
            end
        end
    end
end

function _extract_grads!(grad_storage, args::NTuple{K,Tracker}, grads::Tuple) where K
    for (t, g) in zip(args, grads)
        accumulate_gradient!(grad_storage, t.id, g)
    end
end

function hessian(f, x::DiffTensor)
    return jacobian(x->gradient(f, x)[1], x)
    # gx, = gradient(f, x)
    # slices = typeof(x)[]
    # for i=1:length(x)
    #     grad_storage = Dict{UInt,Any}()
    #     hx = zero(gx)
    #     hx.data[i] = one(T)
    #     accumulate_gradient!(grad_storage, gx.tracker.id, hx)
    #     back!(gx.tracker, grad_storage)
    #     push!(slices, getgrad(grad_storage, x))
    # end
    # return cat(slices...; dims=2)
end

function jacobian(f, x::DiffTensor{T}) where T
    slices = typeof(x)[]
    y = f(x)
    @debug y.tracker
    for i=1:length(y)
        @debug "hessian, row $i"
        grad_storage = Dict{UInt,Any}()
        gy = zero(y)
        gy.data[i] = one(T)
        accumulate_gradient!(grad_storage, y.tracker.id, gy)
        back!(y.tracker, grad_storage)
        push!(slices, getgrad(grad_storage, x))
    end
    return transpose(cat(slices...; dims=2))
end

include("rules.jl")
end