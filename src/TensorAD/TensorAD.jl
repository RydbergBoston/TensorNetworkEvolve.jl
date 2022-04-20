module TensorAD
export DiffTensor

using OMEinsum, LinearAlgebra

const ADTypes = Union{Float32, Float64, ComplexF64, ComplexF32}

struct PartialBack
    input_ids::Union{UInt,Tuple}
    output_ids::Union{UInt,Tuple}
    back
end
struct Instruction
    info::String
    backs::NTuple{N,PartialBack} where N
end
struct Tape
    instructs::Vector{Instruction}
end
const GLOBAL_TAPE = Tape(Instruction[])

# to make objectid work consistently
mutable struct DiffTensor{T<:ADTypes,N,AT<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::AT
    requires_grad::Bool
    function DiffTensor(data::AT, requires_grad::Bool=true) where {T,N,AT<:AbstractArray{T,N}}
        if AT <: DiffTensor
            error("DiffTensor in DiffTensor is forbidden to prevent errors.")
        end
        new{T,N,AT}(data, requires_grad)
    end
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
Instruction(info::String, y::Union{DiffTensor, Tuple}, backs::Pair{<:Union{DiffTensor, NTuple{N,DiffTensor} where N}}...) = Instruction(info, PartialBack.(Ref(y), backs))
function PartialBack(y::Union{DiffTensor, NTuple}, x::Pair)
    PartialBack(getid(x.first), getid(y), x.second)
end

getid(x::DiffTensor) = objectid(x)
getid(x::Tuple) = getid.(x)
Base.zero(x::DiffTensor) = DiffTensor(zero(x.data), false)
Base.size(x::DiffTensor, indices...) = Base.size(x.data, indices...)
getdata(x::DiffTensor) = x.data
getgrad(d::AbstractDict, x::DiffTensor) = getgrad(d, getid(x), ()->zero(x))
function getgrad(d::AbstractDict, key::UInt, default)
    if haskey(d, key)
        return d[key]
    else
        return default()
    end
end
getgrad(d::AbstractDict, key::Tuple, default) = getgrad(Ref(d), key, Ref(default))
getgrad(d::AbstractDict, key::Tuple) = getgrad.(Ref(d), key)
function accumulate_gradient!(grad_storage, key::UInt, g)
    # accumulate or create
    if !haskey(grad_storage, key)
        grad_storage[key] = g
    else
        grad_storage[key] = grad_storage[key] + g
    end
end

function accumulate_gradient!(grad_storage, keys::Tuple, grads)
    for (t, g) in zip(keys, grads)
        accumulate_gradient!(grad_storage, t, g)
    end
end

Base.show(io::IO, ::MIME"text/plain", x::DiffTensor) = Base.show(io, x)
function Base.show(io::IO, x::DiffTensor)
    sz = join(string.(size(x)), "×")
    s = "$(typeof(x))[$sz]"
    if x.requires_grad
        s *= " ✓"
    end
    print(io, s)
end

Base.show(io::IO, ::MIME"text/plain", tape::Tape) = Base.show(io, tape)
function Base.show(io::IO, tape::Tape)
    for instruct in tape.instructs
        println(io, instruct.info)
    end
end

# does not return gradients in kwargs
# always return a tuple
function gradient(f, args...; kwargs...)
    empty!(GLOBAL_TAPE.instructs)
    y = f(args...; kwargs...)
    if ndims(y) != 0 || !(eltype(y) <: Real)
        @warn "differentiating a tensor not a scalar real number, got eltype `$(eltype(y))` and rank `$(ndims(y))`"
    end
    grad_storage = Dict{UInt,Any}()
    accumulate_gradient!(grad_storage, getid(y), DiffTensor(ones(eltype(y), size(y.data)...), false))
    back!(GLOBAL_TAPE, grad_storage)
    return getgrad(grad_storage, args)
end

function back!(y::Tape, grad_storage::AbstractDict)
    for i=length(y.instructs):-1:1
        instruct = y.instructs[i]
        debug_info, backs = instruct.info, instruct.backs
        @debug "$debug_info"
        for pb in backs
            dy = getgrad(grad_storage, pb.output_ids, ()->nothing)
            if dy !== nothing
                grads = pb.back(dy)
                accumulate_gradient!(grad_storage, pb.input_ids, grads)
            else
                @debug "can not find requested gradient for $pb"
            end
        end
    end
end

function hessian(f, x::DiffTensor)
    return jacobian(x->gradient(f, x)[1], x)
end

function jacobian(f, x::DiffTensor{T}) where T
    empty!(GLOBAL_TAPE.instructs)
    slices = DiffTensor{T,1}[]
    y = f(x)
    @debug GLOBAL_TAPE
    for i=1:length(y)
        @debug "jacobian, row $i"
        grad_storage = Dict{UInt,Any}()
        gy = zero(y)
        gy.data[i] = one(T)
        accumulate_gradient!(grad_storage, getid(y), gy)
        back!(GLOBAL_TAPE, grad_storage)
        push!(slices, vec(getgrad(grad_storage, x)))
    end
    return transpose(cat(slices...; dims=2))
end

include("rules.jl")
end