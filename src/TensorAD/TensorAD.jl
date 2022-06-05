module TensorAD
export DiffTensor

using OMEinsum, LinearAlgebra

const ADTypes = Union{Float32, Float64, ComplexF64, ComplexF32}

struct PartialBack
    input_ids::Union{UInt,Tuple}
    output_ids::Union{UInt,Tuple}
    back
    active::Ref{Bool}
end
function PartialBack(input_ids, output_ids, back; active::Bool=true)
    return PartialBack(input_ids, output_ids, back, Ref(active))
end

struct Instruction
    info::String
    backs::NTuple{N,PartialBack} where N
end
struct Tape
    instructs::Vector{Instruction}
end
const GLOBAL_TAPE = Tape(Instruction[])
# NOTE: we never empty this dict!
const GLOBAL_REQUIRES_GRAD = Dict{UInt,Bool}()

# to make objectid work consistently
mutable struct DiffTensor{T<:ADTypes,N,AT<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::AT
    function DiffTensor(data::AT; requires_grad::Bool=true) where {T,N,AT<:AbstractArray{T,N}}
        if AT <: DiffTensor
            error("DiffTensor in DiffTensor is forbidden to prevent errors.")
        end
        t = new{T,N,AT}(data)
        requires_grad!(t, requires_grad)
        return t
    end
end
function difftensor(data::AbstractArray, debug_info::String, backs::Pair...)
    # filter out not required AD rules.
    mask = [requires_grad(pair.first) for pair in backs]
    t = DiffTensor(data; requires_grad=any(mask))
    push!(GLOBAL_TAPE.instructs, Instruction(debug_info, t, backs[mask]...))
    return t
end

requires_grad(t::DiffTensor) = GLOBAL_REQUIRES_GRAD[getid(t)]
requires_grad(t::Tuple) = any(x->requires_grad(x), t)
function requires_grad!(t::DiffTensor, val::Bool=true)
    GLOBAL_REQUIRES_GRAD[getid(t)] = val
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
Base.zero(x::DiffTensor) = DiffTensor(zero(x.data); requires_grad=false)
Base.size(x::DiffTensor, indices...) = Base.size(x.data, indices...)
getdata(x::DiffTensor) = x.data
getdata(x::AbstractArray) = x
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
    if requires_grad(x)
        s *= " ✓"
    end
    print(io, s)
end

Base.show(io::IO, ::MIME"text/plain", tape::Tape) = Base.show(io, tape)
function Base.show(io::IO, tape::Tape)
    for instruct in tape.instructs
        print(io, instruct.info)
        print(io, "  ")
        for back in instruct.backs
            print(io, back.active[] ? "●" : "○", " ")
        end
        println(io)
    end
end

OMEinsum.asarray(x::Number, y::DiffTensor) = DiffTensor(asarray(x,y.data); requires_grad=y.requires_grad)
OMEinsum.asarray(x::DiffTensor, y::DiffTensor) = x  # fix ambiguity error

# does not return gradients in kwargs
# always return a tuple
function gradient(f, args...; kwargs...)
    empty!(GLOBAL_TAPE.instructs)
    y = f(args...; kwargs...)
    if ndims(y) != 0 || !(eltype(y) <: Real)
        @warn "differentiating a tensor not a scalar real number, got eltype `$(eltype(y))` and rank `$(ndims(y))`"
    end
    grad_storage = init_storage!(y)
    back!(GLOBAL_TAPE, grad_storage)
    return getgrad(grad_storage, args)
end

function init_storage!(y; requires_grad=false)
    grad_storage = Dict{UInt,Any}()
    accumulate_gradient!(grad_storage, getid(y), DiffTensor(ones(eltype(y), size(y.data)...); requires_grad))
    return grad_storage
end

function back!(y::Tape, grad_storage::AbstractDict)
    for i=length(y.instructs):-1:1
        instruct = y.instructs[i]
        debug_info, backs = instruct.info, instruct.backs
        @debug "$debug_info"
        for pb in backs
            pb.active[] || continue
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

# variable dependency analysis
function propagate_requires_grad!(tape::Tape=GLOBAL_TAPE)
    # execute the program virtually, and set requires_grad to false
    for instruct in tape.instructs
        for i in 1:length(instruct.backs)
            pb = instruct.backs[i]
            rg = any(x->GLOBAL_REQUIRES_GRAD[x], pb.input_ids)
            # turn off this partial backward
            pb.active[] = rg
            for id in pb.output_ids
                GLOBAL_REQUIRES_GRAD[id] = rg
            end
        end
    end
end

include("rules.jl")

end