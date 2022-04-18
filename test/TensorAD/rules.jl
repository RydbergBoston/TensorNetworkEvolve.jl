using Test, OMEinsum
using ForwardDiff
using TensorNetworkEvolve.TensorAD

function unpackargs(args, x)
    start = 0
    map(args) do arg
        if arg isa AbstractArray && eltype(arg) <: AbstractFloat
            res = reshape(x[start+1:start+length(arg)], size(arg)...)
            start += length(arg)
            res
        else
            arg
        end
    end
end

function packargs(args)
    vecs = []
    for arg in args
        if arg isa AbstractArray && eltype(arg) <: AbstractFloat
            push!(vecs, vec(arg))
        end
    end
    return cat(vecs...; dims=1)
end

function build_testfunc(f, args...; kwargs...)
    function (x)
        _args = unpackargs(args, x)
        f(_args...; kwargs...)
    end, packargs(args)
end

function match_jacobian(f, args...; atol=1e-5, kwargs...)
    tf, x = build_testfunc(f, args...; kwargs...)
    j1 = ForwardDiff.jacobian(tf, x)
    j2 = TensorAD.jacobian(tf, DiffTensor(x; requires_grad=true))
    return isapprox(j1, j2.data; atol)
end

@testset "jacobians" begin
    for (f, args, kwargs) in [
        (ein"ii->", (randn(4,4),), ()),
        (reshape, (randn(4,4), 2, 8), ()),
        (transpose, (randn(4,4),), ()),
        (TensorAD.accum, (randn(4,4), (:,:), randn(4,4)), ()),
        (cat, (randn(4,4), randn(4, 2)), (;dims=2)),
        (getindex, (randn(4,4), 3:4, 2:3), ()),
    ]
        @info "Differentiating function: $f"
        @test match_jacobian(f, args...; kwargs...)
    end
end