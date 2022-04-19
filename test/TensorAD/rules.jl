using Test, OMEinsum
using ForwardDiff
using TensorNetworkEvolve.TensorAD

@testset "jacobians" begin
    for T in [Float64, ComplexF64]
        for (f, args, kwargs) in [
            (ein"ii->", (randn(T,4,4),), ()),
            (reshape, (randn(T,4,4), 2, 8), ()),
            (transpose, (randn(T,4,4),), ()),
            (TensorAD.accum, (randn(T,4,4), (:,:), randn(T,4,4)), ()),
            (cat, (randn(T,4,4), randn(T,4, 2)), (;dims=2)),
            (copy, (randn(T,4,4),), ()),
            (conj, (randn(T,4,4),), ()),
            (real, (randn(T,4,4),), ()),
            (imag, (randn(T,4,4),), ()),
            (+, (randn(T,4,4), randn(T,4, 4)), ()),
            (*, (randn(T,4,4), randn(T,4, 4)), ()),
            (*, (3.0, randn(T,4, 4)), ()),
            (-, (randn(T,4,4), randn(T,4, 4)), ()),
            (-, (randn(T,4,4),), ()),
            (getindex, (randn(T,4,4), 3:4, 2:3), ()),
            (Base.broadcast, (*, randn(T,4,4), randn(T,4, 4)), ()),
            (Base.broadcast, (/, randn(T,4,4), randn(T,4, 4)), ()),
            (Base.broadcast, (^, randn(T,4,4), 3), ()),
            (Base.broadcast, (sin, randn(T,4,4)), ()),
            (Base.broadcast, (cos, randn(T,4,4)), ()),
            (Base.broadcast, (sign, randn(T,4,4)), ()),
            (Base.broadcast, (abs, randn(T,4,4)), ()),
            (Base.broadcast, (sqrt, 10 .+ randn(T,4,4)), ()),
        ]
            @info "Differentiating function: $f, arg types: $(typeof(args))"
            @test match_jacobian(f, args...; kwargs...)
            @test match_random(f, args...; realpart=true, kwargs...)
        end
    end
end