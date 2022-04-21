using Test, OMEinsum
using ForwardDiff
using TensorNetworkEvolve.TensorAD

@testset "jacobians 1" begin
    n = 3
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
            (Base.broadcast, (^, randn(T,4,4), n), ()),
            (Base.broadcast, (sin, randn(T,4,4)), ()),
            (Base.broadcast, (cos, randn(T,4,4)), ()),
            (Base.broadcast, (sign, randn(T,4,4)), ()),
            (Base.broadcast, (abs, randn(T,4,4)), ()),
            (Base.broadcast, (sqrt, 10 .+ randn(T,4,4)), ()),
            (Base.broadcast, (inv, randn(T,4,4)), ()),
            (Base.broadcast, (Complex, randn(T,4,4)), ()),
        ]
            @info "Differentiating function: $f, arg types: $(typeof(args))"
            @test match_jacobian(f, args...; realpart=true, kwargs...)
            @test match_random(f, args...; realpart=true, kwargs...)
            if T <: Complex
                @info "Differentiating function: $f, arg types: $(typeof(args)), imaginary part"
                @test match_jacobian(f, args...; realpart=false, kwargs...)
                @test match_random(f, args...; realpart=false, kwargs...)
            end
        end
    end
end

@testset "jacobians 1" begin
    n = 3
    _mul(x::Array{T,0}, y::Array{T,0}) where T = fill(x[] * y[])
    _mul(x::Array{T,0}, y::Array) where T = x[] * y
    _mul(x::Array, y::Array{T,0}) where T = x * y[]
    _mul(x, y) = x * y
    _fill(x, args...) = fill(x, args...)
    _fill(x::Array, args...) = fill(x[], args...)
    for T in [Float64, ComplexF64]
        for (f, args, kwargs) in [
            (sum, (randn(T,4,4),), ()),
            (_fill, (fill(randn(T)), 2, 3), ()),
            (_mul, (fill(randn(T)), randn(T,4, 4)), ()),
            (_mul, (randn(T,4,4), fill(randn(T))), ()),
            (_mul, (fill(T(2.0)), fill(T(2.0))), ()),
        ]
            @info "Differentiating function: $f, arg types: $(typeof(args))"
            @test match_jacobian(f, args...; realpart=true, kwargs...)
            @test match_random(f, args...; realpart=true, kwargs...)
            if T <: Complex
                @info "Differentiating function: $f, arg types: $(typeof(args)), imaginary part"
                @test match_jacobian(f, args...; realpart=false, kwargs...)
                @test match_random(f, args...; realpart=false, kwargs...)
            end
        end
    end
end