using Test, OMEinsum
using ForwardDiff
using FiniteDifferences
using TensorNetworkEvolve.TensorAD

include("ADTest.jl")
using .ADTest: match_jacobian, match_random, match_hessian
include("rules.jl")

@testset "diff tensor" begin
    function f(x, y)
        z = ein"ij,jk->ik"(x, y)
        return ein"ii->"(sin.(cos.(cos.(z))) .* x .* y)
    end
    x = randn(10, 10)
    y = randn(10, 10)
    gs = TensorAD.gradient(f, DiffTensor(x), DiffTensor(y))
    hs = ForwardDiff.gradient(x->f(reshape(x[1:100], 10, 10), reshape(x[101:200], 10, 10))[], vcat(vec(x), vec(y)))
    @test vcat(vec(gs[1].data), vec(gs[2].data)) ≈ hs
end

@testset "hessian" begin
    function f1(x, y)
        z = ein"ij,jk->ik"(x, y)
        return ein"ii->"(z)
    end
    @test match_hessian(f1, randn(5, 5), randn(5, 5))
    function f2(xy)
        sin.(sin.(sin.(xy)))
    end
    @test match_hessian(f2, fill(0.5))
    function f3(xy)
        x = xy[1:1]
        y = xy[1:1]
        z = xy[1:1]
        ein"i->"((x .* y) .* z)
    end
    @test match_hessian(f3, [0.2, 0.4, 0.8])
end