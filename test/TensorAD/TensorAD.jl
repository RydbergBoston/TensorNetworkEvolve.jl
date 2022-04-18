using Test, OMEinsum
using ForwardDiff
using FiniteDifferences
using TensorNetworkEvolve.TensorAD
#using TensorNetworkEvolve: TensorAD

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
    function f(x, y)
        z = ein"ij,jk->ik"(x, y)
        return ein"ii->"(z)
    end
    x = DiffTensor(randn(10, 10))
    y = DiffTensor(randn(10, 10))
    X = vcat(vec(x.data), vec(y.data))
    function f2(x)
        f(reshape(x[1:100], 10, 10), reshape(x[101:200], 10, 10))
    end
    j1 = ForwardDiff.jacobian(f2, X)
    j2 = TensorAD.jacobian(f2, DiffTensor(X)).data
    @test j1 ≈ j2
    hs = FiniteDifferences.jacobian(central_fdm(5,1), x->ForwardDiff.gradient(x->f2(x)[], x), X)[1]
    H = TensorAD.hessian(f2, DiffTensor(X))
    @test H.data ≈ hs
    @test sum(H.data) ≈ sum(hs)
end

@testset "hessian 2" begin
    function f(xy)
        n = length(xy) ÷ 2
        x = xy[1:n]
        y = xy[n+1:end]
        ein"i,j->"(x, y)
    end
    n = 1
    X = randn(2n)
    h1 = ForwardDiff.hessian(x->f(x)[], X)
    h2 = TensorAD.hessian(f, DiffTensor(X)).data
    @show h1, h2
    @test h1 ≈ h2
end

@testset "hessian 3" begin
    function f(xy)
        sin.(sin.(sin.(xy)))
        #ein"i->"(sin.(xy[1:1][1:1][1:1][1:1][1:1][1:1]))
    end
    X = fill(0.5)
    h1 = ForwardDiff.hessian(x->f(x)[], X)
    h2 = TensorAD.hessian(f, DiffTensor(X)).data
    @show h1, h2
    @test h1 ≈ h2
end

@testset "hessian 4" begin
    function f(xy)
        x = xy[1:1]
        y = xy[1:1]
        z = xy[1:1]
        #ein"i->"(((x .* y) .* z) .*a)
        ein"i->"((x .* y) .* z)
        #ein"i->"(ein"i,i->i"(ein"i,i->i"(ein"i,i->i"(x, y), z), a))
    end
    X = [0.5, 0.6, 0.7]
    j1 = ForwardDiff.jacobian(x->f(x), X)
    j2 = TensorAD.jacobian(f, DiffTensor(X)).data
    @test j1 ≈ j2
    h1 = ForwardDiff.hessian(x->f(x)[], X)
    h2 = TensorAD.hessian(f, DiffTensor(X)).data
    @show h1, h2
    @test h1 ≈ h2
end