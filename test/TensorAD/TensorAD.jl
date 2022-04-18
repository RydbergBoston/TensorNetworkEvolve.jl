using Test, OMEinsum
using ForwardDiff
using FiniteDifferences
using TensorNetworkEvolve.TensorAD
#using TensorNetworkEvolve: TensorAD

@testset "diff tensor" begin
    function f(x, y)
        z = ein"ij,jk->ik"(x, y)
        return ein"ii->"(sin.(cos.(z)) .* x .* y)
    end
    x = randn(10, 10)
    y = randn(10, 10)
    gs = TensorAD.gradient(f, DiffTensor(x; requires_grad=true), DiffTensor(y; requires_grad=true))
    hs = ForwardDiff.gradient(x->f(reshape(x[1:100], 10, 10), reshape(x[101:200], 10, 10))[], vcat(vec(x), vec(y)))
    @test vcat(vec(gs[1].data), vec(gs[2].data)) ≈ hs
end

@testset "hessian" begin
    function f(x, y)
        z = ein"ij,jk->ik"(x, y)
        return ein"ii->"(z)
    end
    x = DiffTensor(randn(10, 10); requires_grad=true)
    y = DiffTensor(randn(10, 10); requires_grad=true)
    gx, gy = TensorAD.gradient(f, x, y)
    grad_storage = Dict{UInt,Any}()
    TensorAD.accumulate_gradient!(grad_storage, gx.tracker.id, DiffTensor(fill(one(eltype(gx)), size(gx.data)...); requires_grad=true))
    TensorAD.back!(gx.tracker, grad_storage)
    @show grad_storage[y.tracker.id].data

    grad_storage = Dict{UInt,Any}()
    TensorAD.accumulate_gradient!(grad_storage, gy.tracker.id, DiffTensor(fill(one(eltype(gy)), size(gy.data)...); requires_grad=true))
    TensorAD.back!(gy.tracker, grad_storage)
    @show TensorAD.getgrad(grad_storage, x).data

    X = vcat(vec(x.data), vec(y.data))
    function f2(x)
        f(reshape(x[1:100], 10, 10), reshape(x[101:200], 10, 10))
    end
    j1 = ForwardDiff.jacobian(f2, X)
    j2 = TensorAD.jacobian(f2, DiffTensor(X; requires_grad=true)).data
    @test j1 ≈ j2
    hs = FiniteDifferences.jacobian(central_fdm(5,1), x->ForwardDiff.gradient(x->f2(x)[], x), X)[1]
    H = TensorAD.hessian(f2, DiffTensor(X; requires_grad=true))
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
    h2 = TensorAD.hessian(f, DiffTensor(X; requires_grad=true)).data
    @show h1, h2
    @test h1 ≈ h2
end

@testset "hessian 3" begin
    function f(xy)
        ein"i->"(sin.(cos.(sin.(xy))))
    end
    X = [0.5]
    h1 = ForwardDiff.hessian(x->f(x)[], X)
    h2 = TensorAD.hessian(f, DiffTensor(X; requires_grad=true)).data
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
    h1 = ForwardDiff.hessian(x->f(x)[], X)
    h2 = TensorAD.hessian(f, DiffTensor(X; requires_grad=true)).data
    @show h1, h2
    @test h1 ≈ h2
end