using Test, OMEinsum
using ForwardDiff
using FiniteDifferences
using TensorNetworkEvolve.TensorAD
#using TensorNetworkEvolve: TensorAD

@testset "diff tensor" begin
    function f(x, y)
        z = ein"ij,jk->ik"(x, y)
        return ein"ii->"(z)
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
    hs = FiniteDifferences.jacobian(central_fdm(5,1), x->ForwardDiff.gradient(x->f2(x)[], x), X)[1]
    g = TensorAD.gradient(f2, DiffTensor(X; requires_grad=true))
    @show g
    H = TensorAD.hessian(f2, DiffTensor(X; requires_grad=true))
    @show H.data ≈ hs
    @show sum(H.data), sum(hs)
    #@show hs
    #@test vcat(vec(gs[1]), vec(gs[2])) ≈ hs
end