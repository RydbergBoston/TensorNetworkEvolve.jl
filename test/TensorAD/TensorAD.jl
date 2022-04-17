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
    gx, gy = gradient(f, x, y)
    grad_storage = Dict{UInt,Any}()
    accumulate_gradient!(grad_storage, gx, DiffTensor(fill(one(eltype(gx)), size(gx.data)...); requires_grad=true))
    back!(gx, grad_storage)
    @show grad_storage[objectid(y)]

    grad_storage = Dict{UInt,Any}()
    accumulate_gradient!(grad_storage, gy, DiffTensor(fill(one(eltype(gy)), size(gy.data)...); requires_grad=true))
    back!(gy, grad_storage)
    @show getgrad(grad_storage, x)
    hs = FiniteDifferences.jacobian(central_fdm(5,1), x0->ForwardDiff.gradient(x->f(reshape(x[1:100], 10, 10), reshape(x[101:200], 10, 10))[], x0), vcat(vec(x), vec(y)))[1]
    #hs = ForwardDiff.hessian(x->f(reshape(x[1:100], 10, 10), reshape(x[101:200], 10, 10))[], vcat(vec(x), vec(y)))
    #@show hs
    #@test vcat(vec(gs[1]), vec(gs[2])) ≈ hs
end