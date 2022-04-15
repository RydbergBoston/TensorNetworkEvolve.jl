using Test, ForwardDiff, TensorNetworkEvolve
using Cracker, OMEinsum

@testset "OMEinsum" begin
    T = Float64
    A, B, C = track(rand(T, 2, 2)), track(rand(T, 2, 2)), track(rand(T, 2))
    Z = ein"ij, jk, k->ik"(A, B, C)
    @test Z.record.f === einsum
    
    ret = abs2(sum(Z))
    Cracker.backpropagate!(ret, 1.0)
    a, b, c = untrack.((A, B, C))
    gA = ForwardDiff.gradient(a->abs2(sum(ein"ij, jk, k->ik"(a, b, c))), a)
    gB = ForwardDiff.gradient(b->abs2(sum(ein"ij, jk, k->ik"(a, b, c))), b)
    gC = ForwardDiff.gradient(c->abs2(sum(ein"ij, jk, k->ik"(a, b, c))), c)
    @test A.record.grad ≈ gA
    @test B.record.grad ≈ gB
    @test C.record.grad ≈ gC
end