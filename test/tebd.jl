using TensorNetworkEvolve, Yao, Graphs
using LinearAlgebra, Test

@testset "random gate" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2), (1,3), (2,4), (2,5), (3,4), (3,5)]
        add_edge!(g, i, j)
    end
    # simple
    peps = rand_simplepeps(ComplexF64, g, 2; Dmax=10)
    reg = Yao.ArrayReg(vec(state(peps)))
    rydberg_tebd!(peps, g; t=0.3, C=10.0, Δ=0.2, Ω=0.3, nstep=10)
    @test !isapprox(statevec(peps), statevec(reg); atol=1e-3)
    rydberg_tebd!(reg, g; t=0.3, C=10.0, Δ=0.2, Ω=0.3, nstep=10)
    @test isapprox(statevec(peps), statevec(reg); atol=1e-3)
    @show norm(statevec(peps) .- statevec(reg))

    # vidal
    peps = rand_vidalpeps(ComplexF64, g, 2; Dmax=10, ϵ=1e-10)
    reg = Yao.ArrayReg(vec(state(peps)))
    rydberg_tebd!(peps; t=0.3, C=10.0, Δ=0.2, Ω=0.3, nstep=10)
    @test !isapprox(statevec(peps), statevec(reg); atol=1e-3)
    rydberg_tebd!(reg, g; t=0.3, C=10.0, Δ=0.2, Ω=0.3, nstep=10)
    @test isapprox(statevec(peps), statevec(reg); atol=1e-3)
    @show norm(statevec(peps) .- statevec(reg))
end
