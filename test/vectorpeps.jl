using Test, Random, TensorNetworkEvolve, Graphs, Yao
using OMEinsumContractionOrders: TreeSA
using LinearAlgebra


@testset "initial state" begin
    peps = zero_vectorpeps(ComplexF64, 20)
    @test vec(peps)[1] == 1 && vec(peps)[100] == 0

    peps = rand_vectorpeps(ComplexF64, 20)
    @test normalize!(peps) == peps
    @test size(vec(peps))[1]==2^20
end

@testset "single and multi-site operations" begin
    # single qubit operations
    peps = zero_vectorpeps(ComplexF64, 10)
    mat = Array{ComplexF64}([0 1; 1 0])
    apply_onsite!(peps, 2, mat)
    @test peps.vec ≈ (x=zeros(ComplexF64, 1<<10); x[3]=1; x)
    
    apply_onsite!(peps, 8, mat)
    @test peps.vec ≈ (x=zeros(ComplexF64, 1<<10); x[131]=1; x)

    peps2 = apply_onsite(peps, 8, mat)
    @test peps2.vec ≈ (x=zeros(ComplexF64, 1<<10); x[3]=1; x)

    # multi-qubit operations
    mat2 = Array{ComplexF64}([0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0])
    mat2 = reshape(mat2, (2, 2, 2, 2))

    peps = rand_qubitvectorpeps(ComplexF64, 10)
    peps2 = QubitVectorPEPS(peps.vec, 10, 1e-12)
    apply_onsite!(peps, 2, mat)
    apply_onsite!(peps, 8, mat)
    apply_onbond!(peps2, 2, 8, mat2)
    @test peps2.vec ≈ peps.vec

    apply_onbond!(peps2, 1, 7, mat2)
    peps = apply_onbond(peps, 1, 7, mat2)
    @test peps2.vec ≈ peps.vec

    # random single qubit gate 
    mat = rand(2, 2)
    mat = mat + transpose(mat)
    mat = exp(-1im*mat)
    peps = zero_qubitvectorpeps(ComplexF64, 10)
    apply_onsite!(peps, 5, mat)
    vec = zeros(ComplexF64, 2^10)
    vec[1] = 1
    vec = ArrayReg(vec)
    apply!(vec, put(10, 5=>matblock(mat)))
    print(peps.vec)
    print(vec)
    @test peps.vec ≈ state(vec)
end


@testset "conj, variables, and load" begin
    peps = rand_vectorpeps(ComplexF64, 10; nflavor=2)
    @test vec(conj(peps)) ≈ conj(vec(peps))

    P = deepcopy(peps)
    randn!(peps)
    @test !(vec(P) ≈ vec(peps))

    vars = variables(P)
    load_variables!(peps, vars)
    @test vec(P) ≈ vec(peps)

    peps2 = randn!(peps)
    peps2 = load_variables(peps2, vars)
    @test vec(P) ≈ vec(peps2)
end

@testset "inner product and normalize" begin
    p1 = rand_vectorpeps(ComplexF64, 10)
    normalize!(p1)
    @test inner_product(p1, p1) ≈ 1

    p1 = rand_vectorpeps(ComplexF64, 10)
    p2 = rand_vectorpeps(ComplexF64, 10)
    @test inner_product(p1, p2) ≈ statevec(p1)' * statevec(p2)

    peps = rand_vectorpeps(ComplexF64, 10)
    @test norm(vec(peps)) ≈ norm(peps)
    normalize!(peps)
    @test norm(peps) ≈ 1
end