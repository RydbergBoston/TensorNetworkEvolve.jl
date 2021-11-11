using Test, Random, TensorNetworkEvolve, Graphs, Yao
using LinearAlgebra

@testset "initial state" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2), (1,3), (2,4), (2,5), (3,4), (3,5)]
        add_edge!(g, i, j)
    end
    peps = zero_vidalpeps(ComplexF64, g, 2)
    @test virtualbonds(peps) == [(1,2), (1,3), (2,4), (2,5), (3,4), (3,5)]
    @test vec(peps) ≈ (x=zeros(ComplexF64, 1<<5); x[1]=1; x)
    @test newlabel(peps, 2) == 11+2
    @test newlabel(peps, 4) == 11+4
    @test getphysicallabel(peps, 2) == 2
    @test length(getvlabel(peps, 2)) == 4
    apply_onsite!(peps, 1, [0 1; 1 0])
    @test vec(peps) ≈ (x=zeros(ComplexF64, 1<<5); x[2]=1; x)
    apply_onbond!(peps, 1, 2, reshape(Matrix(ConstGate.CNOT), 2, 2, 2, 2))
    @test vec(peps) ≈ (x=zeros(ComplexF64, 1<<5); x[4]=1; x)

    peps = rand_vidalpeps(ComplexF64, g, 2)
    normalize!(peps)
    @test norm(peps) ≈ 1
end

@testset "random gate" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2), (1,3), (2,4), (2,5), (3,4), (3,5)]
        add_edge!(g, i, j)
    end
    peps = rand_vidalpeps(ComplexF64, g, 2; Dmax=4)
    reg = Yao.ArrayReg(vec(peps))

    # unary
    m = rand_unitary(2)
    apply_onsite!(peps, 1, m)
    reg |> put(5, (1,)=>matblock(m))
    @test vec(peps) ≈ statevec(reg)

    m = rand_unitary(4)
    reg |> put(5, (4,2)=>matblock(m))
    apply_onbond!(peps, 4, 2, reshape(m,2,2,2,2))
    @test vec(peps) ≈ statevec(reg)
end


@testset "conj, variables, and load" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2), (1,3), (2,4), (2,5), (3,4), (3,5)]
        add_edge!(g, i, j)
    end
    peps = rand_simplepeps(ComplexF64, g, 2; Dmax=4)
    @test statevec(conj(peps)) ≈ conj(statevec(peps))
    P = deepcopy(peps)
    vars = variables(peps)
    randn!(peps)
    @test !(statevec(P) ≈ statevec(peps))
    load_variables!(peps, vars)
    @test statevec(P) ≈ statevec(peps)
    peps2 = randn!(peps)
    peps2 = load_variables(peps2, vars)
    @test statevec(P) ≈ statevec(peps2)
end

@testset "inner product and normalize" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2), (1,3), (2,4), (2,5), (3,4), (3,5)]
        add_edge!(g, i, j)
    end

    p1 = rand_simplepeps(ComplexF64, g, 2; Dmax=4)
    @test inner_product(p1, p1) ≈ statevec(p1)' * statevec(p1)

    p1 = rand_simplepeps(ComplexF64, g, 2; Dmax=4)
    p2 = rand_simplepeps(ComplexF64, g, 2; Dmax=4)
    @test inner_product(p1, p2) ≈ statevec(p1)' * statevec(p2)

    p1 = rand_vidalpeps(ComplexF64, g, 2; Dmax=4)
    p2 = rand_vidalpeps(ComplexF64, g, 2; Dmax=4)
    @test inner_product(p1, p2) ≈ statevec(p1)' * statevec(p2)

    peps = rand_vidalpeps(ComplexF64, g, 2; Dmax=4)
    @test norm(vec(peps)) ≈ norm(peps)
    normalize!(peps)
    @test norm(peps) ≈ 1
end

@testset "expectation values" begin
    function rand_hamiltonian(g::SimpleGraph)
        blocks = []
        nbit = nv(g)
        for edge in edges(g)
            i,j = edge.src, edge.dst
            push!(blocks, put(nbit,(i,j)=>rand()*kron(X,X)) + rand()*kron(nbit, i=>Y,j=>Y) + control(nbit, i,j=>rand()*Z))
        end
        for i in vertices(g)
            push!(blocks, put(nbit,(i,)=>rand()*X))
        end
        return +(blocks...)
    end
    g = SimpleGraph(5)
    for (i,j) in [(1,2), (1,3), (2,4), (2,5), (3,4), (3,5)]
        add_edge!(g, i, j)
    end

    h = rand_hamiltonian(g)
    p1 = rand_simplepeps(ComplexF64, g, 2; Dmax=4)
    p2 = rand_simplepeps(ComplexF64, g, 2; Dmax=4)
    @test expect(h, p1, p2) ≈ statevec(p1)' * mat(h) * statevec(p2)
    @test expect(h, p1, p1) ≈ statevec(p1)' * mat(h) * statevec(p1)
end