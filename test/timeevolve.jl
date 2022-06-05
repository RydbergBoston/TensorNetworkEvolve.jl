using TensorNetworkEvolve, Random, Test
using Graphs, Yao, ForwardDiff
using TensorNetworkEvolve.TensorAD

@testset "sr" begin
    function rand_hamiltonian(g::SimpleGraph)
        blocks = []
        nbit = nv(g)
        for edge in edges(g)
            i,j = edge.src, edge.dst
            push!(blocks, randn()*kron(nbit,i=>X,j=>X) + randn()*kron(nbit, i=>Y,j=>Y) + randn()*kron(nbit, i=>Z, j=>Z))
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
    vars = TensorAD.DiffTensor(variables(p1))
    gvars = TensorAD.gradient(vars->norm(load_variables(p1, vars)), vars)[1]
    @test gvars isa DiffTensor

    @test TensorNetworkEvolve.iloss2(h, p1, variables(p1)) isa Array{T,0} where T
    fvec = TensorNetworkEvolve.fvec(p1, h)
    @test fvec isa DiffTensor
    _complex(x::AbstractVector) = [Complex(x[2i-1], x[2i]) for i=1:length(x)÷2]
    fvec2 = ForwardDiff.gradient(x->TensorNetworkEvolve.iloss2(h, p1, _complex(x))[], reinterpret(Float64, variables(p1)))
    @test fvec.data ≈ _complex(fvec2) * -im
end

@testset "smatrix apply" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2), (1,3), (2,4), (2,5), (3,4), (3,5)]
        add_edge!(g, i, j)
    end

    p1 = rand_simplepeps(ComplexF64, g, 2; Dmax=4)
    x = randn(ComplexF64, length(variables(p1)))
    v1 = TensorAD.DiffTensor(variables(p1))
    v2 = TensorAD.DiffTensor(variables(p1))
    @show TensorNetworkEvolve.apply_smatrix(p1, v1, v2, x)
end