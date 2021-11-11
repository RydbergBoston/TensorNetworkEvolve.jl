using TensorNetworkEvolve, Random, Test
using Graphs

@testset "sr" begin
    function rand_hamiltonian(g::SimpleGraph)
        blocks = []
        nbit = nv(g)
        for edge in edges(g)
            i,j = edge.src, edge.dst
            push!(blocks, put(nbit,(i,j)=>rand()*kron(X,X)) + put(nbit, (i,j)=>rand()*kron(Y,Y)) + put(nbit, (i,j)=>rand()*kron(Z,Z)))
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
    TensorNetworkEvolve.fvec(p1, h)
end