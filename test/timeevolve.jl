using TensorNetworkEvolve, Random, Test
using Graphs, Yao, ForwardDiff
using Zygote

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
    @test Zygote.gradient(norm, p1)[1] isa NamedTuple
    @test TensorNetworkEvolve.iloss2(h, p1, variables(p1)) isa Real
    fvec = TensorNetworkEvolve.fvec(p1, h)
    @test fvec isa Vector
    _complex(x::AbstractVector) = [Complex(x[2i-1], x[2i]) for i=1:length(x)÷2]
    fvec2 = ForwardDiff.gradient(x->TensorNetworkEvolve.iloss2(h, p1, _complex(x)), reinterpret(Float64, variables(p1)))
    @test fvec ≈ _complex(fvec2) * -im
end