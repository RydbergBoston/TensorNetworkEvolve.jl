using TensorNetworkEvolve, Random, Test
using Graphs, Yao, ForwardDiff, KrylovKit, LinearMaps
using Zygote

@testset "simple_evolve" begin
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

    time = 1
    h = rand_hamiltonian(g)
    state1 = rand_state(5)
    state2 = copy(state1)
    te = time_evolve(h, time)
    apply!(state1, te)

    function tn_exact_evolve(s::ArrayReg, h, time)
        #print(statevec(s))
        #apply_h = LinearMap(x->-1im*statevec(ArrayReg(x)|>h), length(statevec(s)))::LinearMaps.FunctionMap
        b = rand(Float64, (length(statevec(s)), length(statevec(s))))
        b = b + b'
        print(size(b))
        print()
        apply_h = LinearMap(x->b*x, length(statevec(s)))::LinearMaps.FunctionMap

        print(apply_h(statevec(s)))
        #print(b)
        #print(a(b))
        #print(typeof(s|>h))
        print(exponentiate(apply_h, time, statevec(s)))
        #exponentiate(rand(Float64,(2,2)), time*-1im, Array([[0],[1]]))#; ishermitian=True)
        #print(state([s)[1])
        return 0
    end 
    tn_exact_evolve(state2, h, time)
    #@test TensorNetworkEvolve.iloss2(h, p1, variables(p1)) isa Real
    #fvec = TensorNetworkEvolve.fvec(p1, h)
    #@test fvec isa Vector
    #_complex(x::AbstractVector) = [Complex(x[2i-1], x[2i]) for i=1:length(x)÷2]
    #fvec2 = ForwardDiff.gradient(x->TensorNetworkEvolve.iloss2(h, p1, _complex(x)), reinterpret(Float64, variables(p1)))
    #@test fvec ≈ _complex(fvec2) * -im
end