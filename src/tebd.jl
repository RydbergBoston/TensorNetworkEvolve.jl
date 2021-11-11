export rydberg_tebd!

function cterm(C::Real)
    return C*kron(P1, P1)
end

function zterm(Δ::Real)
    return Δ*Z
end

function xterm(Ω::Real)
    return Ω/2*X
end

Yao.apply!(reg::PEPS, b::PutBlock{N,1}) where N = apply_onsite!(reg, b.locs[1], Matrix(b.content))
Yao.apply!(reg::PEPS, b::PutBlock{N,2}) where N = apply_onbond!(reg, b.locs..., reshape(Matrix(b.content),2,2,2,2))

function rydberg_tebd_sweep!(peps, g, C::Real, Δ::Real, Ω::Real, δt)
    for edge in edges(g)
        di = degree(g, edge.src)
        dj = degree(g, edge.dst)
        hi = cterm(C) + put(2, 1=>xterm(Ω/di)) + put(2, 2=>xterm(Ω/dj)) + put(2, 1=>zterm(Δ/di)) + put(2, 2=>zterm(Δ/dj))
        apply!(peps, put(nqubits(peps), (edge.src, edge.dst)=>time_evolve(hi, δt)))
    end
    return peps
end

function rydberg_tebd!(reg, g::SimpleGraph; t::Real, C::Real, Δ::Real, Ω::Real, nstep::Int)
    δt = t/nstep
    for _ = 1:nstep-1
        rydberg_tebd_sweep!(reg, g, C, Δ, Ω, δt)
    end
    return reg
end

function rydberg_tebd!(peps::PEPS; t::Real, C::Real, Δ::Real, Ω::Real, nstep::Int)
    g = SimpleGraph(nqubits(peps))
    for (i, j) in virtualbonds(peps)
        add_edge!(g, i, j)
    end
    rydberg_tebd!(peps, g; t=t, C=C, Δ=Δ, Ω=Ω, nstep=nstep)
end