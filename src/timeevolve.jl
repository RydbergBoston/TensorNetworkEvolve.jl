function sr_step(peps, h)
    vars = linsolve(x->apply_smatrix(peps, x), -im .* fvec(peps, h), variables(peps), GMRES())
    load_variables!(copy(peps), vars)
end

# TODO: this needs to be fixed
function apply_smatrix(peps::PEPS, v1, v2, x)
    # Stage 1: computing gradient of the right branch
    empty!(TensorAD.GLOBAL_TAPE.instructs)
    pl = load_variables(peps, v1)
    pl = pl * inv.(norm(pl))
    pr = load_variables(peps, v2)
    pr = pr * inv.(norm(pr))
    y = inner_product(pl, pr)

    # avoid propagating into the left branch.
    TensorAD.requires_grad!(v1, false)
    TensorAD.propagate_requires_grad!(TensorAD.GLOBAL_TAPE)

    # back propagate!
    grad_storage = TensorAD.init_storage!(y; requires_grad=true)
    TensorAD.back!(TensorAD.GLOBAL_TAPE, grad_storage)
    g1 = TensorAD.getgrad(grad_storage, v2)

    # Stage 2: compute the gradient of (g1' * x) over `pr`
    TensorAD.requires_grad!(v1, true)
    TensorAD.requires_grad!(v2, false)
    TensorAD.propagate_requires_grad!(TensorAD.GLOBAL_TAPE)
    @assert TensorAD.requires_grad(g1)

    @show typeof(x), typeof(g1)
    z = ein"i,i->"(x, g1)
    #@show TensorAD.GLOBAL_TAPE
    @show TensorAD.getid(z)
    TensorAD.propagate_requires_grad!(TensorAD.GLOBAL_TAPE)
    @assert TensorAD.requires_grad(z)

    # back propagate!
    grad_storage = TensorAD.init_storage!(z)
    TensorAD.back!(TensorAD.GLOBAL_TAPE, grad_storage)
    println(TensorAD.GLOBAL_TAPE)
    g2 = TensorAD.getgrad(grad_storage, v1)
    return g2

    g = TensorAD.gradient(x->inner_product(conj(peps), x), peps)
    vg = variables(g)
    res = zero(vg)
    # replace target tensor by x_i
    for i=1:nsite(peps)
        qeqs = copy(peps)
        qeqs.tensors[i] = x.tensors[i]
        g_i = TensorAD.gradient(x->inner_product(conj(qeqs), x), peps)
        res .+= variables(g_i)
    end
    return res .- vg .* (vg' * x)
end

# im*L₂
function iloss2(h, peps, variables)
    pl = load_variables(peps, variables)
    pl = pl * inv.(norm(pl))
    pr = peps * inv.(norm(peps))
    return real(expect(h, conj(pl), pr))
end
function fvec(peps::PEPS, h)
    variables = TensorNetworkEvolve.variables(peps)
    return -im*TensorAD.gradient(x->iloss2(h, wrap_difftensor(peps), x), DiffTensor(variables))[1]
end
function wrap_difftensor(peps::PEPS)
    return replace_tensors(peps, DiffTensor.(alltensors(peps)))
end

# @non_differentiable OMEinsum.optimize_greedy(code, size_dict)
# @non_differentiable replace(vec, pairs...)

# # zygote patch
# using Zygote
# function Zygote.accum(x::Vector, y::Tuple)
#     Zygote.accum.(x, y)
# end