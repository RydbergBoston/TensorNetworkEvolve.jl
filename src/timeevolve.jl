function sr_step(peps, h)
    vars = linsolve(x->apply_smatrix(peps, x), -im .* fvec(peps, h), variables(peps), GMRES())
    load_variables!(copy(peps), vars)
end

function normalized_overlap(peps::PEPS, v1, v2)
    pl = load_variables(peps, v1)
    pl = pl * inv.(norm(pl))
    pr = load_variables(peps, v2)
    pr = pr * inv.(norm(pr))
    return real(inner_product(pl, pr))
end

# TODO: this needs to be fixed
function apply_smatrix(peps::PEPS, v1, v2, x)
    # Stage 1: build up the computational graph for the inner product `y = ⟨peps(v1)|peps(v2)⟩`
    empty!(TensorAD.GLOBAL_TAPE.instructs)
    TensorAD.requires_grad!(v1, true)   # set `true` to make the computational graph complete (avoid the automatic optimization)
    TensorAD.requires_grad!(v2, true)
    pl = load_variables(peps, v1)
    pl = pl * inv.(norm(pl))
    pr = load_variables(peps, v2)
    pr = pr * inv.(norm(pr))
    y = real(inner_product(pl, pr))

    # Stage 2: back propagate into the right branch and obtain `g₂ = ∂y / ∂v₂`
    # NOTE: I tried avoid propagating into the left branch, but the computational graph for this branch can not be removed somehow!
    # TensorAD.requires_grad!(v1, false)
    # TensorAD.propagate_requires_grad!(TensorAD.GLOBAL_TAPE)

    grad_storage = TensorAD.init_storage!(y; requires_grad=true)
    TensorAD.back!(TensorAD.GLOBAL_TAPE, grad_storage)
    g2 = TensorAD.getgrad(grad_storage, v2)

    # Stage 4: compute the second loss `z = g₂' * x`, note both `g₂` and `x` are complex numbers.
    # inner product with `x`, treat real and imag parts as real numbers
    @assert TensorAD.requires_grad(g2)
    z = real(ein"i,i->"(conj(x), g2))

    # Stage 3: compute the gradient of `sx = ∂z / ∂v₁`
    # NOTE: I tried to not computing v2's gradients, but somehow the computational graph becomes incomplete.
    # TensorAD.requires_grad!(v1, true)
    # TensorAD.requires_grad!(v2, false)
    # TensorAD.propagate_requires_grad!(TensorAD.GLOBAL_TAPE)

    # back propagate!
    @assert TensorAD.requires_grad(z)
    grad_storage = TensorAD.init_storage!(z)
    TensorAD.back!(TensorAD.GLOBAL_TAPE, grad_storage)
    sx = TensorAD.getgrad(grad_storage, v1)
    return sx
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