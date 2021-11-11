using KrylovKit
using Yao: Add
using Zygote: gradient, @non_differentiable

function sr_step(peps, h)
    vars = linsolve(x->apply_smatrix(peps, x), -im .* fvec(peps, h), variables(peps), GMRES())
    load_variables!(copy(peps), vars)
end

function apply_smatrix(peps::PEPS, x)
    g = gradient(x->inner_product(conj(peps), x), peps)
    vg = variables(g)
    res = zero(vg)
    # replace target tensor by x_i
    for i=1:nsite(peps)
        qeqs = copy(peps)
        qeqs.tensors[i] = x.tensors[i]
        g_i = gradient(x->inner_product(conj(qeqs), x), peps)
        res .+= variables(g_i)
    end
    return res .- vg .* (vg' * x)
end

function fvec(peps::PEPS, h)
    gh = gradient(x->real(expect(h, conj(peps), x)), peps)
    g = gradient(x->real(inner_product(conj(peps), x)), peps)
    return variables(gh) .- expect(h, peps, peps) .* variables(g)
end

@non_differentiable OMEinsum.optimize_greedy(code, size_dict)