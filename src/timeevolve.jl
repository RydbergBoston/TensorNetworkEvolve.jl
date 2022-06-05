using KrylovKit
using Yao: Add
#using Zygote: gradient, @non_differentiable
using .TensorAD: gradient, DiffTensor

function sr_step(peps, h)
    vars = linsolve(x->apply_smatrix(peps, x), -im .* fvec(peps, h), variables(peps), GMRES())
    load_variables!(copy(peps), vars)
end

# TODO: this needs to be fixed
function apply_smatrix(peps::PEPS, x)
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
    pl = pl * (1 ./norm(pl))
    pr = peps * (1 ./ norm(peps))
    return real(expect(h, conj(pl), pr))
end
function fvec(peps::PEPS, h)
    variables = TensorNetworkEvolve.variables(peps)
    return -im*TensorAD.gradient(x->iloss2(h, peps, x), DiffTensor(variables))[1]
end

# @non_differentiable OMEinsum.optimize_greedy(code, size_dict)
# @non_differentiable replace(vec, pairs...)

# # zygote patch
# using Zygote
# function Zygote.accum(x::Vector, y::Tuple)
#     Zygote.accum.(x, y)
# end