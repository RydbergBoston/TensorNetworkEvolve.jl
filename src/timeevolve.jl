using KrylovKit
using Yao: Add
using Zygote: gradient, @non_differentiable

# function sr_step(peps, h)
#     vars = linsolve(x->apply_smatrix(peps, x), -im .* fvec(peps, h), variables(peps), GMRES())
#     load_variables!(copy(peps), vars)
# end

# # TODO: this needs to be fixed
# function apply_smatrix(peps::PEPS, x)
#     g = gradient(x->inner_product(conj(peps), x), peps)
#     vg = variables(g)
#     res = zero(vg)
#     # replace target tensor by x_i
#     for i=1:nsite(peps)
#         qeqs = copy(peps)
#         qeqs.tensors[i] = x.tensors[i]
#         g_i = gradient(x->inner_product(conj(qeqs), x), peps)
#         res .+= variables(g_i)
#     end
#     return res .- vg .* (vg' * x)
# end

# peps_prime = load_variables(peps, variables(peps))
# peps_prime = load_variables(peps, x)
function loss_G1(h, peps, theta, theta_prime, x)
    g = gradient(theta_prime) do
        inner_product(
            conj(load_variables(peps, theta)),
            load_variables(peps, theta_prime)
        )
    end
    return g .* x
end

function loss1(h, peps, theta, x)
    return gradient(theta) do
        theta_prime = copy(theta)
        loss_G1(h, peps, theta, theta_prime, x)
    end
end

# im*L₂
function iloss2(h, peps, variables)
    pl = load_variables(peps, variables)
    pl = pl * (1/norm(pl))
    pr = peps * (1/norm(peps))
    return real(expect(h, conj(pl), pr))
end
function fvec(peps::PEPS, h)
    variables = TensorNetworkEvolve.variables(peps)
    return -im*gradient(x->iloss2(h, peps, x), variables)[1]
end

@non_differentiable OMEinsum.optimize_greedy(code, size_dict)
@non_differentiable replace(vec, pairs...)

# zygote patch
using Zygote
function Zygote.accum(x::Vector, y::Tuple)
    Zygote.accum.(x, y)
end