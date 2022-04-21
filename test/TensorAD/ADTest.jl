module ADTest

using ForwardDiff, FiniteDifferences, TensorNetworkEvolve.TensorAD, OMEinsum

function unpackargs(args, x, mask)
    start = 0
    map(args, mask) do arg, m
        if m
            @assert arg isa AbstractArray{<:TensorAD.ADTypes}
            if eltype(arg) <: Real
                res = reshape(x[start+1:start+length(arg)], size(arg)...)
                start += length(arg)
                res
            else
                rel = x[start+1:start+length(arg)]
                img = x[start+length(arg)+1:start+2*length(arg)]
                res = reshape(rel + im * img, size(arg)...)
                start += 2*length(arg)
                res
            end
        else
            arg
        end
    end
end

function packargs(args, mask)
    vecs = []
    for (arg, m) in zip(args, mask)
        if m
            @assert arg isa AbstractArray{<:TensorAD.ADTypes}
            if eltype(arg) <: Real
                push!(vecs, vec(arg))
            else
                push!(vecs, vcat(vec(real(arg)), vec(imag(arg))))
            end
        end
    end
    return vcat(vecs...)
end

function build_testfunc(f, args...; mask=map(x->x isa AbstractArray{<:TensorAD.ADTypes}, args), realpart=true, kwargs...)
    function (x)
        _args = unpackargs(args, x, mask)
        if realpart
            OMEinsum.asarray(real(f(_args...; kwargs...)), x)
        else
            OMEinsum.asarray(imag(f(_args...; kwargs...)), x)
        end
    end, packargs(args, mask)
end

function match_jacobian(f, args...; realpart=true, atol=1e-5, kwargs...)
    tf, x = build_testfunc(f, args...; realpart, kwargs...)
    j1 = ForwardDiff.jacobian(tf, x)
    j2 = TensorAD.jacobian(tf, DiffTensor(x))
    return isapprox(j1, j2.data; atol)
end

function match_random(f, args...; realpart=true, atol=1e-5, kwargs...)
    empty!(TensorAD.GLOBAL_TAPE.instructs)
    tf, x = build_testfunc(f, args...; realpart, kwargs...)
    j1 = ForwardDiff.jacobian(tf, x)
    X = DiffTensor(x)
    y = tf(X)
    gy = ndims(y) == 0 ? fill(randn(eltype(y))) : randn(eltype(y), size(y)...)
    g1 = vec(vec(gy)' * j1)

    grad_storage = Dict{UInt,Any}()
    TensorAD.accumulate_gradient!(grad_storage, TensorAD.getid(y), DiffTensor(gy, false))
    TensorAD.back!(TensorAD.GLOBAL_TAPE, grad_storage)
    g2 = vec(TensorAD.getgrad(grad_storage, X))
    @debug "g1 = $(g1), g2 = $(g2.data)"
    return isapprox(g1, g2.data; atol) && typeof(X) == typeof(g2)
end

function match_hessian(f, args...; realpart=true, atol=1e-5, kwargs...)
    tf, x = build_testfunc(f, args...; realpart, kwargs...)
    h2 = TensorAD.hessian(tf, DiffTensor(x))
    h1 = FiniteDifferences.jacobian(central_fdm(5,1), x->ForwardDiff.gradient(x->tf(x)[], x), x)[1]
    return isapprox(h1, h2.data; atol)
end

end