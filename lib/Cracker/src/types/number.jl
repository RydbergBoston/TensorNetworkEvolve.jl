struct TrackedReal{T <: Real} <: Real
    value::T
    record::Record{T}
end

struct TrackedComplex{T <: Complex} <: Number
    value::T
    record::Record{T}
end

const TrackedNumber = Union{TrackedReal, TrackedComplex}

track(x::Real, record::Record=leaf(x)) = TrackedReal(x, record)
track(x::Complex, record::Record=leaf(x)) = TrackedComplex(x, record)

is_tracked(::Union{TrackedReal, TrackedComplex}) = true

function Base.show(io::IO, mime::MIME"text/plain", x::Union{TrackedReal, TrackedComplex})
    show(io, mime, x.value)
    print(io, " (tracked)")
end
