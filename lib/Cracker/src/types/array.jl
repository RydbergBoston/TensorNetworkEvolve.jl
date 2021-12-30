struct TrackedArray{T, N, S <: AbstractArray{T, N}} <: AbstractArray{T, N}
    value::S
    record::Record{S}
end

struct TrackedRealArray{T <: Real, N, S <: AbstractArray{T, N}} <: AbstractArray{TrackedReal{T}, N}
    value::S
    record::Record{S}
end

struct TrackedComplexArray{T <: Complex, N, S <: AbstractArray{T, N}} <: AbstractArray{TrackedComplex{T}, N}
    value::S
    record::Record{S}
end

const TrackedArrayType = Union{TrackedArray, TrackedRealArray, TrackedComplexArray}

function Base.show(io::IO, mime::MIME"text/plain", x::TrackedArrayType)
    print(io, "tracked ")
    show(io, mime, x.value)
end

function track(A::AbstractArray, record::Record=leaf(A))
    return if eltype(A) <: Real
        TrackedRealArray(A, record)
    elseif eltype(A) <: Complex
        TrackedComplexArray(A, record)
    else
        TrackedArray(A, record)
    end
end

is_tracked(::TrackedArrayType) = true
Base.IndexStyle(X::TrackedArrayType) = IndexStyle(untrack(X))
Base.size(X::TrackedArrayType, idx::Int...) = size(untrack(X), idx...)
Base.length(X::TrackedArrayType) = length(untrack(X))
