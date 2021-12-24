module Cracker

using ChainRules: rrule
using LinearAlgebra

export track, untrack

include("types/record.jl")
include("types/number.jl")
include("types/array.jl")

const TrackedType = Union{TrackedNumber, TrackedArrayType}
function Base.show(io::IO, x::TrackedType)
    print(io, "track(")
    show(io, x.value)
    print(io, ")")
end


include("trace.jl")
include("rrule.jl")

end
