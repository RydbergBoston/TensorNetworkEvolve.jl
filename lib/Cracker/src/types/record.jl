mutable struct Record{T}
    f
    args
    pullback
    grad::T
    is_leaf::Bool
end

leaf(x) = Record(nothing, nothing, nothing, zero(x), true)
# we use traits over types
is_tracked(x) = false
untrack(x) = is_tracked(x) ? x.value : x
record(x) = x.record
