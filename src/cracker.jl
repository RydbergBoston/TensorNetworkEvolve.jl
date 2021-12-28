using OMEinsum, Cracker
using Cracker: TrackedArrayType

for EC in [:DynamicEinCode, :StaticEinCode]
    @eval function OMEinsum.einsum(code::$EC, @nospecialize(xs::NTuple{N,TrackedArrayType} where N), size_dict::Dict)
        Cracker.trace(einsum, code, xs, size_dict)
    end
end