using TensorNetworkEvolve
using Test

@testset "peps.jl" begin
    include("peps.jl")
end

@testset "tebd.jl" begin
    include("tebd.jl")
end

@testset "TensorAD.jl" begin
    include("TensorAD/TensorAD.jl")
end

@testset "timeevolve.jl" begin
    include("timeevolve.jl")
end

