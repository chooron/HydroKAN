using Lux
using StableRNGs
using HydroKAN
using KolmogorovArnold
using Test

@testset "HydroKAN.jl" begin
    include("run_multikan.jl")
end