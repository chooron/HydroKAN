module HydroKAN

using Random: AbstractRNG

using Lux
using KolmogorovArnold
using ComponentArrays

include("MultiKAN.jl")

greet() = print("Hello HydroKAN!")

end # module HydroKAN
