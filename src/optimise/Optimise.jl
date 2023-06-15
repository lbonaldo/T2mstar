module Optimise


using Random
using Parameters
using StatsBase: sqL2dist

using SharedArrays

using Mstar2t: ParabBand, BandStructure, ScModel,
      electrical_conductivity, seebeck_coefficient, carrier_concentration, thermal_conductivity

using ..Utils: checkerange!, create_runfolder
using ..Plot: plot_χ, plot_best, plot_evoacept, plot_bandcomp, savefig

export  Args, GroundTruth,
        RMC

include("RMC.jl")

end