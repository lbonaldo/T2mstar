module Plot

using CairoMakie
using PlotUtils
using LaTeXStrings

using Mstar2t: ParabBand, BandStructure, ScModel

export  plot_χ,
        plot_best,
        plot_evoacept,
        plot_bandcomp,
        savefig

include("_plot.jl")

end