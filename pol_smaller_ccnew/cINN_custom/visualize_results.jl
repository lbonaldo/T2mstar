using CSV
using DataFrames
import CairoMakie
using Polynomials
using LaTeXStrings

using Mstar2t
import Mstar2t: Scattering, Plot

titlesize = 22
xlabelsize = 20
ylabelsize = 20

if length(ARGS) < 2
    error("[FATAL] Experiment path missing and/or polydegree") 
end

######### PARAMS ######### 
data2plot_folder = ARGS[1]
export_folder = joinpath(data2plot_folder,"plots")
if !isdir(export_folder) # if there is not a results folder -> create it
    mkdir(export_folder)
end
datafile = joinpath(data2plot_folder, "enS_band_comb.txt")
polydegrees = [parse(Int64,i) for i in split(ARGS[2],"_")]
n_coeff = polydegrees .+ 1
en_1 = 0.3;
en_2 = 0.0;
type_1 = 1;
type_2 = -1;
minima = [en_1, en_2];
ts = collect(50.:5:600);
τ_form = Scattering.constant()
##########################

conf = """\n### BAND STRUCTURE CONFIGURATIONS ###
poly degree = $(polydegrees)
en_1 = $(en_1);
en_2 = $(en_2);
type_1 = $(type_1);
type_2 = $(type_2);
ts = $(ts);
τ_form = $(Scattering.constant())
test_folder = $(ARGS[1])
#####################################
"""
println(conf)

xlabels = [L"$T$ $[K]$",L"$T$ $[K]$",L"$T$ $[K]$"];
ylabels = [L"$\sigma$ $[(\Omega m)^{-1}]$",L"$S$ $[\mu VK^{-1}]$",L"n"];

df = Matrix(CSV.read(datafile, DataFrame, header=false))
rows = Int(size(df,1)/2)

for i in 1:2:rows
    # true_value
    band_1 = Band(vcat(df[i,1:3],fill(0.0,3)),en_1,type_1,1)
    band_2 = Band(vcat(df[i,4:6],fill(0.0,3)),en_2,type_2,1)
    model_true = BandStructure(2,[band_1,band_2],df[i,7])
    σ_true = electrical_conductivity(model_true,ts,τ_form)
    n_true = carrier_concentration(model_true,ts,τ_form)
    S_true = seebeck_coefficient(model_true,ts,τ_form)
    try
        # prediction
        band_1 = Band(vcat(df[i+1,1:3],fill(0.0,3)),en_1,type_1,1)
        band_2 = Band(vcat(df[i+1,4:6],fill(0.0,3)),en_2,type_2,1)
        model_pred = BandStructure(2,[band_1,band_2],df[i+1,7])
        σ_pred = electrical_conductivity(model_pred,ts,τ_form)
        n_pred = carrier_concentration(model_pred,ts,τ_form)
        S_pred = seebeck_coefficient(model_pred,ts,τ_form)
        
        # plots
        fig = CairoMakie.Figure(backgroundcolor = CairoMakie.RGBf(0.98, 0.98, 0.98), resolution = (600*3, 500))
        
        μ = round(df[i,7];digits = 2)
        titles = [L"$σ$ vs $T$, $μ_true$ = %$μ",
        L"$S$ vs $T$, $μ_true$ = %$μ",
        L"$n$ vs $T$, $μ_true$ = %$μ"];

        for (i,(t_true,t_pred)) in enumerate(zip([σ_true,S_true,n_true],[σ_pred,S_pred,n_pred]))
            ax = CairoMakie.Axis(fig[1,i], xlabel=xlabels[i], ylabel=ylabels[i], xlabelsize=xlabelsize, ylabelsize=ylabelsize)
            CairoMakie.Label(fig[1,i,CairoMakie.Top()], titles[i], padding = (0, 0, 25, 10), textsize=titlesize)
            CairoMakie.lines!(ax, ts, vec(t_true), label="True", linewidth=1.5)
            CairoMakie.lines!(ax, ts, vec(t_pred), label="Pred", linewidth=1.5)
            CairoMakie.axislegend(ax)
        end
        CairoMakie.save(joinpath(export_folder,"tensors_$(i).png"), fig)

        bandfig = Plot.plot_bandcomp(model_true,model_pred)
        savefig(joinpath(export_folder,"band_comp_$(i).png"), bandfig)
    catch e
        if isa(e, DomainError) || isa(e, TaskFailedException)
            continue
        else
            throw(e)
        end
    end
end

println("Plots exported.")


