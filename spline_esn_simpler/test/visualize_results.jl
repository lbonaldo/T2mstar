using CSV
using DataFrames
using CairoMakie
using Polynomials

using Mstar2t

######### PARAMS ######### 
nplots = 50
polydegree = 6
n_coeff = polydegree+1
en_1 = 1.0;
en_2 = 0.0;
minima = [en_1, en_2];
types = [1,-1];
deg = [1,1];
T = collect(50.:5:600);
τ_form = "constant";
export_folder = joinpath("spline_esn_simpler", "test")
datafile = joinpath(export_folder, "band_comb.txt")
##########################

df = Matrix(CSV.read(datafile, DataFrame, header=false))
rows = Int(size(df,1)/2)
indices = collect(1:100)
# indices = rand(1:rows,nplots)

# df = CSV.read(datafile, DataFrame, header=[:mx_c, :my_c, :mz_c, :mx_v, :my_v, :mz_v, :μ])

for i in indices
    i *= 2
    models_true = [vcat(fill(df[i-1,1],3),[0.0,0.0,0.0]),vcat(fill(df[i-1,2],3),[0.0,0.0,0.0])]
    μ_true = df[i-1,3]
    σ_true = electrical_conductivity(models_true,minima,types,deg,μ_true,T,τ_form)[1,:]
    n_true = carrier_concentration(models_true,minima,types,deg,μ_true,T,τ_form)[1,:]
    S_true = seebeck_coefficient(models_true,minima,types,deg,μ_true,T,τ_form)[1,:]
    models_pred = [vcat(fill(df[i,1],3),[0.0,0.0,0.0]),vcat(fill(df[i,2],3),[0.0,0.0,0.0])]
    μ_pred = df[i,3]
    try
        σ_pred = electrical_conductivity(models_pred,minima,types,deg,μ_pred,T,τ_form)[1,:]
        n_pred = carrier_concentration(models_pred,minima,types,deg,μ_pred,T,τ_form)[1,:]
        S_pred = seebeck_coefficient(models_pred,minima,types,deg,μ_pred,T,τ_form)[1,:]
        fig,ax,_ = lines(T, σ_true, label="True")
        lines!(T, σ_pred, label="Pred")
        axislegend(ax)
        save(joinpath(export_folder,"sigma_$(i).png"), fig)
        
        fig,ax,_ = lines(T, n_true, markerstrokewidth=0, label="True")
        lines!(T, n_pred, label="Pred")
        axislegend(ax)
        save(joinpath(export_folder,"n_$(i).png"), fig)
        
        fig,ax,_ = lines(T, S_true, markerstrokewidth=0, label="True")
        lines!(T, S_pred, label="Pred")
        axislegend(ax)
        save(joinpath(export_folder,"seebeck_$(i).png"), fig)
    catch e
        if isa(e, DomainError)
            continue
        end
    end
end


