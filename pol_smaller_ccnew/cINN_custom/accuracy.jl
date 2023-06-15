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
export_folder = joinpath(data2plot_folder,"accuracy_test")
if !isdir(export_folder) # if there is not a results folder -> create it
    mkdir(export_folder)
end
datafile = joinpath(data2plot_folder, "S_band_comb.txt")
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

df = Matrix(CSV.read(datafile, DataFrame, header=false))
rows = Int(size(df,1))
num_models = Int(size(df,1)/2)

# band params
mstar_1x = Matrix{Float64}(undef,num_models,2)
mstar_1y = Matrix{Float64}(undef,num_models,2)
mstar_1z = Matrix{Float64}(undef,num_models,2)
mstar_2x = Matrix{Float64}(undef,num_models,2)
mstar_2y = Matrix{Float64}(undef,num_models,2)
mstar_2z = Matrix{Float64}(undef,num_models,2)
mstar_trace = Matrix{Float64}(undef,num_models,2)
mu = Matrix{Float64}(undef,num_models,2)

# transport coefficients
num_temps = length(ts)
sigma   = Matrix{Float64}(undef,num_models*num_temps,2)
seebeck = Matrix{Float64}(undef,num_models*num_temps,2)
n       = Matrix{Float64}(undef,num_models*num_temps,2)

counter = 1

let j = 1
    global counter
    for i in 1:2:(rows-1)
        # band params
        mstar_1x[j,:] = [df[i,1],df[i+1,1]] # first entry: true, second entry: pred
        mstar_1y[j,:] = [df[i,2],df[i+1,2]] # first entry: true, second entry: pred
        mstar_1z[j,:] = [df[i,3],df[i+1,3]] # first entry: true, second entry: pred
        mstar_2x[j,:] = [df[i,4],df[i+1,4]] # first entry: true, second entry: pred
        mstar_2y[j,:] = [df[i,5],df[i+1,5]] # first entry: true, second entry: pred
        mstar_2z[j,:] = [df[i,6],df[i+1,6]] # first entry: true, second entry: pred
        mstar_trace[j,:] = [1/3*(df[i,1]+df[i,2]+df[i,3]),1/3*(df[i+1,1]+df[i+1,2]+df[i+1,3])]  # traces of mstar
        mu[j,:] = [df[i,7],df[i+1,7]]
        j += 1
    end
end
let j = 1
    global counter
    for i in 1:2:(rows-1)       
        try
            # transport coefficients
            band_1 = Band(vcat(df[i,1:3],fill(0.0,3)),en_1,type_1,1)
            band_2 = Band(vcat(df[i,4:6],fill(0.0,3)),en_2,type_2,1)
            model_true = BandStructure(2,[band_1,band_2],df[i,7])
            sigma[(j-1)*num_temps+1:j*num_temps,1] = electrical_conductivity(model_true,ts,τ_form)
            n[(j-1)*num_temps+1:j*num_temps,1] = carrier_concentration(model_true,ts,τ_form)
            seebeck[(j-1)*num_temps+1:j*num_temps,1] = seebeck_coefficient(model_true,ts,τ_form)

            band_1 = Band(vcat(df[i+1,1:3],fill(0.0,3)),en_1,type_1,1)
            band_2 = Band(vcat(df[i+1,4:6],fill(0.0,3)),en_2,type_2,1)
            model_pred = BandStructure(2,[band_1,band_2],df[i+1,7])
            sigma[(j-1)*num_temps+1:j*num_temps,2] = electrical_conductivity(model_pred,ts,τ_form)
            n[(j-1)*num_temps+1:j*num_temps,2] = carrier_concentration(model_pred,ts,τ_form)
            seebeck[(j-1)*num_temps+1:j*num_temps,2] = seebeck_coefficient(model_pred,ts,τ_form)
            j += 1
        catch e
            if isa(e, DomainError) || isa(e, TaskFailedException)
                continue
            else
                throw(e)
            end
        end
    end
    counter = j
end

println(rows)
println(size(mstar_1x))
println(num_models)
println(counter)

# get only saved values
sigma = sigma[1:counter*num_temps,:]
n = n[1:counter*num_temps,:]
seebeck = seebeck[1:counter*num_temps,:]

println(size(seebeck))

# plots band params
fig = CairoMakie.Figure(backgroundcolor = CairoMakie.RGBf(0.98, 0.98, 0.98), resolution = (1500,800))
    
xlabels = ["Predicted"];
ylabels = ["True"]
titles = ["mstar_1x","mstar_1y","mstar_1z","mstar_2x","mstar_2y","mstar_2z","mstar_trace","mu"]

for (k,band_param) in enumerate([mstar_1x,mstar_1y,mstar_1z,mstar_2x,mstar_2y,mstar_2z,mstar_trace,mu])
    i,j = (k-1)÷4+1,(k-1)%4+1
    ax = CairoMakie.Axis(fig[i,j], xlabel=xlabels[1], ylabel=ylabels[1], xlabelsize=xlabelsize, ylabelsize=ylabelsize)
    CairoMakie.Label(fig[i,j,CairoMakie.Top()], titles[k], padding = (0, 0, 25, 10), textsize=titlesize)
    CairoMakie.lines!(ax, [minimum(band_param[:,1]),maximum(band_param[:,1])], [minimum(band_param[:,1]),maximum(band_param[:,1])], color=:orange)
    CairoMakie.scatter!(ax, band_param[:,2], band_param[:,1], markersize = 3)
end
CairoMakie.save(joinpath(export_folder,"band_params.png"), fig)
println("Band params plots done.")


# plots transport coefficients
fig = CairoMakie.Figure(backgroundcolor = CairoMakie.RGBf(0.98, 0.98, 0.98), resolution = (650*3,500))
    
xlabels = ["Predicted"];
ylabels = ["True"]
titles = ["El conductivity","Seebeck","Carrier density"]

for (i,tensor) in enumerate([sigma,seebeck,n])
    # tensor = tensor[sortperm(tensor[:,1]),:]
    ax = CairoMakie.Axis(fig[1,i], xlabel=xlabels[1], ylabel=ylabels[1], xlabelsize=xlabelsize, ylabelsize=ylabelsize)
    CairoMakie.Label(fig[1,i,CairoMakie.Top()], titles[i], padding = (0, 0, 25, 10), textsize=titlesize)
    CairoMakie.lines!(ax, [minimum(tensor[:,1]),maximum(tensor[:,1])], [minimum(tensor[:,1]),maximum(tensor[:,1])], color=:orange)
    CairoMakie.scatter!(ax, tensor[:,2], tensor[:,1], markersize = 3)

end
CairoMakie.save(joinpath(export_folder,"tensors.png"), fig)
println("Coeffs plots done.")

println("Plots exported.")
