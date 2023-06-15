using CSV
using JLD2
using DataFrames
using Polynomials
using Statistics
using CairoMakie
using Random
using Statistics

using Mstar2t

rng = MersenneTwister(1234);

function filterdf!(df::DataFrame)
    columns = [:σ_coeff1,:σ_coeff2,:σ_coeff3,:σ_coeff4,:σ_coeff5,:σ_coeff6,:σ_coeff7,:n_coeff1,:n_coeff2,:n_coeff3,:n_coeff4,:n_coeff5,:n_coeff6,:n_coeff7]
    for col in columns
        data = dataset[!,col]
        average = mean(data)
        stdev = stdm(data, average)
        filter!(col => x -> x .> average-2stdev && x .< average+2stdev, df)
    end
end

# function check_dataset(dataset::DataFrame,σ::DataFrame,S::DataFrame,n::DataFrame,polydegree::Int64,ts::Int64,x_cols::Int64)
function check_dataset(dataset::DataFrame,σ::DataFrame,n::DataFrame,polydegree::Int64,ts::Int64,x_cols::Int64)
    n_coeff = polydegree+1
    rows = size(dataset,1)
    T_start = findfirst(isequal("T1"), names(σ))
    T = Vector(σ[1,T_start:T_start+ts-1])
    sigma_pol = Array{Float64,2}(undef,rows,ts)
    n_pol = Array{Float64,2}(undef,rows,ts)
    #seebeck_pol = Array{Float64,2}(undef,rows,ts)
    sigma_res = Array{Float64,2}(undef,2rows,ts)
    n_res = Array{Float64,2}(undef,2rows,ts)
    #seebeck_res = Array{Float64,2}(undef,2rows,ts)
    for i in 1:2:2rows
        j = Int((i+1)/2)
        sigma_res[i,:] = Vector(σ[j,T_start+ts:end])
        n_res[i,:] = Vector(n[j,T_start+ts:end])
        #seebeck_res[i,:] = Vector(S[j,T_start+ts:end])
        p_sigma = Polynomial(dataset[j,x_cols+1:x_cols+n_coeff],:x)
        sigma_res[i+1,:] = p_sigma.(T)
        p_n = Polynomial(dataset[j,x_cols+n_coeff+1:x_cols+2n_coeff],:x)
        n_res[i+1,:] = p_n.(T) 
        #p_seebeck = Polynomial(dataset[j,x_cols+2n_coeff+1:x_cols+3n_coeff],:x)
        #seebeck_res[i+1,:] = p_seebeck.(T)
    end

    ### CSV files
    export_folder = joinpath(".","spline_esn_simpler","check_dataset")
    CSV.write(joinpath(export_folder,"check_sigma.csv"), Tables.table(sigma_res), writeheader=false)
    CSV.write(joinpath(export_folder,"check_n.csv"), Tables.table(n_res), writeheader=false)
    # CSV.write(joinpath(export_folder,"check_seebeck.csv"), Tables.table(seebeck_res), writeheader=false)
end

# function test_dataset(dataset::DataFrame,σ::DataFrame,S::DataFrame,cc::DataFrame,polydegree::Int64,ts::Int64,x_cols::Int64,nplots::Int64)
function test_dataset(dataset::DataFrame,σ::DataFrame,cc::DataFrame,polydegree::Int64,ts::Int64,x_cols::Int64,nplots::Int64)
    rows = size(dataset,1)
    indices = rand(1:rows,nplots)
    n_coeff = polydegree+1
    T_start = findfirst(isequal("T1"), names(σ))
    export_folder = joinpath(".","spline_esn_simpler","check_dataset")
    for i in indices
        T = Vector(σ[i,T_start:T_start+ts-1])
        sigma = Vector(σ[i,T_start+ts:end])
        n = Vector(cc[i,T_start+ts:end])
        # seebeck = Vector(S[i,T_start+ts:end])
    
        p_sigma = Polynomial(dataset[i,x_cols+1:x_cols+n_coeff],:x)
        pol_sigma = p_sigma.(T)
        p_n = Polynomial(dataset[i,x_cols+n_coeff+1:x_cols+2n_coeff],:x)
        pol_n = p_n.(T) 
        # p_seebeck = Polynomial(dataset[i,x_cols+2n_coeff+1:x_cols+3n_coeff],:x)
        # pol_seebeck = p_seebeck.(T)
    
        fig,_ = scatter(T, sigma, markerstrokewidth=0, label="Data")
        lines!(T, pol_sigma, label="Fit")
        save(joinpath(export_folder,"sigma_$(i).png"), fig)
    
        fig,_ = scatter(T, n, markerstrokewidth=0, label="Data")
        lines!(T, pol_n, label="Fit")
        save(joinpath(export_folder,"n_$(i).png"), fig)
    
        # fig,_ = scatter(T, seebeck, markerstrokewidth=0, label="Data")
        # lines!(T, pol_seebeck, label="Fit")
        # save(joinpath(export_folder,"seebeck_$(i).png"), fig)
    end
end

# filenames: array of fullpath of results.csv files, rows: number of models
# function create_dataset(σ::DataFrame,S::DataFrame,n::DataFrame,polydegree::Int64,ts::Int64)
function create_dataset(σ::DataFrame,n::DataFrame,polydegree::Int64,ts::Int64)
    n_coeff = polydegree+1
    rows = size(σ,1)
    x = select(σ, Not([1,5,6,7,11,12,13,14,15,16,17,18,19]))[:,[1,4,7]]  # three m* for each band, μ
    T_start = findfirst(isequal("T1"), names(σ))
    T = Vector(σ[1,T_start:T_start+ts-1])
    y_sigma = Matrix(σ[:,T_start+ts:end])
    y_n = Matrix(n[:,T_start+ts:end])
    # y_seebeck = Matrix(S[:,T_start+ts:end])
    # dataset = Array{Float64,2}(undef,rows,size(x,2)+3*n_coeff)
    dataset = Array{Float64,2}(undef,rows,size(x,2)+2*n_coeff)
    println(size(dataset))
    x_cols = size(x,2)
    for i in 1:rows
        dataset[i,1:x_cols] = Vector(x[i,:])
        pol_sigma = fit(T, y_sigma[i,:], polydegree)
        dataset[i,x_cols+1:x_cols+n_coeff] = pol_sigma.coeffs
        pol_n = fit(T, y_n[i,:], polydegree)
        dataset[i,x_cols+n_coeff+1:x_cols+2n_coeff] = pol_n.coeffs
    end
    # for i in 1:rows
    #     pol_seebeck = fit(T, y_seebeck[i,:], polydegree)
    #     tmp = pol_seebeck.coeffs
    #     while length(tmp) < n_coeff
    #         push!(tmp, 0.0)
    #     end
    #     dataset[i,x_cols+2n_coeff+1:x_cols+3n_coeff] = tmp
    # end
    σ_list_coeff = ["σ_coeff$(i)" for i in 1:n_coeff]
    n_list_coeff = ["n_coeff$(i)" for i in 1:n_coeff]
    # S_list_coeff = ["S_coeff$(i)" for i in 1:n_coeff]
    # header = vcat(names(x),σ_list_coeff,n_list_coeff,S_list_coeff)
    header = vcat(names(x),σ_list_coeff,n_list_coeff)
    return DataFrame(dataset,header),x_cols
end

######### PARAMS ######### 
# inputs
polydegree = 6
n_coeff = polydegree+1
traineval_perc = 0.85
eval_perc = 0.1
##########################

# MODEL DEFINITION
m_star = [1.,1.5,2.,2.5,3.,3.5,4.,4.5,5,5,5,6.,6.5]
models = Vector{Vector{Vector{Float64}}}(undef,length(m_star)*length(m_star));
for i in 1:length(m_star)
    for j in 1:length(m_star)
        models[(i-1)*length(m_star)+j] = [[m_star[i],m_star[i],m_star[i],0.,0.,0.], [m_star[j],m_star[j],m_star[j],0.,0.,0.]];
    end
end;
shuffle!(rng,models);

en_1 = 1.0;
en_2 = 0.0;
minima = [en_1, en_2];
types = [1,-1];
deg = [1,1];
μ = collect(-.1:0.0005:.1);
T = collect(50.:5:600);
τ_form = "constant";

σ = electrical_conductivity(models,minima,types,deg,μ,T,τ_form,exportasdf=true);
n = carrier_concentration(models,minima,types,deg,μ,T,τ_form,exportasdf=true);
# S = seebeck_coefficient(models,minima,types,deg,μ,T,τ_form,exportasdf=true);

ts = length(T)

dataset,x_cols = create_dataset(σ,n,polydegree,ts)
# dataset,x_cols = create_dataset(σ,S,n,polydegree,ts)
check_dataset(dataset,σ,n,polydegree,ts,x_cols)
# check_dataset(dataset,σ,S,n,polydegree,ts,x_cols)
test_dataset(dataset,σ,n,polydegree,ts,x_cols,50)
# test_dataset(dataset,σ,S,n,polydegree,ts,x_cols,50)
 
# plot distribution
export_folder = joinpath(".","spline_esn_simpler","check_dataset")
# for t in ["σ","n","S"]
for t in ["σ","n"]
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (1500, 1000))
    for i in 1:3
        for j in 1:3
            if (i-1)*3+j == (n_coeff+1)
                break
            else
                ax = Axis(fig[i,j])
                data = dataset[!,t*"_coeff"*string((i-1)*3+j)]
                average = mean(data)
                stdev = stdm(data, average) 
                println(t*"_coeff"*string((i-1)*3+j)*" Average: ", average, " Std: ", stdev)
                hist!(ax, data, bins = 100, strokewidth = 1, strokearound = true)
                vlines!(ax, average, color=:blue, linestyle=:dash)
                vlines!(ax, [average-stdev, average+stdev], color=:orange, linestyle=:dash)
            end
        end
    end
    save(joinpath(export_folder, "density_plot_"*t*".png"),fig)
end

# filter distribution
# for t in ["σ","n","S"]
for t in ["σ","n"]
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (1500, 1000))
    for i in 1:3
        for j in 1:3
            if (i-1)*3+j == (n_coeff+1)
                break
            else
                ax = Axis(fig[i,j])
                data = dataset[!,t*"_coeff"*string((i-1)*3+j)]
                len = length(data)
                average = mean(data)
                stdev = stdm(data, average) 
                data = data[data .> average-stdev]
                data = data[data .< average+stdev]
                println(t*"_coeff"*string((i-1)*3+j)*": Average: ", average, "\tStd: ", stdev, "\tReduc: ", (1-length(data)/len)*100)
                hist!(ax, data, bins = 100, strokewidth = 1, strokearound = true)
                vlines!(ax, average, color=:blue, linestyle=:dash)
                vlines!(ax, [average-stdev, average+stdev], color=:orange, linestyle=:dash)
            end
        end
    end
    save(joinpath(export_folder, "filtered_density_plot_"*t*".png"),fig)
end

## filter dataset
filterdf!(dataset)

println("Beginning split...")
total_size = size(dataset,1)
traineval_size = Int64(round(traineval_perc*total_size))
test_size = total_size - traineval_size
eval_size = Int64(round(eval_perc*traineval_size))
train_size = traineval_size - eval_size
println("Dataset size: ", total_size)
println("Train size: ", train_size)
println("Eval size: ", eval_size)
println("Test size: ", test_size)
println("Total sum: ", train_size+eval_size+test_size)

println("Setup complete, generating training and test data...")

shuffle_idx = Random.shuffle(collect(1:size(dataset,1)))

idx_train = shuffle_idx[1:train_size]
idx_eval = shuffle_idx[train_size+1:train_size+eval_size]
idx_test = shuffle_idx[train_size+eval_size+1:train_size+eval_size+test_size]

## a couple of checks
println("Intersect test 1: ", intersect(Set(idx_train),Set(idx_eval)))
println("Intersect test 2: ", intersect(Set(idx_train),Set(idx_test)))
println("Intersect test 3: ", intersect(Set(idx_eval),Set(idx_test)))
println("Num train data: ", length(idx_train), if length(idx_train)==train_size " (ok)" else "error" end)
println("Num eval data: ", length(idx_eval), if length(idx_eval)==eval_size " (ok)" else "error" end)
println("Num test data: ", length(idx_test), if length(idx_test)==test_size " (ok)" else "error" end)


using PyCall
np = pyimport("numpy");

# EXPORTING TRAIN DATA
x_train = dataset[idx_train,1:x_cols]
np.save("spline_esn_simpler/python_dataset/x_train.npy",np.asarray(Matrix(x_train)))
y_sigma_train = dataset[idx_train,x_cols+1:x_cols+n_coeff]
np.save("spline_esn_simpler/python_dataset/y_sigma_train.npy",np.asarray(Matrix(y_sigma_train)))
y_n_train = dataset[idx_train,x_cols+n_coeff+1:x_cols+2n_coeff]
np.save("spline_esn_simpler/python_dataset/y_n_train.npy",np.asarray(Matrix(y_n_train)))
# y_seebeck_train = dataset[idx_train,x_cols+2n_coeff+1:x_cols+3n_coeff]
# np.save("spline_esn_simpler/python_dataset/y_seebeck_train.npy",np.asarray(Matrix(y_seebeck_train)))

# EXPORTING VALIDATION DATA
x_eval = dataset[idx_eval,1:x_cols]
np.save("spline_esn_simpler/python_dataset/x_eval.npy",np.asarray(Matrix(x_eval)))
y_sigma_eval = dataset[idx_eval,x_cols+1:x_cols+n_coeff]
np.save("spline_esn_simpler/python_dataset/y_sigma_eval.npy",np.asarray(Matrix(y_sigma_eval)))
y_n_eval = dataset[idx_eval,x_cols+n_coeff+1:x_cols+2n_coeff]
np.save("spline_esn_simpler/python_dataset/y_n_eval.npy",np.asarray(Matrix(y_n_eval)))
# y_seebeck_eval = dataset[idx_eval,x_cols+2n_coeff+1:x_cols+3n_coeff]
# np.save("spline_esn_simpler/python_dataset/y_seebeck_eval.npy",np.asarray(Matrix(y_seebeck_eval)))

# EXPORTING TEST DATA
x_test = dataset[idx_test,1:x_cols]
np.save("spline_esn_simpler/python_dataset/x_test.npy",np.asarray(Matrix(x_test)))
y_sigma_test = dataset[idx_test,x_cols+1:x_cols+n_coeff]
np.save("spline_esn_simpler/python_dataset/y_sigma_test.npy",np.asarray(Matrix(y_sigma_test)))
y_n_test = dataset[idx_test,x_cols+n_coeff+1:x_cols+2n_coeff]
np.save("spline_esn_simpler/python_dataset/y_n_test.npy",np.asarray(Matrix(y_n_test)))
# y_seebeck_test = dataset[idx_test,x_cols+2n_coeff+1:x_cols+3n_coeff]
# np.save("spline_esn_simpler/python_dataset/y_seebeck_test.npy",np.asarray(Matrix(y_seebeck_test)))

## simple check
df_check = hcat(x_train,y_sigma_train,y_n_train)
# df_check = hcat(x_train,y_sigma_train,y_n_train,y_seebeck_train)
append!(df_check,hcat(x_eval,y_sigma_eval,y_n_eval))
# append!(df_check,hcat(x_eval,y_sigma_eval,y_n_eval,y_seebeck_eval))
append!(df_check,hcat(x_test,y_sigma_test,y_n_test))
# append!(df_check,hcat(x_test,y_sigma_test,y_n_test,y_seebeck_test))
dataset[shuffle_idx,:] == df_check


