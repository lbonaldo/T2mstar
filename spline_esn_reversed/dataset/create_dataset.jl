using CSV
using JLD2
#using Plots
using DataFrames
using Polynomials
using Statistics
using CairoMakie
using Random

function check_dataset(filenames::Array{String}, x_sigma::Array{Float64,2}, x_seebeck::Array{Float64,2}, x_n::Array{Float64,2}, rows::Int64, tn::Int64)
    n_files = length(filenames)
    x_sigma_pol = Array{Float64,2}(undef,rows*n_files,tn)
    x_seebeck_pol = Array{Float64,2}(undef,rows*n_files,tn)
    x_n_pol = Array{Float64,2}(undef,rows*n_files,tn)
    x_sigma_true = Array{Float64,2}(undef,rows*n_files,tn)
    x_seebeck_true = Array{Float64,2}(undef,rows*n_files,tn)
    x_n_true = Array{Float64,2}(undef,rows*n_files,tn)
    for (idx,file) in enumerate(filenames)
        data = Matrix(select(CSV.read(file, DataFrame), Not([1,5,6,7,11,12,13,14,15,16,17])))
        for i in 1:rows
            x_sigma_true[(idx-1)*rows+i,:] = data[i,10+tn:10+2tn-1]
            x_seebeck_true[(idx-1)*rows+i,:] = data[i,10+2tn:10+3tn-1]
            x_n_true[(idx-1)*rows+i,:] = data[i,10+3tn:10+4tn-1]
            ts = data[i,10:10+tn-1]
            p_sigma = Polynomial(x_sigma[(idx-1)*rows+i,:],:x)
            x_sigma_pol[(idx-1)*rows+i,:] = p_sigma.(ts)
            p_seebeck = Polynomial(x_seebeck[(idx-1)*rows+i,:],:x)
            x_seebeck_pol[(idx-1)*rows+i,:] = p_seebeck.(ts)
            p_n = Polynomial(x_n[(idx-1)*rows+i,:],:x)
            x_n_pol[(idx-1)*rows+i,:] = p_n.(ts) 
            #sigma += Statistics.mean()
            #seebeck += Statistics.mean(p_seebeck.(ts) - data[i,10+2tn:10+3tn-1])            
            #n += Statistics.mean(p_n.(ts) - data[i,10+3tn:10+4tn-1])
        end
    end

    ### CSV files
    res_sigma = [x_sigma_true x_sigma_pol]
    CSV.write("check_sigma.csv", Tables.table(res_sigma), writeheader=false)
    res_seebeck = [x_seebeck_true x_seebeck_pol]
    CSV.write("check_seebeck.csv", Tables.table(res_seebeck), writeheader=false)
    res_n = [x_n_true x_n_pol]
    CSV.write("check_n.csv", Tables.table(res_n), writeheader=false)

    ### VISUALIZATION
    # noto_sans = assetpath("fonts", "NotoSans-Regular.ttf")
    # noto_sans_bold = assetpath("fonts", "NotoSans-Bold.ttf")
    # fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98),
    # resolution = (1000, 1000), font = noto_sans)

    # axtop = Axis(fig[1,1])
    # axmain = Axis(fig[2,1])
    # axright = Axis(fig[2,2])

    # density!(axtop, Float64.(Iterators.flatten(x_sigma_pol)))
    # density!(axtop, Float64.(Iterators.flatten(x_sigma_true)))
    # density!(axmain, Float64.(Iterators.flatten(x_seebeck_pol)))
    # density!(axmain, Float64.(Iterators.flatten(x_seebeck_true)))
    # density!(axright, Float64.(Iterators.flatten(x_n_pol)))
    # density!(axright, Float64.(Iterators.flatten(x_n_true)))
    # save("distributions.png", fig)

end

function test_data(filenames::Array{String}, f_index::Array{Int64}, indices::Array{Int64}, tn::Int64)
    for idx in f_index
        data = Matrix(select(CSV.read(filenames[idx], DataFrame), Not([1,5,6,7,11,12,13,14,15,16,17])))
        for i in indices
            y = data[i,1:9]
            ts = data[i,10:10+tn-1]
            sigma = data[i,10+tn:10+2tn-1]
            seebeck = data[i,10+2tn:10+3tn-1]
            n = data[i,10+3tn:10+4tn-1]
        
            pol_sigma = fit(ts, sigma, polydegree)
            pol_seebeck = fit(ts, seebeck, polydegree)
            pol_n = fit(ts, n, polydegree)
        
            scatter(ts, sigma, markerstrokewidth=0, label="Data")
            plot!(pol_sigma, extrema(ts)..., label="Fit")
            savefig("sigma_$(i)_$idx.png")
        
            scatter(ts, seebeck, markerstrokewidth=0, label="Data")
            plot!(pol_seebeck, extrema(ts)..., label="Fit")
            savefig("seebeck_$(i)_$idx.png")
        
            scatter(ts, n, markerstrokewidth=0, label="Data")
            plot!(pol_n, extrema(ts)..., label="Fit")
            savefig("n_$(i)_$idx.png")
        end
    end
end

function create_dataset(filenames::Array{String}, rows::Int64, polydegree::Int64, tn::Int64)
    n_coeff = polydegree+1
    n_files = length(filenames)
    y = Array{Float64,2}(undef,rows*n_files,9)
    x_sigma = Array{Float64,2}(undef,rows*n_files,n_coeff)
    x_seebeck = Array{Float64,2}(undef,rows*n_files,n_coeff)
    x_n = Array{Float64,2}(undef,rows*n_files,n_coeff)
    # 1. loop over all files
    for (idx,file) in enumerate(filenames)
        # 2. read the file
        data = Matrix(select(CSV.read(file, DataFrame), Not([1,5,6,7,11,12,13,14,15,16,17])))
        y[(idx-1)*rows+1:idx*rows,:] = data[:,1:9]
        for i in 1:rows
            ts = data[i,10:10+tn-1]
            sigma = data[i,10+tn:10+2tn-1]
            pol_sigma = fit(ts, sigma, polydegree)
            x_sigma[(idx-1)*rows+i,:] = pol_sigma.coeffs
            n = data[i,10+3tn:10+4tn-1]
            pol_n = fit(ts, n, polydegree)
            x_n[(idx-1)*rows+i,:] = pol_n.coeffs
        end
        for i in 1:rows
            ts = data[i,10:10+tn-1]
            seebeck = data[i,10+2tn:10+3tn-1]
            pol_seebeck = fit(ts, seebeck, polydegree)
            tmp = pol_seebeck.coeffs
            while length(tmp) < n_coeff
                push!(tmp, 0.0)
            end
            x_seebeck[(idx-1)*rows+i,:] = tmp
        end
    end
    return x_sigma, x_seebeck, x_n, y
end

######### PARAMS ######### 
# inputs
tn = 19
polydegree = 5
rows = 4961

traineval_perc = 0.85
eval_perc = 0.1
##########################

# get file names into a string Vector
filenames = readlines("folders.txt")

# test_data(filenames, [11,12,13,14,15,16,17,18,19,20], [2531], tn)
x_sigma, x_seebeck, x_n, y = create_dataset(filenames, rows, polydegree, tn)

println(size(x_sigma))
println(size(x_seebeck))
println(size(x_n))
println(size(y))

# check_dataset(filenames, x_sigma, x_seebeck, x_n, rows, tn)

println("Beginning split...")
total_size = size(x_sigma,1)
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

shuffle_idx = Random.shuffle(collect(1:size(x_sigma,1)))

idx_train = shuffle_idx[1:train_size]
idx_eval = shuffle_idx[train_size+1:train_size+eval_size]
idx_test = shuffle_idx[train_size+eval_size+1:train_size+eval_size+test_size]

println("Intersect test 1: ", intersect(Set(idx_train),Set(idx_eval)))
println("Intersect test 2: ", intersect(Set(idx_train),Set(idx_test)))
println("Intersect test 3: ", intersect(Set(idx_eval),Set(idx_test)))

println("Num train data: ", length(idx_train))
println("Num eval data: ", length(idx_eval))
println("Num test data: ", length(idx_test))

### TRAIN DATA
y_train = y[idx_train,:]
@save "y_train.out" y_train
x_sigma_train = x_sigma[idx_train,:]
@save "x_sigma_train.out" x_sigma_train
x_seebeck_train = x_seebeck[idx_train,:]
@save "x_seebeck_train.out" x_seebeck_train
x_n_train = x_n[idx_train,:]
@save "x_n_train.out" x_n_train

### VALIDATION DATA
y_eval = y[idx_eval,:]
@save "y_eval.out" y_eval
x_sigma_eval = x_sigma[idx_eval,:]
@save "x_sigma_eval.out" x_sigma_eval
x_seebeck_eval = x_seebeck[idx_eval,:]
@save "x_seebeck_eval.out" x_seebeck_eval
x_n_eval = x_n[idx_eval,:]
@save "x_n_eval.out" x_n_eval

### TEST DATA
y_test = y[idx_test,:]
@save "y_test.out" y_test
x_sigma_test = x_sigma[idx_test,:]
@save "x_sigma_test.out" x_sigma_test
x_seebeck_test = x_seebeck[idx_test,:]
@save "x_seebeck_test.out" x_seebeck_test
x_n_test = x_n[idx_test,:]
@save "x_n_test.out" x_n_test
