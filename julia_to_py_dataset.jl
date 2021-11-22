# (inverse) inverse> $env:JULIA_NUM_THREADS=4
# (inverse) inverse> julia --project=. .\julia_to_py_dataset.jl

using Random
using PyCall
using JLD2: @load
np = pyimport("numpy")

train_file = "data/train_data.out"
test_file = "data/test_data.out"
@load train_file traindata
@load test_file testdata

xmin = 0.05
dx = 0.01
xmax = 5.0
beta = collect(xmin:dx:xmax)

frac = 0.1
num_rows = Int64(round(length(traindata)*frac))
num_column = length(traindata[1][1])
num_params = length(traindata[1][2])+1

println(num_rows)

x_train = Array{Float64,2}(undef, num_rows*num_column, num_params)
y_train = Array{Float64,1}(undef, num_rows*num_column)

index = Vector(0:num_rows*num_column-1)
shuffle!(MersenneTwister(1234), index)

Threads.@threads for k in eachindex(index)
    a = index[k]
    b = num_column
    i = div(a,b)
    j = Int64(round((a/b-i)*b))+1
    i += 1
    t = beta[j]
    x_train[k,:] = push!(copy(traindata[i][2]),t)
    y_train[k] = traindata[i][1][j]
end

data = np.asarray(x_train)
np.save("data/coeff_train.npy",data)
data = np.asarray(y_train)
np.save("data/I_train.npy",data)

num_rows = Int64(round(length(testdata)))
num_column = length(testdata[1][1])
num_params = length(testdata[1][2])+1

println(num_rows)

x_test = Array{Float64,2}(undef, num_rows*num_column, num_params)
y_test = Array{Float64,1}(undef, num_rows*num_column)

index = Vector(0:num_rows*num_column-1)
shuffle!(MersenneTwister(1234), index)

Threads.@threads for k in eachindex(index)
    a = index[k]
    b = num_column
    i = div(a,b)
    j = Int64(round((a/b-i)*b))+1
    i += 1
    t = beta[j]
    x_test[k,:] = push!(copy(testdata[i][2]),t)
    y_test[k] = testdata[i][1][j]
end

data = np.asarray(x_test)
np.save("data/coeff_test.npy",data)
data = np.asarray(y_test)
np.save("data/I_test.npy",data)
