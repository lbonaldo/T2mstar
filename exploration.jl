using Statistics
using JLD2: @load
train_file = "data/train_data.out"
test_file = "data/test_data.out"
@load train_file traindata
@load test_file testdata

frac = 0.1
num_rows = Int64(round(length(traindata)*frac))
num_column = length(traindata[1][1])
num_params = length(traindata[1][2])+1

x_train = Array{Float64,2}(undef, num_rows*num_column, num_params)
y_train = Array{Float64,1}(undef, num_rows*num_column)

xmin = 0.05
dx = 0.01
xmax = 5.0
beta = collect(xmin:dx:xmax)

Threads.@threads for i in 1:num_rows
    for j in 1:num_column
        x_train[(i-1)*num_column+j,:] = push!(copy(traindata[i][2]), beta[j])
        y_train[(i-1)*num_column+j] = traindata[i][1][j]
    end
end

frac = 0.1
num_rows = Int64(round(length(testdata)*frac))
num_column = length(testdata[1][1])
num_params = length(testdata[1][2])+1

x_test = Array{Float64,2}(undef, num_rows*num_column, num_params)
y_test = Array{Float64,1}(undef, num_rows*num_column)

xmin = 0.05
dx = 0.01
xmax = 5.0
beta = collect(xmin:dx:xmax)

Threads.@threads for i in 1:num_rows
    for j in 1:num_column
        x_test[(i-1)*num_column+j,:] = push!(copy(testdata[i][2]), beta[j])
        y_test[(i-1)*num_column+j] = testdata[i][1][j]
    end
end

println(mean!([1. 1. 1. 1. 1. 1.], x_train))
println(mean!([1. 1. 1. 1. 1. 1.], x_test))
println(std(x_train[:,1]), "\t", std(x_test[:,1]))
println(std(x_train[:,2]), "\t", std(x_test[:,2]))
println(std(x_train[:,3]), "\t", std(x_test[:,3]))
println(std(x_train[:,4]), "\t", std(x_test[:,4]))
println(std(x_train[:,5]), "\t", std(x_test[:,5]))
println(std(x_train[:,6]), "\t", std(x_test[:,6]))
println(mean(y_train), "\t", mean(y_test))
println(std(y_train), "\t", std(y_test))
