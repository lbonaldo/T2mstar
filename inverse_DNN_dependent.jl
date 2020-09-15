using Flux
using CuArrays
using Flux: throttle, @epochs
using Flux.Optimise
using JLD2: @load
using BSON: @save
using Optim

#The actual neural network
model = Chain(
    Dense(496, 248, logcosh),
    Dense(248, 124, logcosh),
    Dense(124, 62, logcosh),
    Dense(62, 7, logcosh),
    #Dense(62, 31, relu),
    #Dense(31, 15, leakyrelu),
    #Dense(15, 7, elu)

) |> gpu

#=
model = Chain(
    Dense(496, 992, logcosh),
    #Dense(992, 992, elu),
    Dense(992, 496, logcosh),
    Dense(496, 248, elu),
    Dense(248, 124, elu),
    Dense(124, 62, leakyrelu),
    #Dense(62, 7, tanh)
    Dense(62, 31, leakyrelu),
    Dense(31, 15, tanh),
    Dense(15, 7, tanh)

) |> gpu
=#
if length(ARGS) != 2
    println("Invalid arguments. Usage: julia something.jl traindata.out testdata.out")
    exit(1)
end
#trainfile = open(ARGS[1], "r")
#testfile = open(ARGS[2], "r")
#traindata = gpu.(read(trainfile)) #Zips the input and output data in the training set together and moves it to a GPU (if availibile and CuArrays is set up correctly)
#testdata = gpu.(read(testfile))
#Imports the data
@load ARGS[1] traindata
@load ARGS[2] testdata
println(typeof(traindata))
#Moves the data to the GPU for more efficient training
traindata = gpu.(traindata)
testdata = gpu.(testdata)
#mse(ŷ, y) = sum((ŷ .- y).^2) * 1 // length(y)
#weighted_mse(yhat, y) = (sum((yhat .- y).^2) + 9*(yhat[end] - y[end])^2) * 1 // length(y)

#Mean squared error loss function
function loss(x, y)
    yhat = model(x)
    #return weighted_mse(y, yhat)
    return Flux.mse(yhat, y)
end
loss((x, y)) = loss(x, y)

#This function calculates the average loss for an entire set of data.
function determineMeanAccuracy(data)
    earr = loss.(data)
    return sum(earr)/length(earr)
end
etot_b = determineMeanAccuracy(testdata)
println("Average MSE of test set (before training): $etot_b")

ps = Flux.params(model)
#opt = Optim.LBFGS()
opt = AdaMax(0.001) #Gradient Descent optimizer with a learning rate of 0.001.
#opt = Momentum(0.005)
#opt = ADAM(0.01)

#This is the callback function, that in this case prints the loss function of the first element of the training
# set every five seconds
evalcb = throttle(() -> @show(loss(traindata[1])), 5) #Every five seconds prints a loss to give an idea of how the training is going
println("Beginning training...")
@epochs 3 Flux.train!(loss, ps, traindata, opt, cb = evalcb) #Trains the network over the data set 10 times.
@show out = model(testdata[1][1]) |> cpu
println(testdata[1][2])
#testout = cpu.(testout)

etot = determineMeanAccuracy(testdata)
println("Average MSE of training set (after training): $etot")
#Moves the model back to the CPU so it can be written to disk
model = cpu(model)
@save "model.bson" model
