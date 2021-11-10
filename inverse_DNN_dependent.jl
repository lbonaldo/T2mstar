using Flux
#using CuArrays
using Flux: throttle, @epochs
using Flux.Optimise
using JLD2: @load
#using BSON: @save
#using Optim
#using Plots

#The actual neural network
#=
model = Chain(
    Dense(496, 248, tanh),
    Dense(248, 124, tanh),
    Dense(124, 62, tanh),
    Dense(62, 5, tanh),
    #Dense(62, 31, relu),
    #Dense(31, 15, leakyrelu),
    #Dense(15, 7, elu)

) |> gpu
=#

model = Chain(
    Dense(496, 992, leakyrelu),
    Dense(992, 496, leakyrelu),
    Dense(496, 248, leakyrelu),
    Dense(248, 124, leakyrelu),
    Dense(124, 62, leakyrelu),
    Dense(62, 31, leakyrelu),
    Dense(31, 15, leakyrelu),
    Dense(15, 5, σ)

) |> gpu

#trainfile = open(ARGS[1], "r")
#testfile = open(ARGS[2], "r")
#traindata = gpu.(read(trainfile)) #Zips the input and output data in the training set together and moves it to a GPU (if availibile and CuArrays is set up correctly)
#testdata = gpu.(read(testfile))
#Imports the data
train_file = "train.out"
test_file = "test.out"
@load train_file traindata
@load test_file testdata
println(traindata[100001])
println(typeof(traindata))
exit()
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
#etot_b = determineMeanAccuracy(testdata)
#println("Average MSE of test set (before training): $etot_b")

epochs = 5
training_rate = 0.001
callback_rate = 5 #The amount of time between error callback, in seconds
etot_train = zeros(Float64, epochs+1)
etot_test = zeros(Float64, epochs+1)
etot_train[1] = determineMeanAccuracy(traindata)
etot_test[1] = determineMeanAccuracy(testdata)
eval_train = Float64[]
epoch_points = zeros(Int64, epochs)

ps = Flux.params(model)
#opt = Optim.LBFGS()
#opt = AdaMax(training_rate) #Gradient Descent optimizer with a learning rate of 0.001.
opt = Momentum(training_rate)
#opt = ADAM(0.01)

# This is the callback function, that in this case prints the loss function of the first element of the training
# set every five seconds
evalcb = throttle(() -> push!(eval_train, loss(traindata[1])), callback_rate) #Every five seconds prints a loss to give an idea of how the training is going
println("Beginning training...")
for i in 1:epochs
    global epoch_points
    global etot_train
    global etot_test
    println("Beginning Epoch [$i]")
    Flux.train!(loss, ps, traindata, opt, cb=evalcb)
    etot_train[i+1] = determineMeanAccuracy(traindata)
    etot_test[i+1] = determineMeanAccuracy(testdata)
    model_i = cpu(model)
    @save "model_$i.bson" model_i
    epoch_points[i] = length(eval_train)
end
#@epochs epochs Flux.train!(loss, ps, traindata, opt, cb = evalcb) #Trains the network over the data set 10 times.
#@epochs epochs cust_train(loss, ps, traindata, opt, evalcb)
@show out = model(testdata[1][1]) |> cpu
println(testdata[1][2])
#testout = cpu.(testout)

println("Training finished, generating plots...")
pyplot()

function detailed_loss(data)
    a = zeros(length(data))
    b = zeros(length(data))
    c = zeros(length(data))
    d = zeros(length(data))
    μ = zeros(length(data))
    a_hat = zeros(length(data))
    b_hat = zeros(length(data))
    c_hat = zeros(length(data))
    d_hat = zeros(length(data))
    μ_hat = zeros(length(data))
    for i in eachindex(data)
        x, y = data[i]
        yhat = model(x)
        a[i] = y[1]
        b[i] = y[2]
        c[i] = y[3]
        d[i] = y[4]
        μ[i] = y[5]
        a_hat[i] = yhat[1]
        b_hat[i] = yhat[2]
        c_hat[i] = yhat[3]
        d_hat[i] = yhat[4]
        μ_hat[i] = yhat[5]
    end
    a_error = Flux.mse(a, a_hat)
    b_error = Flux.mse(b, b_hat)
    c_error = Flux.mse(c, c_hat)
    d_error = Flux.mse(d, d_hat)
    μ_error = Flux.mse(μ, μ_hat)
    return (a_error, b_error, c_error, d_error, μ_error)
end


println("coefficient-specific loss:")
println(detailed_loss(testdata))

cb_plot = plot(eval_train, title="First Element Callback Error", label="Loss Function Value")
plot!(cb_plot, epoch_points[1:end-1], map(x->eval_train[x], epoch_points[1:end-1]), seriestype=:scatter, label="Epochs")
savefig(cb_plot, "Callback.png")

train_plot = plot(0:epochs, etot_train, title="Loss function at Epochs", label="Training Loss")
plot!(train_plot, 0:epochs, etot_test, label="Validation Loss")
savefig(train_plot, "Training.png")

#Moves the model back to the CPU so it can be written to disk
model = cpu(model)
@save "model_final.bson" model
