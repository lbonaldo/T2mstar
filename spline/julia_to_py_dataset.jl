# (inverse) inverse> $env:JULIA_NUM_THREADS=4
# (inverse) inverse> julia --project=.. .\julia_to_py_dataset.jl

using Random
using PyCall
using JLD2: @load
np = pyimport("numpy")

@load "dataset/x_train.out" x_train
@load "dataset/y_train.out" y_train
@load "dataset/x_eval.out" x_eval
@load "dataset/y_eval.out" y_eval
@load "dataset/x_test.out" x_test
@load "dataset/y_test.out" y_test

train_size = size(x_train)[1]
eval_size = size(x_eval)[1]
test_size = size(x_test)[1]
println("Training data size: ", train_size)
println("Validation data size: ", eval_size)
println("Test data size: ", test_size)
input_x_size = size(x_train)[2]
output_y_size = size(y_train)[2]
println("Input size: ", input_x_size)
println("Output size: ", output_y_size)

println("Are they correct [yes/no]?")
answer = readline()

if answer == "yes"
    np.save("dataset/x_train.npy", np.asarray(x_train))
    np.save("dataset/y_train.npy", np.asarray(y_train))
    np.save("dataset/x_eval.npy", np.asarray(x_eval))
    np.save("dataset/y_eval.npy", np.asarray(y_eval))
    np.save("dataset/x_test.npy", np.asarray(x_test))
    np.save("dataset/y_test.npy", np.asarray(y_test))
else
    exit("exiting...")
end