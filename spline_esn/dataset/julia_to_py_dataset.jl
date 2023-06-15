# (inverse) inverse> $env:JULIA_NUM_THREADS=4
# (inverse) inverse> julia --project=. .\julia_to_py_dataset.jl

using Random
using PyCall
using JLD2: @load
np = pyimport("numpy")

jroot_folder = "data_julia"
pyroot_folder = "data_python"

# macro getvarname(arg)
#     string(arg)
# end

# function loadconvertsave(jfilename)
#     (filename,jext) = splitext(jfilename)
#     jfile = joinpath(jroot_folder, jfilename)
#     jdata = eval(@load(jfile)[1])
#     pydata = np.asarray(jdata)
#     np.save(joinpath(pyroot_folder, filename*".npy"),pydata)
# end

@load "data_julia/y_train.out" y_train
filename = "y_train"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(y_train))
@load "data_julia/y_eval.out" y_eval
filename = "y_eval"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(y_eval))
@load "data_julia/y_test.out" y_test
filename = "y_test"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(y_test))
@load "data_julia/x_sigma_train.out" x_sigma_train
filename = "x_sigma_train"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(x_sigma_train))
@load "data_julia/x_sigma_eval.out" x_sigma_eval
filename = "x_sigma_eval"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(x_sigma_eval))
@load "data_julia/x_sigma_test.out" x_sigma_test
filename = "x_sigma_test"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(x_sigma_test))
@load "data_julia/x_seebeck_train.out" x_seebeck_train
filename = "x_seebeck_train"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(x_seebeck_train))
@load "data_julia/x_seebeck_eval.out" x_seebeck_eval
filename = "x_seebeck_eval"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(x_seebeck_eval))
@load "data_julia/x_seebeck_test.out"   x_seebeck_test
filename = "x_seebeck_test"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(x_seebeck_test))
@load "data_julia/x_n_train.out" x_n_train
filename = "x_n_train"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(x_n_train))
@load "data_julia/x_n_eval.out" x_n_eval
filename = "x_n_eval"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(x_n_eval))
@load "data_julia/x_n_test.out" x_n_test
filename = "x_n_test"
np.save(joinpath(pyroot_folder, filename*".npy"), np.asarray(x_n_test))