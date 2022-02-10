using Pkg
Pkg.activate("../../")

using Random
using Plots
using Distributions
using JLD2: @load

@load "x_train.out" x_train
@load "y_train.out" y_train
@load "x_eval.out" x_eval
@load "y_eval.out" y_eval
@load "x_test.out" x_test
@load "y_test.out" y_test

coeff = [a,b,c,d,μ]
coeff_name = ['a','b','c','d','μ']

for i in eachindex(coeff)
    histogram(coeff[i])
    savefig(coeff_name[i]*".png")
end

# size = 2000

# a = 1 .- rand(Float64, size)
# c = 1 .- rand(Float64, size)
# b = (1 .- rand(Float64, size)) .* sign.(randn(Float64, size))
# d = (1 .- rand(Float64, size)) .* sign.(randn(Float64, size))
# μ = rand(Uniform(-1, 1), size)