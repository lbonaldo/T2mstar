# (inverse) inverse> $env:JULIA_NUM_THREADS=4
# (inverse) inverse> julia --project=. .\julia_to_py_dataset.jl

using JLD2
using QuadGK
using Roots
using Polynomials

######### PARAMS ######### 
tmin = 0.05
dt = 0.01
tmax = 5.0
polydegree = 9

total_size = 1000000
traineval_perc = 0.8
eval_perc = 0.2
##########################

println("Beginning setup...")
ts = collect(tmin:dt:tmax)
traineval_size = Int64(traineval_perc*total_size)
test_size = total_size - traineval_size
eval_size = Int64(eval_perc*traineval_size)
train_size = traineval_size - eval_size
println("Setup complete, generating training and test data...")

# Defines the Left function.
function l(x::Float64, a::Float64, b::Float64)
    if -a*x - b > 0
        return sqrt(-a*x - b)
    else
        return 0
    end
end

function r(x::Float64, c::Float64, d::Float64)
    if c*x - d > 0
        return sqrt(c*x - d)
    else
        return 0
    end
end

# This is the meat of this script. This actually calculate F(β, params).
function forwardProb(β::Float64, a::Float64, b::Float64, c::Float64, d::Float64, μ::Float64)
    # Defines the normalized gaussian function. It is defined here and not above with L and R because the numerical routines want a function of one variable.
    gaussian(x) = sqrt(β/π)*exp(-β*(x - μ)^2)
    # Defines the functions used by Roots.jl to find the intersection points
    l_min(x) = l(x, a, b) - gaussian(x)
    r_min(x) = r(x, c, d) - gaussian(x)
    # If the bands cross and β is sufficiently small, we only end up integrating over the gaussian, which is normalized. So we get 1. 
    # gaussian((d-b)/(a+c)) is the height of the gaussian at the intersection point, and sqrt(-(a*d + b*c)/(a + c)) is the height at which the crossing bands touch.
    bandcross = (d/c) < (-b/a) ? true : false
    if bandcross == true && gaussian((d-b)/(a+c)) < sqrt(-(a*d + b*c)/(a + c))
        return 1.0
    end
    # We need to find the intersection between the gaussian and the bands. Unfortunately, there is no analytical solution to this. I use a root finding routing (Roots.jl) to find these two points.
    # However, we can be intelligent about where to find the roots. It is possible to show mathematically using the max height of the gaussian that:
    # x0 ∈ [-(β + πb)/πa, -b/a] and x3 ∈ [d/c, (β + πd)/πc]
    # So we will look for the roots in these intervals. These values are independent of whether the bands cross.
    x0 = find_zero(l_min, (-(β + pi*b)/(pi*a), -b/a), Bisection())
    x3 = find_zero(r_min, (d/c, (β + pi*d)/(pi*c)), Bisection())
    # Perform the gaussian integrals. These are independent of whether the bands cross.
    A = quadgk(gaussian, -Inf, x0)[1]
    B = quadgk(gaussian, x3, Inf)[1]
    # Now for the other integrals, which do depend on whether the bands cross.
    # I solved these integrals analytically, so only the result is here.
    if !bandcross
        A += (2/(3a)) * abs(-a*x0 - b)^1.5 # this can occasionally be slightly negative due to numerical imprecision
        B += (2/(3c)) * abs(c*x3 - d)^1.5 #...
    else # I apoligize in advance, this algebra is ugly
        A += (2/(3a)) * (abs(-a*x0 - b)^1.5 - abs(-(a*d + b*c)/(a+c))^1.5)
        B += (2/(3c)) * (abs(-(a*d + b*c)/(a+c))^1.5 - abs(c*x3-d)^1.5)
    end
    return B-A
end

# This function generates a single data "point", consisting of a set of parameters and the corresponding function F(β, params)
function generateDataPoint(ts::Vector{Float64}, polydegree::Int64)
    # a and c are strictly positive, since they determine the direction of the function.
    a,c = rand(Float64, 2)
    # b and d can be either positive of negative, since bands are allowed to cross. 
    b,d = rand(Float64, 2) .* sign.(randn(Float64, 2))
    # Since the intersection point for both bands is -b/a and d/c, respectively, if a or c are sufficiently small our band positions can be pretty crazy. 
    # This is just a sanity check to makes sure that they aren't, so that μ always lies between -1 and 1. This makes it a bit more ML friendly.
    if abs(a) < abs(b)
        bs = sign(b)
        temp = a
        a = abs(b)
        b = bs*temp
    end
    if abs(c) < abs(d)
        cs = sign(d)
        temp = c
        c = abs(d)
        d = cs*temp
    end
    # no band crossing for this test
    while d/c < -b/a
        tmp, d = d, -b
        b = -tmp
        a, c = c, a
    end
    # Increases the chance that μ lies between the bands. The bands hit zero at -b/a and d/c, respectively. μ is selected at a random spot between these two points.
    μ_t = rand(Float64)
    μ = (1-μ_t)*(-b/a) + μ_t*(d/c)
    # Calculate I(β, params) by calling forwardProb for each value of β.
    I = map(t -> forwardProb(t, a, b, c, d, μ), ts)
    pol = fit(ts, I, polydegree)
    return (pol.coeffs, [a, b, c, d, μ]) # Input/Output pair
end

x_train = Array{Float64,2}(undef,train_size,10)
y_train = Array{Float64,2}(undef,train_size,5)
x_eval = Array{Float64,2}(undef,eval_size,10)
y_eval = Array{Float64,2}(undef,eval_size,5)
x_test = Array{Float64,2}(undef,test_size,10)
y_test = Array{Float64,2}(undef,test_size,5)

time = time_ns()
# TRAINING DATA
Threads.@threads for i in 1:train_size # This is the number of points in the training set
    local g = generateDataPoint(ts, polydegree)
    x_train[i,:] = g[1]
    y_train[i,:] = g[2]
end
@save "x_train.out" x_train
@save "y_train.out" y_train
println("[1/3] Training data created.")
# VALID DATA
Threads.@threads for i in 1:eval_size # This is the number of points in the valdiation set
    local g = generateDataPoint(ts, polydegree)
    x_eval[i,:] = g[1]
    y_eval[i,:] = g[2]
end
@save "x_eval.out" x_eval
@save "y_eval.out" y_eval
println("[2/3] Validation data created.")
# TEST DATA
Threads.@threads for i in 1:test_size # This is the number of points in the test set
    local g = generateDataPoint(ts, polydegree)
    x_test[i,:] = g[1]
    y_test[i,:] = g[2]
end
@save "x_test.out" x_test
@save "y_test.out" y_test
println("[3/3] Test data created.")
gen_time = time_ns() - time

print("\n")