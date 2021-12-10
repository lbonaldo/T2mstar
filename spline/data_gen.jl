using JLD2
using QuadGK
using Roots

tol = 1e-5 # Difference tolerance. Shouldn't need to be changed
size = 100 # The number of elements in the training set to generate
test_frac = 0.1 # The size of the test set relative to the size of the train set.

#root_heur = Dict{Array{Float64, 1}, Array{Float64, 1}}()

# Defines the Left function.
function l(x::Float64, a::Float64, b::Float64)
    if -a*x - b > 0
        return sqrt(-a*x - b)
    else
        return 0
    end
end
# Defines the Right function.
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
function generateDataPoint(tmin::Float64, tx::Float64, tmax::Float64)
    # A grid of β values.
    ts = collect(tmin:tx:tmax)
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
    if (d/c) < (-b/a)
        tmp, d = d, -b
        b = -tmp
        a, c = c, a
    end

    # Increases the chance that μ lies between the bands. The bands hit zero at -b/a and d/c, respectively. μ is selected at a random spot between these two points.
    μ_t = rand(Float64)
    μ = (1-μ_t)*(-b/a) + μ_t*(d/c)
    # Calculate I(β, params) by calling forwardProb for each value of β.
    I = map(t -> forwardProb(t, a, b, c, d, μ), ts)
    
    return (I, [a, b, c, d, μ]) # Input/Output pair
end

if length(ARGS) != 2
    println("Invalid arguments. Usage: julia data_gen.jl traindata.out testdata.out")
    exit(1)
end

println("Beginning setup...")
tmin = 0.05
tx = 0.01
tmax = 5.0
println("Setup complete, generating training and test data...")

# Initializes the arrays. This is because I'm lazy and am expanding the arrays below.
g = generateDataPoint(xmin, dx, xmax)
g2 = generateDataPoint(xmin, dx, xmax)
testout = generateDataPoint(xmin, dx, xmax)
xs = [g[1]]
ys = [g[2]]
xg = [g2[1]]
yg = [g2[2]]

time = time_ns()
# Prints a progress bar that will be updated during the calculation.
print("Progress: |" * "-"^50 * ">")
percstore = 0.0
for i in 1:size # This is the number of points in the training set
    global percstore
    local g = generateDataPoint(xmin, dx, xmax)
    # Pushes the new data to the arrays created above. This is actually reasonably efficient, unlike np.append().
    push!(xs, g[1])
    push!(ys, g[2])
    # Math for the progress bar.
    perc = Int(round(i/((1 + test_frac)*size)*50))
    if perc != percstore
        bar = "#"^perc * "-"^(50 - perc)
        print("\r" * "Progress: |" * bar * ">")
    end
    percstore = perc
end
# Does the same thing but for the test/validation set.
for i in 1:test_frac*size # This is the number of points in the test set
    global percstore
    local g = generateDataPoint(xmin, dx, xmax)
    push!(xg, g[1])
    push!(yg, g[2])
    perc = Int(round((i + size)/((1 + test_frac)*size)*50))
    if perc != percstore
        bar = "#"^perc * "-"^(50 - perc)
        print("\r" * "Progress: |" * bar * ">")
    end
    percstore = perc
end
gen_time = time_ns() - time

# Bundles the data together.
traindata = collect(zip(xs, ys))
testdata = collect(zip(xg, yg))

# Saves the data to files specified by the command line arguments.
@save ARGS[1] traindata
@save ARGS[2] testdata
print("\n")