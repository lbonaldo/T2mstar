using JLD2
using QuadGK
using Roots

tol = 1e-5 #Difference tolerance. Shouldn't need to be changed
size = 40000 #The number of elements in the training set to generate
test_frac = 0.2 #The size of the test set relative to the size of the train set.

#root_heur = Dict{Array{Float64, 1}, Array{Float64, 1}}()

#Defines the Left function.
function l(x::Float64, a::Float64, b::Float64, c::Float64)
    if -b*x -c > 0
        return a*sqrt(-b*x - c)
    else
        return 0
    end
end
#Defines the Right function.
function r(x::Float64, d::Float64, f::Float64, g::Float64)
    if f*x -g > 0
        return d*sqrt(f*x - g)
    else
        return 0
    end
end

#This is the meat of this script. This actually calculate F(β, params).
function forwardProb(β::Float64, params::Array{Float64, 1})
    α = sqrt(β/π) #normalizes the gaussian
    a,b,d,f,c,g,μ = params
    #Defines the gaussian function. It is defined here and not above with L and R because the numerical routines want a function of one variable.
    gaussian(x) = α*exp(-β*(x - μ)^2)
    l_min(x) = l(x, a, b, c) - gaussian(x) #The function that needs to be set to 0 to find the intersection of L and the gaussian.
    r_min(x) = r(x, d, f, g) - gaussian(x) # ... R and the gaussian.
    l_int(x) = l(x, a, b, c) #Defines an integrable function of one variable.
    r_int(x) = r(x, d, f, g)
    #Here, we are permitting our "bands" to cross, ie L and R can intersect as well. This results in some strange math. The bandcross variable gets flagged true if such a crossing exists.
    bandcross = false
    if (-c/b) > (g/f) #Checks if the bands cross.
        bandcross = true
    end
    #These are the four key points for integration, set to 0.0 before calculation.
    # x0 is the intersection of L and the gaussian, x1 is the intersection of L with 0,
    # x2 is the intersection of R with 0, and x3 is the intersection of R with the gaussian.
    #
    # If the bands cross, than x1 = x2 is the intersection of R and L.
    x0 = 0.0
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0

    if bandcross #The bands cross
        x1 = (d^2*g - a^2*c)/(a^2*b + d^2*f)
        x2 = x1
    else
        x1 = -c/b
        x2 = g/f
    end
    #Finds the intersection between the outside functions and the gaussian.
    #Places an outer bound based on the height of the gaussian. This CAN result in a failed root-finding, hence the try-catch.
    try
        if bandcross
            x3 = find_zero(l_min, (-c/b, x1), Bisection())
            x0 = find_zero(r_min, (x2, g/f), Bisection())
        else
            x0 = find_zero(l_min, (-(α^2+a^2*c)/(a^2*b), x1), Bisection())
            x3 = find_zero(r_min, (x2, (α^2+d^2*g)/(d^2*f)), Bisection())
        end
    #This triggers if the above root finding failes.
    catch err #The root-finding here is slower, as the bounds are much more imprecise.
        if bandcross
            x3 = find_zero(l_min, (-7.5, 7.5), Bisection())
            x0 = find_zero(r_min, (-7.5, 7.5), Bisection())
        else
            x0 = find_zero(l_min, (-7.5, 7.5), Bisection())
            x3 = find_zero(r_min, (-7.5, 7.5), Bisection())
        end
    end
    #This numerically computes the integrals of the gaussian using the bounds found above.
    A = quadgk(gaussian, -Inf, x0)[1]
    B = quadgk(gaussian, x3, Inf)[1]
    #To save computation time, the integrals of l and r can be done analytically.

    #It is possible for this to return very small imaginary numbers if a value very close
    # to zero was raised to 1.5
    if abs(gaussian(x0) - l_int(x0)) < tol || !bandcross
        #A += quadgk(l_int, x0, x1)[1]
        A += 2a/(3b) * real((-b*x0-c + 0im)^1.5 - (-b*x1-c + 0im)^1.5)
    elseif abs(gaussian(x0) - r_int(x0)) < tol
        #A += quadgk(r_int, x0, x1)[1]
        A += 2d/(3f) * real((f*x1-g + 0im)^1.5 - (f*x0-g + 0im)^1.5)
    else
        #This shouldn't trigger. If it does, then the bands cross and the handling techniques failed.
        #The process of handling band crossing is still (to some extent) a work in progress, which is why this is neccessary.
        println("FATAL ERROR IN CALCULATION for parameters $params")
    end
    #Same as above but for the opposite side.
    if abs(gaussian(x3) - r_int(x3)) < tol || !bandcross
        #B += quadgk(r_int, x2, x3)[1]
        B += 2d/(3f) * real((f*x3-g + 0im)^1.5 - (f*x2-g + 0im)^1.5)
    elseif abs(gaussian(x3) - l_int(x3)) < tol
        #B += quadgk(l_int, x2, x3)[1]
        B += 2a/(3b) * real((-b*x2-c + 0im)^1.5 - (-b*x3-c + 0im)^1.5)
    else
        println("FATAL ERROR IN CALCULATION for parameters $params")
    end
    return A+B #The final sum.
end

#This function generates a single data "point", consisting of a set of parameters and the corresponding function F(β, params)
function generateDataPoint(xmin::Float64, dx::Float64, xmax::Float64)
    #A grid of β values.
    data1 = collect(xmin:dx:xmax)
    #Generates the parameters randomly.
    params = map(x -> abs(x) + 0.5, rand(Float64, 4))
    params = vcat(params, randn(Float64, 2))
    #Increases the chance that μ lies between the bands
    μ_t = 0.5randn()
    push!(params, (1-μ_t)*(-params[5]/params[2]) + μ_t*(params[6]/params[4]))
    #Calculate F(β, params) by calling forwardProb for each value of β.
    data1 = map(x -> forwardProb(x, params), data1)
    return (data1, params) #Input/Output pair
end
if length(ARGS) != 2
    println("Invalid arguments. Usage: julia data_gen.jl traindata.out testdata.out")
    exit(1)
end
#trainfile = open(ARGS[1], "w")
#testfile = open(ARGS[2], "w")
println("Beginning setup...")
xmin = 0.05
dx = 0.01
xmax = 5.0
#Initializes the arrays. This is because I'm lazy and am expanding the arrays below.
g = generateDataPoint(xmin, dx, xmax)
g2 = generateDataPoint(xmin, dx, xmax)
testout = generateDataPoint(xmin, dx, xmax)
xs = [g[1]]
ys = [g[2]]
xg = [g2[1]]
yg = [g2[2]]
println("Setup complete, generating training and test data...")
time = time_ns()
#Prints a progress bar that will be updated during the calculation.
print("Progress: |" * "-"^50 * ">")
percstore = 0.0
for i in 1:size #This is the number of points in the training set
    global percstore
    g = generateDataPoint(xmin, dx, xmax)
    #Pushes the new data to the arrays created above. This is actually reasonably efficient, unlike np.append().
    push!(xs, g[1])
    push!(ys, g[2])
    #Math for the progress bar.
    perc = Int(round(i/((1 + test_frac)*size)*50))
    if perc != percstore
        bar = "#"^perc * "-"^(50 - perc)
        print("\r" * "Progress: |" * bar * ">")
    end
    percstore = perc
end
#Does the same thing but for the test/validation set.
for i in 1:test_frac*size #This is the number of points in the test set
    global percstore
    g = generateDataPoint(xmin, dx, xmax)
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

#Bundles the data together.
traindata = collect(zip(xs, ys))
testdata = collect(zip(xg, yg))
#println(typeof(traindata))
#println(typeof(traindata))
#write(trainfile, traindata)
#write(testfile, testdata)

#Saves the data to files specified by the command line arguments.
@save ARGS[1] traindata
@save ARGS[2] testdata
