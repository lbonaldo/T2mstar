using Plots
using QuadGK
using Roots
using LaTeXStrings
#pyplot()

# Generates the coeffcients for the sqrt functions. 
# a and c are strictly positive, since they determine the direction of the function.
# b and d can be either positive of negative, since bands are allowed to cross.
# 
a, c = rand(Float64, 2)
b, d = rand(Float64, 2) .* sign.(randn(Float64, 2))

# Since the intersection point for both bands is -b/a and d/c, respectively, 
# if a or c are sufficiently small our band positions can be pretty crazy. 
# This is just a sanity check to makes sure that they aren't, so 
# that μ always lies between -1 and 1. This makes it a bit more ML friendly.

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


# Increases the chance that μ lies between the bands
# The bands hit zero at -b/a and d/c, respectively.
# μ is selected at a random spot between these two points.
μ_t = rand(Float64)
μ = (1-μ_t)*(-b/a) + μ_t*(d/c)
@show a,c,b,d,μ
bandcross = false
if (d/c) < (-b/a)
    bandcross = true
    println("Bands cross!")
end

function l(x)
    if -a*x - b > 0
        return sqrt(-a*x - b)
    else
        return 0
    end
end

function r(x)
    if c*x - d > 0
        return sqrt(c*x - d)
    else
        return 0
    end
end

function forwardProb(β)
    α = sqrt(β/π) #The gaussian normalization coefficient
    xarr = -5:0.05:5
    #Plots the l and r functions
    fig = plot(xarr, r.(xarr), label=L"R(d, f, g, t)", lw=2, xlabel="t", ylabel="L, R, G")
    plot!(fig, xarr, l.(xarr), label=L"L(a, b, c, t)", lw=2)
    gaussian(x) = α*exp(-β*(x - μ)^2)
    #Defines the functions used by Roots.jl to find the intersection points
    l_min(x) = l(x) - gaussian(x)
    r_min(x) = r(x) - gaussian(x)
    #If the bands cross and β is sufficiently small, we only end up
    # integrating over the gaussian, which is normalized. So we get 1.
    # g((d-b)/(a+c)) is the height of the gaussian at the intersection point,
    # and sqrt(-(a*d + b*c)/(a + c)) is the height at which the crossing 
    # bands touch.
    if bandcross == true && gaussian((d-b)/(a+c)) < sqrt(-(a*d + b*c)/(a + c))
        return 1.0
    end
    #We need to find the intersection between the gaussian and the bands.
    # Unfortunately, there is no analytical solution to this.
    # I use a root finding routing (Roots.jl) to find these two points.
    # However, we can be intellegent about where to find the roots.
    # It is possible to show mathematically using the max height of the gaussian that:
    # x0 ∈ [-(β + πb)/πa, -b/a] and x3 ∈ [d/c, (β + πd)/πc]
    # So we will look for the roots in these intervals.
    # These values are independent of whether the bands cross.
    @show a,c,b,d,μ,β
    println(l_min)
    println((-(β + pi*b)/(pi*a), -b/a))
    x0 = find_zero(l_min, (-(β + pi*b)/(pi*a), -b/a), Bisection())
    x3 = find_zero(r_min, (d/c, (β + pi*d)/(pi*c)), Bisection())
    #Perform the gaussian integrals. These are independent of whether the bands cross.
    A = quadgk(gaussian, -Inf, x0)[1]
    B = quadgk(gaussian, x3, Inf)[1]
    #Now for the other integrals, which do depend on whether the bands cross.
    # I solved these integrals analytically, so only the result is here.
    if !bandcross
        A += (2/(3a)) * (-a*x0 - b)^1.5
        B += (2/(3c)) * (c*x3 - d)^1.5
    else #I apoligize in advance, this algebra is ugly
        A += (2/(3a)) * ((-a*x0 - b)^1.5 - (-(a*d + b*c)/(a+c))^1.5)
        B += (2/(3c)) * ((-(a*d + b*c)/(a+c))^1.5 - (c*x3-d)^1.5)
    end
    #Adds the gaussian and intersection points to the plot
    plot!(fig, xarr, gaussian.(xarr), label=L"G(μ, t, β)", lw=2)
    if !bandcross
        plot!(fig, [x0, -b/a, d/c, x3], [l(x0), l(-b/a), r(d/c), r(x3)], seriestype=:scatter, label="Intersection Points", lw=3)
    else
        plot!(fig, [x0, (d-b)/(a+c), x3], [l(x0), sqrt(-(a*d + b*c)/(a + c)), r(x3)], seriestype=:scatter, label="Intersection Points", lw=3)
    end
    display(fig)
    return B-A
end
#println(forwardProb(1.0))
X = collect(0.05:0.01:5.0)
Y = forwardProb.(X)
#plt = plot(X, Y, label=L"$F($  $\underbar \!\!\! x, β)$", lw=2, xlabel=L"\beta", ylabel="F")
#savefig(plt, "example_plot.png")
#display(plt)
