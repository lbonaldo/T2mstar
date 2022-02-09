using CSV
using DataFrames
using CairoMakie
import ColorSchemes
using LaTeXStrings

using QuadGK
using Roots

## PARAMETERS
test_name = "test2"
##

## FUNCTION DEFINITION
function l(x,a::Float64,b::Float64)
    if -a*x - b > 0
        return sqrt(-a*x - b)
    else
        return 0
    end
end

function r(x,c::Float64,d::Float64)
    if c*x - d > 0
        return sqrt(c*x - d)
    else
        return 0
    end
end

function forwardProb(β)
    (a, c, b, d, μ) = (0.42408792957505614, 0.23518179321838595, -0.5171051560919273, 0.3980596866750412, 1.5614801151887576)
    bandcross = false
    if (d/c) < (-b/a)
        bandcross = true
        println("Bands cross!")
    end
    α = sqrt(β/π)
    xarr = -5:0.05:5
    lines!(axtopright, xarr, r.(xarr,c,d), label=L"R(x;c,d)", color=ColorSchemes.tab10[3], lw=2, xlabel="t", ylabel="L, R, G")
    lines!(axtopright, xarr, l.(xarr,a,b), label=L"L(x;a,b)", color=ColorSchemes.tab10[1], lw=2)
    gaussian(x) = α*exp(-β*(x - μ)^2)
    l_min(x) = l(x,a,b) - gaussian(x)
    r_min(x) = r(x,c,d) - gaussian(x)
    if bandcross == true && gaussian((d-b)/(a+c)) < sqrt(-(a*d + b*c)/(a + c))
        return 1.0
    end
    x0 = find_zero(l_min, (-(β + pi*b)/(pi*a), -b/a), Bisection())
    x3 = find_zero(r_min, (d/c, (β + pi*d)/(pi*c)), Bisection())
    A = quadgk(gaussian, -Inf, x0)[1]
    B = quadgk(gaussian, x3, Inf)[1]
    if !bandcross
        A += (2/(3a)) * (-a*x0 - b)^1.5
        B += (2/(3c)) * (c*x3 - d)^1.5
    else
        A += (2/(3a)) * ((-a*x0 - b)^1.5 - (-(a*d + b*c)/(a+c))^1.5)
        B += (2/(3c)) * ((-(a*d + b*c)/(a+c))^1.5 - (c*x3-d)^1.5)
    end
    lines!(axtopright, xarr, gaussian.(xarr), label=L"G(x;β,μ)", color=ColorSchemes.tab10[5], lw=2)
    if !bandcross
        scatter!(axtopright, [x0, -b/a, d/c, x3], [l(x0,a,b), l(-b/a,a,b), r(d/c,c,d), r(x3,c,d)], label="Intersection Points", color=ColorSchemes.tab10[6], lw=3)
    else
        scatter!(axtopright, [x0, (d-b)/(a+c), x3], [l(x0,a,b), sqrt(-(a*d + b*c)/(a + c)), r(x3,c,d)], label="Intersection Points", color=ColorSchemes.tab10[6], lw=3)
    end
    axislegend()
    return B-A
end
##

## CREATING DATA
coeff_type = Vector{String}(["_true", "_pred", "_diff"])
coeff_name = Vector{String}(["b","d","a","c","mu"])
header = Vector{String}(undef, length(coeff_name)*length(coeff_type))
for i in eachindex(coeff_type)
    header[(i-1)*5+1:i*5] = coeff_name .* coeff_type[i]
end
res = CSV.read(joinpath(test_name, "y_results.txt"), DataFrame, header=header)

## PLOTS
noto_sans = assetpath("fonts", "NotoSans-Regular.ttf")
noto_sans_bold = assetpath("fonts", "NotoSans-Bold.ttf")

fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98),
resolution = (1000, 1000), font = noto_sans)

axtop = Axis(fig[1,1])
axmain = Axis(fig[2,1], xlabel = "prediction", ylabel = "true")
axright = Axis(fig[2,2])
axtopright = Axis(fig[1,2], xlabel = "x")

linkyaxes!(axmain, axright)
linkxaxes!(axmain, axtop)

for (i,coeff) in enumerate(coeff_name)
    color = ColorSchemes.tab10[i]
    println(color)
    scatter!(axmain, res[:, coeff*"_pred"], res[:, coeff*"_true"], markersize = 5, color = (color, 0.2), label = coeff)
    density!(axtop, res[:, coeff*"_pred"], color = (color, 0.2), strokecolor = color, strokewidth = 1, strokearound = true)
    density!(axright, res[:, coeff*"_true"], direction = :y, color = (color, 0.2), strokecolor = color, strokewidth = 1, strokearound = true)
end

X = 2.0
Y = forwardProb(X)

#axislegend(axmain)
hidexdecorations!(axtop, grid = false)
hideydecorations!(axright, grid = false)
colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)
Legend(fig[1:2, 3], axmain, markersize = 5)
Label(fig[0,:], text = "NN results: reconstruction of coefficients' distributions", fontsize = 30)
save(joinpath(test_name, "leg.png"), fig)


# x_start = 1000
# x_stop = 2000
# y_true = res[x_start:x_stop,1:5]
# y_pred = res[x_start:x_stop,6:10]

# ### DEVIATIONS
# for i in coeff_name   
#     i_true = y_true[:, i*"_true"]
#     i_pred = y_pred[:, i*"_pred"]
#     fig2, ax1, plot1 = scatter(1000..1500, i_true, figure = (backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (1100, 800), font = noto_sans), colormap = :viridis )
#     plot2 = scatter!(ax1, 1000..1500, i_pred)
#     lines(fig2[2, 1], 1000..1500, i_true-i_pred, color = :green )
#     Legend(fig2[1:2, 2], [plot1, plot2], ["true", "pred"])
#     Label(fig2[0,:], text = "Scatter plot for "*i)
#     save(joinpath(test_name, i*"_plot.png"), fig2)
# end