# (inverse) inverse> $env:JULIA_NUM_THREADS=4
# (inverse) inverse> julia --project=. .\julia_to_py_dataset.jl

using CSV
using JLD2: @load
using QuadGK
using Roots
using Polynomials
using DataFrames
using GLMakie
using GeometryBasics
using ColorSchemes
#using CairoMakie
import Plots: theme_palette

######### PARAMS ######### 
tmin = 0.05
dt = 0.01
tmax = 5.0
polydegree = 9
test_name = "test2"
coeff_type = Vector{String}(["_true", "_pred", "_diff"])
coeff_name = Vector{String}(["a","b","c","d","mu"])
header = Vector{String}(undef, length(coeff_name)*length(coeff_type))
for i in eachindex(coeff_type)
    header[(i-1)*5+1:i*5] = coeff_name .* coeff_type[i]
end
##########################

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
    # to implement
end

# This function computes the spline from the parameters of the forwardProb
function coeff2spline(ts::Vector{Float64}, polydegree::Int64, coeffs::Vector{Float64})
    (a, b, c, d, μ) = coeffs
    I = map(t -> forwardProb(t, a, b, c, d, μ), ts)
    pol = fit(ts, I, polydegree)
    return pol.coeffs
end

function test3dBars(x_len::Int64, y_len::Int64, z::Matrix{Float64})
    x = 1:x_len
    y = 1:y_len
    δx = (x[2] - x[1]) / 2
    δy = (y[2] - y[1]) / 2
    cbarPal = :Spectral_11
    texture = reshape(get(colorschemes[cbarPal], 0:0.01:1), 1, 101)
    fig = Figure(resolution=(1200, 800), fontsize=26)
    ax = Axis3(fig[1, 1]; aspect=(1, 1, 1), elevation=π / 6, perspectiveness=0.5)
    for (idx, i) in enumerate(x), (idy, j) in enumerate(y)
        rectMesh = FRect3D(Vec3f0(i - δx, j - δy, 0), Vec3f0(2δx, 2δy, z[idx, idy]))
        recmesh = GeometryBasics.normal_mesh(rectMesh)
        uvs = [Point2f(p[3], 0) for p in coordinates(recmesh)] # normalize this so zmax = 1
        recmesh = GeometryBasics.Mesh(
            meta(coordinates(recmesh); normals=normals(recmesh), uv = uvs), 
            faces(recmesh)
        )
        mesh!(ax, recmesh; color=texture, shading=false)
    end
    save(joinpath(plots_path, "spline_plot.png"), fig)
end

println("Beginning setup...")
ts = collect(tmin:dt:tmax)
@load "../dataset/run1/julia/x_test.out" x_test # Nrows x 10 tensor
# y_pred = CSV.read(joinpath("../tests", test_name, "y_results.txt"), DataFrame, header=header)[:,6:10]
@load "../dataset/run1/julia/y_test.out" y_test # Nrows x 10 tensor
println("Setup complete, generating training and test data...")

# get slines from parameters
arr = Array{Float64,2}(undef, size(x_test,1), length(ts))
for i in 1:size(x_test,1)
    p = Polynomial(x_test[i,:], :x)
    arr[i,:] = p.(ts)
end

test3dBars(size(arr,1), size(arr,2), arr)

exit(-1)

### PLOTS
noto_sans = assetpath("fonts", "NotoSans-Regular.ttf")
noto_sans_bold = assetpath("fonts", "NotoSans-Bold.ttf")
fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98),
    resolution = (1000, 1000), font = noto_sans)

#Axis(f[1,1], title = "Coeff a")
topleft = Axis(fig[1,1])
topcenter = Axis(fig[1,2])
topright = (fig[1,3])
centerright = Axis(fig[2,3])
bottomright = Axis(fig[end, end])
bottomleft = Axis(fig[2:4, 1:2])

# scatter!(topleft, res[:, res[:, coeff*"_pred"], coeff*"_true"], color = (c, 0.2), label = coeff)
# scatter!(topcenter, res[:, res[:, coeff*"_pred"], coeff*"_true"], color = (c, 0.2), label = coeff)
# scatter!(topright, res[:, res[:, coeff*"_pred"], coeff*"_true"], color = (c, 0.2), label = coeff)
# scatter!(centerright, res[:, res[:, coeff*"_pred"], coeff*"_true"], color = (c, 0.2), label = coeff)
# scatter!(bottomright, res[:, res[:, coeff*"_pred"], coeff*"_true"], color = (c, 0.2), label = coeff)
heatmap(bottomleft, arr)
Label(fig[0,:], text = "Spline distribution")
save(joinpath(plots_path, "spline_plot.png"), fig)