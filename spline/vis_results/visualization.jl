using CSV
using DataFrames
using CairoMakie
import Plots: theme_palette

test_name = "test3"

## DATA READING
coeff_type = Vector{String}(["_true", "_pred", "_diff"])
coeff_name = Vector{String}(["a","b","c","d","mu"])
header = Vector{String}(undef, length(coeff_name)*length(coeff_type))
for i in eachindex(coeff_type)
    header[(i-1)*5+1:i*5] = coeff_name .* coeff_type[i]
end

res = CSV.read(joinpath(test_name, "y_results.txt"), DataFrame, header=header)

noto_sans = assetpath("fonts", "NotoSans-Regular.ttf")
noto_sans_bold = assetpath("fonts", "NotoSans-Bold.ttf")

### DISTRIBUTIONS
fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98),
    resolution = (1000, 1000), font = noto_sans)

axtop = Axis(fig[1,1])
axmain = Axis(fig[2,1], xlabel = "y_pred", ylabel = "y_true")
axright = Axis(fig[2,2])

linkyaxes!(axmain, axright)
linkxaxes!(axmain, axtop)

for (i,coeff) in enumerate(coeff_name)
    c = theme_palette(:tol_light).colors.colors[i]
    density!(axtop, res[:, coeff*"_pred"], color = (c, 0.2))
    scatter!(axmain, res[:, res[:, coeff*"_pred"], coeff*"_true"], color = (c, 0.2), label = coeff)
    density!(axright, res[:, coeff*"_true"], direction = :y, color = (c, 0.2))
end
axislegend(axmain)
hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)
colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)
Label(fig[0,:], text = "Coefficient distributions")
save(joinpath(plots_path, "dists_all.png"), fig)

x_start = 1000
x_stop = 2000
y_true = res[x_start:x_stop,1:5]
y_pred = res[x_start:x_stop,6:10]

### DEVIATIONS
for i in coeff_name   
    i_true = y_true[:, i*"_true"]
    i_pred = y_pred[:, i*"_pred"]
    fig2, ax1, plot1 = scatter(1000..1500, i_true, figure = (backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (1100, 800), font = noto_sans), colormap = :viridis )
    plot2 = scatter!(ax1, 1000..1500, i_pred)
    lines(fig2[2, 1], 1000..1500, i_true-i_pred, color = :green )
    Legend(fig2[1:2, 2], [plot1, plot2], ["true", "pred"])
    Label(fig2[0,:], text = "Scatter plot for "*i)
    save(joinpath(plots_path, i*"_plot.png"), fig2)
end