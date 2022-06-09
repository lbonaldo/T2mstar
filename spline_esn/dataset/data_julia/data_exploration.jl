using CairoMakie
using DataFrames
using JLD2: @load, @save

cd("spline_esn/dataset/data_julia")

# ANALYSIS COEFFS
function mergeandfilter(spline_numcoeff::Int64,sigma::Matrix{Float64},seebeck::Matrix{Float64},n::Matrix{Float64},model_numcoeff::Int64, model::Matrix{Float64})
    σ_list_coeff = ["σcoeff_$(i)" for i in 1:spline_numcoeff]
    S_list_coeff = ["Scoeff_$(i)" for i in 1:spline_numcoeff]
    n_list_coeff = ["ncoeff_$(i)" for i in 1:spline_numcoeff]
    mdlist_coeff = ["mdcoeff_$(i)" for i in 1:model_numcoeff]
    df = hcat(DataFrame(model,mdlist_coeff),DataFrame(sigma,σ_list_coeff),DataFrame(seebeck,S_list_coeff),DataFrame(n,n_list_coeff))
end

function filterplot(data::Matrix{Float64}, xliml::Vector{Float64}, xlimr::Vector{Float64}, filename::String)
    list_coeff = ["coeff1","coeff2","coeff3","coeff4","coeff5","coeff6"]
    columns = [:coeff1,:coeff2,:coeff3,:coeff4,:coeff5,:coeff6]
    df = DataFrame(data,list_coeff)
    for i in eachindex(list_coeff)
        filter!(columns[i] => x -> x .> xliml[i] && x .< xlimr[i], df)
    end
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (1500, 1000))
    for i in 1:2
        for j in 1:3
            gr = fig[i,j] = GridLayout()
            coeff = data[:,(i-1)*3+j]
            ax1 = Axis(gr[1,1])
            density!(ax1, coeff, npoints = 20000, color = (:blue, 0.3), strokecolor = :blue, strokewidth = 1, strokearound = true)    
            coeff = coeff[coeff .> xliml[(i-1)*3+j]]
            coeff = coeff[coeff .< xlimr[(i-1)*3+j]]
            ax2 = Axis(gr[2,1])
            xlims!(ax2, [xliml[(i-1)*3+j],xlimr[(i-1)*3+j]])
            density!(ax2, coeff, npoints = 20000, color = (:blue, 0.3), strokecolor = :blue, strokewidth = 1, strokearound = true)
            density!(ax2, df[!,(i-1)*3+j], npoints = 20000, color = (:red, 0.3), strokecolor = :red, strokewidth = 1, strokearound = true)
        end
    end
    save(joinpath("filter_dataset", filename*".png"),fig)
    @save joinpath("filter_dataset", filename*".out") df
end

# N
@load "./x_n_train.out" x_n_train
@load "./x_n_eval.out" x_n_eval
@load "./x_n_test.out" x_n_test

size(x_n_train)
size(x_n_eval)
size(x_n_test)

xliml = [-1e10,-5e7,-5e5,-150,-1,-5e-4]
xlimr = [1e10,5e7,5e5,150,2,5e-4]
filterplot(x_n_train, xliml, xlimr, "y_n_train_filt")
filterplot(x_n_eval, xliml, xlimr, "y_n_eval_filt")
filterplot(x_n_test, xliml, xlimr, "y_n_test_filt")

# sigma
@load "./x_sigma_train.out" x_sigma_train
@load "./x_sigma_eval.out" x_sigma_eval
@load "./x_sigma_test.out" x_sigma_test

size(x_sigma_train)
size(x_sigma_eval)
size(x_sigma_test)

xliml = [-5e2,-1e1,-0.1,-1e-4,-1e-7,-1e-11]
xlimr = [5e2,1e1,0.1,1e-4,1e-7,1e-11]
filterplot(x_sigma_train, xliml, xlimr, "y_sigma_train_filt")
filterplot(x_sigma_eval, xliml, xlimr, "y_sigma_eval_filt")
filterplot(x_sigma_test, xliml, xlimr, "y_sigma_test_filt")

# seebeck
@load "./x_seebeck_train.out" x_seebeck_train
@load "./x_seebeck_eval.out" x_seebeck_eval
@load "./x_seebeck_test.out" x_seebeck_test

size(x_seebeck_train)
size(x_seebeck_eval)
size(x_seebeck_test)

xliml = [-0.002,-3e-5,-6e-8,-6e-11,-4e-14,0]
xlimr = [0.002,3e-5,6e-8,6e-11,4e-14,1]
filterplot(x_seebeck_train, xliml, xlimr,"y_seebeck_train_filt")
filterplot(x_seebeck_eval, xliml, xlimr,"y_seebeck_eval_filt")
filterplot(x_seebeck_test, xliml, xlimr,"y_seebeck_test_filt")


# model parameters
@load "./y_train.out" y_train
size(y_train)

spline_numcoeff = 6
model_numcoeff = 9

σ_list_coeff = ["σcoeff_$(i)" for i in 1:spline_numcoeff]
S_list_coeff = ["Scoeff_$(i)" for i in 1:spline_numcoeff]
n_list_coeff = ["ncoeff_$(i)" for i in 1:spline_numcoeff]
mdlist_coeff = ["mdcoeff_$(i)" for i in 1:model_numcoeff]
df = hcat(DataFrame(y_train,mdlist_coeff),DataFrame(x_sigma_train,σ_list_coeff),DataFrame(x_seebeck_train,S_list_coeff),DataFrame(x_n_train,n_list_coeff))

y_train[1,:]
