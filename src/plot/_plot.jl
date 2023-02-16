titlesize   = 22
xlabelsize  = 20
ylabelsize  = 20
zlabelsize  = 18
annsize     = 14
const mu0   = 1.602176e-19 # 1 ev in Joules (J/eV) 

jlblue = Colors.JULIA_LOGO_COLORS.blue
jlred = Colors.JULIA_LOGO_COLORS.red
jlgreen = Colors.JULIA_LOGO_COLORS.green


# plot χsq evolution over iterations
function plot_χ(n_iter::Int64,χ_vec::Array{Float64},χ_acpt::Vector{Vector{Float64}})
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (650*2, 500));
    ax1 = Axis(fig[1,1], xlabel="iter", ylabel="χsq", xlabelsize=xlabelsize, ylabelsize=ylabelsize);
    Label(fig[1,1,Top()], "χ random evolution", padding = (0, 0, 25, 10), textsize=titlesize);
    scatter!(ax1, collect(1:n_iter), χ_vec, markersize=2.5);
    ax2 = Axis(fig[1,2], xlabel="steps", ylabel="χsq", xlabelsize=xlabelsize, ylabelsize=ylabelsize);
    Label(fig[1,2,Top()], "χ-accepted evolution", padding = (0, 0, 25, 10), textsize=titlesize);
    scatter!(ax2, [el[1] for el in χ_acpt], [el[2] for el in χ_acpt], markersize=5);
    return fig
end


# plot only some predicted model, less at the beginning of the iterations, and more at the end.
function get_predindices(n_models::Int64)
    indices = collect(1:n_models)
    new_indices = Int64[]
    for i in 1:n_models
        if rand(Bernoulli(indices[i]/n_models)) # linear increasing probability
            push!(new_indices,indices[i])
        end
    end
    return new_indices
end


function get_transport(num_bands::Int64,bandtype::Vector{Int64},τ_form::ScModel,T::Union{Vector{Float64},Vector{Int64}} ,ftransport::Vector{Function},model_acpt::Vector{Float64})

    # random params -> new model candidate
    pred_bands = Array{ParabBand}(undef,num_bands)
    for b in 1:num_bands
        # effective masses
        m_idx = (b-1)*4 # index to select the correct parameters, 4 because each band has 3 m* + e0
        m = vcat(model_acpt[m_idx+1:m_idx+3],[0.0,0.0,0.0]);
        # position
        e0 = model_acpt[m_idx+4]
        # same type of original band structure (cond or val)
        type = bandtype[b]
        # create the band 
        pred_bands[b] = ParabBand(m,e0,type,1);
    end
    model_pred = BandStructure(num_bands,pred_bands,model_acpt[end]);   # build the band structure

    pred_tensors = Vector{Vector{Float64}}(undef,length(ftransport))
    for (i,f) in enumerate(ftransport)
        pred_tensors[i] = vec(f(model_pred,T,τ_form))
    end
    return pred_tensors
end


function get_plotlabels(tensor_names::Vector{String})
    titles = LaTeXString[]; ylabels = LaTeXString[]
    for name in tensor_names
        if name == "el_cond"
            push!(titles,   L"$σ$ vs $T$, $τ = const$")
            push!(ylabels,  L"$\sigma$ $[(\Omega m)^{-1}]$")
        elseif name == "seebeck"
            push!(titles,   L"$S$ vs $T$, $τ = const$")
            push!(ylabels,  L"$S$ $[\mu VK^{-1}]$")
        elseif name == "cconc"
            push!(titles,   L"$n$ vs $T$, $τ = const$");
            push!(ylabels,  L"n")
        elseif name == "th_cond"    
            push!(titles,   L"$K$ vs $T$, $τ = const$")
            push!(ylabels,  L"$K$ $[W$ $(mK)^{-1}]$")
        end
    end
    xlabels = [L"$T$ $[K]$", L"$T$ $[K]$", L"$T$ $[K]$"];
    return titles,xlabels,ylabels
end


# plot best selected model
function plot_best(num_bands::Int64,bandtype::Vector{Int64},τ_form::ScModel,T::Union{Vector{Float64},Vector{Int64}},ftransport::Vector{Function},tensor_names::Vector{String},true_tensors::Vector{Vector{Float64}},model_acpt::Vector{Float64})

    num_tensors = length(tensor_names)
    titles,xlabels,ylabels = get_plotlabels(tensor_names)
    pred_tensors = get_transport(num_bands,bandtype,τ_form,T,ftransport,model_acpt);

    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (650*num_tensors, 500));
    # plot true vs best_pred
    for i in 1:num_tensors
        ax = Axis(fig[1,i], xlabel=xlabels[i], ylabel=ylabels[i], xlabelsize=xlabelsize, ylabelsize=ylabelsize)
        Label(fig[1,i,Top()], titles[i], padding = (0, 0, 25, 10), textsize=titlesize)
        if tensor_names[i] == "seebeck"
            lines!(ax, T, vec(true_tensors[i])*1e6, label="True")
            lines!(ax, T, vec(pred_tensors[i])*1e6, label="Pred")
        else
            lines!(ax, T, vec(true_tensors[i]), label="True")
            lines!(ax, T, vec(pred_tensors[i]), label="Pred")
        end
        axislegend(ax)
    end
    return fig
end


# plot evolution of accepted models
function plot_evoacept(num_bands::Int64,bandtype::Vector{Int64},τ_form::ScModel,T::Union{Vector{Float64},Vector{Int64}} ,ftransport::Vector{Function},tensor_names::Vector{String},true_tensors::Vector{Vector{Float64}},model_acpt::Vector{Vector{Float64}},χ_acpt::Vector{Vector{Float64}})

    num_tensors = length(tensor_names)
    titles,xlabels,ylabels = get_plotlabels(tensor_names)
    
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (650*num_tensors, 500))
    palette = cgrad(:viridis, length(model_acpt), categorical=true);     

    # create the axes
    axes = Vector{Axis}(undef,num_tensors)
    for i in 1:num_tensors
        ax = Axis(fig[1,i], xlabel=xlabels[i], ylabel=ylabels[i], xlabelsize=xlabelsize, ylabelsize=ylabelsize)
        Label(fig[1,i,Top()], titles[i], padding = (0, 0, 25, 10), textsize=titlesize)
        axes[i] = ax
    end
    # pred
    pred_indices = get_predindices(length(model_acpt))
    for idx in pred_indices
        pred_tensors = get_transport(num_bands,bandtype,τ_form,T,ftransport,model_acpt[idx])
        for i in 1:num_tensors
            if tensor_names[i] == "seebeck"
                lines!(axes[i], T, vec(pred_tensors[i])*1e6, color=palette[idx])
            else
                lines!(axes[i], T, vec(pred_tensors[i]), color=palette[idx])
            end
        end
    end

    # true
    for i in 1:num_tensors
        if tensor_names[i] == "seebeck"
            scatter!(axes[i], T, vec(true_tensors[i])*1e6, label="True", color=:red, markersize=3,markerstyle='+')
        else
            scatter!(axes[i], T, vec(true_tensors[i]), label="True", color=:red, markersize=3,markerstyle='+')
        end
        # lines!(ax, T, vec(v_true), label="True", color=:red, linewidth=1.5,linestyle=:dash)
        axislegend(axes[i])
    end

    Colorbar(fig[1,num_tensors+1], limits=(Int64(χ_acpt[1][1]),Int64(χ_acpt[end][1])),colormap=:viridis,label="iter",labelsize=zlabelsize);
    return fig
end


function plot_bandstructure!(axes::Array{Axis},bs::BandStructure,xaxis::AbstractArray,color::ColorTypes.RGBA{Float64},label::String)

    n_bands = bs.n
    bands = bs.bands
    μ = bs.μ

    # parabolic function to represent the bands
    f(type,m,en) = (type)*m*xaxis.^2 .+ en

    for (i,ax) in enumerate(axes)
        for b in 1:n_bands
            m = bands[b].mstar  # eff masses
            e0 = bands[b].ϵ₀    # energy min/max
            type = bands[b].type    # bandtype
            if b == 1
                lines!(ax, xaxis, f(type,m[i],e0), color=color, label=label)	# band
            else
                lines!(ax, xaxis, f(type,m[i],e0), color=color)	# band
            end
        end
        hlines!(ax, μ, color=jlgreen, linestyle = :dash)	# fermi level
        hidedecorations!(ax, grid=false);
        axislegend(ax)
    end

    return axes
end


function plot_bandcomp(bs_true::BandStructure,bs_pred::BandStructure,xaxis::AbstractArray=range(-1, 1, length=100))
    titles = [L"$m_x$", L"$m_y$", L"$m_z$"]
    fig = Figure(resolution = (500*3, 500))
    axes = [Axis(fig[1,i], title=titles[i], titlesize=titlesize) for i in 1:3]

    plot_bandstructure!(axes,bs_true,xaxis,jlblue,"true")
    plot_bandstructure!(axes,bs_pred,xaxis,jlred,"pred")
    return fig
end


function savefig(fullpath::String, fig::Figure)
    save(fullpath, fig)
end
