@with_kw mutable struct Args
    n_iter::Int64   = 2000  # number of interations
    n_restart       = 1     # number of restart of the algorithm
    rnd_seed::Union{Int64,Array{Int64}} = 1234  # seed for random number generator
    err::Float64    = 0.01  # ground truth error
    β_init::Union{Float64,Array{Float64}}    = 1e-1  # temperature for simulating annealing
    β_step::Int64   = 1 # num iterations between steps of simulating annealing
end

@with_kw struct GroundTruth
    τ::ScModel                                  # relaxation time
    T::Union{Vector{Float64},Vector{Int64}}     # temperature
    σ::Vector{Float64} = Float64[]              # electrical conductivity
    S::Vector{Float64} = Float64[]              # seebeck coefficient
    n::Vector{Float64} = Float64[]              # carrier concentration
    K::Vector{Float64} = Float64[]              # thermal conductivity
end
# outer ctor
function GroundTruth(τ::ScModel,T::Union{Vector{Float64},Vector{Int64}},σ::Union{Vector{Float64},Matrix{Float64}},S::Union{Vector{Float64},Matrix{Float64}},n::Union{Vector{Float64},Matrix{Float64}})
    σ = !isempty(σ) ? vec(σ) : Float64[]
    S = !isempty(S) ? vec(S) : Float64[]
    n = !isempty(n) ? vec(n) : Float64[]
    return GroundTruth(τ,T,σ,S,n,Float64[])
end

struct Model
    bs::BandStructure
    τ::ScModel
end


function get_randcand(n_bands::Int64,rng::MersenneTwister,mrange::AbstractVector,erange::AbstractVector,μrange::AbstractVector,τrange::AbstractVector,bandtype::Array{Int64})
    # new model candidate from random params 
    bands_cand = Array{ParabBand}(undef,n_bands)
    for b in 1:n_bands
        # effective masses
        m_start = (b-1)*3 # index to select the correct mass range (see mranges above), 3 because each band has 3 m*
        m = [Float64(rand(rng,mrange[m_start+1])),
            Float64(rand(rng,mrange[m_start+2])),
            Float64(rand(rng,mrange[m_start+3])),
            0.0,0.0,0.0];
        # position
        e0 = rand(rng,erange[b])
        # same type of original band structure (cond or val)
        type = bandtype[b]
        # create the band 
        bands_cand[b] = ParabBand(m,e0,type,1);
    end
    # chemical potential
    μ_cand = rand(rng,μrange);
    bs = BandStructure(n_bands,bands_cand,μ_cand);   # build the band structure
    # relaxation time
    τ_A_cand,τ_B_cand,τ_C_cand = rand(rng,τrange[1]),rand(rng,τrange[2]),rand(rng,τrange[3])
    τ(t) = τ_A_cand*(1/(t-τ_B_cand)^τ_C_cand)
    τ_form = Scattering.T_fun(τ)
    # return the model candidate
    return Model(bs,τ_form)
end


function select_params2change()
    # create a vector of parameters to change
    params = Vector(undef,2n_bands+1)   # 2b+1: masses,en for each band + μ
    for b in 1:n_bands
        m_start = (b-1)*3+1             # 3 masses
        params[b] = mrange[m_start:m_start+2]
    end
    for b in 1:n_bands
        params[b+n_bands] = erange[b]   # energy
    end
    

    # randomly select which parameter to change
    idx = rand(1:length(params))

end


function updatecand(model::Model,n_bands::Int64,rng::MersenneTwister,mrange::AbstractVector,erange::AbstractVector,μrange::AbstractVector)



    bands = Array{ParabBand}(undef,n_bands)
    # effective masses
    if (idx <= n_bands)    # change the masses
        for b in 1:n_bands
            local m
            tmp_band = bs.bands[b]
    
            if b == idx
                ranges = params[idx]
                m = [Float64(rand(rng,ranges[1])),
                     Float64(rand(rng,ranges[2])),
                     Float64(rand(rng,ranges[3])),
                     0.0,0.0,0.0];
            else
                m = [tmp_band.mstar[1],
                     tmp_band.mstar[2],
                     tmp_band.mstar[3],
                     0.0,0.0,0.0];
            end
            e0 = tmp_band.ϵ₀
            type = tmp_band.type
            bands[b] = ParabBand(m,e0,type,1);
        end
        return BandStructure(n_bands,bands,bs.μ)
    
    # energy (position)
    elseif (idx > n_bands) && (idx <= 2n_bands) # change band min/max
        for b in 1:n_bands
            local e0
            tmp_band = bs.bands[b]

            m = tmp_band.mstar;

            range = params[idx]
            band_idx = idx - n_bands
            if b == band_idx
                e0 = rand(rng,range)
            else
                e0 = tmp_band.ϵ₀
            end
            type = tmp_band.type
            bands[b] = ParabBand(m,e0,type,1);
        end
        return BandStructure(n_bands,bands,bs.μ)
 
    # fermi level position
    elseif idx == 2n_bands+1
        μ = rand(rng,μrange);
        return BandStructure(n_bands,bs.bands,μ)
    end
end

function update_singleparam(model::BandStructure,n_bands::Int64,rng::MersenneTwister,mrange::AbstractVector,erange::AbstractVector,μrange::AbstractVector)

    # randomly select which parameter to change: 
    idx = rand(1:2n_bands+1)    # 2b+1: masses,en for each band + μ 
    
    # array to store the new band structure
    bands = Array{ParabBand}(undef,n_bands)
    # design choice: first n_bands are masses, then energies, then μ
    if (idx <= n_bands)    # change the masses
        for b in 1:n_bands
            idx_mass = rand(1:3)    # randomly select the mass to modify
            band = model.bands[b]
            m = zeros(6)
            for m_i in 1:3    # three masses
                if (m_i == idx_mass) && (b == idx)
                    m[m_i] = Float64(rand(rng,mrange[idx]))   # new mass candidate
                else
                    m[m_i] = band.mstar[m_i]  # same mass
                end
            end
            e0 = band.ϵ₀
            type = band.type
            bands[b] = ParabBand(m,e0,type,1);
        end
        return BandStructure(n_bands,bands,model.μ)
    
    # energy (~position)
    elseif (idx > n_bands) && (idx <= 2n_bands) # change band min/max
        band_idx = idx - n_bands    # which band to modify
        for b in 1:n_bands
            local e₀
            band = model.bands[b]

            m = band.mstar

            if b == band_idx
                e0 = rand(rng,erange[b])   # new energy candidate
            else
                e0 = band.ϵ₀    # same energy
            end
            type = band.type
            bands[b] = ParabBand(m,e0,type,1);
        end
        return BandStructure(n_bands,bands,model.μ)

    # fermi level position
    elseif idx == 2n_bands+1
        μ = rand(rng,μrange);
        return BandStructure(n_bands,model.bands,μ)
    end
end


function shiftenergies!(model_best::SharedMatrix{Float64},energies::Vector{Float64},num_bands::Int64)

    min_idx = argmin(energies)
    model_best_shift = Matrix{Float64}(undef,size(model_best,1),size(model_best,2))
    for (i,col) in enumerate(eachcol(model_best))
        var = Float64[]
        for b in 1:num_bands
            push!(var,col[4(b-1)+4])
        end
        push!(var,col[end])
        var2 = Array{Float64}(undef,size(model_best,1))
        Δ = energies[min_idx] - var[min_idx]
        for b in 1:num_bands
            m = model_best[4(b-1)+1:4(b-1)+3,i]
            e0 = model_best[4(b-1)+4,i]
            var2[4(b-1)+1:4(b-1)+4] = cat(m,e0+Δ,dims=1)
        end
        var2[end] = model_best[end,i]+Δ
        model_best_shift[:,i] = var2
    end
    return model_best_shift 
end


# RMC algorithm
function RMC(model_true::BandStructure,grtrue::GroundTruth,args::Args,ranges::Dict{String,AbstractVector};anneal=false,plot::Bool=false)

    # set some general parameters
    n_bands = model_true.n  # number of bands
    n_params = 4n_bands+1   # 4 params for each band (3m*+ϵ₀) + chemical potential
    bandtype = [band.type for band in model_true.bands] # get the type of bands from true model
    T = grtrue.T        # temperature
    τ_form = grtrue.τ   # relaxation time
    
    # candidate models - get parameters constraints
    mrange = ranges["m"]
    erange = checkerange!(ranges["ϵ₀"])
    μrange = ranges["μ"]
    τrange = ranges["τ"]
    
    # number of total possible combinations
    tot_comb = prod([length(v) for v in mrange]) * prod([length(v) for v in erange]) * length(μrange) * prod([length(v) for v in τrange]);
    println("Total possible combinations: ", tot_comb)
    
    # create folder to export results
    outpath = create_runfolder(".")
    println("---> 1. Output folder created.")
    
    # set RMC parameters
    n_iter = args.n_iter
    n_restart = args.n_restart
    err = args.err
    β = args.β_init
    β_step = args.β_step
    idx2plot = unique(rand(1:n_restart,4))  # export only 4 chains
    
    # simulated annealing - exp decay
    exp_annealing(x::Real,n::Int64,α::Float64) = exp(60(x+α)/(n+α))
    exp_annealing(x) = exp_annealing(x,n_iter,-1.6)

    # get (non-zero) true/experimental transport coefficients
    not_empty_tensors = [!isempty(grtrue.σ),!isempty(grtrue.S),!isempty(grtrue.n),!isempty(grtrue.K)]
    n_tensors = sum(not_empty_tensors)
    true_tensors = [grtrue.σ,grtrue.S,grtrue.n,grtrue.K][not_empty_tensors]
    
    # select which function to call in Mstar2t
    ftransport = [electrical_conductivity,seebeck_coefficient,carrier_concentration,thermal_conductivity][not_empty_tensors]

    # Compute the variance for each not-empty transport tensor
    σsq = Array{Float64}(undef,length(true_tensors))
    for i in eachindex(true_tensors)
        # σsq[i] = sum((true_tensors[i].*err).^2)/length(true_tensors[i]);
        σsq[i] = sum(true_tensors[i].^2)
    end

    # vector to store results across all the runs
    χ_best = SharedArray{Float64}(n_restart)
    model_best = SharedArray{Float64}(n_params,n_restart)

    # run all the algorithm
    println("\n---> 2. RMC started.")
    t_start = time_ns()
    @sync Threads.@threads for run_idx in 1:n_restart

        # get local random seed
        rng = MersenneTwister()
        
        # vector to store results
        χ_vec = Vector{Float64}(undef,n_iter);
        χ_acpt = Vector{Float64}[];
        models_acpt = Vector{Float64}[];
        
        χsq_running = Inf;  # best χ square during the loop
        β = args.β_init     # reset β
        i_acpt = 1;         # index on accepted models

        model_running = get_randcand(n_bands,rng,mrange,erange,μrange,τrange,bandtype)
        for iter in 1:n_iter

            # annealing step - exp decay
            β = exp_annealing(iter)

            model_cand = updatecand(model_running,n_bands,rng,mrange,erange,μrange)

            # calculate transport coefficients for candidate model and compute χsq
            χsq_new = 0.0   # χ square between candidate and true
            cand_tensors = Vector{Vector{Float64}}(undef,n_tensors)
            for (i,f) in enumerate(ftransport)
                cand_tensor = vec(f(model_cand,T,τ_form))
                χsq_new += sqL2dist(true_tensors[i],cand_tensor)/σsq[i]
                cand_tensors[i] = cand_tensor
            end

            # accept/reject
            if χsq_new < χsq_running
                χsq_running = χsq_new   # accept
                push!(χ_acpt,[iter,χsq_new])   # add new χ accepted
                model_running = model_cand  # new best model
                model_cand_params = Array{Float64}(undef,4n_bands)    # 4 params each band
                for b in 1:n_bands
                    band = model_cand.bands[b]
                    idx = (b-1)*4     # index to select the correct position in the model_cand_params array
                    model_cand_params[idx+1:idx+4] = cat(band.mstar[1:3],band.ϵ₀,dims=1)
                end
                push!(models_acpt,cat(model_cand_params,model_cand.μ, dims=1))    # add new accepted model
            else
                if χsq_running == 0.0
                    break
                end
                Δχsq = χsq_new - χsq_running
                # P = min(1,exp(-β*(Δχsq)))
                # if (rand(Bernoulli(P))) && (Δχsq > 0.0)
                if (Δχsq != 0) && rand() <= exp(-β*(Δχsq))   # Metropolis acceptance probability
                    # println(iter,"\taccepted even if not smaller", "\t", β)
                    # println(Δχsq)
                    # println(exp(-β*(Δχsq)))
                    χsq_running = χsq_new   # accept
                    push!(χ_acpt,[iter,χsq_new])   # add new χ accepted
                    model_running = model_cand  # new best model
                    model_cand_params = Array{Float64}(undef,4n_bands)    # 4 params each band 
                    for b in 1:n_bands
                        band = model_cand.bands[b]
                        idx = (b-1)*4     # index to select the correct position in the model_cand_params array
                        model_cand_params[idx+1:idx+4] = cat(band.mstar[1:3],band.ϵ₀,dims=1)
                    end
                    push!(models_acpt,cat(model_cand_params,model_cand.μ, dims=1))    # add new accepted model
                else
                end
            end
            open(joinpath(outpath,"models_$(run_idx).txt"), "w") do f
                for i in eachindex(models_acpt) 
                    println(f, models_acpt[i],"\t", χ_acpt[i])
                end
            end
            # save χs for printing and export
            χ_vec[iter] = χsq_new
            # model_iter[iter,:] = cat(band_1_cand[1:3],band_2_cand[1:3],minima_cand,μ_cand, dims=1)
        end
        # plot results
        if plot && (run_idx ∈ idx2plot)
            # println("---> 3 Plotting results...")
            tensor_names = ["el_cond","seebeck","cconc","th_cond"][not_empty_tensors]
            χfig = plot_χ(n_iter,χ_vec,χ_acpt)
            savefig(joinpath(outpath,"χ_result_$(run_idx).png"), χfig);
            tensorfig = plot_best(n_bands,bandtype,τ_form,T,ftransport,tensor_names,true_tensors,models_acpt[end])
            savefig(joinpath(outpath,"tensors_result_$(run_idx).png"), tensorfig);
            evfig = plot_evoacept(n_bands,bandtype,τ_form,T,ftransport,tensor_names,true_tensors,models_acpt,χ_acpt)
            savefig(joinpath(outpath,"allaccept_result_$(run_idx).png"), evfig);
        end

        χ_best[run_idx] = χsq_running
        model_best[:,run_idx] = models_acpt[end]
    end
    t_end = time_ns()

    # print true model structure
    energies = Array{Float64}(undef,n_bands+1)
    model_true_params = Array{Float64}(undef,4n_bands)    # 4 params each band
    for b in 1:n_bands
        band = model_true.bands[b]
        idx = (b-1)*4     # index to select the correct position in the model_cand_params array
        model_true_params[idx+1:idx+4] = cat(band.mstar[1:3],band.ϵ₀,dims=1)
        energies[b] = band.ϵ₀
    end
    energies[end] = model_true.μ
    model_best = shiftenergies!(model_best,energies,n_bands)

    # plot best model among all runs
    println("\n---> 3. Plotting results.")
    idx = argmin(χ_best)
    tensor_names = ["el_cond","seebeck","cconc","th_cond"][not_empty_tensors]
    tensorfig = plot_best(n_bands,bandtype,τ_form,T,ftransport,tensor_names,true_tensors,model_best[:,idx])
    savefig(joinpath(outpath,"tensors_result_best.png"), tensorfig);
    # plot band structure
    bands_pred = Array{ParabBand}(undef,n_bands)
    best_model_overall = model_best[:,idx]
    for b in 1:n_bands
        # effective masses
        m_idx = (b-1)*4 # index to select the correct mass range (each band has 4 params)
        m = best_model_overall[m_idx+1:m_idx+3]
        # position
        e0 = best_model_overall[m_idx+4]
        # same type of original band structure (cond or val)
        type = bandtype[b]
        # create the band 
        bands_pred[b] = ParabBand(m,e0,type,1);
    end
    μ_pred = best_model_overall[end];
    bandfig = plot_bandcomp(model_true,BandStructure(n_bands,bands_pred,μ_pred))
    savefig(joinpath(outpath,"band_comp.png"), bandfig);

    open(joinpath(outpath,"results.txt"), "w") do f 
        println(f,"Simulation time: \t", round((t_end-t_start)*1e-9,digits=3), " s")
        println(f,"Selected chisq: \t", χ_best)
        println(f,"Best chisq: \t\t", χ_best[idx])
        println(f,"\nSelected models: ",model_best[:,1])
        for i in 2:n_restart
            println(f,"                 ",model_best[:,i])
        end
        println(f,"\nBest model:      ",model_best[:,idx])
        printstyled(f,"\nTrue model:      "; color = :green, bold=true); 
        println(f,cat(model_true_params,model_true.μ,dims=1))
    end

    println("\n##### Results of the simulation.##### ")
    println("Simulation time: \t", round((t_end-t_start)*1e-9,digits=3), " s")
    printstyled("\nSelected χsq: "; color = :blue, bold=true)
    println(χ_best)
    printstyled("Best χsq: "; color = :blue, bold=true)
    println(χ_best[idx])
    printstyled("\nSelected models: "; color = :blue, bold=true)
    println(model_best[:,1])
    for i in 2:n_restart
        println("                 ",model_best[:,i])
    end
    printstyled("\nBest model:      "; color = :blue, bold=true)
    println(model_best[:,idx])
    printstyled("\nTrue model:      "; color = :green, bold=true)
    println(cat(model_true_params,model_true.μ,dims=1))

    println("\nSimulation completed.")
    return χ_best,model_best
end

# TODO: check band creation from model array: m* has 6 component. 
# change it to this vcat(model[m_idx+1:m_idx+3],zeros(3))

# f(β::Float64,α::Float64) = β / (1-αt) # linear
# f(β::Float64,α::Float64) = β / α      # geometric
# f(β::Float64,α::Float64) = β + α      # homographic 
# f(β::Float64,α::Float64) = β          # constant
