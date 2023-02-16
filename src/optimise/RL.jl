using Parameters

using Mstar2t
import Mstar2t: Scattering

using StatsBase: sqL2dist

using POMDPs, QuickPOMDPs, POMDPModelTools
using DiscreteValueIteration


State = Vector{Float64}
Action = Int64

# utility functions
function incr(value::Real,collection::AbstractVector) 
	idx = findfirst(isequal(value),collection)
	return idx+1 <= length(collection) ? collection[idx+1] : value
end

function decr(value::Real,collection::AbstractVector) 
	idx = findfirst(isequal(value),collection)
	return idx-1 >= 1 ? collection[idx-1] : value
end

function bs2state(bs::BandStructure)::State
	s = Array{Float64}(undef,4rl_params.n_bands+1)
	bands = bs.bands
	for b in 1:rl_params.n_bands
		m_idx = (b-1)*4 # index to select the correct mass range (each band has 4 params)
		s[m_idx+1:m_idx+3] = bands[b].mstar[1:3]
		s[b*4] = bands[b].e0
	end
	s[end] = bs.μ
	return s
end

function state2bs(s::State)::BandStructure
	bands = Array{Band}(undef,rl_params.n_bands)
    for b in 1:rl_params.n_bands
        # effective masses
        m_idx = (b-1)*4 # index to select the correct mass range (each band has 4 params)
        m = vcat(s[m_idx+1:m_idx+3],zeros(3))
        # position
        e0 = s[b*4]
        # same type of original band structure (cond or val)
        type = bandtype[b]
        # create the band 
        bands[b] = Band(m,e0,type,1)
    end
	return BandStructure(rl_params.n_bands,bands,s[end])
end


# FINAL/TRUE STATE
n_bands = 1
m_1 = [1.2, 2.2, 0.2, 0.0, 0.0, 0.0];
bandtype = [-1]
band_1_true = Band(m_1,0.2,-1,1)
μ_true = 0.1
model_true = BandStructure(n_bands,band_1_true,μ_true)

ts = collect(50:50:600)
τ_form = Scattering.constant()

# TRUE TENSORS -> to compute reward
σ_true = electrical_conductivity(model_true,ts,τ_form)
S_true = seebeck_coefficient(model_true,ts,τ_form)
n_true = carrier_concentration(model_true,ts,τ_form)
true_tensors = [σ_true,S_true,n_true]
n_tensors = length(true_tensors)
σ² = Array{Float64}(undef,n_tensors)
for i in eachindex(true_tensors)
    σ²[i] = sum(true_tensors[i].^2)
end

# Ex: 2-band system
@with_kw struct BSPredictorParameters
	n_bands = n_bands
	mrange = 0.2:1:3;
	# ϵ₀c_range = .0:.2:0.8;
	ϵ₀v_range = 0.0:0.0;
	μrange = -.5:.1:.5;
	end_thr = 1e-18
	χ²_running = Inf
end
rl_params = BSPredictorParameters();

mranges = [rl_params.mrange for i in 1:3*n_bands] 
# eranges = [params.ϵ₀c_range,params.ϵ₀v_range]
eranges = [rl_params.ϵ₀v_range]
μrange = rl_params.μrange

# MDP/RL objects
# 2. Action space
𝒜 = collect(1:2(3n_bands+(n_bands-1)+1)) # increse/decrease each params

# 4. Reward function
function R(s::State,χ²_best::Float64)

	σₛ = electrical_conductivity(state2bs(s),ts,τ_form)
	Sₛ = seebeck_coefficient(state2bs(s),ts,τ_form)
	nₛ = carrier_concentration(state2bs(s),ts,τ_form)
	state_tensors = [σₛ,Sₛ,nₛ]	

	χ² = 0.0
	for i in 1:3
	    χ² += sqL2dist(true_tensors[i],state_tensors[i])/σ²[i]
	end

	Δχ² = χ²-χ²_best

	if χ² == 0
		return 1e6,χ²
	elseif Δχ² > 0	# worsen
		return -1,χ²_best
	elseif Δχ² < 0	# improvement 
		return 100,χ²
	end
end

# discount factor
γ = 0.95

# model_true
model_true_v = bs2state(model_true)

# initialstate
initialstate_v = Array{Float64}(undef,4rl_params.n_bands+1)
for b in 1:rl_params.n_bands
	for j in 1:3
		initialstate_v[(b-1)*4+j] = rand(mranges[(b-1)*3+j])
	end
end
for b in 1:rl_params.n_bands-1
	initialstate_v[b*4] = rand(eranges[b])
end
initialstate_v[end-1] = model_true_v[end-1]
initialstate_v[end] = rand(μrange)

# termination
function termination(s::State) 
	σₛ = electrical_conductivity(state2bs(s),ts,τ_form)
	Sₛ = seebeck_coefficient(state2bs(s),ts,τ_form)
	nₛ = carrier_concentration(state2bs(s),ts,τ_form)
	state_tensors = [σₛ,Sₛ,nₛ]	

	χ² = 0.0
	for i in 1:3
		χ² += sqL2dist(true_tensors[i],state_tensors[i])/σ²[i]
	end
	χ² == 0	# win
end

# MDP formualation
abstract type BSPredictor <: MDP{State, Action} end
χ²_running = rl_params.χ²_running

mdp = QuickMDP(
    function gen(s, a, rng)

		global χ²_running

		if a == 8n_bands-1  # ↑μ: increse μ
			s[end] = incr(s[end],μrange)
			r,χ²_running = R(s,χ²_running)
			return (sp=s, r=r)	# minimize #steps
		elseif a == 8n_bands # ↓μ: decrease μ
			s[end] = decr(s[end],μrange)
			r,χ²_running = R(s,χ²_running)
			return (sp=s, r=r)	# minimize num steps
		end
	
		# decode action index
		b = (a-1) ÷ 8 + 1	# band index
		rem = (a-1) % 8
		up = Bool(rem % 2)	# 0:descrese (false), 1:increse (true)
		idx_par = rem ÷ 2 + 1	# which params to change
		ismass = idx_par < 4 # 1,2,3: masses, 4: energy
	
		if ismass
			if up	# increse
				s[(b-1)*4+idx_par] = incr(s[(b-1)*4+idx_par],mranges[(b-1)*3+idx_par])
			else 	# decrese
				s[(b-1)*4+idx_par] = decr(s[(b-1)*4+idx_par],mranges[(b-1)*3+idx_par])
			end
		elseif b < n_bands	# move only n_bands-1 bands + μ
			if up	# increse
				s[(b-1)*4+idx_par] = incr(s[(b-1)*4+idx_par],eranges[b])
			else 	# decrese
				s[(b-1)*4+idx_par] = decr(s[(b-1)*4+idx_par],eranges[b])
			end
		end
		r,χ²_running = R(s,χ²_running)
		return (sp=s, r=r)	# minimize num steps
	end,
    initialstate = [initialstate_v],
    actions      = 𝒜,
    discount     = γ,
    isterminal   = termination
	);

using CUDA
using DeepQLearning
using Flux
using POMDPModels
using POMDPSimulators
using POMDPPolicies

layer1 = Dense(4n_bands+1, 64, relu)
layer2 = Dense(64, 64, relu)
layer3 = Dense(64, 64, relu)
layer4 = Dense(64, length(actions(mdp)), tanh)
model = Chain(layer1, layer2, layer3, layer4)

exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2))

solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, 
							 num_ep_eval=500,max_episode_length=1000,
							 exploration_policy = exploration,
                             learning_rate=0.005,log_freq=500,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true)

policy = solve(solver, mdp)

sim = RolloutSimulator(max_steps=500)
# search_agent = rand(POMDPs.initialstate(mdp))
search_agent =  [1.2,2.2,0.2,0.2,1.2,1.2,1.2,0.0,0.5]
agent_recorder = HistoryRecorder(max_steps=300)
search_history = simulate(sim, mdp, policy,search_agent)
println("Total discounted reward for 1 simulation: $search_history")
search_history = simulate(agent_recorder, mdp, policy, search_agent)





using LocalApproximationValueIteration
using LocalFunctionApproximation
using GridInterpolations

grid = RectangleGrid(mranges[1],mranges[2],mranges[3],eranges[1],
					 mranges[4],mranges[5],mranges[6],eranges[2],μrange);

interpolation = LocalGIFunctionApproximator(grid);

solver = LocalApproximationValueIterationSolver(interpolation,
											max_iterations=100,
											is_mdp_generative=true,
											n_generative_samples=1)


policy = solve(solver, mdp);

import POMDPSimulators: HistoryRecorder

search_agent = rand(POMDPs.initialstate(mdp))
agent_recorder = HistoryRecorder(max_steps=300)
search_history = simulate(agent_recorder, mdp, policy, search_agent)

agent_step = 2
current_state = [search_history[agent_step].s...]


# # VALUE ITERATION
# solver = ValueIterationSolver(max_iterations=30);

# # policy extraction
# policy = solve(solver, mdp)

# using POMDPPolicies
# using TabularTDLearning
# ql_solver = QLearningSolver(n_episodes=30,
# 							learning_rate=0.8,
# 							exploration_policy=EpsGreedyPolicy(mdp, 0.5),
# 							verbose=false);

# ql_policy = solve(ql_solver, mdp);