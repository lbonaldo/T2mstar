using Mstar2t
using Random

# ground true model
m_1 = [3.0, 3.0, 3.0, 0.0, 0.0, 0.0];
band_1_true = Band(m_1,0.5,1,1);

m_2 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
band_2_true = Band(m_2,0.0,-1,1);

μ_true = -0.1;
model_true = BandStructure(2,[band_1_true,band_2_true],μ_true);

T = collect(50:50:600);
τ_form = constant();

σ_true = electrical_conductivity(model_true,T,τ_form);
S_true = seebeck_coefficient(model_true,T,τ_form);
n_true = carrier_concentration(model_true,T,τ_form);

rng = MersenneTwister(1234);

args = Args(2000,5,rand(rng,1:5000),0.01,1000.,100);
coeff_true = GroundTruth(T=T,τ=τ_form,σ=σ_true,S=S_true,n=n_true);

# ranges 
mrange = 1:4;
e0c_range = .0:.1:0.8;
e0v_range = 0.0;
μrange = -.5:0.1:.5;

ranges = Dict("m"  => [mrange,mrange,mrange,mrange,mrange,mrange], 
              "e0" => [e0c_range,e0v_range],
              "μ"  => μrange   );

# run the algorithm
χ_best,model_best = Mstar2t.RMC(model_true,coeff_true,args,ranges,anneal=true,plot=false);