using Mstar2t
using Mstar2t: Scattering

using T2mstar

# ground true model
m_1 = [3.0, 3.0, 3.0, 0.0, 0.0, 0.0];
band_1_true = ParabBand(m_1,0.5,1,1);

m_2 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
band_2_true = ParabBand(m_2,0.0,-1,1);

μ_true = -0.1;
model_true = BandStructure(2,[band_1_true,band_2_true],μ_true);

T = collect(50:50:600);
τ_form = Scattering.constant();

σ_true = electrical_conductivity(model_true,T,τ_form);
S_true = seebeck_coefficient(model_true,T,τ_form);
n_true = carrier_concentration(model_true,T,τ_form);

args = Args(1000,5,1234,0.01,1e8,200);
coeff_true = GroundTruth(τ_form,T,σ_true,S_true,n_true);

# ranges 
mrange = 1:4;
ϵ₀c_range = .0:.1:0.8;
ϵ₀v_range = 0.0;
μrange = -.5:0.1:.5;
τ_A_range = (40:1:50)*1e-11
τ_B_range = 0:50:100
τ_C_range = 2:0.1:3

ranges = Dict("m"  => [mrange,mrange,mrange,mrange,mrange,mrange], 
              "ϵ₀" => [ϵ₀c_range,ϵ₀v_range],
              "μ"  => μrange,
              "τ"  => [τ_A_range,τ_B_range,τ_C_range] );

# run the algorithm
χ_best,model_best = Mstar2t.RMC(model_true,coeff_true,args,ranges,anneal=true,plot=false);