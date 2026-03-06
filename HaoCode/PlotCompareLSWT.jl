using Sunny
using Plots

include("triangular_nlswt.jl")

function Run_LSWT(Js,B; dims = (1,1,1))
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    units = Units(:meV, :angstrom)
    a = 4.1149
    c = 22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)
    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst,qs,12)

    # g-tensor for KYbSe₂
    gab, gcc = 3.13*1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    #sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(6, 6, 1))
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [B, 0, 0] * units.T)

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))

    polarize_system!(sys)

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    res = intensities_bands(swt, path)

    return res
end

Jsize = 5
nsample = 2
Js = zeros(Jsize, nsample)
# Js[:, 1] = [0.6703412, 0.42209536, 0.59768635, 0.23526795, 0.63349897]
# Js[:, 2] = [0.67246586, 0.42533526, 0.61599827, -0.09850078, 0.61625373]
# Js[:, 1] = [0.18784128, 0.27096057, 0.52975523, 0.2925026, 0.3287352]
# Js[:, 2] = [0.16038370, 0.2801616, 0.5479578, 0.08557, 0.32782653]
Js[:, 1] = [0.14007732, 0.9819725, 0.08231799, -0.34657982, 0.09762552]
Js[:, 2] = [0.18598570, 0.88416976, 0.07183129, -0.11190467, 0.11418682]

res = zeros(12, nsample)

# Randomly sample Js from the Jsize dimensional space with each J in range -1 to 1
for sample_idx in 1:nsample
    B_sat, _ = determine_B_sat(Js[:, sample_idx])
    restemp = Run_LSWT(Js[:, sample_idx], B_sat)
    res[:, sample_idx] = vec(restemp.disp)
end


# plot the two curves
scatter(1:12, res[:,1], label="J true", xlabel="q-point index", ylabel="Energy (meV)", title="LSWT anisotropy", legend=:top)
scatter!(1:12, res[:,2], label="J prediction")
#savefig("~/Desktop/SpinWaveDispersionComparison.pdf")
savefig("~/Desktop/fig3.pdf")
