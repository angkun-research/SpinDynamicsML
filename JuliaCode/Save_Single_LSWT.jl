using Sunny
using DataFrames, CSV

# Reuse Hao's Lanczos and B_sat helpers
include("../JuliaCode/triangular_nlswt.jl")

# -----------------------------
# User parameters
# -----------------------------
#Js = [ 1.008152, -0.874677,  0.864058, 0.0, -1.004507]   # [J1, Δ, Jpmpm, Jzpm, J2]
Js = [1.0, 0.7, 0.2, 0.0, 0.051] 
dir = [1.0, 0.0, 0.0]
B_sat, _ = determine_B_sat(Js, dir)
B_sat = B_sat * 1.1 #7.7017794 #B_sat * 1.15
println("B_sat = ", B_sat)

# -----------------------------
# Build LSWT result 
# -----------------------------
function run_lswt(Js; B=60, dims=(1, 1, 1), nqp=12)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    units = Units(:meV, :angstrom)

    a = 4.1149
    c = 22.6911
    dir = [1.0, 0.0, 0.0]
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    cryst = Crystal(latvecs, [[0, 0, 0]], 162)

    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst, qs, nqp)  # 12 points along (H,H,0)

    gab, gcc = 3.13 * 1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten = [J1 + 2Jpmpm 0 Jzpm;
              0 J1 - 2Jpmpm 0;
              Jzpm 0 J1 * Δ]

    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, B * units.T * dir)

    polarize_system!(sys, -dir)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    return intensities_bands(swt, path)
end

results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[])

res = run_lswt(Js; B=B_sat)
push!(results, (
    copy(Js[[1, 2, 3, 5]]), # only keep J1, Δ, J1pmpm, J2
    vec(res.data),
    vec(res.disp),
))

CSV.write("data/LSWT_Js4_Bsat_Nk$(12)_test_1.csv", results)
println("saved results to data")

