using Sunny
using DataFrames, CSV

# Reuse Hao's Lanczos and B_sat helpers
include("../JuliaCode/triangular_nlswt.jl")

# -----------------------------
# User parameters
# -----------------------------
#Js = [ 0.996519, -0.95559 ,  0.55927 , 0.0, -0.180572]   # [J1, Δ, Jpmpm, Jzpm, J2]
Js = [1.0, 0.7, 0.2, 0.0, 0.051] 
#Js0 = [1.0, -0.368421, -1.0, 0.0, 0.473684]
dir = [1.0, 0.0, 0.0]
B_sat, _ = determine_B_sat(Js, dir)
B_sat = B_sat * 1.1 #7.7017794 #B_sat * 1.15
println("B_sat: ", B_sat)

# -----------------------------
# Build Lanczos result 
# -----------------------------
function Run_THED(Js, B;
    clustersize=(11, 11, 1),
    dir=[1.0, 0.0, 0.0],
    energies=collect(0.0:0.01:50.0),
    eta=0.2,
)
    res = triangular_nlswt_lanczos_calculations(Js, clustersize, copy(dir); B=B)
    isnothing(res) && error("Lanczos calculation failed for Js=$(Js)")

    B_used, qs_path, as_path, bs_path, norm2s_path = res

    img = map_lanczos_results_to_neutron_intensities(
        qs_path, as_path, bs_path, norm2s_path, energies, eta
    )  # size: (nq, nE)
    Es_all, Ss_all = Float64[], Float64[]
    for k_ind in 1:length(qs_path)
        keep_indices = argmax(img[k_ind, :]) # find the index of the maximum intensity for this q-point
        E_temp, S_temp = energies[keep_indices], img[k_ind, keep_indices]
        append!(Es_all, E_temp)
        append!(Ss_all, S_temp)
    end
    return (data=vec(Ss_all), disp=vec(Es_all))
end

results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[])

res = Run_THED(Js, B_sat)
push!(results, (
    copy(Js[[1, 2, 3, 5]]), # only keep J1, Δ, J1pmpm, J2
    vec(res.data),
    vec(res.disp),
))

CSV.write("data/THED_Js4_Nk12_Ne1.csv", results)
println("saved results to data")


# # -----------------------------
# # Build LSWT result 
# # -----------------------------
# function run_lswt(Js; B=60, dims=(1, 1, 1), nqp=12)
#     J1, Δ, Jpmpm, Jzpm, J2 = Js
#     units = Units(:meV, :angstrom)

#     a = 4.1149
#     c = 22.6911
#     dir = [1.0, 0.0, 0.0]
#     latvecs = lattice_vectors(a, a, c, 90, 90, 120)
#     cryst = Crystal(latvecs, [[0, 0, 0]], 162)

#     qs = [[0, 0, 0], [1, 1, 0]]
#     path = q_space_path(cryst, qs, nqp)  # 12 points along (H,H,0)

#     gab, gcc = 3.13 * 1.09310, 1.01
#     gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
#     sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

#     J1_ten = [J1 + 2Jpmpm 0 Jzpm;
#               0 J1 - 2Jpmpm 0;
#               Jzpm 0 J1 * Δ]

#     set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
#     set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
#     set_field!(sys, B * units.T * dir)

#     polarize_system!(sys, -dir)
#     swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
#     res = intensities_bands(swt, path)
#     return (data=vec(res.data), disp=vec(res.disp))
# end

# res_thed = Run_THED(Js, B_sat)
# res_lswt = run_lswt(Js; B=B_sat)

# println("THED disp: ", res_thed.disp)
# println("LSWT disp: ", res_lswt.disp)
# println("THED data: ", res_thed.data)
# println("LSWT data: ", res_lswt.data)


# using Plots
# using LaTeXStrings
# using LinearAlgebra
# using Measures

# xinds = 1:length(res_thed.disp)

# plot(xinds, res_thed.disp, label="THED", lw=2)
# plot!(xinds, res_lswt.disp, label="LSWT", lw=2)
# xlabel!("Energy (meV)")
# ylabel!("Intensity")
# title!("Comparison of THED and LSWT")
# savefig("plots/THED_vs_LSWT_disp.png")

# plot(xinds, res_thed.data, label="THED", lw=2)
# plot!(xinds, res_lswt.data, label="LSWT", lw=2)
# xlabel!("Energy (meV)")
# ylabel!("Intensity")
# title!("Comparison of THED and LSWT")
# savefig("plots/THED_vs_LSWT_data.png")
# print("Saved plots to plots/ \n")