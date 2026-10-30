using Sunny
using DataFrames, CSV

include("triangular_ladder.jl")

# -----------------------------
# User parameters
# -----------------------------
# [J1, Δ, Jpmpm, Jzpm, J2]
Js0 = [1,0.541,-0.625,0,0.062] #[1.0, 0.7, 0.2, 0.0, 0.051] 
dir = [1.0, 0.0, 0.0]
B_sat, sys, _ = polarized_system(Js0, dir)

# -----------------------------
# Build Lanczos result 
# -----------------------------
function Run_THL(Js, B;
    η = 0.2,
    N_E=100,
)
    _, sys, _ = polarized_system(Js, [1,0,0])
    swt = SpinWaveTheory(sys; measure=dssf_measure(sys))
    qpath = [[0, 0, 0], [1, 1, 0]]
    nq = 60
    qs = q_space_path(sys.crystal, qpath, nq).qs
    e_max = energy_max(swt; n=16)
    energies = collect(0.0:0.01:e_max)
    data, nbreak = ladder_dssf(swt, qs; energies, η)

    img = data[7,:,:]
    Es_all, Ss_all = Float64[], Float64[]
    for k_ind in 1:length(qs)
        keep_indices = argmax(img[k_ind, :]) # find the index of the maximum intensity for this q-point
        E_temp, S_temp = energies[keep_indices], img[k_ind, keep_indices]
        append!(Es_all, E_temp)
        append!(Ss_all, S_temp)
    end
    return (data=vec(Ss_all), disp=vec(Es_all))
end

results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[])

res = Run_THL(Js0, B_sat)
push!(results, (
    copy(Js0[[1, 2, 3, 5]]), # only keep J1, Δ, J1pmpm, J2
    vec(res.data),
    vec(res.disp),
))
# print(res.data)
CSV.write("../data/THL_Js4_Nk60_Ne1.csv", results)
println("saved results to data")



