using Sunny
using DataFrames, CSV

# Reuse Hao's Lanczos and B_sat helpers
include("../HaoCode/triangular_nlswt.jl")

# -----------------------------
# User parameters
# -----------------------------
#Js = [ 0.996519, -0.95559 ,  0.55927 , 0.0, -0.180572]   # [J1, Δ, Jpmpm, Jzpm, J2]
Js = [1.0, 0.7, 0.2, 0.0, 0.051] 
#Js0 = [1.0, -0.368421, -1.0, 0.0, 0.473684]
dir = [1.0, 0.0, 0.0]
B_sat, _ = determine_B_sat(Js0, dir)
B_sat = B_sat * 1.1 #7.7017794 #B_sat * 1.15

function adaptive_knots(ω, S; N=10)
    n = length(ω)
    @assert n == length(S)
    keep = sort([1, n])

    # linear interpolation prediction between two kept indices
    linpred(i, j, k) = S[i] + (S[j]-S[i]) * (ω[k]-ω[i]) / (ω[j]-ω[i])

    while length(keep) < N
        keep = sort(keep)
        best_err = -Inf
        best_k = nothing

        for seg in 1:(length(keep)-1)
            i, j = keep[seg], keep[seg+1]
            if j <= i + 1
                continue
            end
            # find worst point in this segment
            for k in (i+1):(j-1)
                err = abs(S[k] - linpred(i, j, k))
                if err > best_err
                    best_err = err
                    best_k = k
                end
            end
        end

        best_k === nothing && break # no more points to add
        push!(keep, best_k)
    end

    keep = sort(unique(keep))
    return ω[keep], S[keep], keep
end

# -----------------------------
# Build Lanczos result 
# -----------------------------
function Run_Lanczos(Js, B;
    clustersize=(11, 11, 1),
    dir=[1.0, 0.0, 0.0],
    energies=collect(0.0:0.01:50.0),
    eta=0.2,
    N_E=100,
)
    res = triangular_nlswt_lanczos_calculations(Js, clustersize, copy(dir); B=B)
    isnothing(res) && error("Lanczos calculation failed for Js=$(Js)")

    B_used, qs_path, as_path, bs_path, norm2s_path = res

    img = map_lanczos_results_to_neutron_intensities(
        qs_path, as_path, bs_path, norm2s_path, energies, eta
    )  # size: (nq, nE)
    Es_all, Ss_all = Float64[], Float64[]
    for k_ind in 1:length(qs_path)
        E_temp, S_temp, keep_indices = adaptive_knots(energies, img[k_ind, :], N=N_E)
        append!(Es_all, E_temp)
        append!(Ss_all, S_temp)
    end
    return (data=vec(Ss_all), disp=vec(Es_all))
end

results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[])

res = Run_Lanczos(Js, B_sat)
push!(results, (
    copy(Js[[1, 2, 3, 5]]), # only keep J1, Δ, J1pmpm, J2
    vec(res.data),
    vec(res.disp),
))

#CSV.write("data/Lanczos_Js4_Nk$(12)_Ne$(100)_test_1.csv", results)
#CSV.write("data/Lanczos_Js4_Nk12_Ne100_idx9_sigma0.0.csv", results)
CSV.write("data/Lanczos_Js4_Nk12_Ne100_idx9_sigma0.1.csv", results)
println("saved results to data")

