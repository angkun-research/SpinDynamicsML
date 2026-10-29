using Sunny
using DataFrames, CSV

include("triangular_ladder.jl")

# -----------------------------
# User parameters
# -----------------------------
# [J1, Δ, Jpmpm, Jzpm, J2]
#Js0 = [1.0, 0.7, 0.2, 0.0, 0.051] 
Js0 = [1. , 1. , 1. ,0.0 , 0.5]
Js = Js0
#Js = [1.018004, 1.028298, 1.016899, 0.0, 0.5007  ]
#Js = [1.004141, 0.973033, 1.003829,0.0, 0.509457]
dir = [1.0, 0.0, 0.0]
B_sat, sys, _ = polarized_system(Js0, dir)

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
        E_temp, S_temp, keep_indices = adaptive_knots(energies, img[k_ind, :], N=N_E)
        append!(Es_all, E_temp)
        append!(Ss_all, S_temp)
    end
    return (data=vec(Ss_all), disp=vec(Es_all))
end

results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[])

res = Run_THL(Js, B_sat)
push!(results, (
    copy(Js[[1, 2, 3, 5]]), # only keep J1, Δ, J1pmpm, J2
    vec(res.data),
    vec(res.disp),
))

CSV.write("../data/THL_Js4_Nk60_Ne100_idx8.csv", results)
#CSV.write("../data/THL_Js4_Nk60_Ne100_idx8_sigma0.0.csv", results)
#CSV.write("../data/THL_Js4_Nk60_Ne100_idx8_sigma0.1.csv", results)
println("saved results to data")

