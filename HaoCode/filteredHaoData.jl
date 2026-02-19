using Sunny
using JLD2
using CairoMakie
using DataFrames, CSV,ProgressBars

include("triangular_nlswt.jl")

@show Threads.nthreads()

"""
Select N representative points (ω[k], S[k]) using greedy max interpolation error.
Good for preserving peaks with few points.
"""
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

# The lower bound and upper bound of energy range (in meV)
energy_min, energy_max = 0.0, 30.0
energy_step = 0.01
energies = collect(energy_min:energy_step:energy_max)
# The Lorentzian broadening factor (in meV) FWHM=2η
η = 0.2

num_tasks = 250
#results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[])
# Store rows in a thread-safe way (no shared mutation inside threads)
const RowT = NamedTuple{(:Js, :data, :disp), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
rows = RowT[]

# Randomly sample Js from the Jsize dimensional space with each J in range -1 to 1
for taskid in ProgressBar(1:num_tasks)
    filetemp = load("data/HaoResults/results_task$(taskid)_of_$(num_tasks).jld2");
    num_jobs = length(filetemp["Js_list"])
    local_rows = Vector{RowT}(undef, num_jobs)
    Threads.@threads for jobid in 1:num_jobs
        Js = filetemp["Js_list"][jobid]
        as = filetemp["as_list"][jobid]
        bs = filetemp["bs_list"][jobid]
        norm2s = filetemp["norm2s_list"][jobid]
        qs_path = filetemp["qs_list"][jobid]
        B = filetemp["B_list"][jobid]
        intensities = map_lanczos_results_to_neutron_intensities(qs_path, as, bs, norm2s, energies, η)
        Es_all, Ss_all = Float64[], Float64[]
        for k_ind in 1:length(qs_path)
            E_temp, S_temp, keep_indices = adaptive_knots(energies, intensities[k_ind, :], N=20)
            append!(Es_all, E_temp)
            append!(Ss_all, S_temp)
        end
        println("Task $(taskid), Job $(jobid)")
        # push!(results, (
        #     copy(Js),
        #     vec(Es_all),
        #     vec(Ss_all),
        # ))
        # DO NOT push! to a shared DataFrame here
        local_rows[jobid] = (Js = copy(Js), data = vec(Es_all), disp = vec(Ss_all))
    end
    append!(rows, local_rows)
    if length(rows) >= 1000
        break # stop early for testing
    end
end

results = DataFrame(rows)
CSV.write("data/HaoResults/Hao_anisotropy_Js5_Filter20_N1009.csv", results)