using Sunny
using JLD2
using CairoMakie
using DataFrames, CSV, ProgressMeter #ProgressBars

include("triangular_ladder.jl")

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

# ------------------------------------------------------------
# Parameter conversion at the existing helper's boundary
# triangular_ladder.jl expects [J1, Jzz, Jpmpm, Jzpm, J2].
# ------------------------------------------------------------
function paper_to_ladder_Js(Js)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    return [J1, J1 * Δ, Jpmpm, Jzpm, J2]
end

# The lower bound and upper bound of energy range (in meV)
E_min, E_max = 0.0, 50.0
E_step = 0.01
energies = collect(E_min:E_step:E_max)
# The Lorentzian broadening factor (in meV) FWHM=2η
η = 0.2

nq = 12
qpoints = [[0, 0, 0], [1, 1, 0]]
dir = [1.0, 0.0, 0.0]
NE = 20

Jsize = 4 #5 #4
N_uniform = 20 #30
Nsamples = 2*N_uniform^(Jsize-1)
Js_all = Vector{Float64}[]
for J10 in [-1,1]
    for ΔJ10 in collect(range(-1, 1, length=N_uniform+1))
        Δ = ΔJ10 / J10
        if Δ == 0.0
            continue
        end
        for Jpmpm in collect(range(-1, 1, length=N_uniform+1))
            if Jpmpm == 0.0
                continue
            end
            for J20 in collect(range(-1, 1, length=N_uniform+1))
                if J20 == 0.0
                    continue
                end
                if Jsize == 4
                    Js = [J10, Δ, Jpmpm, 0.0, J20]
                    push!(Js_all, Js)
                else
                    for Jzpm in collect(range(-1, 1, length=N_uniform+1))
                        if Jzpm == 0.0
                            continue
                        end
                        Js = [J10, Δ, Jpmpm, Jzpm, J20]
                        push!(Js_all, Js)
                    end
                end
            end
        end
    end
end


# Store rows in a thread-safe way (no shared mutation inside threads)
const RowT = NamedTuple{(:Js, :disp, :data, :B), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}
rows = RowT[]
p = Progress(Nsamples)
# Uniformly sample Js from the Jsize dimensional space with each J in range -1 to 1
Threads.@threads for taskid in 1:Nsamples #ProgressBar(1:Nsamples)
    Js = Js_all[taskid]
    try
        B_sat, sys = polarized_system(paper_to_ladder_Js(Js), dir)
        path = q_space_path(sys.crystal, qpoints, nq)
        swt_ladder = SpinWaveTheory(sys; measure=dssf_measure(sys))
        #intensities_bands(swt_ladder, path)
        ladder_full, nbreak = ladder_dssf(swt_ladder, path; energies,η, threaded=false)
        intensities = @view ladder_full[7, :, :] # perpendicular channel
        if nbreak > 0 || !all(isfinite, intensities)
            @warn "Skipping task $taskid: invalid ladder spectrum" nbreak
            continue
        end
        Es_all, Ss_all = Float64[], Float64[]
        for k_ind in 1:size(intensities, 1)
            E_temp, S_temp, keep_indices = adaptive_knots(energies, intensities[k_ind, :], N=NE)
            append!(Es_all, E_temp)
            append!(Ss_all, S_temp)
        end
        if Jsize == 4
            Js = Js[[1, 2, 3, 5]]
        end
        temp = (Js = copy(Js), disp = vec(Es_all), data = vec(Ss_all), B = copy(B_sat))
        push!(rows, temp)
    catch e
        @warn "Error processing task $taskid: $e"
    end
    next!(p)
end

results = DataFrame(rows)
pt = "/Users/angkunwu/SpinDynamic.jl/data/"
CSV.write(pt * "Ladder_anisotropy_Js$(Jsize)_Filter$(NE).csv", results)