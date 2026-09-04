using Sunny
using JLD2
using CairoMakie
using DataFrames, CSV, ProgressBars

include("triangular_nlswt.jl")

@show Threads.nthreads()

gauss(x, μ, σ) = exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))

"""
Fit an N-component Gaussian mixture to S(ω) via weighted EM, with weights π_k
also learned (not fixed uniform).
Returns μ (means), σ (std devs), A (amplitudes), π_k (mixture weights).
"""
function fit_gmm_full(ω, S; N=10, n_iter=200, tol=1e-10)
    Δω = ω[2] - ω[1]
    w = max.(S, 0.0)
    wsum = sum(w)
    @assert wsum > 0 "signal has no positive mass"
    p = w ./ wsum

    cumw = cumsum(p)
    μ = [ω[searchsortedfirst(cumw, q)] for q in range(1/(2N), 1 - 1/(2N); length=N)]
    span = maximum(ω) - minimum(ω)
    σ = fill(span / (2N), N)
    π_k = fill(1.0 / N, N)

    n = length(ω)
    r = zeros(n, N)
    prev_ll = -Inf

    for _ in 1:n_iter
        for i in 1:n
            tot = 0.0
            for k in 1:N
                r[i, k] = π_k[k] * gauss(ω[i], μ[k], σ[k])
                tot += r[i, k]
            end
            tot = max(tot, 1e-300)
            @views r[i, :] ./= tot
        end

        for k in 1:N
            wk = p .* @view r[:, k]
            Nk = sum(wk)
            π_k[k] = Nk
            Nk < 1e-12 && continue
            μ[k] = sum(wk .* ω) / Nk
            σ2 = sum(wk .* (ω .- μ[k]) .^ 2) / Nk
            σ[k] = sqrt(max(σ2, 1e-6))
        end
        π_k ./= sum(π_k)

        ll = 0.0
        for i in 1:n
            s = 0.0
            for k in 1:N
                s += π_k[k] * gauss(ω[i], μ[k], σ[k])
            end
            ll += p[i] * log(max(s, 1e-300))
        end
        abs(ll - prev_ll) < tol && break
        prev_ll = ll
    end

    A = π_k .* wsum .* Δω
    return μ, σ, A, π_k
end

# The lower bound and upper bound of energy range (in meV)
energy_min, energy_max = 0.0, 50.0
energy_step = 0.01
energies = collect(energy_min:energy_step:energy_max)
# The Lorentzian broadening factor (in meV) FWHM=2η
η = 0.2
dir = [1.0, 0.0, 0.0]
Npts = 20

num_tasks = 250
# 3-channel storage per (component, k-point): mean, variance, weight
const RowT = NamedTuple{(:Js, :means, :variances, :weights, :B, :Bsats), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64, Vector{Float64}}}
rows = RowT[]

pt = "/Users/angkunwu/SpinDynamic.jl/data/HaoResults/"
for taskid in ProgressBar(1:num_tasks)
    filetemp = load(pt * "results_task$(taskid)_of_$(num_tasks).jld2");
    num_jobs = length(filetemp["Js_list"])
    local_rows = Vector{RowT}(undef, num_jobs)
    Threads.@threads for jobid in 1:num_jobs
        Js = filetemp["Js_list"][jobid]
        as = filetemp["as_list"][jobid]
        bs = filetemp["bs_list"][jobid]
        norm2s = filetemp["norm2s_list"][jobid]
        qs_path = filetemp["qs_list"][jobid]
        B = filetemp["B_list"][jobid]
        B_sat = zeros(3)
        B_sat[1], _ = determine_B_sat(Js, [1.0, 0.0, 0.0])
        B_sat[2], _ = determine_B_sat(Js, [0.0, 1.0, 0.0])
        B_sat[3], _ = determine_B_sat(Js, [0.0, 0.0, 1.0])
        dssf = map_lanczos_results_to_dssf_tensor(qs_path, as, bs, norm2s, energies, η)
        means_all, vars_all, weights_all = Float64[], Float64[], Float64[]
        for component in 1:6
            for k_ind in 1:length(qs_path)
                μ_temp, σ_temp, A_temp, π_temp = fit_gmm_full(energies, dssf[component, k_ind, :], N=Npts)
                append!(means_all, μ_temp)
                append!(vars_all, σ_temp .^ 2)
                append!(weights_all, π_temp)
            end
        end
        println("Task $(taskid), Job $(jobid)")
        local_rows[jobid] = (Js = copy(Js), means = vec(means_all), variances = vec(vars_all),
                              weights = vec(weights_all), B = copy(B), Bsats = vec(B_sat))
    end
    append!(rows, local_rows)
end
println("Total rows collected: ", length(rows))
results = DataFrame(rows)
CSV.write(pt * "Hao_anisotropy_Js5_S6tensor_GMM$(Npts)_Bsats.csv", results)