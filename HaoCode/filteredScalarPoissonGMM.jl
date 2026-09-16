using Sunny
using JLD2
using CairoMakie
using DataFrames, CSV, ProgressBars

include("triangular_nlswt.jl")

@show Threads.nthreads()

function gaussian_broaden(
    data::AbstractMatrix,
    energies::AbstractVector;
    fwhm::Real = 0.8,
    nσ::Real = 5,
)
    nω = size(data, 2)
    length(energies) == nω ||
        error("data columns and energies must have the same length")

    dE = energies[2] - energies[1]
    σ = fwhm / (2 * sqrt(2 * log(2)))

    if σ <= 0
        return copy(data)
    end

    radius = ceil(Int, nσ * σ / dE)
    offsets = -radius:radius
    kernel = @. exp(-0.5 * (offsets * dE / σ)^2)
    kernel ./= sum(kernel)

    output = zeros(Float64, size(data))

    for i in axes(data, 1)
        for j in axes(data, 2)
            for k in eachindex(kernel)
                source_j = j + k - radius - 1

                if 1 <= source_j <= nω
                    output[i, j] += data[i, source_j] * kernel[k]
                end
            end
        end
    end

    return output
end

function PoissonCountIntensity(intensities, T; bgrate=1.0, rng::AbstractRNG = Random.default_rng())
    counts = zeros(Int64,size(intensities))
    for i in eachindex(intensities)
        λ = (intensities[i] + bgrate) * T
        counts[i] = λ > 0 ? rand(rng, Poisson(λ)) : 0
    end
    # s' = (n_s+n_{bg})/T - bg,  
    # E(s') = E(n_s+ n_{bg})/T - bg = s, 
    # Var(s') = (s+bg)T/T^2 = (s+bg)/T
    ints_new = counts / T .- bgrate
    return ints_new # New noisy, binned intensities
end


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
Npts = 20
T = 100 # T=100, 1000, 1000000

num_tasks = 250
# 3-channel storage: mean, variance, weight (instead of energy, intensity)
const RowT = NamedTuple{(:Js, :means, :variances, :weights, :B, :Bsats), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64, Vector{Float64}}}
rows = RowT[]

for taskid in ProgressBar(1:num_tasks)
    pt = "/Users/angkunwu/SpinDynamic.jl/data/HaoResults/"
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
        intensities = map_lanczos_results_to_neutron_intensities(qs_path, as, bs, norm2s, energies, η)
        intensities = gaussian_broaden(intensities, energies)
        intensities = PoissonCountIntensity(intensities, T)
        means_all, vars_all, weights_all = Float64[], Float64[], Float64[]
        for k_ind in 1:length(qs_path)
            μ_temp, σ_temp, A_temp, π_temp = fit_gmm_full(energies, intensities[k_ind, :], N=Npts)
            append!(means_all, μ_temp)
            append!(vars_all, σ_temp .^ 2)
            append!(weights_all, π_temp)
        end
        println("Task $(taskid), Job $(jobid)")
        local_rows[jobid] = (Js = copy(Js), means = vec(means_all), variances = vec(vars_all),
                              weights = vec(weights_all), B = copy(B), Bsats = vec(B_sat))
    end
    append!(rows, local_rows)
end

results = DataFrame(rows)
pt = "/Users/angkunwu/SpinDynamic.jl/data/HaoResults/"
CSV.write(pt * "Hao_anisotropy_Js4_Scalar_Poisson_GMM$(Npts)_Bsats.csv", results)