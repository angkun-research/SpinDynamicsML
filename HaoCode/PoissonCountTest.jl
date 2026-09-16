using Sunny
using Plots
using LaTeXStrings
using LinearAlgebra
using Measures
using Random
using Distributions: Poisson

# Reuse Hao's Lanczos and B_sat helpers
include("../HaoCode/triangular_nlswt.jl")
include("../HaoCode/UtilDiscretizations.jl")

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
        # λi = intensities[i] * T
        # λbg = bgrate * T
        # counts[i] = λi > 0 ? rand(rng, Poisson(λi)) : 0
        # counts[i] += λbg > 0 ? rand(rng, Poisson(λbg)) : 0
        # Two independent Poisson processes is also Poisson
        λ = (intensities[i] + bgrate) * T
        counts[i] = λ > 0 ? rand(rng, Poisson(λ)) : 0
    end
    # s' = (n_s+n_{bg})/T - bg,  
    # E(s') = E(n_s+ n_{bg})/T - bg = s, 
    # Var(s') = (s+bg)T/T^2 = (s+bg)/T
    ints_new = counts / T .- bgrate
    return ints_new # New noisy, binned intensities
end

# -----------------------------
# User parameters
# -----------------------------
Js = [1.0, -0.368421, -1.0, 0.0, 0.473684]  # [J1, Delta, Jpmpm, Jzpm, J2]
dir = [1.0, 0.0, 0.0]

clustersize = (11, 11, 1)
eta = 0.1
energy_min, energy_max = 0.0, 15.0
energy_step = 0.01
energies = collect(energy_min:energy_step:energy_max)

# -----------------------------
# Compute Lanczos DSSF
# -----------------------------
function run_lanczos_dssf(Js, dir, clustersize, energies, eta; B=nothing)
    res = triangular_nlswt_lanczos_calculations(Js, clustersize, copy(dir); B=B)
    isnothing(res) && error("Lanczos calculation failed for Js=$(Js).")
    B_used, qs_path, as_path, bs_path, norm2s_path = res
    scalar_img = map_lanczos_results_to_neutron_intensities(
        qs_path, as_path, bs_path, norm2s_path, energies, eta
    )
    return B_used, qs_path, scalar_img
end

B_sat, _ = determine_B_sat(Js, dir)
B = B_sat * 1.1
B1, qs1, scalar1 = run_lanczos_dssf(Js, dir, clustersize, energies, eta; B=B)
# Instrumental Gaussian broadening
fwhm = 0.8
scalar1 = gaussian_broaden(scalar1, energies; fwhm = fwhm)


T = 100
scalar_poisson_1 = PoissonCountIntensity(scalar1, T)
T = 1000
scalar_poisson_2 = PoissonCountIntensity(scalar1, T)
T = 100000
scalar_poisson_3 = PoissonCountIntensity(scalar1, T)

# plot heatmaps
heatmap(scalar1'; title = "Original Scalar Intensity", color = :viridis)
heatmap(scalar_poisson_1'; title = "Poisson Count Intensity (T=100)", color = :viridis)
heatmap(scalar_poisson_2'; title = "Poisson Count Intensity (T=1000)", color = :viridis)
heatmap(scalar_poisson_3'; title = "Poisson Count Intensity (T=100000)", color = :viridis)


# plot curve
plot(energies, scalar1[1, :]; label = "Original Scalar Intensity", linewidth = 2)
scatter!(energies, scalar_poisson_1[1, :]; label = "Poisson Count (T=100)", linewidth = 2)
scatter!(energies, scalar_poisson_2[1, :]; label = "Poisson Count (T=1000)", linewidth = 2)
scatter!(energies, scalar_poisson_3[1, :]; label = "Poisson Count (T=100000)", linewidth = 2)

S_full = scalar_poisson_2[1, :]   # Γ-point curve, same one used in Plot_interpolation.jl

N_knots = 20
E_adapt, S_adapt, _ = adaptive_knots(energies, S_full; N=N_knots)
μ_gmm, σ_gmm, A_gmm, π_gmm = fit_gmm_full(energies, S_full; N=N_knots)
S_gmm_recon = reconstruct_gmm(energies, μ_gmm, σ_gmm, A_gmm)

# -----------------------------
# Plot and save
# -----------------------------
p1 = plot(energies, S_full;
    label = "Full", color = :royalblue, linewidth = 3,
    title = "Adaptive knots (greedy interpolation)",
    xlabel = L"\omega\ \mathrm{(meV)}", ylabel = L"I(\mathbf{k}=\Gamma, \omega)",
    xlims = (0, 15), legend = :topright, guidefontsize = 14, titlefontsize = 12)
scatter!(p1, E_adapt, S_adapt; label = "Knots (N=$N_knots)", color = :crimson, markersize = 4)

p2 = plot(energies, S_full;
    label = "Full", color = :royalblue, linewidth = 3,
    title = "Gaussian mixture (flexible weight π_k=1/N)",
    xlabel = L"\omega\ \mathrm{(meV)}", ylabel = L"I(\mathbf{k}=\Gamma, \omega)",
    xlims = (0, 15), legend = :topright, guidefontsize = 14, titlefontsize = 12)
plot!(p2, energies, S_gmm_recon; label = "GMM reconstruction", color = :darkorange, linewidth = 2, linestyle = :dash)
scatter!(p2, μ_gmm, A_gmm ./ (σ_gmm .* sqrt(2π)); label = "Component means (N=$N_knots)", color = :darkorange, markersize = 4)

fig = plot(p1, p2; layout = (1, 2), size = (900, 400),margin = 6mm)

# savefig(fig, "./plots/GMM_vs_AdaptiveKnots.pdf")
# println("Saved: ./plots/GMM_vs_AdaptiveKnots.pdf")

function plot_poisson_discretizations(energies, int_curve, T; N_knots = 20)
    intensity_T = PoissonCountIntensity(int_curve, T)
    S_full = intensity_T   # Γ-point curve, same one used in Plot_interpolation.jl
    E_adapt, S_adapt, _ = adaptive_knots(energies, S_full; N=N_knots)
    μ_gmm, σ_gmm, A_gmm, π_gmm = fit_gmm_full(energies, S_full; N=N_knots)
    S_gmm_recon = reconstruct_gmm(energies, μ_gmm, σ_gmm, A_gmm)

    p1 = plot(energies, S_full;
    label = "Full", color = :royalblue, linewidth = 3,
    title = "Linear, T="*"$T",
    xlabel = L"\omega\ \mathrm{(meV)}", ylabel = L"I(\mathbf{k}=\Gamma, \omega)",
    xlims = (0, 15), legend = :topright, guidefontsize = 14, titlefontsize = 12)
    scatter!(p1, E_adapt, S_adapt; label = "Knots (N=$N_knots)", color = :crimson, markersize = 4)

    p2 = plot(energies, S_full;
    label = "Full", color = :royalblue, linewidth = 3,
    title = "GMM",
    xlabel = L"\omega\ \mathrm{(meV)}", ylabel = L"I(\mathbf{k}=\Gamma, \omega)",
    xlims = (0, 15), legend = :topright, guidefontsize = 14, titlefontsize = 12)
    plot!(p2, energies, S_gmm_recon; label = "GMM reconstruction", color = :darkorange, linewidth = 2, linestyle = :dash)
    scatter!(p2, μ_gmm, A_gmm ./ (σ_gmm .* sqrt(2π)); label = "Component means (N=$N_knots)", color = :darkorange, markersize = 4)
    return p1, p2
end

Ts = [50, 100, 1000] #[100, 1000, 1000000]
ps_linear = []
ps_gmm = []
for T in Ts
    p1, p2 = plot_poisson_discretizations(energies, scalar1[1,:], T; N_knots = 20)
    push!(ps_linear, p1)
    push!(ps_gmm, p2)
end

fig = plot(ps_linear..., ps_gmm...; layout = (2, length(Ts)), size = (1200, 600), margin = 5mm)
