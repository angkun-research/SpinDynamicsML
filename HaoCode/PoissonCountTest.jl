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

fwhm = 0.8
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

function run_lswt(Js, B; field_dir=[1.0, 0.0, 0.0], dims=(1, 1, 1), nqp=12)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    units = Units(:meV, :angstrom)

    a = 4.1149
    c = 22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    cryst = Crystal(latvecs, [[0, 0, 0]], 162)

    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst, qs, nqp)

    gab, gcc = 3.13 * 1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    sys = System(cryst, [1 => Moment(s=1 / 2, g=gtens)], :dipole; dims=dims)

    J1_ten = [J1 + 2Jpmpm  0             Jzpm;
              0            J1 - 2Jpmpm   0;
              Jzpm         0             J1 * Δ]

    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, B * units.T * field_dir)

    polarize_system!(sys, -field_dir)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    res = intensities_bands(swt, path)
    return (vec(res.disp), vec(res.data), path)
end

function broaden_lswt_peaks(disp, intensities; fwhm=0.8, n_sigma_padding=6.0)
    #fwhm = 0.8
    sigma = fwhm / (2 * sqrt(2 * log(2)))
    energy_step = 0.01
    # n_sigma_padding = 6.0

    energy_min = minimum(disp) - n_sigma_padding * sigma
    energy_max = maximum(disp) + n_sigma_padding * sigma
    energies = collect(energy_min:energy_step:energy_max)
    gauss(x, mu, sigma) = exp(-0.5 * ((x - mu) / sigma)^2) / (sigma * sqrt(2 * pi))
    inten_mat = zeros(length(energies), length(disp))
    for i in eachindex(disp)
        for j in eachindex(energies)
            inten_mat[j, i] = intensities[i] * gauss(energies[j], disp[i], sigma)
        end
    end
    # transpose to match the expected output format
    inten_mat = inten_mat'
    return energies,inten_mat
end

function fit_single_gaussian(energies, intensity_mat)
    dE = energies[2] - energies[1]  # assuming uniform spacing
    weights = intensity_mat #max.(intensity_mat, 0.0)
    # intensity_mat is length(disp) \times length(energies)
    N_p, N_E = size(intensity_mat)
    disp_new = zeros(N_p)
    inten_new = zeros(N_p)
    for i in 1:N_p
        mean_temp = sum(weights[i, :] .* energies) / sum(weights[i, :])
        # variance_temp = sum(weights[i, :] .* (energies .- mean_temp).^2) / sum(weights[i, :])
        # sigma_temp = sqrt(variance_temp)
        disp_new[i] = mean_temp
        total_int = sum(weights[i, :]) * dE
        inten_new[i] = total_int #/ (sigma_temp * sqrt(2 * pi))
    end
    return disp_new, inten_new
end

function fit_gmm_full(ω, S; N=10, n_iter=200, tol=1e-10)
    Δω = ω[2] - ω[1]
    w = S #max.(S, 0.0)
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

function fit_all_qs(energies, intensity_mat)
    N_q, N_E = size(intensity_mat)
    disp_new = zeros(N_q)
    inten_new = zeros(N_q)
    for i in 1:N_q
        μ, σ, A, π_k = fit_gmm_full(energies, intensity_mat[i, :]; N=1)
        disp_new[i] = μ[1]
        inten_new[i] = A[1]
    end
    return disp_new, inten_new
end

B_sat, _ = determine_B_sat(Js, dir)
B = B_sat * 1.1

# B1, qs1, scalar1 = run_lanczos_dssf(Js, dir, clustersize, energies, eta; B=B)
# # Instrumental Gaussian broadening
# scalar1 = gaussian_broaden(scalar1, energies; fwhm = fwhm)

disp, inten, path = run_lswt(Js, B)
energies, scalar1 = broaden_lswt_peaks(disp, inten; fwhm=fwhm)
#inten_mat_fitted_disp, inten_mat_fitted_inten = fit_single_gaussian(disp, scalar1)

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

disp_new, inten_new = fit_single_gaussian(energies, scalar_poisson_1)
#disp_new, inten_new = fit_all_qs(energies, scalar_poisson_1) # same as fit_single_gaussian for N=1
plot(disp; label = "Original Scalar Intensity", linewidth = 2)
plot!(disp_new; label = "Fitted Gaussian", linewidth = 2)

plot([inten inten_new])

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
