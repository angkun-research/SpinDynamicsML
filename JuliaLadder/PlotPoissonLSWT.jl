using Sunny
using Plots
using LaTeXStrings
using LinearAlgebra
using Measures
using Random
using Distributions: Poisson

# Reuse Hao's Lanczos and B_sat helpers
include("triangular_ladder.jl")

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


# -----------------------------
# User parameters
# -----------------------------
Js = [1.0, -0.368421, -1.0, 0.0, 0.473684]  # [J1, Delta, Jpmpm, Jzpm, J2]
dir = [1.0, 0.0, 0.0]

fwhm = 0.8
Ts = [10, 100, 1000]
seed = 1234
nqp = 100
qs = [[0, 0, 0], [1, 1, 0]]

Random.seed!(seed)

B_sat, sys, _ = polarized_system(Js, [1,0,0])
path = q_space_path(sys.crystal, qs, nqp)
swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
res = intensities_bands(swt, path)
disp, inten = vec(res.disp), vec(res.data)
energies, scalar1 = broaden_lswt_peaks(disp, inten; fwhm=fwhm)

# Momentum path: q = (h, h, 0).
q = collect(range(0.0, 1.0; length=length(disp)))

# -----------------------------
# Poisson noise and moment extraction
# -----------------------------
noisy_spectra = [
    PoissonCountIntensity(
        scalar1, T;
        rng=MersenneTwister(seed + i),
    )
    for (i, T) in enumerate(Ts)
]

fitted_bands = [
    fit_single_gaussian(energies, spectrum)
    for spectrum in noisy_spectra
]

# Use fixed color ranges across all four panels.
S_range = (
    minimum(minimum.(noisy_spectra)),
    maximum(maximum.(noisy_spectra)),
)

I_range = (
    minimum(minimum(band[2]) for band in fitted_bands),
    maximum(maximum(band[2]) for band in fitted_bands),
)

curve_cmap = cgrad(:magma)

function intensity_color(I, limits, cmap)
    lo, hi = limits
    t = hi == lo ? 0.5 : clamp((I - lo) / (hi - lo), 0.0, 1.0)
    return cmap[t]
end


# -----------------------------
# Four vertically stacked panels
# -----------------------------
panels = Plots.Plot[]

for (i, T) in enumerate(Ts)
    E_fit, I_fit = fitted_bands[i]
    bottom_panel = i == length(Ts)

    p = heatmap(
        q, energies, noisy_spectra[i]';
        color=:viridis,
        clims=S_range,
        colorbar=false, #true,
        colorbar_title=L"S(q,\omega)",
        xlims=(first(q), last(q)),
        ylims=(first(energies), last(energies)),
        xlabel=bottom_panel ? L"\mathbf{q}=(h,h,0)\;\mathrm{(r.l.u.)}" : "",
        ylabel=L"\omega/|J_1|",
        xticks=bottom_panel ? collect(0.0:0.2:1.0) : false,
        framestyle=:box,
        grid=false,
        legend=false,
        title="",
        left_margin=5mm,
        right_margin=5mm,
        top_margin=0mm,
        bottom_margin=0mm,
        guidefontsize=12,
        tickfontsize=10,
    )

    # Clean dispersion as a white dashed reference.
    plot!(
        p, q, disp;
        color=:black,
        linestyle=:solid,
        linewidth=1.5,
        label="True", # Dispersion",
    )

    # Black outline beneath the extracted dispersion.
    plot!(
        p, q, E_fit;
        color=:red,#:black,
        linestyle=:dot, # other styles: :solid, :dash, :dashdot
        linewidth=2,
        label="Extracted", # Dispersion",
    )

    annotate!(
        p,
        first(q) + 0.025 * (last(q) - first(q)),
        last(energies) - 0.07 * (last(energies) - first(energies)),
        text("($(Char('a' + i - 1)))  T = $T", 12, :white, :left),
    )
    # add legend to the first panel only
    if i == 1
        plot!(p, legend=:bottomright,legendfontsize=12)
    end

    push!(panels, p)
end

ptot = plot(
    panels...;
    layout=grid(3, 1),
    link=:x,
    size=(500, 500),
    dpi=300,
    top_margin=0mm,
    bottom_margin=-3.5mm,
)
plot!(ptot; subplot=3, bottom_margin=0mm)

# Shared color scale, drawn as a narrow heatmap.
color_values = collect(range(S_range[1], S_range[2]; length=256))

shared_bar = heatmap(
    [0.0, 1.0],
    color_values,
    hcat(color_values, color_values);
    color=:viridis,
    clims=S_range,
    colorbar=false,
    xlims=(0, 1),
    ylims=S_range,
    xticks=false,
    ymirror=true,
    ylabel=L"S(q,\omega)",
    yguideposition=:right,
    framestyle=:box,
    grid=false,
    legend=false,
    guidefontsize=12,
    tickfontsize=10,
    left_margin=-10mm,
    right_margin=2mm,
    top_margin=0mm,
    bottom_margin=0mm,
)
layout_shared = @layout [panels{0.95w} bar]

fig = plot(
    ptot,
    shared_bar;
    layout=layout_shared,
    size=(600, 500),
    dpi=1000,
)

display(fig)

# -----------------------------
# Save
# -----------------------------
savefig(fig, "../plots/Poisson_LSWT_demo.pdf")


