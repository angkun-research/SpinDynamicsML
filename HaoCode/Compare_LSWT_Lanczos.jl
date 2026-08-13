using Sunny
using CairoMakie

# Reuse Hao's Lanczos and B_sat helpers
include("../HaoCode/triangular_nlswt.jl")

# -----------------------------
# User parameters
# -----------------------------
#Js = [1.0, 0.2, 0.1, 0.0, 0.5]   # [J1, Δ, Jpmpm, Jzpm, J2]
Js = [1.0, 0.7, 0.2, 0.2, 0.051]  # [J1, Delta, Jpmpm, Jzpm, J2]
dir = [1.0, 0.0, 0.0]

clustersize = (11, 11, 1)        # Hao's Lanczos cluster
η = 0.2                            # Lorentzian broadening (meV)
energy_min, energy_max = 0.0, 8.0
energy_step = 0.01
energies = collect(energy_min:energy_step:energy_max)

# -----------------------------
# Build LSWT result on 12 q-points to match Lanczos
# -----------------------------
function run_lswt(Js, B; dims=(1, 1, 1), nqp=12)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    units = Units(:meV, :angstrom)

    a = 4.1149
    c = 22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    cryst = Crystal(latvecs, [[0, 0, 0]], 162)

    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst, qs, nqp)  # 12 points along (H,H,0)

    gab, gcc = 3.13 * 1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten = [J1 + 2Jpmpm 0 Jzpm;
              0 J1 - 2Jpmpm 0;
              Jzpm 0 J1 * Δ]

    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, B * units.T * dir)

    polarize_system!(sys, -dir)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    return intensities_bands(swt, path)
end

# Convert LSWT discrete bands/intensities to S(q, ω) heatmap
function lswt_to_heatmap(disp, data, energies; η=0.2)
    nbands, nq = size(disp)
    img = zeros(Float64, nq, length(energies))
    normalization = 1.0 / (sqrt(2π) * η)
    for iq in 1:nq
        for ib in 1:nbands
            E0 = disp[ib, iq]
            A = data[ib, iq]
            #img[iq, :] .+= A .* (η / π) ./ ((energies .- E0).^2 .+ η^2)
            img[iq, :] .+= A .* normalization .* exp.(-0.5 * ((energies .- E0) ./ η).^2)
        end
    end
    return img
end

# -----------------------------
# Compute B_sat once (shared physics scale)
# -----------------------------

# Lanczos (Hao)
B_sat, qs_path, as_path, bs_path, norm2s_path =
    triangular_nlswt_lanczos_calculations(Js, clustersize, copy(dir))
println("Lanczos B_sat: $B_sat T")
lanczos_img = map_lanczos_results_to_neutron_intensities(
    qs_path, as_path, bs_path, norm2s_path, energies, η
)
println("Lanczos image computed.")
# LSWT
lswt_res = run_lswt(Js, B_sat; nqp=12)
lswt_img = lswt_to_heatmap(lswt_res.disp, lswt_res.data, energies; η=η)
println("LSWT image computed.")
# -----------------------------
# Plot and save
# -----------------------------
# For 12 momentum bins, use 13 bin edges in x
hs = [i / 12 for i in 0:12]

# LSWT heatmap
with_theme(theme_latexfonts()) do
    fig = Figure(size=(760, 620))
    ax = Axis(
        fig[1, 1],
        xlabel=L"(H,H,0)",
        ylabel="E (meV)",
        #title="Sunny LSWT: S(q, \omega)",
        xlabelsize=24,
        ylabelsize=24,
        titlesize=24,
        xticklabelsize=18,
        yticklabelsize=18
    )
    hm = heatmap!(ax, hs, energies, lswt_img; colormap=:viridis)
    ylims!(ax, energy_min, energy_max)
    Colorbar(fig[1, 2], hm, label=L"S^{LSWT}_{\perp}(\mathbf{q}, E)")
    text!(ax, 0.03, energy_max - 1.0,
          text="Js = $(round.(Js, digits=3)), B = $(round(B_sat, digits=3)) T",
          align=(:left, :top), fontsize=14, color=:white)
    save("./plots/Spin_LSWT.pdf", fig)
end

# Lanczos heatmap
with_theme(theme_latexfonts()) do
    fig = Figure(size=(760, 620))
    ax = Axis(
        fig[1, 1],
        xlabel=L"(H,H,0)",
        ylabel=L"E (\mathrm{meV})",
        #title="Hao Lanczos: S(q, \omega)",
        xlabelsize=24,
        ylabelsize=24,
        titlesize=24,
        xticklabelsize=18,
        yticklabelsize=18
    )
    hm = heatmap!(ax, hs, energies, lanczos_img; colormap=:viridis)
    ylims!(ax, energy_min, energy_max)
    Colorbar(fig[1, 2], hm, label=L"S^{TED}_{\perp}(\mathbf{q}, E)")
    text!(ax, 0.03, energy_max - 1.0,
          text="Js = $(round.(Js, digits=3)), B = $(round(B_sat, digits=3)) T",
          align=(:left, :top), fontsize=14, color=:white)
    save("./plots/Spin_Lanczos.pdf", fig)
end

println("Saved: Spin_LSWT.pdf and Spin_Lanczos.pdf")

diff_img = lswt_img .- lanczos_img
diff_absmax = maximum(abs, diff_img)
diff_absmax = diff_absmax > 0 ? diff_absmax : 1e-12

# Difference heatmap (LSWT - Lanczos)
with_theme(theme_latexfonts()) do
    fig = Figure(size=(760, 620))
    ax = Axis(
        fig[1, 1],
        xlabel=L"(H,H,0)",
        ylabel=L"E (\mathrm{meV})",
        xlabelsize=24,
        ylabelsize=24,
        titlesize=24,
        xticklabelsize=18,
        yticklabelsize=18
    )
    hm = heatmap!(
        ax, hs, energies, diff_img;
        colormap=:balance,
        colorrange=(-diff_absmax, diff_absmax)
    )
    ylims!(ax, energy_min, energy_max)
    Colorbar(fig[1, 2], hm, label=L"\Delta S_{\perp} (\mathrm{LSWT} - \mathrm{Lanczos})")
    text!(ax, 0.03, energy_max - 1.0,
          text="Js = $(round.(Js, digits=3)), B = $(round(B_sat, digits=3)) T",
          align=(:left, :top), fontsize=14, color=:white)
    save("./plots/Spin_Diff.pdf", fig)
end

println("Saved: Spin_Diff.pdf")
