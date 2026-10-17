using Sunny
using CairoMakie

# Reuse Hao's Lanczos and B_sat helpers
include("../JuliaLadder/triangular_ladder.jl")

# -----------------------------
# User parameters
# -----------------------------
#Js = [1.0, 0.2, 0.1, 0.0, 0.5]   # [J1, Δ, Jpmpm, Jzpm, J2]
#Js = [1.0, 0.7, 0.2, 0.2, 0.051]  # [J1, Delta, Jpmpm, Jzpm, J2]
Js = [1.0, 0.7, 0.2, 0.2, 0.051] 
dir = [1.0, 0.0, 0.0]

η = 0.2                            # Lorentzian broadening (meV)
E_min, E_max = 0.0, 8.0
energy_step = 0.01
energies = collect(E_min:energy_step:E_max)

nq = 12
qpoints = [[0, 0, 0], [1, 1, 0]]

# ------------------------------------------------------------
# Parameter conversion at the existing helper's boundary
# triangular_ladder.jl expects [J1, Jzz, Jpmpm, Jzpm, J2].
# ------------------------------------------------------------
function paper_to_ladder_Js(Js)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    return [J1, J1 * Δ, Jpmpm, Jzpm, J2]
end

# -----------------------------
# Build LSWT result on 12 q-points to match Lanczos
# -----------------------------
function run_lswt(Js, B; dims=(1, 1, 1), nqp=12)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    #units = Units(:meV, :angstrom)

    a = 1 #4.1149
    c = 5 #22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    cryst = Crystal(latvecs, [[0, 0, 0]], 162)

    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst, qs, nqp)  # 12 points along (H,H,0)

    # gab, gcc = 3.13 * 1.09310, 1.01
    # gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    gtens = 1
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten = [J1 + 2Jpmpm 0 Jzpm;
              0 J1 - 2Jpmpm 0;
              Jzpm 0 J1 * Δ]

    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    #set_field!(sys, B * units.T * dir)
    set_field!(sys, B * dir)

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

# ------------------------------------------------------------
# Shared physical system and momentum path
# ------------------------------------------------------------
# Keep the original Js unchanged; convert only for the existing helper.
B_sat, sys = polarized_system(paper_to_ladder_Js(Js), dir)

path = q_space_path(sys.crystal, qpoints, nq)

# polarized_system normally returns an applied field of 1.1 * B_sat.
println("Applied field: $B_sat")
# ------------------------------------------------------------
# LSWT
# ------------------------------------------------------------
swt_lswt = SpinWaveTheory(sys; measure=ssf_perp(sys))
lswt_result = intensities_bands(swt_lswt, path)
lswt_img = lswt_to_heatmap(lswt_result.disp, lswt_result.data, energies; η=η)

println("LSWT image computed.")

# ------------------------------------------------------------
# Ladder
# ------------------------------------------------------------
swt_ladder = SpinWaveTheory(sys; measure=dssf_measure(sys))
data, nbreak = ladder_dssf(
    swt_ladder, path;
    energies,
    η,
    verbose=true,
)
ladder_img = data[7, :, :]  # perpendicular channel

println("Ladder image computed.")
println("Ladder expansion breakdowns: $nbreak / $(size(ladder_img, 1))")

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
    ylims!(ax, E_min, E_max)
    Colorbar(fig[1, 2], hm, label=L"S^{LSWT}_{\perp}(\mathbf{q}, E)")
    text!(ax, 0.03, E_max - 1.0,
          text="Js = $(round.(Js, digits=3)), B = $(round(B_sat, digits=3)) T",
          align=(:left, :top), fontsize=14, color=:white)
    save("../plots/Spin_LSWT.pdf", fig)
end

# Ladder heatmap
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
    hm = heatmap!(ax, hs, energies, ladder_img; colormap=:viridis)
    ylims!(ax, E_min, E_max)
    Colorbar(fig[1, 2], hm, label=L"S^{Ladder}_{\perp}(\mathbf{q}, E)")
    text!(ax, 0.03, E_max - 1.0,
          text="Js = $(round.(Js, digits=3)), B = $(round(B_sat, digits=3)) T",
          align=(:left, :top), fontsize=14, color=:white)
    save("../plots/Spin_Ladder.pdf", fig)
end

println("Saved: Spin_LSWT.pdf and Spin_Ladder.pdf")

diff_img = lswt_img .- ladder_img
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
    ylims!(ax, E_min, E_max)
    Colorbar(fig[1, 2], hm, label=L"\Delta S_{\perp} (\mathrm{LSWT} - \mathrm{Ladder})")
    text!(ax, 0.03, E_max - 1.0,
          text="Js = $(round.(Js, digits=3)), B = $(round(B_sat, digits=3)) T",
          align=(:left, :top), fontsize=14, color=:white)
    save("../plots/Spin_Diff.pdf", fig)
end

println("Saved: Spin_Diff.pdf")
