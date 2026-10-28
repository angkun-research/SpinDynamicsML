# Fig. 2 of the manuscript (fig:distinguishable): DSSF channels of J and the
# difference J′ − J, where J′ only flips the sign of Jzpm. Computed with THL,
# the `:ladder` resummation of Sunny.
#
#     julia --project=.. -t auto fig2_distinguishable.jl
#
# from this folder. Writes ../../InverseTriangular/figures/fig2_distinguishable.pdf.

using CairoMakie, Printf
using LaTeXStrings
#include(joinpath(@__DIR__, "..", "triangular_ladder.jl"))
include("triangular_ladder.jl")

# ---- Settings ----
Js1 = [1.0, 0.7, 0.2, 0.2, 0.051]        # J = [J1, J1zz, Jpmpm, Jzpm, J2]
Js2 = [1.0, 0.7, 0.2, -0.2, 0.051]       # J′: same, with Jzpm → -Jzpm
dir = [1.0, 0.0, 0.0]                    # field direction; B = B_sat + |J1| of Js1
# q = (h, h, 0): the path and number of q points of the training data
qpath = [[0, 0, 0], [1, 1, 0]]
nq = 60
η = 0.2                                  # Lorentzian HWHM
dE = 0.01                                # energy step, ≤ η/2
e_max = 8.0
#outfile = joinpath(@__DIR__, "..", "..", "InverseTriangular", "figures", "fig2_distinguishable.pdf")
outfile = joinpath(@__DIR__, "..", "plots", "fig2_distinguishable.pdf")

# One BLAS thread: the matrices are tiny, and threading is over q
Sunny.load_blas_for_threading()
BLAS.set_num_threads(1)

# ---- Compute ----
# Both Hamiltonians at the field of Js1
B, sys1, _ = polarized_system(Js1, dir)
_, sys2, _ = polarized_system(Js2, dir; B)
qs = q_space_path(sys1.crystal, qpath, nq).qs
hs = [q[1] for q in qs]
energies = collect(0.0:dE:e_max)

function channels(sys)
    swt = SpinWaveTheory(sys; measure=dssf_measure(sys))
    data, nbreak = ladder_dssf(swt, qs; energies, η)
    nbreak > 0 && @warn "Expansion breaks down at $nbreak q points"
    return data
end
data1 = channels(sys1)
data2 = channels(sys2)

# ΔS = S(J′) − S(J). Panels, as indices into DSSF_CHANNELS: perp, yy, zz, zx+xz
cols = [7, 2, 3, 6]
# Upright bold J, as \mathbf{J} in the manuscript (MathTeXEngine's \mathbf alone is bold italic)
titles1 = [L"S_\perp(\mathrm{\mathbf{J}})", L"S_{yy}(\mathrm{\mathbf{J}})", L"S_{zz}(\mathrm{\mathbf{J}})", L"S_{zy}+S_{yz}\,(\mathrm{\mathbf{J}})"]
titles2 = [L"\Delta S_\perp", L"\Delta S_{yy}", L"\Delta S_{zz}", L"\Delta(S_{zy}+S_{yz})"]
S1 = [data1[c, :, :] for c in cols]
ΔS = [data2[c, :, :] - data1[c, :, :] for c in cols]
# Relative L2 difference over the (q, ω) grid
rel = [norm(d) / norm(s) for (d, s) in zip(ΔS, S1)]

@printf("B = %.4f\n", B)
for (c, r, d) in zip(cols, rel, ΔS)
    @printf("%-6s  max|ΔS| = %.2e   ‖ΔS‖/‖S‖ = %.3f\n", DSSF_CHANNELS[c], maximum(abs, d), r)
end

# ---- Plot ----
# Colorbar tick labels in LaTeX, compact
sci_ticks(vals) = map(vals) do v
    iszero(v) && return L"0"
    s = @sprintf("%.2g", v)
    if occursin('e', s)
        a, b = split(s, 'e')
        latexstring(a == "1" ? "" : "$a\\times", "10^{$(parse(Int, b))}")
    else
        latexstring(s)
    end
end

# Sized for the full text width of a two-column revtex page (in pt)
fig = with_theme(theme_latexfonts(); fontsize=9) do
    fig = Figure(size=(510, 300), figure_padding=(2, 6, 2, 2))
    # One color scale for the whole difference row, so that zero looks zero
    dmax = maximum(d -> maximum(abs, d), ΔS)
    for (i, c) in enumerate(cols)
        signed = c ≥ 4 && c ≤ 6
        m = max(maximum(abs, S1[i]), 1e-12)
        common = (; xticks=([0, 0.5, 1], [L"0", L"0.5", L"1"]), yticks=0:2:8,
                    ylabel=L"\omega/J_1", ylabelvisible=i == 1, yticklabelsvisible=i == 1,
                    titlesize=10, titlegap=2, xticklabelsize=8, yticklabelsize=8)

        ax1 = Axis(fig[2, i]; title=titles1[i], xticklabelsvisible=false, common...)
        hm1 = heatmap!(ax1, hs, energies, S1[i]; colormap=signed ? :bluesreds : :viridis,
                       colorrange=signed ? (-m, m) : (0, m), rasterize=4)
        Colorbar(fig[1, i], hm1; vertical=false, height=5, ticklabelsize=7, ticksize=2,
                 ticks=WilkinsonTicks(3; k_min=3), tickformat=sci_ticks)

        ax2 = Axis(fig[4, i]; title=titles2[i], xlabel=L"(h,h,0)", common...)
        hm2 = heatmap!(ax2, hs, energies, ΔS[i]; colormap=:bluesreds,
                       colorrange=(-dmax, dmax), rasterize=4)
        Colorbar(fig[3, i], hm2; vertical=false, height=5, ticklabelsize=7, ticksize=2,
                 ticks=WilkinsonTicks(3; k_min=3), tickformat=sci_ticks)
        text!(ax2, 0.04, 7.6; text=latexstring(@sprintf("\\|\\Delta S\\|/\\|S\\| = %.2f", rel[i])),
              align=(:left, :top), fontsize=8)

        for ax in (ax1, ax2)
            ylims!(ax, 0, e_max)
        end
    end
    rowgap!(fig.layout, 3)
    rowgap!(fig.layout, 2, 6)
    colgap!(fig.layout, 8)
    fig
end

mkpath(dirname(outfile))
save(outfile, fig; pt_per_unit=1)
println("Wrote ", normpath(outfile))
fig
