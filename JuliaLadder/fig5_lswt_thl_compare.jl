# Fig. 5 of the manuscript (appfig:compare): unpolarized DSSF S⊥ from LSWT and
# from THL (the `:ladder` resummation of Sunny), and their difference, for the
# parameter set of Fig. 2.
#
#     julia --project=.. -t auto fig5_lswt_thl_compare.jl
#
# from this folder. Writes ../../InverseTriangular/figures/fig5_lswt_thl_compare.pdf.

using CairoMakie, Printf
using LaTeXStrings
#include(joinpath(@__DIR__, "..", "triangular_ladder.jl"))
include("triangular_ladder.jl")

# ---- Settings ----
Js = [1.0, 0.7, 0.2, 0.0, 0.051]         # [J1, J1zz, Jpmpm, Jzpm, J2]
dir = [1.0, 0.0, 0.0]                    # field direction; B = B_sat + |J1|
# q = (h, h, 0): the path and number of q points of the training data
qpath = [[0, 0, 0], [1, 1, 0]]
nq = 60
η = 0.2                                  # Lorentzian HWHM, for both methods
dE = 0.01                                # energy step, ≤ η/2
e_max = 8.0
#outdir = joinpath(@__DIR__, "..", "..", "InverseTriangular", "figures")
outdir = joinpath(@__DIR__, "..", "plots")
    
# One BLAS thread: the matrices are tiny, and threading is over q
Sunny.load_blas_for_threading()
BLAS.set_num_threads(1)

# ---- Compute ----
B, sys, _ = polarized_system(Js, dir)
swt = SpinWaveTheory(sys; measure=dssf_measure(sys))
qs = q_space_path(sys.crystal, qpath, nq).qs
hs = [q[1] for q in qs]
energies = collect(0.0:dE:e_max)
iperp = findfirst(==("perp"), DSSF_CHANNELS)

res = intensities(swt, qs; energies, kernel=lorentzian(; fwhm=2η))
S_lswt = permutedims(stack(res.data), (1, 3, 2))[iperp, :, :]
data, nbreak = ladder_dssf(swt, qs; energies, η)
nbreak > 0 && @warn "Expansion breaks down at $nbreak q points"
S_thl = data[iperp, :, :]
ΔS = S_lswt - S_thl

@printf("B = %.4f\n", B)
@printf("max S⊥: LSWT %.3f, THL %.3f; max|ΔS⊥| = %.3f; ‖ΔS⊥‖/‖S⊥(THL)‖ = %.3f\n",
        maximum(S_lswt), maximum(S_thl), maximum(abs, ΔS), norm(ΔS) / norm(S_thl))

# ---- Plot ----
# LSWT and THL on one color scale, so that their intensities compare directly
smax = max(maximum(S_lswt), maximum(S_thl))
dmax = maximum(abs, ΔS)
panels = [
    (S_lswt, L"S_\perp^\mathrm{LSWT}(\mathrm{\mathbf{q}},\omega)", :viridis, (0, smax)),
    (S_thl, L"S_\perp^\mathrm{THL}(\mathrm{\mathbf{q}},\omega)", :viridis, (0, smax)),
    (ΔS, L"S_\perp^\mathrm{LSWT}-S_\perp^\mathrm{THL}", :bluesreds, (-dmax, dmax)),
]

# Sized for the full text width of a two-column revtex page (in pt), in the
# style of Fig. 2: colorbar on top, then the panel title
fig = with_theme(theme_latexfonts(); fontsize=9) do
    fig = Figure(size=(510, 195), figure_padding=(2, 6, 2, 2))
    for (i, (S, title, colormap, colorrange)) in enumerate(panels)
        ax = Axis(fig[2, i]; title, xlabel=L"(h,h,0)", ylabel=L"\omega/J_1",
                  ylabelvisible=i == 1, yticklabelsvisible=i == 1,
                  xticks=([0, 0.5, 1], [L"0", L"0.5", L"1"]), yticks=0:2:8,
                  titlesize=10, titlegap=2, xticklabelsize=8, yticklabelsize=8)
        hm = heatmap!(ax, hs, energies, S; colormap, colorrange, rasterize=4)
        ylims!(ax, 0, e_max)
        Colorbar(fig[1, i], hm; vertical=false, height=5, ticklabelsize=7, ticksize=2,
                 ticks=WilkinsonTicks(4; k_min=3))
        # APS-style panel label, on its own row above the colorbar
        Label(fig[0, i], "$('a' + i - 1)"; fontsize=10, halign=:left, tellwidth=false)
    end
    rowgap!(fig.layout, 2)
    colgap!(fig.layout, 10)
    fig
end

outfile = joinpath(outdir, "fig5_lswt_thl_compare.pdf")
mkpath(outdir)
save(outfile, fig; pt_per_unit=1)
println("Wrote ", normpath(outfile))
fig
