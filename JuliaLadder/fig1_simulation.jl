using CairoMakie, LaTeXStrings
include(joinpath(@__DIR__, "triangular_ladder.jl"))

Js1 = [1.0, 0.2, 0.4, 0.0, 0.5] #[1.0, 0.2, 0.4, 0.4, 0.5]
dir = [1.0, 0.0, 0.0]
qpath = [[0, 0, 0], [1, 1, 0]]
nq = 60
η = 0.2
dE = 0.01
e_max = 10.0
col = 7

Sunny.load_blas_for_threading()
BLAS.set_num_threads(1)

B, sys1, _ = polarized_system(Js1, dir)
qs = q_space_path(sys1.crystal, qpath, nq).qs
hs = [q[1] for q in qs]
energies = collect(0.0:dE:e_max)

swt = SpinWaveTheory(sys1; measure=dssf_measure(sys1))
data1, nbreak = ladder_dssf(swt, qs; energies, η)
nbreak > 0 && @warn "Expansion breaks down at $nbreak q points"
Sperp = data1[col, :, :]

fig = with_theme(theme_latexfonts(); fontsize=12) do
    fig = Figure(size=(340, 300))
    ax = Axis(fig[1, 1];
        title=L"S_\perp(\mathrm{\mathbf{J}})",
        xlabel=L"(h,h,0)",
        ylabel=L"\omega/J_1",
        xticks=[0, 0.5, 1],
        yticks=0:2:8,
    )
    hm = heatmap!(ax, hs, energies, Sperp;
        colormap=:viridis,
        colorrange=(0, max(maximum(abs, Sperp), 1e-12)),
        rasterize=4,
    )
    Colorbar(fig[1, 2], hm)
    ylims!(ax, 0, e_max)
    fig
end

outfile = joinpath(@__DIR__, "..", "plots", "fig1_simulation.pdf")
mkpath(dirname(outfile))
save(outfile, fig; pt_per_unit=1)
fig
