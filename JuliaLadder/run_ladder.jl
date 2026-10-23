# One parameter set with `:ladder`: compute, time and plot the DSSF channels.
#
#     julia --project=. -t auto run_job.jl
#
# or `include("run_job.jl")` from a REPL started with `--project=ladder -t auto`
# to see the figure. Edit the settings below. The figure is also written to
# figures/run_job.png.

using CairoMakie, Printf
using LaTeXStrings
include(joinpath(@__DIR__, "triangular_ladder.jl"))

# ---- Settings ----
# The parameter sets below corresponds to Fig. 2 of the manuscript
Js = [1.0, 0.7, 0.2, 0.2, 0.051]          # [J1, J1zz, Jpmpm, Jzpm, J2]
dir = [1.0, 0.0, 0.0]                    # field direction; B = 1.1 B_sat
qpath = [[0, 0, 0],[1, 1, 0]] #[[0, 0, 0], [1/3, 1/3, 0], [1/2, 1/2, 0], [1, 1, 0]]
labels = nothing                         # automatic coordinate labels; or one string per qpath point
nq = 12                                  # q points along the path (approximate)
η = 0.2                                  # Lorentzian HWHM
dE = 0.01                                # energy step, ≤ η/2

# One BLAS thread: the matrices are tiny, and threading is over q
Sunny.load_blas_for_threading()
BLAS.set_num_threads(1)

# ---- Compute ----
t_B = @elapsed B, sys, sat = polarized_system(Js, dir)
swt = SpinWaveTheory(sys; measure=dssf_measure(sys))
path = q_space_path(sys.crystal, qpath, nq; labels)
energies = collect(0.0:dE:ceil(energy_max(swt)))

# Compile on two q points, so that the timing below is the run time
t_compile = @elapsed ladder_dssf(swt, path.qs[1:2]; energies, η)
t_ladder = @elapsed data, nbreak = ladder_dssf(swt, path; energies, η, verbose=true)
disp = dispersion(swt, path)

nq_path = length(path.qs)
println()
@printf("Js = %s, B = %.4f (B_sat = %.4f)\n", string(Js), B, sat.B_sat)
@printf("%d q × %d energies, %d threads; breakdown at %d q\n", nq_path, length(energies), Threads.nthreads(), nbreak)
@printf("B_sat bisection   %6.2f s  (mostly compilation; about 0.02 s when warm)\n", t_B)
@printf("compilation       %6.2f s  (first call, 2 q)\n", t_compile)
@printf(":ladder           %6.2f s  (%.1f ms per q)\n", t_ladder, 1000t_ladder / nq_path)

# ---- Plot ----
xs = 1:nq_path
xticks = path.xticks
fig = with_theme(theme_latexfonts()) do
    fig = Figure(size=(1800, 950))
    for (c, ch) in enumerate(DSSF_CHANNELS)
        S = data[c, :, :]
        m = max(maximum(abs, filter(isfinite, S)), 1e-12)
        diag = c ≤ 3 || c == 7
        ax = Axis(fig[2 * ((c - 1) ÷ 4) + 1, mod1(c, 4)]; title=ch, xticks, ylabel=L"E/J",)
        hm = heatmap!(ax, xs, energies, S; colormap=diag ? :viridis : :bluesreds,
                    colorrange=diag ? (0, 0.5m) : (-0.5m, 0.5m))
        lines!(ax, xs, vec(disp); color=(:white, 0.6), linestyle=:dash, linewidth=1)
        Colorbar(fig[2 * ((c - 1) ÷ 4) + 2, mod1(c, 4)], hm; vertical=false)
        ylims!(ax, 0, 8)
    end
    Label(fig[0, :], "dyson = :ladder,  Js = $Js,  B = $(round(B; digits=3)) ∥ $dir,  η = $η   (dashed: LSWT)"; fontsize=25)
    fig
end
# mkpath(joinpath(@__DIR__, "figures"))
# save(joinpath(@__DIR__, "figures", "run_job.png"), fig)
fig
println("success")