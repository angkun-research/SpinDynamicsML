include("triangular_nlswt.jl")
# Example usage
# Js = [2rand()-1, rand(), 2rand()-1, 2rand()-1, 2rand()-1]
Js = [ -0.13845397277671623
  0.2722621890898467
 -0.2158110342079964
  0.9811099998016632
 -0.5413876779816684]
# Js = [2rand()-1, rand(), 0, 0, 2rand()-1]
clustersize = (11, 11, 1)
B, qs_path, as_path, bs_path, norm2s_path = triangular_nlswt_lanczos_calculations(Js, clustersize)
energy_min, energy_max = 0.0, 20.0
energy_step = 0.01
η = 0.25
energies = collect(energy_min:energy_step:energy_max)
intensities = map_lanczos_results_to_neutron_intensities(qs_path, as_path, bs_path, norm2s_path, energies, η)

using CairoMakie
# Visualization
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 600))
    ax  = Axis(fig[1,1]; xlabel=L"(H,H,0)", ylabel="Energy (meV)", title="Intensity Heatmap", xlabelsize=30, ylabelsize=30, titlesize=30, xticklabelsize=25, yticklabelsize=25)
    heatmap!(ax, hs, energies, intensities; colormap=:viridis, colorrange=(0, 2.5))
    text!(ax, 0.05, 15, text="J1, Δ, J1pmpm, J1zpm, J2 = $(round.(Js, digits=3))", align = (:left, :top), fontsize=16, color=:white)
    text!(ax, 0.05, 13, text="B = $(round(B, digits=3)) T", align = (:left, :top), fontsize=16, color=:white)
    ylims!(ax, 0, 20)
    ax2 = Axis(fig[1,2]; xlabel="Energy (meV)", ylabel="Intensity", title="Intensity Cut", xlabelsize=30, ylabelsize=30, titlesize=30, xticklabelsize=25, yticklabelsize=25)
    lines!(ax2, energies, intensities[1, :], label=L"\Gamma")
    lines!(ax2, energies, intensities[6, :], label=L"M")
    axislegend(ax2; labelsize=18)
    xlims!(ax2, 0, 20)
    fig
end