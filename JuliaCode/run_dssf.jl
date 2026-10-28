using CairoMakie
using LaTeXStrings
include("triangular_nlswt.jl")

Js_true1 = [-0.14914677, 0.21312027, 0.3567458, 0.92679834, 0.6190808]
Js_pred1 = [-0.20039411, 0.6151507, 0.43861523, -0.7318158, 0.59355915]

# Js_true2 = [-0.22519903, 0.46690437, -0.6312011, 0.83504575, -0.65450096]
# Js_pred2 = [-0.18858238, 0.5688328, -0.64751595, 0.72200924, -0.57967013]
Js_true2 = [0.26805058, 0.89494973, -0.01945474, -0.4242556, 0.12087087]
#Js_pred2 = [0.28811148, 0.727085, -0.04912354, -0.44243002, 0.12818626]
Js_pred2 = [0.28811148, 0.727085, -0.04912354, -0.4242556, 0.12818626]

Js_true3 = [0.1415774, 0.07344072, -0.8469103, -0.5265461, 0.7317381]
Js_pred3 = [0.09587734, 0.25093865, -0.8052857, -0.07487219, 0.7793585]

Js_true = copy(Js_true2)
Js_pred = copy(Js_pred2)
sample_id = 2

clustersize = (11, 11, 1)

dir = [1.0, 0.0, 0.0]
# dir = [√3/2, 1/2, 0]
θ = rad2deg(atan(dir[2], dir[1]))

B, qs_path, as_path, bs_path, norm2s_path = triangular_nlswt_lanczos_calculations(Js_true, clustersize, dir)
energy_min, energy_max = 0.0, 20.0
energy_step = 0.01
η = 0.1
energies = collect(energy_min:energy_step:energy_max)
dssf = map_lanczos_results_to_dssf_tensor(qs_path, as_path, bs_path, norm2s_path, energies, η)

B, qs_path, as_path_pred, bs_path_pred, norm2s_path_pred = triangular_nlswt_lanczos_calculations(Js_pred, clustersize, dir)#; B=B)
energy_min, energy_max = 0.0, 20.0
energy_step = 0.01
η = 0.1
energies = collect(energy_min:energy_step:energy_max)
dssf_pred = map_lanczos_results_to_dssf_tensor(qs_path, as_path_pred, bs_path_pred, norm2s_path_pred, energies, η)

dssf_diff = dssf - dssf_pred
norms = zeros(6)
for i in 1:6
    norms[i] = norm(dssf[i, :, :] - dssf_pred[i, :, :])
end

hs = [i/12 for i in 0:12]

channels_ref = latexstring.(["S_{xx} \\, \\text{(Ref)}", "S_{yy} \\, \\text{(Ref)}", "S_{zz} \\, \\text{(Ref)}", "S_{xy}+S_{yx} \\, \\text{(Ref)}", "S_{yz}+S_{zy} \\, \\text{(Ref)}", "S_{zx}+S_{xz} \\, \\text{(Ref)}"])
channels_pred = latexstring.(["S_{xx} \\, \\text{(Pred)}", "S_{yy} \\, \\text{(Pred)}", "S_{zz} \\, \\text{(Pred)}", "S_{xy}+S_{yx} \\, \\text{(Pred)}", "S_{yz}+S_{zy} \\, \\text{(Pred)}", "S_{zx}+S_{xz} \\, \\text{(Pred)}"])
channels_diff = latexstring.(["S_{xx} \\, \\text{(Diff)}", "S_{yy} \\, \\text{(Diff)}", "S_{zz} \\, \\text{(Diff)}", "S_{xy}+S_{yx} \\, \\text{(Diff)}", "S_{yz}+S_{zy} \\, \\text{(Diff)}", "S_{zx}+S_{xz} \\, \\text{(Diff)}"])
# Plot the reference results
labelsize = 50
ticklabelsize = 35
with_theme(theme_latexfonts()) do
    fig = Figure(size=(4500, 3500))
    for i in 1:6
        ax = Axis(fig[1, i]; xlabel=L"(h,h,0)", ylabel="Energy (meV)", title="$(channels_ref[i])", xlabelsize=labelsize, ylabelsize=labelsize, titlesize=labelsize, xticklabelsize=ticklabelsize, yticklabelsize=ticklabelsize)
        cm = heatmap!(ax, hs, energies, dssf[i, :, :]; colormap=:viridis)
        Colorbar(fig[0, i], cm, ticklabelsize=ticklabelsize, vertical=false)
        ylims!(ax, 0, 20)

        ax2 = Axis(fig[3, i]; xlabel=L"(h,h,0)", ylabel="Energy (meV)", title="$(channels_pred[i])", xlabelsize=labelsize, ylabelsize=labelsize, titlesize=labelsize, xticklabelsize=ticklabelsize, yticklabelsize=ticklabelsize)
        cm2 = heatmap!(ax2, hs, energies, dssf_pred[i, :, :]; colormap=:viridis)
        Colorbar(fig[2, i], cm2, ticklabelsize=ticklabelsize, vertical=false)
        ylims!(ax2, 0, 20)

        ax3 = Axis(fig[5, i]; xlabel=L"(h,h,0)", ylabel="Energy (meV)", title="$(channels_diff[i])", xlabelsize=labelsize, ylabelsize=labelsize, titlesize=labelsize, xticklabelsize=ticklabelsize, yticklabelsize=ticklabelsize)
        cm3 = heatmap!(ax3, hs, energies, dssf_diff[i, :, :]; colormap=:bluesreds, colorrange=(-1, 1))
        Colorbar(fig[4, i], cm3, ticklabelsize=ticklabelsize, vertical=false)
        ylims!(ax3, 0, 20)
        text!(ax3, 0.05, 15, text="L2 Norm = $(round(norms[i], digits=3))", align = (:left, :top), fontsize=labelsize, color=:black)
    end
    fig
    #save(joinpath(homedir(), "Desktop", "sample$(sample_id)_B||$(round(θ, digits=0))deg_dssf_2.png"), fig)
    save("./plots/Spin_LSWT_full_tensor.pdf", fig)
end