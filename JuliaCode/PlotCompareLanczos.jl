include("triangular_nlswt.jl")

nsample = 2
Js = zeros(5, nsample)
# Js[:, 1] = [-0.31751448, 0.37577713, 0.03958666, 0.56317496, -0.9516833]
# Js[:, 2] = [-0.36470276, 0.31307775, -0.00717631, 0.22353317, -0.9270824]
# Js[:, 1] = [-0.28523913, 0.23764205, -0.00544375, 0.6276385, -0.8542301]
# Js[:, 2] = [-0.32080898, 0.17617194, -0.03999586, 0.4133444, -0.849618]
# Js[:, 1] = [-0.25998485, 0.64280397, -0.82634705, -0.54688674, -0.48267987]
# Js[:, 2] = [-0.20252854, 0.73818296, -0.83325434, -0.12743643, -0.57101625]

# Js[:, 1] = [-0.14914677, 0.21312027, 0.3567458, 0.92679834, 0.6190808]
# Js[:, 2] = [-0.20039411, 0.6151507, 0.43861523, -0.7318158, 0.59355915]
# Js[:, 1] = [-0.22519903, 0.46690437, -0.6312011, 0.83504575, -0.65450096]
# Js[:, 2] = [-0.18858238, 0.5688328, -0.64751595, 0.72200924, -0.57967013]
# Js[:, 1] = [0.14157740, 0.07344072, -0.8469103, -0.5265461, 0.7317381]
# Js[:, 2] = [0.09587734, 0.25093865, -0.8052857, -0.07487219, 0.7793585]

# Js[:, 1] = [0.26805058, 0.89494973, -0.01945474, -0.4242556, 0.12087087]
# Js[:, 2] = [0.28811148, 0.727085, -0.04912354, -0.44243002, 0.12818626]
# Js[:, 1] = [0.54158115, 0.62906706, -0.72885025, 0.9616471, 0.57017493]
# Js[:, 2] = [0.674732, 0.6464902, -0.80049103, 0.8315382, 0.58519995]
# Js[:,1] = [-0.62963295, 0.8469429, -0.20859149, -0.69922495, 0.85480577]
# Js[:,2] = [-0.70979524, 0.7278443, -0.21127568, -0.7666664, 0.9204104]

# Js[:,1] = [0.21400326, 0.7771115, 0.5002325, -0.31930724, 0.8521418]
# Js[:,2] = [0.23267163, 0.53185964, 0.4400487, -0.33313343, 0.84743136]
Js[:,1] = [-0.22182386, 0.9545119, -0.88626814, -0.24695198, -0.8764502]
Js[:,2] = [-0.4369124, 0.6825587, -0.82195246, -0.21397357, -0.8218367]

# Js = [2rand()-1, rand(), 0, 0, 2rand()-1]
clustersize = (11, 11, 1)
energy_min, energy_max = 0.0, 50.0
energy_step = 0.01
η = 0.2
energies = collect(energy_min:energy_step:energy_max)
dir = [1.0, 0.0, 0.0]

res = zeros(12, length(energies), nsample)
Bs = zeros(nsample)
for sample_idx in 1:nsample
    B, qs_path, as_path, bs_path, norm2s_path = triangular_nlswt_lanczos_calculations(Js[:, sample_idx], clustersize,dir)
    intensity_temp = map_lanczos_results_to_neutron_intensities(qs_path, as_path, bs_path, norm2s_path, energies, η)
    res[:, :, sample_idx] = intensity_temp
    Bs[sample_idx] = B
    println("Sample $sample_idx completed.")
end


# using CairoMakie
# # Visualization
# hs = [i/12 for i in 0:12]
# with_theme(theme_latexfonts()) do
#     fig = Figure(size=(1200, 600))
#     ax  = Axis(fig[1,1]; xlabel=L"(H,H,0)", ylabel="Energy (meV)", title="True value", xlabelsize=30, ylabelsize=30, titlesize=30, xticklabelsize=25, yticklabelsize=25)
#     heatmap!(ax, hs, energies, res[:,:,1]; colormap=:viridis, colorrange=(0, 2.5))
#     text!(ax, 0.05, 15, text="J1, Δ, J1pmpm, J1zpm, J2 = $(round.(Js[:,1], digits=3))", align = (:left, :top), fontsize=16, color=:white)
#     text!(ax, 0.05, 13, text="B = $(round(Bs[1], digits=3)) T", align = (:left, :top), fontsize=16, color=:white)
#     ylims!(ax, 0, 20)
#     ax  = Axis(fig[1,2]; xlabel=L"(H,H,0)", ylabel="Energy (meV)", title="Predicted value", xlabelsize=30, ylabelsize=30, titlesize=30, xticklabelsize=25, yticklabelsize=25)
#     heatmap!(ax, hs, energies, res[:,:,2]; colormap=:viridis, colorrange=(0, 2.5))
#     text!(ax, 0.05, 15, text="J1, Δ, J1pmpm, J1zpm, J2 = $(round.(Js[:,2], digits=3))", align = (:left, :top), fontsize=16, color=:white)
#     text!(ax, 0.05, 13, text="B = $(round(Bs[2], digits=3)) T", align = (:left, :top), fontsize=16, color=:white)
#     ylims!(ax, 0, 20)
#     fig
#     save("../Desktop/fig1.pdf", fig) # save figure
# end

# with_theme(theme_latexfonts()) do
#     fig = Figure(size=(600, 600))
#     ax2 = Axis(fig[1,1]; xlabel="Energy (meV)", ylabel="Intensity", title="True value", xlabelsize=30, ylabelsize=30, titlesize=30, xticklabelsize=25, yticklabelsize=25)
#     lines!(ax2, energies, res[1, :, 1], label=L"\Gamma, \mathrm{true}", color=:blue)
#     lines!(ax2, energies, res[1, :, 2], label=L"\Gamma, \mathrm{predicted}", color=:blue, linestyle=:dash,linewidth=3)
#     lines!(ax2, energies, res[6, :, 1], label=L"M, \mathrm{true}", color=:red)
#     lines!(ax2, energies, res[6, :, 2], label=L"M, \mathrm{predicted}", color=:red, linestyle=:dash,linewidth=3)
#     axislegend(ax2; labelsize=18)
#     xlims!(ax2, 0, 20)
#     fig
#     save("../Desktop/fig2.pdf", fig) # save figure
# end