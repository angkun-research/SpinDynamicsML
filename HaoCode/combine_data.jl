using Sunny
using JLD2
using CairoMakie

include("triangular_nlswt.jl")
# The lower bound and upper bound of energy range (in meV)
energy_min, energy_max = 0.0, 30.0
energy_step = 0.01
energies = collect(energy_min:energy_step:energy_max)
# The Lorentzian broadening factor (in meV) FWHM=2η
η = 0.2

taskid = 15
num_tasks = 250
result = load("data/HaoResults/results_task$(taskid)_of_$(num_tasks).jld2");
jobid = 3 #33, 2
Js = result["Js_list"][jobid]
# as = result["as_list"][jobid]
# bs = result["bs_list"][jobid]
# norm2s = result["norm2s_list"][jobid]
# qs_path = result["qs_list"][jobid]
# B = result["B_list"][jobid]
#Js[4] = 0.0 # set Jzpm to zero for now

clustersize = (11, 11, 1)
dir = [1.0, 0.0, 0.0]
B, qs_path, as_path, bs_path, norm2s_path = triangular_nlswt_lanczos_calculations(Js, clustersize,dir)
intensities = map_lanczos_results_to_neutron_intensities(qs_path, as, bs, norm2s, energies, η)

hs = [i/12 for i in 0:12]
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
    #save("../Desktop/fig1.pdf", fig) # save figure
end


function Run_LSWT(Js,B; dims = (1,1,1))
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    units = Units(:meV, :angstrom)
    a = 4.1149
    c = 22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)
    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst,qs,12)

    # g-tensor for KYbSe₂
    gab, gcc = 3.13*1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    #sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(6, 6, 1))
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [B, 0, 0] * units.T)

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))

    randomize_spins!(sys)
    minimize_energy!(sys)

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    res = intensities_bands(swt, path)

    #plot_intensities(res; units, title="Triangular Lattice")
    return res
end


res = Run_LSWT(Js, B)
units = Units(:meV, :angstrom)

#scatter(collect(1:12), res.disp[16,:])
fig = plot_intensities(res; units, title="Triangular Lattice")

#save("../Desktop/fig2.pdf", fig) # save figure

function lswt_to_heatmap(disp, data, energies; η=0.2)
    nbands, nq = size(disp)
    img = zeros(Float64, nq, length(energies))

    for j in 1:nq
        for b in 1:nbands
            E0 = disp[b, j]
            A = data[b, j]
            img[j, :] .+= A .* (η / π) ./ ((energies .- E0).^2 .+ η^2)
        end
    end

    return img
end
lswt_img = lswt_to_heatmap(res.disp, res.data, energies; η=η)
hs_lswt = collect(LinRange(0, 1, 13))

# Js5 = Js
# Js4 = Js
# Js4[4] = 0.0
# B, qs_path, as_path, bs_path, norm2s_path = triangular_nlswt_lanczos_calculations(Js5, clustersize,dir)
# intensities1 = map_lanczos_results_to_neutron_intensities(qs_path, as_path, bs_path, norm2s_path, energies, η)
# B, qs_path, as_path, bs_path, norm2s_path = triangular_nlswt_lanczos_calculations(Js4, clustersize,dir)
# intensities2 = map_lanczos_results_to_neutron_intensities(qs_path, as_path, bs_path, norm2s_path, energies, η)
# intensities = abs.(intensities1 .- intensities2)

# res = Run_LSWT(Js5, B)
# lswt_img1 = lswt_to_heatmap(res.disp, res.data, energies; η=η)
# res = Run_LSWT(Js4, B)
# lswt_img2 = lswt_to_heatmap(res.disp, res.data, energies; η=η)
# lswt_img = abs.(lswt_img1 .- lswt_img2)

with_theme(theme_latexfonts()) do
    fig = Figure(size=(900, 1200))

    ax1 = Axis(fig[1, 1];
        xlabel=L"(H,H,0)",
        ylabel="Energy (meV)",
        title="Hao's Result",
        xlabelsize=30, ylabelsize=30, titlesize=30,
        xticklabelsize=25, yticklabelsize=25)

    hm1 = heatmap!(ax1, hs, energies, intensities;
        colormap=:viridis, colorrange=(0, 2.5))

    text!(ax1, 0.05, 15,
        text="J1, Δ, J1pmpm, J1zpm, J2 = $(round.(Js, digits=3))",
        align=(:left, :top), fontsize=16, color=:white)
    text!(ax1, 0.05, 13,
        text="B = $(round(B, digits=3)) T",
        align=(:left, :top), fontsize=16, color=:white)
    ylims!(ax1, 0, 20)

    ax2 = Axis(fig[2, 1];
        xlabel=L"(H,H,0)",
        ylabel="Energy (meV)",
        title="LSWT Heatmap",
        xlabelsize=30, ylabelsize=30, titlesize=30,
        xticklabelsize=25, yticklabelsize=25)

    hm2 = heatmap!(ax2, hs_lswt, energies, lswt_img;
        colormap=:viridis, colorrange=(0, 2.5))
    ylims!(ax2, 0, 20)

    linkxaxes!(ax1, ax2)
    linkyaxes!(ax1, ax2)

    Colorbar(fig[:, 2], hm1)
    fig
    save("../Desktop/fig2.pdf", fig)
end

"""
Select N representative points (ω[k], S[k]) using greedy max interpolation error.
Good for preserving peaks with few points.
"""
function adaptive_knots(ω, S; N=10)
    n = length(ω)
    @assert n == length(S)
    keep = sort([1, n])

    # linear interpolation prediction between two kept indices
    linpred(i, j, k) = S[i] + (S[j]-S[i]) * (ω[k]-ω[i]) / (ω[j]-ω[i])

    while length(keep) < N
        keep = sort(keep)
        best_err = -Inf
        best_k = nothing

        for seg in 1:(length(keep)-1)
            i, j = keep[seg], keep[seg+1]
            if j <= i + 1
                continue
            end
            # find worst point in this segment
            for k in (i+1):(j-1)
                err = abs(S[k] - linpred(i, j, k))
                if err > best_err
                    best_err = err
                    best_k = k
                end
            end
        end

        best_k === nothing && break
        push!(keep, best_k)
    end

    keep = sort(unique(keep))
    return ω[keep], S[keep], keep
end


E_temp, S_temp, keep_indices = adaptive_knots(energies, intensities[6, :], N=3) # 20

fig = Figure(size=(600, 400))
ax = Axis(fig[1,1], xlabel="Energy (meV)", ylabel="Intensity", title="Adaptive Knots", xlabelsize=20, ylabelsize=20, titlesize=20, xticklabelsize=15, yticklabelsize=15)
scatter!(ax, E_temp, S_temp,label="Adaptive Knots", color=:red)
lines!(ax, energies, intensities[6, :], alpha=0.3, label="Original")
axislegend(ax; labelsize=12)
fig
#save("../Desktop/fig1.pdf", fig) # save figure

