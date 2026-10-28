using Sunny
using CairoMakie
using LaTeXStrings
using LinearAlgebra

# Reuse Hao's Lanczos and B_sat helpers
include("triangular_ladder.jl")

# -----------------------------
# User parameters
# -----------------------------
#Js = [1.0, 0.2, 0.1, -0.1, 0.5]  # [J1, Delta, Jpmpm, Jzpm, J2]
#Js = [1.0, 0.7, 0.2, 0.2, 0.051]  # [J1, Delta, Jpmpm, Jzpm, J2]
Js = [1.0, 0.2, 0.4, 0.0, 0.5]
dir = [1.0, 0.0, 0.0]

eta = 0.2                          # Lorentzian broadening (meV)

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

        best_k === nothing && break # no more points to add
        push!(keep, best_k)
    end

    keep = sort(unique(keep))
    return ω[keep], S[keep], keep
end

# -----------------------------
# Compute Lanczos DSSF for two parameter sets
# -----------------------------
B_sat, sys, _ = polarized_system(Js, dir)

swt = SpinWaveTheory(sys; measure=dssf_measure(sys))
qpath = [[0, 0, 0], [1, 1, 0]]
nq = 12
qs = q_space_path(sys.crystal, qpath, nq).qs
e_max = energy_max(swt)
energies = collect(0.0:0.01:e_max)
data, nbreak = ladder_dssf(swt, qs; energies, η = eta)

scalar1 = data[7,:,:]

E_int, S_int, _ = adaptive_knots(energies, scalar1[1, :]; N=20)


# Heatmap edges
nq = size(scalar1, 1)
hs = collect(range(0.0, 1.0; length=nq + 1))

# -----------------------------
# Plot and save
# -----------------------------
labelsize = 28
ticklabelsize = 20

fig = Figure(size = (500, 400))
ax = Axis(
    fig[1, 1],
    xlabel = L"\omega / J_1",
    ylabel = L"I(\mathbf{k}=\Gamma, \omega)",
    xlabelsize = labelsize,
    ylabelsize = labelsize,
    xticklabelsize = ticklabelsize,
    yticklabelsize = ticklabelsize,
)

# (1) Full curve: scalar1[1, :] vs energies
lines!(ax, energies, scalar1[1, :];
    color = :royalblue, linewidth = 3, label = "Full")

# (2) Reduced/interpolated curve: S_int vs E_int
scatter!(ax, E_int, S_int; color = :crimson, markersize = 8,label = "Interpolated")

axislegend(ax, position = :rt, framevisible = true, labelsize = 18)
xlims!(ax, 0, e_max)

save("../plots/Interpolation_knots_THL.pdf", fig)


println("Saved: ../plots/Interpolation_knots_THL.pdf")

