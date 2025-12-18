using Sunny
using GLMakie
using LinearAlgebra
using ProgressMeter
using JLD2

"""
    triangular_nlswt_model_construction(J1, Δ, Jpmpm, Jzpm, J2, B, clustersize)

Constructs a non-linear spin wave theory model for the triangular lattice antiferromagnet
with the specified parameters.
On the nearest-neighbor bond, the 3×3 exchange tensor is given by:

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]

On the next-nearest-neighbor bond, the exchange is isotropic with strength J2. `Lmag` is the linear system size for finite-size clusters used in the non-perturbative calculation.
"""
function triangular_nlswt_model_construction(J1, Δ, Jpmpm, Jzpm, J2, B, clustersize)
    units = Units(:meV, :angstrom)
    latvecs = lattice_vectors(4, 4, 12, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)

    gab, gcc = 3.13*1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(1, 1, 1))

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [B, 0, 0] * units.T)

    randomize_spins!(sys)
    minimize_energy!(sys)

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    Nu = prod(clustersize)
    num_1ps = Sunny.nbands(swt)
    num_2ps = Int(binomial(Nu*num_1ps+2-1, 2) / Nu)
    println("Dimension of truncated Hilbert space: ", num_1ps + num_2ps)
    npt = Sunny.NonPerturbativeTheory(swt, clustersize)
    return npt
end

# Example
Lmag = 25
# Units in meV
J1 = 0.551
J2 = 0.071 * J1
Δ  = 0.9
Jpmpm = 0.0J1
Jzpm = 0.2J1
# Units in Tesla
B = 14.0

npt = triangular_nlswt_model_construction(J1, Δ, Jpmpm, Jzpm, J2, B, (Lmag, Lmag, 1))

# The q-grid along a particular path, here in this example, we have (hh3) direction
qs = [[k/Lmag, k/Lmag, 3] for k in 0:Lmag-1]
# Energy grid
energies = collect(0.0:0.01:3.5)

########### Hyperparameters for the calculation ############
# We can fix them for the current values and play with them later
# Artificial broadening
η = 0.06J1
# Lanczos iterations
n_iters = 2000
# Tolerence for integration that calculates the magnon-magnon interaction
atol = 1e-5

# Buffers to store intensities
intensitiesMat = zeros(length(qs), length(energies))

# Calculations
pm = Progress(length(qs), "Calculating intensities along (hh3) path")
# We can parallelize over q-points for speedup. On my machine, each q points takes about 1.5 minutes.
Threads.@threads for i in eachindex(qs)
    q = qs[i]
    intensity = Sunny.intensities_continued_fraction(npt, q, energies, η, n_iters; atol=atol)
    intensitiesMat[i, :] = intensity
    next!(pm)
end

# The results for "learning" is stored in `intensities` (25 x 351 matrix)
result_path = "./data"#joinpath(@__DIR__, "data")
isdir(result_path) || mkpath(result_path)
file_name = joinpath(result_path, "NPintensities.jld2")
#jldsave(file_name; intensitiesMat=intensitiesMat, Lmag=Lmag, energies=energies)

res = load(file_name)

intensitiesMat = res["intensitiesMat"]
Lmag = res["Lmag"]
energies = res["energies"]

# Visualization
with_theme(theme_latexfonts()) do
    fig = Figure()
    ax  = Axis(fig[1,1]; xlabel=L"(h,h,3)", ylabel="Energy (meV)", title="NLSWT Intensity", xlabelsize=30, ylabelsize=30, titlesize=30, xticklabelsize=25, yticklabelsize=25)
    heatmap!(ax, range(0, 1-1/Lmag, length=Lmag), energies, intensitiesMat; colormap=:viridis)
    save("/Users/angkunwu/Desktop/NPintensities.png", fig)
    fig
end

# plot a scatter plot of intensities at fixed energy cuts
fig = Figure()
ax1 = Axis(fig[1,1]; xlabel="E", ylabel="Intensity")
for k in [8,10,12,13,14,16]
    scatter!(ax1, energies, intensitiesMat[k, :]; label="E=$(energies[k]) meV")
end
fig
