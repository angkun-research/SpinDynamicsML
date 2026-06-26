# Compute Fisher Information Matrix (FIM) for LSWT outputs using central finite differences.
#
# Usage:
#   julia --project=. HPCtemp/FIM_identifiability.jl
# or from the REPL:
#   include("HPCtemp/FIM_identifiability.jl"); res = compute_FIM(J0; ...)

using LinearAlgebra
using Statistics
using ProgressBars
using Sunny

# We polarize the system along the x direction to be consistent with KYbSe₂ experiments
function polarize_system!(sys)
    for i in CartesianIndices(sys.dipoles)
        set_dipole!(sys, (-1, 0, 0), i)
    end
end

function Run_LSWT(Js,B; dims = (1,1,1),Nk=12)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    units = Units(:meV, :angstrom)
    a = 4.1149
    c = 22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)
    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst,qs,Nk)

    # g-tensor for KYbSe₂
    gab, gcc = 3.13*1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    #sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(6, 6, 1))
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [B, 0, 0] * units.T)

    #swt = SpinWaveTheory(sys; measure=ssf_perp(sys))

    # randomize_spins!(sys)
    # minimize_energy!(sys)
    polarize_system!(sys)

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    res = intensities_bands(swt, path)

    #plot_intensities(res; units, title="Triangular Lattice")
    return res
end

# Build the observable vector S(J): concatenation of dispersion then intensity
function S_vec(Js; B=60, Nk=1000, dims=(1,1,1))
    d = length(Js)
    if d == 4
        Js = vcat(Js[1:3], 0.0, Js[4])  # insert Jzpm=0 for 4-parameter case
    end
    res = Run_LSWT(Js, B; dims=dims, Nk=Nk)
    dvec = vec(res.disp)
    idvec = vec(res.data)
    return vcat(dvec, idvec)
end

"""
compute_FIM(Js0; δ=1e-3, B=60, Nk=1000, dims=(1,1,1), sigma=nothing, check_steps=[1e-2,1e-3,1e-4])

Returns a NamedTuple with:
- F: the d×d Fisher information matrix (whitened if `sigma` provided)
- evals: eigenvalues (sorted descending)
- evecs: eigenvectors
- J_S: the (m × d) forward Jacobian (unwhitened)
- info: diagnostic info (cond, smallest nonzero eigen, step used)
Options:
- `sigma`: optional vector of length m providing measurement std; if provided rows of J_S are divided by sigma (whitening).
- `check_steps`: list of δ multipliers to test convergence (optional).
"""
function compute_FIM(Js0; δ=1e-3, B=60, Nk=1000, dims=(1,1,1), sigma=nothing)
    d = length(Js0)
    S0 = S_vec(Js0; B=B, Nk=Nk, dims=dims)
    m = length(S0)
    J_S = zeros(Float64, m, d)

    # central finite differences
    for i in 1:d
        step = δ * max(abs(Js0[i]), 1.0)
        dp = zeros(d); dp[i] = step
        Sp = S_vec(Js0 .+ dp; B=B, Nk=Nk, dims=dims)
        Sm = S_vec(Js0 .- dp; B=B, Nk=Nk, dims=dims)
        J_S[:, i] = (Sp .- Sm) ./ (2 * step)
    end

    # Optional whitening by measurement std (sigma)
    J_S_used = copy(J_S)
    if sigma !== nothing
        @assert length(sigma) == m "sigma must have length = number of observables"
        invsigma = 1.0 ./ sigma
        # scale each row
        for r in 1:m
            J_S_used[r, :] .*= invsigma[r]
        end
    end

    # Form F (d x d)
    F = J_S_used' * J_S_used
    F = Symmetric(F)  # ensure symmetric for eigen decomposition

    # eigen / spectrum
    eig = eigen(F)
    # sort descending
    idx = sortperm(eig.values, rev=true)
    evals = eig.values[idx]
    evecs = eig.vectors[:, idx]

    cond_est = (evals[1] / (evals[end] > 0 ? evals[end] : eps()))
    info = Dict(
        :cond => cond_est,
        :min_eval => evals[end],
        :max_eval => evals[1],
        :m => m,
        :d => d
    )

    return (F=F, evals=evals, evecs=evecs, J_S=J_S, info=info)
end

# Optional helper: convergence check over multiple δ values (prints relative changes)
function delta_convergence(Js0; deltas=[1e-2,1e-3,1e-4], B=60, Nk=200, dims=(1,1,1))
    println("Running δ-convergence for: ", deltas)
    results = Dict()
    prev_JS = nothing
    for δ in deltas
        res = compute_FIM(Js0; δ=δ, B=B, Nk=Nk, dims=dims)
        results[δ] = res
        if prev_JS !== nothing
            # compare columns
            diff = norm(prev_JS.J_S - res.J_S) / max(norm(prev_JS.J_S), eps())
            println("δ=", δ, " rel change in J_S: ", diff)
        end
        prev_JS = res
    end
    return results
end

# Small example runner when invoked directly
println("FIM_identifiability.jl — example run")
    # Example 4-parameter anisotropy point (adjust to your model dimensionality)
J0 = [1.0, 0.9, 0.1, 0.0, 0.2]   # length d must match your Run_LSWT expectation slice
J0 = [1.0, 0.9, 0.1, 0.2]
println("Computing FIM at J0 = ", J0)
res = compute_FIM(J0; δ=1e-3, B=60, Nk=200)
println("Eigenvalues (desc): ", res.evals)
println("Condition est: ", res.info[:cond])

J0s = [
    [1.0, 0.2, 0.1, 0.05, 0.5],
    [1.0, 0.3, 0.1, 0.1, 0.8],
    [1.0, 0.3, -0.2, -0.1, -0.5],
    [-1.0, 0.5, 0.2, 0.1, 0.3],
    [-1.0, 0.4, 0.3, 0.2, 0.1]
]
for J0 in J0s
    println("\nComputing FIM at J0 = ", J0)
    res = compute_FIM(J0; δ=1e-3, B=60, Nk=200)
    println("Eigenvalues (desc): ", res.evals)
    println("Condition est: ", res.info[:cond])
    J04 = J0[[1,2,3,5]]  # drop Jzpm for 4-parameter case
    println("\nComputing FIM at 4-parameter J0 = ", J04)
    res4 = compute_FIM(J04; δ=1e-3, B=60, Nk=200)
    println("Eigenvalues (desc): ", res4.evals)
    println("Condition est: ", res4.info[:cond])
end