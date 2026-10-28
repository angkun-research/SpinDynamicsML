# Compute Fisher Information Matrix (FIM) for LSWT or ladder outputs using central finite differences.
using LinearAlgebra
using Statistics
using ProgressBars
using Sunny

include("../JuliaLadder/triangular_ladder.jl")

function Run_LSWT(Js,B; dims = (1,1,1),Nk=12, observable=:perp, Qz=0)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    #units = Units(:meV, :angstrom)
    a = 1 #4.1149
    c = 5 #22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)
    qs = [[0, 0, Qz], [1, 1, Qz]] #[[0, 0, 5], [1, 1, 5]]#[[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst,qs,Nk)

    # g-tensor for KYbSe₂
    # gab, gcc = 3.13*1.09310, 1.01
    # gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    gtens = 1
    #sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(6, 6, 1))
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    #set_field!(sys, [B, 0, 0] * units.T)
    set_field!(sys, [B, 0, 0])

    polarize_system!(sys,[-1,0,0])

    measure = if observable == :perp
        ssf_perp(sys)
    elseif observable == :tensor
        ssf_custom((q, ssf) -> ssf, sys)
    else
        error("Unknown observable=$(observable). Use :perp or :tensor")
    end

    swt = SpinWaveTheory(sys; measure=measure)
    res = intensities_bands(swt, path)

    return res
end

"""
    Run_Ladder(Js, B; dims=(1,1,1), Nk=12, ...)

Ladder spectrum for paper parameters `[J1, Δ, Jpmpm, Jzpm, J2]` at fixed B.
Returns `(B, qs, data)`, with data shaped `(nq, nE)` for `:perp` or
`(6, nq, nE)` for `:tensor`. Tensor channels follow `DSSF_CHANNELS[1:6]`:
xx, yy, zz, xy+yx, yz+zy, zx+xz. Energies must be equally spaced (step ≤ eta/2).
"""
function Run_Ladder(Js, B;
    dims=(1, 1, 1),
    Nk=12,
    dir=[1.0, 0.0, 0.0],
    eta=0.2,
    observable=:perp,
    tensor_components=:sym6,
    Qz=0,
)
    observable in (:perp, :tensor) ||
        error("Unknown observable=$(observable). Use :perp or :tensor")
    if observable == :tensor && tensor_components != :sym6
        error("Ladder backend currently supports tensor_components=:sym6 only")
    end

    # The ladder helper takes Jzz, whereas the FIM parameters use Δ = Jzz/J1.
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    a = 1 #4.1149
    c = 5 #22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)
    gtens = 1
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [B, 0, 0])

    polarize_system!(sys, [-1,0,0])  
    path = q_space_path(sys.crystal, [[0, 0, Qz], [1, 1, Qz]], Nk)
    swt = SpinWaveTheory(sys; measure=dssf_measure(sys))
    e_max = 10 #energy_max(swt) #10
    energies = collect(0.0:0.01:e_max)
    data, nbreak = ladder_dssf(swt, path; energies=energies, η=eta,
                              threaded=false, verbose=false)
    if nbreak > 0 || !all(isfinite, data)
        error("Invalid ladder spectrum for Js=$(Js), B=$(B): $(nbreak) momentum breakdowns")
    end

    spectrum = observable == :perp ? data[7, :, :] : data[1:6, :, :]
    return (B=B, qs=path.qs, data=spectrum)
end

function flatten_tensor_data(res; tensor_components=:sym6)
    if tensor_components == :sym6
        sxx = real.(getindex.(res.data[1, :], 1, 1))
        syy = real.(getindex.(res.data[1, :], 2, 2))
        szz = real.(getindex.(res.data[1, :], 3, 3))
        sxy = real.(getindex.(res.data[1, :], 1, 2))
        sxz = real.(getindex.(res.data[1, :], 1, 3))
        syz = real.(getindex.(res.data[1, :], 2, 3))
        return vcat(sxx, syy, szz, sxy, sxz, syz)
    elseif tensor_components == :full9
        nq = size(res.data, 2)
        out = Vector{Float64}(undef, 9 * nq)
        k = 1
        for iq in 1:nq
            M = res.data[1, iq]
            for a in 1:3, b in 1:3
                out[k] = real(M[a, b])
                k += 1
            end
        end
        return out
    else
        error("Unknown tensor_components=$(tensor_components). Use :sym6 or :full9")
    end
end

function S_vec(Js;
    backend=:lswt,
    B=60,
    Nk=12,
    dims=(1,1,1),
    observable=:perp,
    tensor_components=:sym6,
    dir=[1.0,0.0,0.0],
    eta=0.2,
    Qz=0,
)
    d = length(Js)
    if d == 4
        Js = vcat(Js[1:3], 0.0, Js[4])  # insert Jzpm=0 for 4-parameter case
    end

    if backend == :lswt
        res = Run_LSWT(Js, B; dims=dims, Nk=Nk, observable=observable, Qz=Qz)
        dvec = vec(res.disp)
        idvec = observable == :perp ? vec(res.data) : flatten_tensor_data(res; tensor_components=tensor_components)
        return vcat(dvec, idvec)

    elseif backend == :ladder
        res = Run_Ladder(Js, B;
            dims=dims,
            Nk=Nk,
            dir=dir,
            eta=eta,
            observable=observable,
            tensor_components=tensor_components,
            Qz=Qz
        )

        # Ladder returns an energy-resolved spectrum, without a dispersion branch.
        return vec(res.data)

    else
        error("Unknown backend=$(backend). Use :lswt or :ladder")
    end
end
"""
compute_FIM(Js0; δ=1e-3, B=60, Nk=1000, dims=(1,1,1), sigma=nothing)

Returns a NamedTuple with:
- F: the d×d Fisher information matrix (whitened if `sigma` provided)
- evals: eigenvalues (sorted descending)
- evecs: eigenvectors
- J_S: the (m × d) forward Jacobian (unwhitened)
- info: diagnostic info (cond, smallest nonzero eigen, step used)
Options:
- `sigma`: optional vector of length m providing measurement std; if provided rows of J_S are divided by sigma (whitening).
"""
function compute_FIM(Js0;
    delta=1e-3,
    backend=:lswt,
    B=60,
    Nk=12,
    dims=(1,1,1),
    observable=:perp,
    tensor_components=:sym6,
    dir=[1.0,0.0,0.0],
    eta=0.2,
    Qz=0,
)
    d = length(Js0)
    S0 = S_vec(Js0; backend=backend, B=B, Nk=Nk, dims=dims, observable=observable,
               tensor_components=tensor_components, dir=dir,eta=eta, Qz=Qz)
    m = length(S0)
    J_S = zeros(Float64, m, d)

    for i in 1:d
        step = delta * max(abs(Js0[i]), 1.0)
        dp = zeros(d); dp[i] = step
        println("Computing column $(i) of $(d) in Jacobian")
        Sp = S_vec(Js0 .+ dp; backend=backend, B=B, Nk=Nk, dims=dims, observable=observable,
                   tensor_components=tensor_components, dir=dir,
                   eta=eta, Qz=Qz)
        println("done 1.")
        Sm = S_vec(Js0 .- dp; backend=backend, B=B, Nk=Nk, dims=dims, observable=observable,
                   tensor_components=tensor_components, dir=dir,
                   eta=eta, Qz=Qz)
        println("done 2.")
        J_S[:, i] = (Sp .- Sm) ./ (2 * step)
    end

    J_S_used = copy(J_S)
    F = Symmetric(J_S_used' * J_S_used)
    eig = eigen(F)
    idx = sortperm(eig.values, rev=true)
    evals = eig.values[idx]
    evecs = eig.vectors[:, idx]
    cond_est = evals[1] / (evals[end] > 0 ? evals[end] : eps())

    info = Dict(
        :cond => cond_est,
        :min_eval => evals[end],
        :max_eval => evals[1],
        :m => m,
        :d => d,
        :backend => backend,
        :observable => observable,
        :tensor_components => tensor_components
    )

    return (F=F, evals=evals, evecs=evecs, J_S=J_S, info=info)
end

# Small example runner when invoked directly
println("FIM_identifiability.jl — example run")
    # Example 4-parameter anisotropy point (adjust to your model dimensionality)
#J0 = [1.0,0.7,0.2,0.01,0.051] #[1.0,0.2,0.1,0.05, 0.5]
J0 = [1.0,0.7,0.2,0.2,0.051]
Qz = 1
println("Qz: ", Qz)
println("Computing FIM at J0 = ", J0)
observable = :perp #:tensor #:perp
println("Observable: ", observable)
tensor_components = :sym6
backend = :ladder #:ladder #:lswt
println("Backend: ", backend)
if length(J0) == 4
    Jtemp = vcat(J0[1:3], 0.0, J0[4])
    B, sys, sat = polarized_system(Jtemp, [1,0,0])
else
    B, sys, sat = polarized_system(J0, [1,0,0])
end
println("Initial B_sat: ", sat.B_sat)
Nk = 12
println("B_set: ", B, ", Nk: ", Nk)
res = compute_FIM(J0; backend=backend, delta=1e-3, B=B, Nk=Nk, 
    observable=observable, tensor_components=tensor_components, Qz=Qz)
println("Eigenvalues (desc): ", res.evals)
println("Condition est: ", res.info[:cond])

# J0s = [
#     [1.0, 0.2, 0.1, 0.05, 0.5],
#     [1.0, 0.3, 0.1, 0.1, 0.8],
#     [1.0, 0.3, -0.2, -0.1, -0.5],
#     [-1.0, 0.5, 0.2, 0.1, 0.3],
#     [-1.0, 0.4, 0.3, -0.2, 0.1]
# ]
# for J0 in J0s
#     println("\nComputing FIM at J0 = ", J0)
#     local res = compute_FIM(J0; delta=1e-3, B=60, Nk=200, observable=observable, tensor_components=tensor_components)
#     println("Eigenvalues (desc): ", res.evals)
#     println("Condition est: ", res.info[:cond])
#     J04 = J0[[1,2,3,5]]  # drop Jzpm for 4-parameter case
#     println("\nComputing FIM at 4-parameter J0 = ", J04)
#     local res4 = compute_FIM(J04; delta=1e-3, B=60, Nk=200, observable=observable, tensor_components=tensor_components)
#     println("Eigenvalues (desc): ", res4.evals)
#     println("Condition est: ", res4.info[:cond])
# end