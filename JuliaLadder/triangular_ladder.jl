# Field-polarized triangular-lattice model and its dynamical structure factor
# from the `:ladder` resummation of `Sunny.corrected_intensities`. The model is
# that of ../THED/triangular_nlswt.jl; the saturation field and the field of the
# calculation are determined as described below.

using Sunny, LinearAlgebra, StaticArrays

# Channels of `ladder_dssf`, in order: the six of THED's
# `map_lanczos_results_to_dssf_tensor`, then the unpolarized neutron intensity
const DSSF_CHANNELS = ["xx", "yy", "zz", "xy+yx", "yz+zy", "zx+xz", "perp"]

# KYbSe₂-like cell
triangular_crystal() = Crystal(lattice_vectors(1.0, 1.0, 5.0, 90, 90, 120), [[0, 0, 0]], 162)

# Js = [J1, J1zz, Jpmpm, Jzpm, J2]. With g = -1 the field is in units of the
# exchange.
function triangular_system(Js; dims=(1, 1, 1), seed=nothing)
    J1, J1zz, Jpmpm, Jzpm, J2 = Js
    sys = System(triangular_crystal(), [1 => Moment(s=1/2, g=-1)], :dipole; dims, seed)
    set_exchange!(sys, [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1zz], Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    return sys
end

function polarize_system!(sys, dir)
    for site in eachsite(sys)
        set_dipole!(sys, dir, site)
    end
end

# ---- Saturation field ----
#
# B_sat is the field above which the state polarized along the field direction
# `dir` is the classical ground state. Two criteria enter:
#
# - Local: below B_loc the polarized state is not even a local energy minimum.
#   It is one exactly when the LSWT matrix H(q) has no negative eigenvalue at
#   any q. For spins along the field, the Zeeman term only shifts H(q) by c B·I,
#   so B_loc = -min_q λ_min(H₀(q)) / c, with H₀ the matrix at zero field. The
#   minimum is searched over continuous q, so a soft mode at an incommensurate
#   wavevector is found as precisely as a commensurate one.
#
# - Energetic: bisection on whether energy minimization from random spins in a
#   6×6 supercell finds a state below the polarized one (the original THED
#   criterion). It detects first-order transitions, at which another state wins
#   while the polarized one is still a local minimum, but misses instabilities
#   at wavevectors incommensurate with the supercell.
#
# B_sat is the larger of the two. In a survey of 300 random parameter sets, the
# energetic criterion alone underestimated B_sat by up to 22%, crashing LSWT at
# 1.1 B_sat in 20 cases; 19 sets, all with J1 < 0, had first-order transitions.

# Component of the exchange field perpendicular to `dir` in the state polarized
# along `dir`. The polarized state is stationary at every field only if it
# vanishes, as it does by symmetry for `dir` in the plane or along c.
function exchange_torque(Js, dir)
    dir = normalize(dir)
    sys = triangular_system(Js)
    h = sum(Matrix(c.bilin * I(3)) * dir + transpose(Matrix(c.bilin * I(3))) * dir
            for c in sys.interactions_union[1].pair if !c.isculled)
    return norm(h - dot(dir, h) * dir)
end

# The local saturation field B_loc and the soft-mode wavevector Q, in RLU. The
# minimum of λ_min(H₀(q)) is searched on an n×n grid over the zone (the model has
# no interlayer couplings, so q_z is irrelevant), then refined three times by a
# factor of 4 around the five lowest grid points. Costs about 10 ms.
function local_saturation_field(Js, dir; n=96)
    dir = normalize(dir)
    τ = exchange_torque(Js, dir)
    τ < 1e-8 || error("The state polarized along $dir is not stationary (exchange torque $τ); \
                       B_loc requires a field in the plane or along c.")
    swt0, swt1 = map((0.0, 1.0)) do B
        sys = triangular_system(Js)
        set_field!(sys, B * dir)
        polarize_system!(sys, dir)
        SpinWaveTheory(sys; measure=nothing, regularization=0.0)
    end
    L = Sunny.nbands(swt0)
    H = zeros(ComplexF64, 2L, 2L)
    hmat(swt, h, k) = (Sunny.dynamical_matrix!(H, swt, Sunny.Vec3(h, k, 0)); copy(H))

    # The Zeeman shift c B·I, verified at an arbitrary q
    ΔH = hmat(swt1, 0.123, 0.456) - hmat(swt0, 0.123, 0.456)
    c = real(ΔH[1, 1])
    c > 0 && norm(ΔH - c * I) < 1e-10 * c || error("Zeeman term does not shift H(q) by a multiple of the identity")

    λ(h, k) = eigmin(Hermitian(hmat(swt0, h, k)))
    grid = sort!(vec([(λ(i/n, j/n), i/n, j/n) for i in 0:n-1, j in 0:n-1]))
    best = grid[1]
    for (_, h, k) in grid[1:5]
        step = 1/n
        for _ in 1:3
            _, h, k = minimum((λ(h + a*step/4, k + b*step/4), h + a*step/4, k + b*step/4) for a in -4:4, b in -4:4)
            step /= 4
        end
        best = min(best, (λ(h, k), h, k))
    end
    λmin, h, k = best
    return max(-λmin / c, 0.0), Sunny.Vec3(mod(h, 1), mod(k, 1), 0)
end

# The energetic criterion: bisection on whether energy minimization from random
# spins in a 6×6 supercell finds a state below the polarized one. A single random
# start can stall in a local minimum and miss the lower state; in the survey this
# made B_sat vary by up to 0.35 between runs for 23 of 300 sets. Each trial field
# therefore gets `restarts` random starts, and a fixed `seed` makes the result
# reproducible. Any lower state found is a valid one, so more restarts can only
# move the result up toward the true first-order field.
function energetic_saturation_field(Js, dir; B_diff_tol=1e-2, energy_tol=1e-8, max_iters=100, restarts=8, seed=0)
    dir = normalize(dir)
    sys = triangular_system(Js; dims=(6, 6, 1), seed)
    function lower_state_exists(B)
        set_field!(sys, B * dir)
        polarize_system!(sys, dir)
        e_ref = energy_per_site(sys)
        return any(1:restarts) do _
            randomize_spins!(sys)
            minimize_energy!(sys)
            energy_per_site(sys) < e_ref - energy_tol
        end
    end
    B_min, B_max = 0.0, 100.0
    iter = 0
    while B_max - B_min > B_diff_tol && iter < max_iters
        iter += 1
        B = (B_min + B_max) / 2
        lower_state_exists(B) ? (B_min = B) : (B_max = B)
    end
    return (B_min + B_max) / 2
end

# B_sat = max(B_loc, B_energy), with both criteria and the soft-mode wavevector Q
function saturation_field(Js, dir)
    B_loc, Q = local_saturation_field(Js, dir)
    B_energy = energetic_saturation_field(Js, dir)
    return (; B_sat=max(B_loc, B_energy), B_loc, B_energy, Q)
end

# The field at which spectra are computed, B_sat + |J1|. A margin relative to
# B_sat (e.g. 1.1 B_sat) is too small where B_sat is small and quantum
# corrections are large, as for J1 < 0 with strong U(1)-breaking couplings: in
# the survey, 1.1 B_sat left 23 of 300 sets in breakdown, B_sat + |J1| none of
# the 23. For J1 > 0 this field is about 1.13 - 1.2 B_sat.
measurement_field(B_sat, Js) = B_sat + abs(Js[1])

# The polarized single-site system along `dir` at field B, by default the
# measurement field. Returns (B, sys, sat), with `sat` the result of
# `saturation_field`, or `nothing` if B is given.
function polarized_system(Js, dir; B=nothing)
    dir = normalize(dir)
    sat = isnothing(B) ? saturation_field(Js, dir) : nothing
    B = @something B measurement_field(sat.B_sat, Js) 
    sys = triangular_system(Js)
    set_field!(sys, B * dir)
    polarize_system!(sys, dir)
    return B, sys, sat
end

# The measure for `DSSF_CHANNELS`. Off-diagonal channels are the symmetric
# combinations S^{ab} + S^{ba}; "perp" is Sunny's `ssf_perp`.
function dssf_measure(sys)
    return ssf_custom(sys) do q, ssf
        q2 = dot(q, q)
        perp = iszero(q2) ? (2/3) * real(tr(ssf)) : real(tr(ssf) - dot(q, ssf, q) / q2)
        SVector(real(ssf[1, 1]), real(ssf[2, 2]), real(ssf[3, 3]),
                real(ssf[1, 2] + ssf[2, 1]), real(ssf[2, 3] + ssf[3, 2]), real(ssf[3, 1] + ssf[1, 3]),
                perp)
    end
end

# Upper end of an energy window that holds the two-magnon continuum: 2.2 times
# the largest LSWT energy on a grid over the zone
function energy_max(swt; n=16)
    qs = [[i/n, j/n, 0] for i in 0:n-1, j in 0:n-1]
    return 2.2 * maximum(dispersion(swt, vec(qs)))
end

# The channels of DSSF_CHANNELS at `qpts` from the `:ladder` resummation, as a
# (channel × q × energy) array, plus the number of q at which the expansion
# breaks down (intensity set to NaN near ω = 0). `energies` must be equally
# spaced, with step ≤ η/2. `swt` must carry `dssf_measure`. The line shape is a
# Lorentzian of half-width η; an optional `kernel`, e.g. `gaussian(; σ)`, is
# convolved on top.
function ladder_dssf(swt, qpts; energies, η, kernel=nothing, threaded=false, verbose=false)
    res = Sunny.corrected_intensities(swt, qpts; energies, η, kernel, dyson=:ladder, threaded, verbose)
    data = permutedims(stack(res.data), (1, 3, 2))
    nbreak = count(iq -> any(isnan, view(data, :, iq, :)), axes(data, 2))
    return data, nbreak
end
