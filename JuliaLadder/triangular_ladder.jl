# Field-polarized triangular-lattice model and its dynamical structure factor
# from the `:ladder` resummation of `Sunny.corrected_intensities`. The model and
# the B_sat criterion are those of ../THED/triangular_nlswt.jl.

using Sunny, LinearAlgebra, StaticArrays

# Channels of `ladder_dssf`, in order: the six of THED's
# `map_lanczos_results_to_dssf_tensor`, then the unpolarized neutron intensity
const DSSF_CHANNELS = ["xx", "yy", "zz", "xy+yx", "yz+zy", "zx+xz", "perp"]

# KYbSe₂-like cell
triangular_crystal() = Crystal(lattice_vectors(1.0, 1.0, 5.0, 90, 90, 120), [[0, 0, 0]], 162)

# Js = [J1, J1zz, Jpmpm, Jzpm, J2]. With g = -1 the field is in units of the
# exchange.
function triangular_system(Js; dims=(1, 1, 1))
    J1, J1zz, Jpmpm, Jzpm, J2 = Js
    sys = System(triangular_crystal(), [1 => Moment(s=1/2, g=-1)], :dipole; dims)
    set_exchange!(sys, [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1zz], Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    return sys
end

function polarize_system!(sys, dir)
    for site in eachsite(sys)
        set_dipole!(sys, dir, site)
    end
end

# Saturation field along `dir`, by bisection on whether energy minimization from
# random spins on a 6×6 supercell finds a state below the polarized one
function determine_B_sat(Js, dir; B_diff_tol=1e-2, energy_tol=1e-8, max_iters=100)
    dir = normalize(dir)
    sys = triangular_system(Js; dims=(6, 6, 1))
    B_min, B_max = 0.0, 100.0
    iter = 0
    while B_max - B_min > B_diff_tol && iter < max_iters
        iter += 1
        B = (B_min + B_max) / 2
        set_field!(sys, B * dir)
        polarize_system!(sys, dir)
        e_ref = energy_per_site(sys)
        randomize_spins!(sys)
        minimize_energy!(sys)
        energy_per_site(sys) < e_ref - energy_tol ? (B_min = B) : (B_max = B)
    end
    return (B_min + B_max) / 2
end

# The polarized single-site system at field B along `dir`, by default 1.1 B_sat
# (shifted by 5 if tiny, as in THED). Returns (B, sys).
function polarized_system(Js, dir; B=nothing)
    dir = normalize(dir)
    B = @something B 1.1 * determine_B_sat(Js, dir)
    B = B < 1e-1 ? B + 5.0 : B
    sys = triangular_system(Js)
    set_field!(sys, B * dir)
    polarize_system!(sys, dir)
    return B, sys
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
# spaced, with step ≤ η/2. `swt` must carry `dssf_measure`.
function ladder_dssf(swt, qpts; energies, η, threaded=true, verbose=false)
    res = Sunny.corrected_intensities(swt, qpts; energies, η, dyson=:ladder, threaded, verbose)
    data = permutedims(stack(res.data), (1, 3, 2))
    nbreak = count(iq -> any(isnan, view(data, :, iq, :)), axes(data, 2))
    return data, nbreak
end
