using Sunny
using LinearAlgebra

function map_lanczos_results_to_neutron_intensities(qs_path, as_path, bs_path, norm2s_path, ωs, η)
    ret_buff = zeros(length(ωs), 6)
    num_obs = 3
    niters = size(as_path, 2)
    intensities = zeros(length(qs_path), length(ωs))

    # The original crystal
    # a = 4.1149
    # c = 22.6911
    a, c = 1, 5
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)

    for (iq, q) in enumerate(qs_path)
        ret_buff .= 0.0
        as = view(as_path, iq, :, :)
        bs = view(bs_path, iq, :, :)
        norm2s = view(norm2s_path, iq, :)

        for i in 1:num_obs
            as_i = view(as, :, i)
            bs_i = view(bs, :, i)
            for (iω, ω) in enumerate(ωs)
                z = ω + 1im*η
                G = z - as_i[niters]
                for j in niters-1:-1:1
                    A = bs_i[j]^2 / G
                    G = z - as_i[j] - A
                end
                G = inv(G) * norm2s[i]
                ret_buff[iω, i] = - imag(G) / π
            end
            # Off-diagonal elements
            as_i_plus = view(as, :, i+3)
            bs_i_plus = view(bs, :, i+3)
            for (iω, ω) in enumerate(ωs)
                z = ω + 1im*η
                G = z - as_i_plus[niters]
                for j in niters-1:-1:1
                    A = bs_i_plus[j]^2 / G
                    G = z - as_i_plus[j] - A
                end
                G = inv(G) * norm2s[i+3]
                ret_buff[iω, i+3] = - imag(G) / π
            end
        end
        q_global = cryst.recipvecs * q
        q2 = Sunny.norm2(q_global)

        ret = zeros(length(ωs))
        if q2 < 1e-6
            # Later we may add the 2/3 factor to be consistent with the Sunny main
            for i in 1:num_obs
                @. ret += 2/3*ret_buff[:, i]
            end
        else
            for i in 1:num_obs
                @. ret += ret_buff[:, i] * (1 - q_global[i]^2 / q2)
                @. ret -= (ret_buff[:, i+3] - ret_buff[:, i] - ret_buff[:, mod1(i+1, 3)]) * q_global[i] * q_global[mod1(i+1, 3)] / q2
            end
        end
        intensities[iq, :] = ret
    end

    return intensities
end

function map_lanczos_results_to_dssf_tensor(qs_path, as_path, bs_path, norm2s_path, ωs, η)
    ret_buff = zeros(length(ωs), 6)
    num_obs = 3
    niters = size(as_path, 2)
    dssf = zeros(6, length(qs_path), length(ωs))

    # The original crystal
    # a = 4.1149
    # c = 22.6911
    a, c = 1, 5
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)

    for iq in eachindex(qs_path)
        ret_buff .= 0.0
        as = view(as_path, iq, :, :)
        bs = view(bs_path, iq, :, :)
        norm2s = view(norm2s_path, iq, :)

        for i in 1:num_obs
            as_i = view(as, :, i)
            bs_i = view(bs, :, i)
            for (iω, ω) in enumerate(ωs)
                z = ω + 1im*η
                G = z - as_i[niters]
                for j in niters-1:-1:1
                    A = bs_i[j]^2 / G
                    G = z - as_i[j] - A
                end
                G = inv(G) * norm2s[i]
                ret_buff[iω, i] = - imag(G) / π
            end
            dssf[i, iq, :] = ret_buff[:, i]
            # Off-diagonal elements:
            # (x+y)^2, (y+z)^2, (z+x)^2 correlations
            as_i_plus = view(as, :, i+3)
            bs_i_plus = view(bs, :, i+3)
            for (iω, ω) in enumerate(ωs)
                z = ω + 1im*η
                G = z - as_i_plus[niters]
                for j in niters-1:-1:1
                    A = bs_i_plus[j]^2 / G
                    G = z - as_i_plus[j] - A
                end
                G = inv(G) * norm2s[i+3]
                ret_buff[iω, i+3] = - imag(G) / π
            end
            # Here we store xy+yx, yz+zy, zx+xz correlations
            dssf[i+3, iq, :] = ret_buff[:, i+3] - ret_buff[:, i] - ret_buff[:, mod1(i+1, 3)]
        end
    end

    return dssf
end

# We polarize the system along the x direction to be consistent with KYbSe₂ experiments
function polarize_system!(sys, dir)
    for i in CartesianIndices(sys.dipoles)
        set_dipole!(sys, dir, i)
    end
end

# Use bisection to search for the saturation field B_sat for given exchange parameters. Here we use the classical energy as a criterion for detecting the saturation field.
function determine_B_sat(Js, dir; B_diff_tol=1e-2, energy_tol=1e-8, max_iters=100)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    normalize!(dir)
    # units = Units(:meV, :angstrom)
    a = 1 #4.1149
    c = 5 #22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)

    # g-tensor for KYbSe₂
    gab, gcc = 3.13*1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    #gtens = 1
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(6, 6, 1))

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))

    # Bisection method to find B_sat
    B_min, B_max = 0.0, 100.0
    B_sat = (B_min + B_max) / 2
    iter = 0
    while B_max - B_min > B_diff_tol && iter < max_iters
        iter += 1
        #set_field!(sys, B_sat * units.T * dir)
        set_field!(sys, B_sat * dir)
        polarize_system!(sys, -dir)
        e_ref = energy_per_site(sys)
        randomize_spins!(sys)
        minimize_energy!(sys)
        e_run = energy_per_site(sys)
        if e_run < e_ref - energy_tol
            B_min = B_sat
        else
            B_max = B_sat
        end
        B_sat = (B_min + B_max) / 2
    end

    sys_min = reshape_supercell(sys, diagm(ones(3)))
    polarize_system!(sys_min, -dir)

    return B_sat, sys_min
end


"""
    triangular_nlswt_lanczos_calculations(Js, clustersize)

Constructs a non-linear spin wave theory model for the triangular lattice antiferromagnet
with the specified parameters.
On the nearest-neighbor bond, the 3×3 exchange tensor is given by:

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]

On the next-nearest-neighbor bond, the exchange is isotropic with strength J2.
"""
function triangular_nlswt_lanczos_calculations(Js, clustersize, dir; B=nothing, Qz=0, opts...)
    units = Units(:meV, :angstrom)

    B_sat, sys_min = determine_B_sat(Js, dir; opts...)
    #@show B_sat
    # If B is not provided, pick a value between 1.1 and 1.2 times the saturation field.
    #B = isnothing(B) ? (1.1 + 0.1 * rand()) * B_sat : B
    B = isnothing(B) ? 1.1 * B_sat : B

    # Avoid extremely small fields.
    B = B < 1e-1 ? B + 5.0 : B

    #set_field!(sys_min, B * units.T * dir)
    set_field!(sys_min, B * dir)

    swt = SpinWaveTheory(sys_min; measure=ssf_perp(sys_min))
    try
        npt = Sunny.NonPerturbativeTheory(swt, clustersize)
        # Let's focus on the [H,H,0] path
        qs = [[0, 0, Qz], [1, 1, Qz]]
        path = Sunny.q_space_path_npt(npt, qs)
        niters = 1000
        atol = 1e-5
        as_path = zeros(length(path.qs), niters, 6)
        bs_path = zeros(length(path.qs), niters-1, 6)
        norm2s_path = zeros(length(path.qs), 6)

        for i in eachindex(path.qs)
            q = path.qs[i]
            as, bs, norm2s = Sunny.modified_lanczos(npt, q, niters; atol=atol)
            as_path[i, :, :] = as
            bs_path[i, :, :] = bs
            norm2s_path[i, :] = norm2s
        end
        return B, path.qs, as_path, bs_path, norm2s_path
    catch _
        return nothing
    end
end