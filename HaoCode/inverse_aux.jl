using Sunny

function map_lanczos_to_intensities(as, bs, norm2s, npt::NonPerturbativeTheory, q, ωs, η::Float64)
    (; swt) = npt
    ret_buff = zeros(length(ωs), 6)
    num_obs = num_observables(swt.measure)
    niters = size(as, 1)

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

    # Apply the neutron polarization factor
    cryst = orig_crystal(swt.sys)
    q_global = cryst.recipvecs * q
    q2 = norm2(q_global)

    ret = zeros(length(ωs))
    if q2 < 1e-6
        # Later we may add the 2/3 factor to be consistent with the Sunny main
        for i in 1:num_obs
            @. ret += ret_buff[:, i]
        end
    else
        for i in 1:num_obs
            @. ret += ret_buff[:, i] * (1 - q_global[i]^2 / q2)
            @. ret -= (ret_buff[:, i+3] - ret_buff[:, i] - ret_buff[:, mod1(i+1, 3)]) * q_global[i] * q_global[mod1(i+1, 3)] / q2
        end
    end

    return ret
end