# LSWT DSSF channels of J and their difference under Jzpm → -Jzpm.
#
# From JuliaLadder:
#     julia --project=. -t auto fig2_distinguishable_lswt.jl
#
# Writes ../plots/fig2_distinguishable_lswt.pdf.

using CairoMakie, Printf
using LaTeXStrings
include(joinpath(@__DIR__, "triangular_ladder.jl"))

# ---- Settings ----
Js1 = [1.0, 0.7, 0.2,  0.2, 0.051]
Js2 = [1.0, 0.7, 0.2, -0.2, 0.051]
dir = [1.0, 0.0, 0.0]

qpath = [[0, 0, 0], [1, 1, 0]]
nq = 60
η = 0.2                         # Gaussian standard deviation
dE = 0.01
e_max = 8.0
outfile = joinpath(
    @__DIR__, "..", "plots", "fig2_distinguishable_lswt.pdf"
)

Sunny.load_blas_for_threading()
BLAS.set_num_threads(1)

# ---- Compute ----
# Both Hamiltonians at the field of Js1.
B, sys1, _ = polarized_system(Js1, dir)
_, sys2, _ = polarized_system(Js2, dir; B)

qs = q_space_path(sys1.crystal, qpath, nq).qs
hs = [q[1] for q in qs]
energies = collect(0.0:dE:e_max)

# Convert discrete LSWT bands to an intensity heatmap.
# Both disp and data must have dimensions (band × q).
function lswt_to_heatmap(disp, data, energies; η=0.2)
    nbands, nq = size(disp)
    img = zeros(Float64, nq, length(energies))
    normalization = 1.0 / (sqrt(2π) * η)

    for iq in 1:nq
        for ib in 1:nbands
            E0 = disp[ib, iq]
            A = data[ib, iq]

            img[iq, :] .+= A .* normalization .*
                exp.(-0.5 .* ((energies .- E0) ./ η).^2)

            # For Lorentzian broadening, replace the expression above with:
            # img[iq, :] .+= A .* (η / π) ./
            #     ((energies .- E0).^2 .+ η^2)
            # Then η denotes the Lorentzian HWHM.
        end
    end
    return img
end

function channels(sys)
    swt_scalar = SpinWaveTheory(sys; measure=ssf_perp(sys))
    measure = ssf_custom((q, ssf) -> ssf, sys)
    swt_tensor = SpinWaveTheory(sys; measure=measure)

    path = q_space_path(sys.crystal, qpath, nq)

    res_scalar = intensities_bands(swt_scalar, path)
    scalar_img = lswt_to_heatmap(
        res_scalar.disp, res_scalar.data, energies; η=η
    )

    res_tensor = intensities_bands(swt_tensor, path)
    T = res_tensor.data

    # Keep all bands and form the symmetric off-diagonal channels.
    S_tensor = [
        real.(getindex.(T, 1, 1)),
        real.(getindex.(T, 2, 2)),
        real.(getindex.(T, 3, 3)),
        real.(getindex.(T, 1, 2) .+ getindex.(T, 2, 1)),
        real.(getindex.(T, 2, 3) .+ getindex.(T, 3, 2)),
        real.(getindex.(T, 3, 1) .+ getindex.(T, 1, 3)),
    ]

    data = zeros(7, length(path.qs), length(energies))
    for i in 1:6
        data[i, :, :] = lswt_to_heatmap(
            res_tensor.disp, S_tensor[i], energies; η=η
        )
    end
    data[7, :, :] = scalar_img
    return data
end

data1 = channels(sys1)
data2 = channels(sys2)

# DSSF_CHANNELS: xx, yy, zz, xy+yx, yz+zy, zx+xz, perp
cols = [7, 4, 5, 6]
titles1 = [
    L"S_\perp(\mathrm{\mathbf{J}})",
    L"(S_{xy}+S_{yx})(\mathrm{\mathbf{J}})",
    L"(S_{yz}+S_{zy})(\mathrm{\mathbf{J}})",
    L"(S_{zx}+S_{xz})(\mathrm{\mathbf{J}})",
]
titles2 = [
    L"\Delta S_\perp",
    L"\Delta(S_{xy}+S_{yx})",
    L"\Delta(S_{yz}+S_{zy})",
    L"\Delta(S_{zx}+S_{xz})",
]

S1 = [data1[c, :, :] for c in cols]
ΔS = [data2[c, :, :] - data1[c, :, :] for c in cols]
rel = [
    iszero(norm(s)) ? NaN : norm(d) / norm(s)
    for (d, s) in zip(ΔS, S1)
]

@printf("B = %.4f\n", B)
for (c, r, d) in zip(cols, rel, ΔS)
    @printf(
        "%-6s  max|ΔS| = %.2e   ‖ΔS‖/‖S‖ = %.3f\n",
        DSSF_CHANNELS[c], maximum(abs, d), r
    )
end

# ---- Plot ----
sci_ticks(vals) = map(vals) do v
    iszero(v) && return L"0"
    s = @sprintf("%.2g", v)
    if occursin('e', s)
        a, b = split(s, 'e')
        latexstring(
            a == "1" ? "" : "$a\\times",
            "10^{$(parse(Int, b))}"
        )
    else
        latexstring(s)
    end
end

fig = with_theme(theme_latexfonts(); fontsize=9) do
    fig = Figure(size=(510, 300), figure_padding=(2, 6, 2, 2))
    dmax = max(maximum(d -> maximum(abs, d), ΔS), 1e-12)

    for (i, c) in enumerate(cols)
        signed = 4 ≤ c ≤ 6
        m = max(maximum(abs, S1[i]), 1e-12)
        common = (;
            xticks=([0, 0.5, 1], [L"0", L"0.5", L"1"]),
            yticks=0:2:8,
            ylabel=L"\omega/J_1",
            ylabelvisible=i == 1,
            yticklabelsvisible=i == 1,
            titlesize=10,
            titlegap=2,
            xticklabelsize=8,
            yticklabelsize=8,
        )

        ax1 = Axis(
            fig[2, i];
            title=titles1[i], xticklabelsvisible=false, common...
        )
        hm1 = heatmap!(
            ax1, hs, energies, S1[i];
            colormap=signed ? :bluesreds : :viridis,
            colorrange=signed ? (-m, m) : (0, m),
            rasterize=4,
        )
        Colorbar(
            fig[1, i], hm1;
            vertical=false, height=5, ticklabelsize=7, ticksize=2,
            ticks=WilkinsonTicks(3; k_min=3), tickformat=sci_ticks,
        )

        ax2 = Axis(
            fig[4, i];
            title=titles2[i], xlabel=L"(h,h,0)", common...
        )
        hm2 = heatmap!(
            ax2, hs, energies, ΔS[i];
            colormap=:bluesreds,
            colorrange=(-dmax, dmax),
            rasterize=4,
        )
        Colorbar(
            fig[3, i], hm2;
            vertical=false, height=5, ticklabelsize=7, ticksize=2,
            ticks=WilkinsonTicks(3; k_min=3), tickformat=sci_ticks,
        )

        annotation = if isnan(rel[i])
            L"\|S\|=0"
        else
            latexstring(
                @sprintf("\\|\\Delta S\\|/\\|S\\| = %.2f", rel[i])
            )
        end
        text!(
            ax2, 0.04, 0.95e_max;
            text=annotation, align=(:left, :top), fontsize=8,
        )

        for ax in (ax1, ax2)
            ylims!(ax, 0, e_max)
        end
    end

    rowgap!(fig.layout, 3)
    rowgap!(fig.layout, 2, 6)
    colgap!(fig.layout, 8)
    fig
end

mkpath(dirname(outfile))
save(outfile, fig; pt_per_unit=1)
println("Wrote ", normpath(outfile))
fig
