using Sunny
using Plots
using LaTeXStrings
using LinearAlgebra
using Measures

# Reuse Hao's Lanczos and B_sat helpers
include("triangular_ladder.jl")

# -----------------------------
# User parameters
# -----------------------------
Js = [1.0, -0.368421, -1.0, 0.0, 0.473684]  # [J1, Delta, Jpmpm, Jzpm, J2]
dir = [1.0, 0.0, 0.0]
nq = 12  # number of q-points along the path
η = 0.2

function paper_to_ladder_Js(Js)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    return [J1, J1 * Δ, Jpmpm, Jzpm, J2]
end

# -----------------------------
# Method 1: greedy adaptive knots (baseline, from Plot_interpolation.jl)
# -----------------------------
function adaptive_knots(ω, S; N=10)
    n = length(ω)
    keep = sort([1, n])
    linpred(i, j, k) = S[i] + (S[j]-S[i]) * (ω[k]-ω[i]) / (ω[j]-ω[i])
    while length(keep) < N
        keep = sort(keep)
        best_err, best_k = -Inf, nothing
        for seg in 1:(length(keep)-1)
            i, j = keep[seg], keep[seg+1]
            j <= i + 1 && continue
            for k in (i+1):(j-1)
                err = abs(S[k] - linpred(i, j, k))
                if err > best_err
                    best_err, best_k = err, k
                end
            end
        end
        best_k === nothing && break
        push!(keep, best_k)
    end
    keep = sort(unique(keep))
    return ω[keep], S[keep], keep
end

# -----------------------------
# Method 2: Gaussian mixture discretization, fixed uniform weight π_k = 1/N
# Treats S(ω) as an unnormalized density over ω and fits (μ_k, σ_k) via
# weighted EM, holding the mixture weights fixed and uniform.
# -----------------------------
gauss(x, μ, σ) = exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))

function fit_gmm_full(ω, S; N=10, n_iter=200, tol=1e-10)
    Δω = ω[2] - ω[1]
    w = max.(S, 0.0)
    wsum = sum(w)
    @assert wsum > 0 "signal has no positive mass"
    p = w ./ wsum

    cumw = cumsum(p)
    μ = [ω[searchsortedfirst(cumw, q)] for q in range(1/(2N), 1 - 1/(2N); length=N)]
    span = maximum(ω) - minimum(ω)
    σ = fill(span / (2N), N)
    π_k = fill(1.0 / N, N)              # initial guess only, now updated below

    n = length(ω)
    r = zeros(n, N)
    prev_ll = -Inf

    for _ in 1:n_iter
        # E-step: responsibilities
        for i in 1:n
            tot = 0.0
            for k in 1:N
                r[i, k] = π_k[k] * gauss(ω[i], μ[k], σ[k])
                tot += r[i, k]
            end
            tot = max(tot, 1e-300)
            @views r[i, :] ./= tot
        end

        # M-step: update π_k, μ_k, σ_k
        for k in 1:N
            wk = p .* @view r[:, k]
            Nk = sum(wk)
            π_k[k] = Nk                  # weight is just the responsibility-weighted mass
            Nk < 1e-12 && continue
            μ[k] = sum(wk .* ω) / Nk
            σ2 = sum(wk .* (ω .- μ[k]) .^ 2) / Nk
            σ[k] = sqrt(max(σ2, 1e-6))
        end
        π_k ./= sum(π_k)                 # guard against roundoff drift

        # weighted log-likelihood for convergence check
        ll = 0.0
        for i in 1:n
            s = 0.0
            for k in 1:N
                s += π_k[k] * gauss(ω[i], μ[k], σ[k])
            end
            ll += p[i] * log(max(s, 1e-300))
        end
        abs(ll - prev_ll) < tol && break
        prev_ll = ll
    end

    A = π_k .* wsum .* Δω
    return μ, σ, A, π_k
end

function reconstruct_gmm(ω, μ, σ, A)
    S_hat = zeros(length(ω))
    for k in eachindex(μ)
        S_hat .+= A[k] .* gauss.(ω, μ[k], σ[k])
    end
    return S_hat
end

# -----------------------------
# Compute Ladder DSSF
# -----------------------------
B, sys, _ = polarized_system(paper_to_ladder_Js(Js), dir)
qpoints = [[0, 0, 0], [1, 1, 0]]
path = q_space_path(sys.crystal, qpoints, nq)
swt_ladder = SpinWaveTheory(sys; measure=dssf_measure(sys))
e_max = energy_max(swt_ladder)
energies = collect(0.0:0.01:e_max)
ladder_full, nbreak = ladder_dssf(swt_ladder, path; energies,η, threaded=false)
intensities = @view ladder_full[7, :, :] # perpendicular channel

S_full = intensities[1, :]   # Γ-point curve, same one used in Plot_interpolation.jl

N_knots = 20
E_adapt, S_adapt, _ = adaptive_knots(energies, S_full; N=N_knots)
μ_gmm, σ_gmm, A_gmm, π_gmm = fit_gmm_full(energies, S_full; N=N_knots)
S_gmm_recon = reconstruct_gmm(energies, μ_gmm, σ_gmm, A_gmm)

# -----------------------------
# Plot and save
# -----------------------------
p1 = plot(energies, S_full;
    label = "Full", color = :royalblue, linewidth = 3,
    title = "Linear adaptive knots",
    xlabel = L"\omega / |J_1|", ylabel = L"I(\mathbf{k}=\Gamma, \omega)",
    xlims = (0, 15), legend = :topright, guidefontsize = 14, titlefontsize = 14)
scatter!(p1, E_adapt, S_adapt; label = "Knots (N=$N_knots)", color = :crimson, markersize = 4)

p2 = plot(energies, S_full;
    label = "Full", color = :royalblue, linewidth = 3,
    title = "Gaussian mixture model",
    xlabel = L"\omega / |J_1|", ylabel = L"I(\mathbf{k}=\Gamma, \omega)",
    xlims = (0, 15), legend = :topright, guidefontsize = 14, titlefontsize = 14)
plot!(p2, energies, S_gmm_recon; label = "GMM", color = :darkorange, linewidth = 2, linestyle = :dash)
scatter!(p2, μ_gmm, A_gmm ./ (σ_gmm .* sqrt(2π)); label = "Means (N=$N_knots)", color = :darkorange, markersize = 4)

# Panel labels, positioned relative to each panel
annotate!(p1, ((-0.10, 1.06), text("a", 18, :black)))
annotate!(p2, ((-0.10, 1.06), text("b", 18, :black)))

fig = plot(p1, p2; layout = (1, 2), size = (900, 400),margin = 6mm, 
    framestyle = :box,
    tickfontsize = 14,
    legendfontsize = 14,
    )

savefig(fig, "../plots/GMM_vs_AdaptiveKnots.pdf")
println("Saved: ../plots/GMM_vs_AdaptiveKnots.pdf")