using LinearAlgebra

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

function fit_gmm_uniform_weight(ω, S; N=10, n_iter=200, tol=1e-10)
    Δω = ω[2] - ω[1]                    # grid spacing, assumes uniform ω
    w = max.(S, 0.0)
    wsum = sum(w)
    @assert wsum > 0 "signal has no positive mass"
    p = w ./ wsum                      # normalized weights (act like a pmf over ω)

    # init means at evenly spaced quantiles of the weighted distribution
    cumw = cumsum(p)
    μ = [ω[searchsortedfirst(cumw, q)] for q in range(1/(2N), 1 - 1/(2N); length=N)]
    span = maximum(ω) - minimum(ω)
    σ = fill(span / (2N), N)
    π_k = fill(1.0 / N, N)              # FIXED, never updated in M-step

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

        # M-step: update μ, σ only (weights stay uniform/fixed)
        for k in 1:N
            wk = p .* @view r[:, k]
            Nk = sum(wk)
            Nk < 1e-12 && continue
            μ[k] = sum(wk .* ω) / Nk
            σ2 = sum(wk .* (ω .- μ[k]) .^ 2) / Nk
            σ[k] = sqrt(max(σ2, 1e-6))
        end

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

    #A = π_k .* wsum   # amplitude (mass) carried by each fixed-weight component
    A = π_k .* wsum .* Δω  
    return μ, σ, A
end

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
