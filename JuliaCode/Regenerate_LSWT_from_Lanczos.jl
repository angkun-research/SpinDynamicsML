using Sunny
using CSV, DataFrames
using ProgressBars
using Base.Threads

# -----------------------------
# I/O
# -----------------------------
const INPUT_CSV  = "./data/HaoResults/Hao_anisotropy_Js4_uniform_trace_Filter20_Bsats.csv"
const OUTPUT_CSV = "./data/HaoResults/LSWT_anisotropy_Js4_uniform_trace_Filter20_B60.csv"

# -----------------------------
# Physics / sampling config
# -----------------------------
const ENERGY_MIN  = 0.0
const ENERGY_MAX  = 50.0
const ENERGY_STEP = 0.01
const ENERGIES    = collect(ENERGY_MIN:ENERGY_STEP:ENERGY_MAX)

const ETA         = 0.2
const NQ          = 12      # momentum points
const NKNOTS      = 20      # adaptive_knots points
const DIMS        = (1, 1, 1)

# -----------------------------
# Helpers
# -----------------------------
function parse_float_vec(x)
    if x === missing
        return Float64[]
    elseif x isa AbstractVector
        return Float64.(x)
    elseif x isa AbstractString
        s = strip(x)
        isempty(s) && return Float64[]
        if startswith(s, "[") && endswith(s, "]")
            s = s[2:end-1]
        end
        s = strip(s)
        isempty(s) && return Float64[]
        return parse.(Float64, strip.(split(s, ",")))
    else
        return [Float64(x)]
    end
end

function parse_float_scalar(x)
    if x === missing
        return NaN
    elseif x isa Number
        return Float64(x)
    else
        return parse(Float64, strip(String(x)))
    end
end

# Same adaptive filter as Hao pipeline
function adaptive_knots(ω, S; N=10)
    n = length(ω)
    @assert n == length(S)
    keep = sort([1, n])

    linpred(i, j, k) = S[i] + (S[j] - S[i]) * (ω[k] - ω[i]) / (ω[j] - ω[i])

    while length(keep) < N
        keep = sort(keep)
        best_err = -Inf
        best_k = nothing

        for seg in 1:(length(keep) - 1)
            i, j = keep[seg], keep[seg + 1]
            if j <= i + 1
                continue
            end
            for k in (i + 1):(j - 1)
                err = abs(S[k] - linpred(i, j, k))
                if err > best_err
                    best_err = err
                    best_k = k
                end
            end
        end

        best_k === nothing && break
        push!(keep, best_k)
    end

    keep = sort(unique(keep))
    return ω[keep], S[keep], keep
end

function polarize_system!(sys)
    for i in CartesianIndices(sys.dipoles)
        set_dipole!(sys, (-1.0, 0.0, 0.0), i)
    end
end

# Scalar LSWT as in collectLSWTdata.jl style (ssf_perp)
function run_lswt_scalar(Js, B; dims=(1,1,1), nqp=12)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    units = Units(:meV, :angstrom)

    a = 4.1149
    c = 22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    cryst = Crystal(latvecs, [[0, 0, 0]], 162)

    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst, qs, nqp)

    gab, gcc = 3.13 * 1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten = [J1 + 2Jpmpm  0           Jzpm;
              0            J1 - 2Jpmpm 0;
              Jzpm         0           J1 * Δ]

    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2,    Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [B, 0.0, 0.0] * units.T)

    polarize_system!(sys)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    return intensities_bands(swt, path)
end

# res.disp/res.data are discrete modes -> broaden to S(q,w)
function lswt_modes_to_heatmap(disp, data, energies; η=0.2)
    nbands, nq = size(disp)
    img = zeros(Float64, nq, length(energies))

    for iq in 1:nq
        for ib in 1:nbands
            E0 = real(disp[ib, iq])
            A  = real(data[ib, iq])
            if !isfinite(E0) || !isfinite(A)
                continue
            end
            img[iq, :] .+= A .* (η / π) ./ ((energies .- E0).^2 .+ η^2)
        end
    end
    return img
end

function filter_scalar_heatmap(img, energies; N=20)
    nq, _ = size(img)
    Es_all = Float64[]
    Ss_all = Float64[]
    for iq in 1:nq
        E_keep, S_keep, _ = adaptive_knots(energies, vec(img[iq, :]); N=N)
        append!(Es_all, E_keep)
        append!(Ss_all, S_keep)
    end
    return Es_all, Ss_all
end

# -----------------------------
# Main
# -----------------------------
println("Reading: ", INPUT_CSV)
df = CSV.read(INPUT_CSV, DataFrame)

# Normalize column name text (handles BOM, spaces, case)
canon(x) = lowercase(strip(replace(String(x), '\ufeff' => "")))
name_map = Dict(canon(nm) => nm for nm in names(df))

# Resolve required columns with flexible aliases
js_col   = get(name_map, "js", nothing)
disp_col = get(name_map, "disp", nothing)
data_col = get(name_map, "data", nothing)
b_col    = get(name_map, "b", nothing)
bsats_col = get(name_map, "bsats", nothing)

# Fallback: CSV may have no header and become Column1..ColumnN
if js_col === nothing && disp_col === nothing && data_col === nothing && ncol(df) >= 3
    println("No standard headers found; assuming positional columns: Js,data,disp,(B),(Bsats)")
    cols = names(df)
    js_col   = cols[1]
    data_col = cols[2]
    disp_col = cols[3]
    b_col    = ncol(df) >= 4 ? cols[4] : nothing
    bsats_col = ncol(df) >= 5 ? cols[5] : nothing
end

@assert js_col !== nothing   "Could not locate Js column. Found columns: $(names(df))"
@assert disp_col !== nothing "Could not locate disp column. Found columns: $(names(df))"
@assert data_col !== nothing "Could not locate data column. Found columns: $(names(df))"
@assert b_col !== nothing    "Could not locate B column. Found columns: $(names(df))"


const OutRow = NamedTuple{(:Js, :disp, :data, :B),
    Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}

rows = Vector{OutRow}(undef, nrow(df))

println("Processing rows: ", nrow(df), " with ", nthreads(), " thread(s)")
Threads.@threads for i in ProgressBar(1:nrow(df))
    row = df[i, :]

    Js = parse_float_vec(row[js_col])
    @assert length(Js) == 4 || length(Js) == 5 "Row $i: Js must have length 4 or 5"

    # If CSV stores Js as [J1, Δ, Jpmpm, J2], insert Jzpm=0 to recover length-5
    Js5 = length(Js) == 5 ? Js : [Js[1], Js[2], Js[3], 0.0, Js[4]]

    B_sat = parse_float_scalar(row[b_col])
    @assert isfinite(B_sat) "Row $i: B is missing or invalid"
    B_sat = 60.0  # Override with computed B_sat for testing

    res = run_lswt_scalar(Js5, B_sat; dims=DIMS, nqp=NQ)
    img = lswt_modes_to_heatmap(res.disp, res.data, ENERGIES; η=ETA)
    E_filtered, S_filtered = filter_scalar_heatmap(img, ENERGIES; N=NKNOTS)

    # Keep Js field shape same as input (4 or 5) for compatibility
    Js_out = length(Js) == 5 ? Js5 : [Js5[1], Js5[2], Js5[3], Js5[5]]

    rows[i] = (
        Js = Js_out,
        disp = E_filtered,   # filtered energies
        data = S_filtered,   # filtered broadened intensity
        B = B_sat,
    )
end

out_df = DataFrame(rows)
println("Writing: ", OUTPUT_CSV)
CSV.write(OUTPUT_CSV, out_df)

println("Done. Rows: ", nrow(out_df))
println("Each row length (disp/data) = ", NQ * NKNOTS)