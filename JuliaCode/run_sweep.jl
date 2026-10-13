using Random
using JLD2

include("triangular_nlswt.jl")

const NUM_SAMPLES = 10_000
const CLUSTERSIZE = (11, 11, 1)

# Slurm array info (defaults allow local testing)
task_id = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
ntasks  = parse(Int, get(ENV, "SLURM_ARRAY_TASK_COUNT", "1"))
seed    = parse(Int, get(ENV, "BASE_SEED", "12345"))

# Create ONE shared output folder (safe if called by many tasks)
const OUTDIR = "results"
mkpath(OUTDIR)

# Split indices [1, NUM_SAMPLES] evenly across array tasks
function chunk_range(task_id::Int, ntasks::Int, n::Int)
    a = floor(Int, (task_id - 1) * n / ntasks) + 1
    b = floor(Int, task_id * n / ntasks)
    return a:b
end

# Reproducible per-sample sampling (independent of chunking)
function sample_Js(i::Int, seed::Int)
    rng = MersenneTwister(hash((seed, i)))
    return [2rand(rng) - 1,
            rand(rng),
            2rand(rng) - 1,
            2rand(rng) - 1,
            2rand(rng) - 1]
end

idxs = chunk_range(task_id, ntasks, NUM_SAMPLES)

# Collect successful results; write once (fast I/O)
Js_list      = Vector{Vector{Float64}}()
B_list       = Float64[]
qs_list      = Any[]
as_list      = Any[]
bs_list      = Any[]
norm2s_list  = Any[]
idx_ok       = Int[]

for i in idxs
    Js = sample_Js(i, seed)
    res = triangular_nlswt_lanczos_calculations(Js, CLUSTERSIZE)
    if res !== nothing
        B, qs_path, as_path, bs_path, norm2s_path = res
        push!(idx_ok, i)
        push!(Js_list, Js)
        push!(B_list, B)
        push!(qs_list, qs_path)
        push!(as_list, as_path)
        push!(bs_list, bs_path)
        push!(norm2s_list, norm2s_path)
    end
end

out = joinpath(OUTDIR, "results_task$(task_id)_of_$(ntasks).jld2")
@save out seed task_id ntasks idxs idx_ok CLUSTERSIZE Js_list B_list qs_list as_list bs_list norm2s_list

println("Done task $(task_id)/$(ntasks): attempted $(length(idxs)) samples, success $(length(idx_ok)). Wrote $out")
