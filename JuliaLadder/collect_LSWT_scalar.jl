using Sunny, DataFrames, CSV, JLD2
#using GLMakie
#using Plots
#using ProgressBars
using ProgressMeter 
using Logging
Logging.disable_logging(Logging.Warn) # Disable warning logs to reduce console clutter

@show Threads.nthreads()
include("triangular_ladder.jl")

function Run_LSWT(sys,B; Nk=60, QZ=0)
    qs = [[0, 0, QZ], [1, 1, QZ]]
    path = q_space_path(sys.crystal,qs,Nk)

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    res = intensities_bands(swt, path)

    return res
end


Jsize = 4 #5 #4
N_uniform = 30 #30
Nk = 60 #12
Nsamples = 2*N_uniform^(Jsize-1)
Js_all = Vector{Float64}[]
for J10 in [-1,1]
    for ΔJ10 in collect(range(-1, 1, length=N_uniform+1))
        Δ = ΔJ10 / J10
        if Δ == 0.0
            continue
        end
        for Jpmpm in collect(range(-1, 1, length=N_uniform+1))
            if Jpmpm == 0.0
                continue
            end
            for J20 in collect(range(-1, 1, length=N_uniform+1))
                if J20 == 0.0
                    continue
                end
                if Jsize == 4
                    Js = [J10, Δ, Jpmpm, 0.0, J20]
                    push!(Js_all, Js)
                else
                    for Jzpm in collect(range(-1, 1, length=N_uniform+1))
                        if Jzpm == 0.0
                            continue
                        end
                        Js = [J10, Δ, Jpmpm, Jzpm, J20]
                        push!(Js_all, Js)
                    end
                end
            end
        end
    end
end
println("Total Js configurations: ", length(Js_all))

QZ = 0

const ScalarRow = NamedTuple{
    (:Js, :data, :disp, :B_sat),
    Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}
}
rows = Vector{ScalarRow}(undef, Nsamples)
p = Progress(Nsamples)

# results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[], B_sat=Float64[])
# results_lock = ReentrantLock()
#for taskid in ProgressBar(1:Nsamples)
Threads.@threads for taskid in 1:Nsamples
    Js = Js_all[taskid]
    B, sys, _ = polarized_system(Js, [1.0, 0.0, 0.0])

    res = Run_LSWT(sys, B;Nk=Nk, QZ=QZ)

    rows[taskid] = (
        Js = Jsize == 4 ? Js[[1, 2, 3, 5]] : copy(Js),
        data = vec(res.data),
        disp = vec(res.disp),
        B_sat = B,
    )
    next!(p) 
    # if Jsize == 4
    #     Js = Js[[1, 2, 3, 5]]
    # end
    # lock(results_lock) do
    #     push!(results, (
    #         copy(Js), # only keep J1, Δ, J1pmpm, J2
    #         vec(res.data),
    #         vec(res.disp),
    #         copy(B)
    #     ))
    #     next!(p)
    # end
end

results = DataFrame(rows)
CSV.write("../data/triangular_scalar_uniform_Js$(Jsize)_Bdep_Nk$(Nk)_NJ$(N_uniform)_QZ$(QZ)_unitless.csv", results)
println("saved results to data")
