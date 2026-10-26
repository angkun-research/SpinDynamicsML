using Sunny, DataFrames, CSV, JLD2
#using ProgressBars
using ProgressMeter 
using Logging
Logging.disable_logging(Logging.Warn) # Disable warning logs to reduce console clutter

@show Threads.nthreads()
include("triangular_ladder.jl")

function Run_LSWT(sys,B; Nk = 60, QZ = 0)
    qs = [[0, 0, QZ], [1, 1, QZ]]
    path = q_space_path(sys.crystal,qs,Nk)

    #measure = ssf_perp(sys)
    measure = ssf_custom((q, ssf) -> ssf, sys)
    #measure = ssf_custom((q, ssf) -> (ssf[1, 1], ssf[2,2], ssf[3,3], ssf[1,2], ssf[1,3], ssf[2,3]), sys)

    swt = SpinWaveTheory(sys; measure=measure)
    res = intensities_bands(swt, path)
    return res
end


Jsize = 4
N_uniform = 20
Nk=60
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
QZ = 0

const RowT = NamedTuple{(:Js,:disp, :B,:IntSxx,:IntSyy,:IntSzz,:IntSxy,:IntSxz,:IntSyz), Tuple{Vector{Float64}, Vector{Float64},Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}
#rows = RowT[]
rows = Vector{RowT}(undef, Nsamples)

p = Progress(Nsamples)
#results_lock = ReentrantLock()
# Uniformly sample Js from the Jsize dimensional space with each J in range -1 to 1
Threads.@threads for taskid in 1:Nsamples
    Js = Js_all[taskid]
    B, sys, _ = polarized_system(Js, [1.0, 0.0, 0.0])
    res = Run_LSWT(sys, B; Nk=Nk, QZ=QZ)
    Sxxtemp = real.(getindex.(res.data[1, :], 1, 1)) 
    Syytemp = real.(getindex.(res.data[1, :], 2, 2))
    Szztemp = real.(getindex.(res.data[1, :], 3, 3))
    Sxytemp = real.(getindex.(res.data[1, :], 1, 2))
    Sxztemp = real.(getindex.(res.data[1, :], 1, 3))
    Syztemp = real.(getindex.(res.data[1, :], 2, 3))
    rows[taskid] = (Js = copy(Js),disp = vec(res.disp),B = B,
        IntSxx = Sxxtemp,IntSyy = Syytemp,IntSzz = Szztemp,
        IntSxy = Sxytemp,IntSxz = Sxztemp,IntSyz = Syztemp,
    )
    next!(p)

    # if Jsize == 4
    #     Js = Js[[1, 2, 3, 5]]
    # end
    # lock(results_lock) do
    #     push!(rows, (Js = copy(Js), disp = vec(res.disp), B = copy(B), 
    #             IntSxx = copy(Sxxtemp), IntSyy = copy(Syytemp), IntSzz = copy(Szztemp), 
    #             IntSxy = copy(Sxytemp), IntSxz = copy(Sxztemp), IntSyz = copy(Syztemp)))
    #     next!(p)
    # end
end

println("Total rows collected: ", length(rows))
results = DataFrame(rows)
CSV.write("data/triangular_tensor_uniform_Js$(Jsize)_Bdep_Nk$(Nk)_NJ$(N_uniform)_unitless.csv", results)
