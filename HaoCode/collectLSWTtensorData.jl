using Sunny, DataFrames, CSV, JLD2
using ProgressBars
# export num_observables from Sunny
#import Sunny: num_observables, num_correlations

# We polarize the system along the x direction to be consistent with KYbSe₂ experiments
function polarize_system!(sys)
    for i in CartesianIndices(sys.dipoles)
        set_dipole!(sys, (-1, 0, 0), i)
    end
end

function Run_LSWT(Js,B; dims = (1,1,1),Nk = 12)
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    units = Units(:meV, :angstrom)
    a = 4.1149
    c = 22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)
    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst,qs,Nk)

    # g-tensor for KYbSe₂
    gab, gcc = 3.13*1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    #sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(6, 6, 1))
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [B, 0, 0] * units.T)

    #measure = ssf_perp(sys)
    measure = ssf_custom((q, ssf) -> ssf, sys)
    #measure = ssf_custom((q, ssf) -> (ssf[1, 1], ssf[2,2], ssf[3,3], ssf[1,2], ssf[1,3], ssf[2,3]), sys)
    # Nobs = num_observables(measure)
    # Ncorr = num_correlations(measure)
    # println("Number of observables: $Nobs, Number of correlations: $Ncorr")

    # randomize_spins!(sys)
    # minimize_energy!(sys)
    polarize_system!(sys)

    swt = SpinWaveTheory(sys; measure=measure)
    res = intensities_bands(swt, path)

    #plot_intensities(res; units, title="Triangular Lattice")
    return res
end


# Jsize = 5
# num_tasks = 250
# #B_same = 20
# #components = [[1, 1], [2, 2], [3, 3], [1, 2], [1, 3], [2, 3]]
# # Store rows in a thread-safe way (no shared mutation inside threads)
# const RowT = NamedTuple{(:Js,:disp, :B,:IntSxx,:IntSyy,:IntSzz,:IntSxy,:IntSxz,:IntSyz), Tuple{Vector{Float64}, Vector{Float64},Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}
# rows = RowT[]

# # Randomly sample Js from the Jsize dimensional space with each J in range -1 to 1
# for taskid in ProgressBar(1:num_tasks)
#     filetemp = load("data/HaoResults/results_task$(taskid)_of_$(num_tasks).jld2");
#     num_jobs = length(filetemp["Js_list"])
#     local_rows = Vector{RowT}(undef, num_jobs)
#     for jobid in 1:num_jobs
#         Js = filetemp["Js_list"][jobid]
#         B = filetemp["B_list"][jobid]
#         res = Run_LSWT(Js, B; dims=(1, 1, 1))
#         Sxxtemp = real.(getindex.(res.data[1, :], 1, 1)) 
#         Syytemp = real.(getindex.(res.data[1, :], 2, 2))
#         Szztemp = real.(getindex.(res.data[1, :], 3, 3))
#         Sxytemp = real.(getindex.(res.data[1, :], 1, 2))
#         Sxztemp = real.(getindex.(res.data[1, :], 1, 3))
#         Syztemp = real.(getindex.(res.data[1, :], 2, 3))
#         local_rows[jobid] = (Js = copy(Js), disp = vec(res.disp), B = copy(B), 
#             IntSxx = copy(Sxxtemp), IntSyy = copy(Syytemp), IntSzz = copy(Szztemp), 
#             IntSxy = copy(Sxytemp), IntSxz = copy(Sxztemp), IntSyz = copy(Syztemp))
#     end
#     append!(rows, local_rows)
# end

# println("Total rows collected: ", length(rows))
# results = DataFrame(rows)
# CSV.write("data/HaoResults/Hao_anisotropy_LSWT_Tensor_Js5_Bdep.csv", results)



Jsize = 5
N_uniform = 20
Nk=20
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
            for Jzpm in collect(range(-1, 1, length=N_uniform+1))
                if Jzpm == 0.0
                    continue
                end
                for J20 in collect(range(-1, 1, length=N_uniform+1))
                    if J20 == 0.0
                        continue
                    end
                    Js = [J10, Δ, Jpmpm, Jzpm, J20]
                    push!(Js_all, Js)
                end
            end
        end
    end
end
B_same = 60
const RowT = NamedTuple{(:Js,:disp, :B,:IntSxx,:IntSyy,:IntSzz,:IntSxy,:IntSxz,:IntSyz), Tuple{Vector{Float64}, Vector{Float64},Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}
rows = RowT[]

# Uniformly sample Js from the Jsize dimensional space with each J in range -1 to 1
for taskid in ProgressBar(1:Nsamples)
    Js = Js_all[taskid]
    res = Run_LSWT(Js, B_same; dims=(1, 1, 1),Nk=Nk)
    # local_rows = Vector{RowT}(undef, 1)
    Sxxtemp = real.(getindex.(res.data[1, :], 1, 1)) 
    Syytemp = real.(getindex.(res.data[1, :], 2, 2))
    Szztemp = real.(getindex.(res.data[1, :], 3, 3))
    Sxytemp = real.(getindex.(res.data[1, :], 1, 2))
    Sxztemp = real.(getindex.(res.data[1, :], 1, 3))
    Syztemp = real.(getindex.(res.data[1, :], 2, 3))
    # local_rows[1] = (Js = copy(Js), disp = vec(res.disp), B = copy(B_same), 
    #         IntSxx = copy(Sxxtemp), IntSyy = copy(Syytemp), IntSzz = copy(Szztemp), 
    #         IntSxy = copy(Sxytemp), IntSxz = copy(Sxztemp), IntSyz = copy(Syztemp))
    # append!(rows, local_rows)
    push!(rows, (Js = copy(Js), disp = vec(res.disp), B = copy(B_same), 
            IntSxx = copy(Sxxtemp), IntSyy = copy(Syytemp), IntSzz = copy(Szztemp), 
            IntSxy = copy(Sxytemp), IntSxz = copy(Sxztemp), IntSyz = copy(Syztemp)))
end

println("Total rows collected: ", length(rows))
results = DataFrame(rows)
CSV.write("data/triangular_anisotropy_uniform_Tensor_Js5_B$(B_same)_Nk$(Nk)_NJ$(N_uniform).csv", results)



# # read back the data
# results = CSV.read("data/triangular_heisenberg_Js10_h20.csv", DataFrame)
# # extract qs
# qs = results.qs[1]

# plot the results
# using GLMakie
# using Plots
# xs = collect(1:length(results.disp[1]))
# ys = results.disp[1]

# fig = Figure(; size=(800,400))
# ax = Axis(fig[1,1], xlabel="q", ylabel="Energy (meV)", title="Dispersion")
# #lines!(ax, xs, ys)
# scatter!(ax, xs, ys, marker=:circle)
# fig |> display


# filetemp = load("data/HaoResults/results_task1_of_250.jld2");
# num_jobs = length(filetemp["Js_list"])
# jobid = 1
# Js = filetemp["Js_list"][jobid]
# B = filetemp["B_list"][jobid]
# Js[4] = 0.0
# res = Run_LSWT(Js, B; dims=(1, 1, 1))


# using Plots
# N = length(res.disp)
# xs = collect(1:N)
# #ys = vec(res.disp)
# ys = real.(getindex.(res.data[1, :], 2, 3)) # only plot the first observable (Sxx)
# scatter(xs, ys, marker=:circle, xlabel="q", ylabel="Energy (meV)", title="Dispersion")
