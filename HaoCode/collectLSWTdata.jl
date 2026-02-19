using Sunny, DataFrames, CSV, JLD2
#using GLMakie
#using Plots
using ProgressBars

function Run_LSWT(Js,B; dims = (1,1,1))
    J1, Δ, Jpmpm, Jzpm, J2 = Js
    units = Units(:meV, :angstrom)
    a = 4.1149
    c = 22.6911
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions, 162)
    qs = [[0, 0, 0], [1, 1, 0]]
    path = q_space_path(cryst,qs,12)

    # g-tensor for KYbSe₂
    gab, gcc = 3.13*1.09310, 1.01
    gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
    #sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(6, 6, 1))
    sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=dims)

    J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [B, 0, 0] * units.T)

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))

    randomize_spins!(sys)
    minimize_energy!(sys)

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    res = intensities_bands(swt, path)

    #plot_intensities(res; units, title="Triangular Lattice")
    return res
end


Jsize = 5
num_tasks = 250
B_same = 20
results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[])

# Randomly sample Js from the Jsize dimensional space with each J in range -1 to 1
for taskid in ProgressBar(1:num_tasks)
    filetemp = load("data/HaoResults/results_task$(taskid)_of_$(num_tasks).jld2");
    num_jobs = length(filetemp["Js_list"])
    for jobid in 1:num_jobs
        Js = filetemp["Js_list"][jobid]
        B = filetemp["B_list"][jobid]
        #res = Run_LSWT(Js, B; dims=(6, 6, 1))
        res = Run_LSWT(Js, B_same)
        push!(results, (
            copy(Js),
            vec(res.data),
            vec(res.disp),
        ))
    end
end

# println(rs)
# println(Js)

CSV.write("data/HaoResults/Hao_anisotropy_Js5_B50.csv", results)

# # read back the data
# results = CSV.read("data/triangular_heisenberg_Js10_h20.csv", DataFrame)
# # extract qs
# qs = results.qs[1]

# plot the results
# xs = collect(1:length(results.disp[1]))
# ys = results.disp[1]

# fig = Figure(; size=(800,400))
# ax = Axis(fig[1,1], xlabel="q", ylabel="Energy (meV)", title="Dispersion")
# #lines!(ax, xs, ys)
# scatter!(ax, xs, ys, marker=:circle)
# fig |> display