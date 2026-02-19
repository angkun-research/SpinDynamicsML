using Sunny, DataFrames, CSV
using GLMakie
#using Plots
using ProgressBars

units = Units(:meV, :angstrom)
a = 1 # (Å)
latvecs = lattice_vectors(a,a,100a,90,90,120) # angles ac, bc, ab
cryst = Crystal(latvecs,[[0,0,0]],162) # has to use 162 space group for anisotropic exchange
#cryst = Crystal(latvecs,[[0,0,0]])

gab, gcc = 3.13*1.09310, 1.01
gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
sys_template = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(1, 1, 1))

h = 20.0 # need to be large for more parameters J with longer range
qs = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
Nkpts = 200
path = q_space_path(cryst, qs, Nkpts)

# # Units in meV
# J1 = 0.551 # Js[1]
# Δ1  = 0.9  # Js[2]
# Jpmpm1 = 0.0J1 # Js[3]
# Jzpm1 = 0.2J1 # Js[4]
# J2 = 0.071 * J1 # Js[5]
# Δ2 = 0.071*Δ1 # Js[6]
# Jpmpm2 = 0.071*Jpmpm1 # Js[7]
# Jzpm2 = 0.071*Jzpm1 # Js[8]
# J1_ten  = [J1+2Jpmpm1 0 Jzpm1; 0 J1-2Jpmpm1 0; Jzpm1 0 J1*Δ1]
# J2_ten = [J2+2Jpmpm2 0 Jzpm2; 0 J2-2Jpmpm2 0; Jzpm2 0 J2*Δ2]
# set_exchange!(sys_template, J1_ten, Bond(1, 1, [1, 0, 0]))
# set_exchange!(sys_template, J2_ten, Bond(1, 1, [1, 2, 0]))
# set_field!(sys_template, [0, 0, h])

Jsize = 8
#Js = zeros(Jsize)
nsample = 10000 

results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[])

function sample_q_triangular(N::Int)
    b2 = [1/3,1/3, 0] #[1/2/sqrt(3), 1/2, 0] # in units of reciprocal lattice vectors
    b3 = [-1/3,2/3, 0] #[ -1/2/sqrt(3), 1/2, 0]
    # find minimal L s.t. triangular number (L+1)*(L+2)/2 >= N
    L = 1
    while (L+1)*(L+2) ÷ 2 < N
        L += 1
    end
    qs = Vector{Vector{Float64}}()
    for i in 0:L
        for j in 0:(L - i)
            u = i / L
            v = j / L
            push!(qs, u*b2 + v*b3) # uniform sampling in one triangle of the BZ
        end
    end
    return qs[1:min(N, length(qs))]
end
qs = sample_q_triangular(Nkpts)

# Randomly sample Js from the Jsize dimensional space with each J in range -1 to 1
for sample_idx in ProgressBar(1:nsample)
    Js = zeros(Jsize)
    for j in 1:Jsize
        Js[j] = rand()*2 - 1 # random value between -1 and 1
    end
    # Js[[4]] = zeros(1) # set anisotropy parameters to zero for Heisenberg model
    # Js[[8]] = zeros(1) # set anisotropy parameters to zero for Heisenberg model
    J1_ten = [Js[1]+2*Js[2] 0 Js[3]; 0 Js[1]-2*Js[2] 0; Js[3] 0 Js[1]*Js[4]]
    J2_ten = [Js[5]+2*Js[6] 0 Js[7]; 0 Js[5]-2*Js[6] 0; Js[7] 0 Js[5]*Js[8]]
    sys = clone_system(sys_template)
    set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J2_ten, Bond(1, 1, [1, 2, 0]))
    set_field!(sys, [0,0,h])
    randomize_spins!(sys)
    minimize_energy!(sys)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    #res = intensities_bands(swt, path)
    res = intensities_bands(swt, qs)
    push!(results, (
        copy(Js),
        vec(res.data),
        vec(res.disp),
    ))
end

# println(rs)
# println(Js)

#CSV.write("data/triangular_anisotropy_Js2_h20_qsBZ$(Nkpts).csv", results)
CSV.write("data/triangular_anisotropy_temp.csv", results)

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