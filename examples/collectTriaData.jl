using Sunny, GLMakie, DataFrames, CSV
using ProgressBars

units = Units(:meV, :angstrom)
a = 1 # (Å)
latvecs = lattice_vectors(a,a,100a,90,90,120) # angles ac, bc, ab
cryst = Crystal(latvecs,[[0,0,0]])

sys_template = System(cryst,[1 => Moment(s=1/2,g=2)],:dipole;dims=(1,1,1))
h = 20.0 # need to be large for more parameters J with longer range
qs = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
path = q_space_path(cryst, qs, 100)

Jvals = range(-1, 1; length=20)
# results = DataFrame(J1=Float64[], J2=Float64[], J3=Float64[], data=Vector{Float64}[], disp=Vector{Float64}[], qs=Vector{Vector{Float64}}[])

# for (J1, J2, J3) in ProgressBar(Iterators.product(Jvals, Jvals, Jvals))
#     sys = clone_system(sys_template)
#     set_exchange!(sys, J1, Bond(1,1,[1,0,0]))
#     set_exchange!(sys, J2, Bond(1,1,[1,2,0]))
#     set_exchange!(sys, J3, Bond(1,1,[2,0,0]))
#     set_field!(sys, [0,0,h])
#     randomize_spins!(sys)
#     minimize_energy!(sys)
#     swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
#     res = intensities_bands(swt, path)
#     push!(results, (
#         J1, J2, J3,
#         vec(res.data),
#         vec(res.disp),
#         res.qpts.qs
#     ))
# end

# create a dictionary to store the kth nearest neighbor bond Vector
bond_vectors = Dict{Int, Vector{Int}}()
bond_vectors[1] = [1,0,0]
bond_vectors[2] = [1,2,0]
bond_vectors[3] = [2,0,0]
bond_vectors[4] = [2,-1,0]
bond_vectors[5] = [3,0,0]
bond_vectors[6] = [2,4,0]
bond_vectors[7] = [3,-1,0]
bond_vectors[8] = [4,0,0]
bond_vectors[9] = [3,-2,0]
bond_vectors[10] = [4,-1,0]
bond_vectors[11] = [5,0,0]
bond_vectors[12] = [3,6,0]
bond_vectors[13] = [4,-2,0]
bond_vectors[14] = [5,-1,0]
bond_vectors[15] = [6,0,0]

Jsize = 10
Js = zeros(Jsize)
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
qs = sample_q_triangular(100)

# Randomly sample Js from the Jsize dimensional space with each J in range -1 to 1
for sample_idx in ProgressBar(1:nsample)
    for j in 1:Jsize
        Js[j] = rand()*2 - 1 # random value between -1 and
    end
    sys = clone_system(sys_template)
    for j in 1:Jsize
        set_exchange!(sys, Js[j], Bond(1,1,bond_vectors[j]))
    end
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


CSV.write("data/triangular_heisenberg_Js10_h20_qsBZ.csv", results)

# # read back the data
# results = CSV.read("data/triangular_heisenberg_Js10_h20.csv", DataFrame)
# # extract qs
# qs = results.qs[1]