using Sunny
using Plots

units = Units(:meV, :angstrom)
a = 1 # (Å)
latvecs = lattice_vectors(a,a,100a,90,90,120) # angles ac, bc, ab
cryst = Crystal(latvecs,[[0,0,0]])

sys_template = System(cryst,[1 => Moment(s=1/2,g=2)],:dipole;dims=(1,1,1))
h = 20.0 # need to be large for more parameters J with longer range
qs = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
Nkpts = 100
path = q_space_path(cryst, qs, Nkpts)

Jvals = range(-1, 1; length=20)
# results = DataFrame(J1=Float64[], J2=Float64[], J3=Float64[], data=Vector{Float64}[], disp=Vector{Float64}[], qs=Vector{Vector{Float64}}[])

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

rs = zeros(10)  #d = a \sqrt{n_1^2 + n_1 n_2 + n_2^2}
rs[1] = 1.0
rs[2] = sqrt(3)
rs[3] = 2.0
rs[4] = sqrt(7)
rs[5] = 3.0
rs[6] = sqrt(12)
rs[7] = sqrt(13)
rs[8] = 4.0
rs[9] = sqrt(19)
rs[10] = sqrt(21)

Jsize = 5 #10
nsample = 2 # 10000 
Js = zeros(Jsize, nsample)
res = zeros(length(path.qs), nsample)

# Js[:, 1] = 1 ./ rs.^2
# #Js[:, 2] = [0.97279334,0.3355829,0.2550886, 0.12769656, 0.13329244, 0.06266326,
# #  0.04556783, 0.07966978, 0.05412785, 0.04124609]
# Js[:, 2] = [0.9722343,0.33919275,0.25549358, 0.13345772, 0.09969644, 0.05931171,
#   0.04195142, 0.03636602, 0.03026602, 0.02043957]

# Js[:, 1] = [0.0672606,0.81141645,-0.66225547,-0.26658773,-0.61779296]
# Js[:, 2] = [0.05755597,0.81726307,-0.6566382,-0.2775978,-0.6139747]
# Js[:, 1] = [0.14411639, -0.65124303, 0.47193098, 0.24249503, 0.9384516]
# Js[:, 2] = [0.14003862, -0.63205945, 0.46725968, 0.24539933, 0.9399781]
Js[:, 1] = [-0.3488161,0.6683957,-0.14896345,0.02395559,0.7439383]
Js[:, 2] = [-0.34946927,0.6603112,-0.15114537,0.01153535,0.7424642]

# Randomly sample Js from the Jsize dimensional space with each J in range -1 to 1
for sample_idx in 1:nsample
    # for j in 1:Jsize
    #     #Js[j] = rand()*2 - 1 # random value between -1 and 1
    #     Js[j] = 1/rs[j]^2  
    # end
    sys = clone_system(sys_template)
    for j in 1:Jsize
        set_exchange!(sys, Js[j,sample_idx], Bond(1,1,bond_vectors[j]))
    end
    set_field!(sys, [0,0,h])
    randomize_spins!(sys)
    minimize_energy!(sys)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    restemp = intensities_bands(swt, path)
    #res = intensities_bands(swt, qs)
    res[:, sample_idx] = vec(restemp.disp)
end


# plot the two curves
scatter(1:Nkpts, res[:,1], label="J true", xlabel="q-point index", ylabel="Energy (meV)", title="Spin Wave Dispersion Comparison", legend=:topright)
scatter!(1:Nkpts, res[:,2], label="J prediction")
#savefig("~/Desktop/SpinWaveDispersionComparison.pdf")
#savefig("~/Desktop/fig3.pdf")