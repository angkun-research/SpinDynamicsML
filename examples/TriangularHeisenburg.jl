using Sunny, GLMakie

units = Units(:meV, :angstrom)
a = 1 # (Å)
latvecs = lattice_vectors(a,a,100a,90,90,120) # angles ac, bc, ab
cryst = Crystal(latvecs,[[0,0,0]])

sys = System(cryst,[1 => Moment(s=1/2,g=2)],:dipole;dims=(1,1,1)) #dims=(6,6,1)) # dipolar moments
J1, J2, J3 = 1, 0.5, 0.2 # (meV) 
# J2/J1 < 1/8 3 sublattice, J2/J1 > 1/8 collinear 
set_exchange!(sys, J1, Bond(1,1,[1,0,0]))
set_exchange!(sys, J2, Bond(1,1,[1,2,0]))
set_exchange!(sys, J3, Bond(1,1,[2,0,0]))
h = 6.0 #6.0 # in J1 # need to be above the saturation field to get the polarized state
set_field!(sys, [0,0,h]) # add magnetic field in z direction
view_crystal(sys, ndims=2,refbonds=3)

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys; color=[S[3] for S in sys.dipoles],ndims=2)
energy_per_site(sys) |> println
print_wrapped_intensities(sys)

#suggest_magnetic_supercell([[1/3, 1/3, 0],[-1/3, -1/3, 0]])
suggest_magnetic_supercell([[0, 0, 0]])

#sys_min = reshape_supercell(sys,  [1 0 0; -1 3 0; 0 0 1])
#randomize_spins!(sys_min)
#minimize_energy!(sys_min);
#plot_spins(sys_min; color=[S[3] for S in sys_min.dipoles], ndims=2,ghost_radius=5)

#swt = SpinWaveTheory(sys_min; measure=ssf_perp(sys_min))
swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
qs = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
path = q_space_path(cryst, qs, 100)
res = intensities_bands(swt, path)
#plot_intensities(res; units, ylims=(0, 10), title="Single Crystal Bands")
plot_intensities(res; units, title="Triangular Lattice")

# ouput
# res.data, res.disp, res.qpts.qs
fig = Figure()
ax1 = Axis(fig[1,1]; xlabel="Q point index", ylabel="Intensity")
#ylims!(ax1, 0, 15)
#scatter!(ax1, 1:length(res.disp[1,:]), res.disp[1,:])
scatter!(ax1, 1:length(res.data[1,:]), res.data[1,:])
fig
