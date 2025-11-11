using Sunny, GLMakie

# Units and lattice (2D honeycomb in hexagonal/120° cell)
units = Units(:meV, :angstrom)
a = 1.0 # (Å)
latvecs = lattice_vectors(a, a, 100a, 90, 90, 120) # long c to make quasi-2D

# honeycomb basis: two sublattices per unit cell (fractional coords)
# common choice: A = (0,0,0), B = (1/3, 2/3, 0), in unit of lattice vectors
cryst = Crystal(latvecs, [[0,0,0], [1/3, 2/3, 0]])

#view_crystal(cryst)

# two moments (one per basis site); same Moment params here
sys = System(cryst, [1 => Moment(s=1/2, g=2)], :dipole; dims=(1,1,1))
view_crystal(sys, ndims=2,refbonds=10)

# Exchange couplings (meV)
J1, J2, J3 = 1.0, 0.5, 0.2

# Nearest-neighbor J1: A<->B bonds (three per site).
# Using unit-cell translation vectors in fractional lattice coordinates.
# The three NN bonds from A to B can be represented by these translations:
set_exchange!(sys, J1, Bond(1,2, [0, 0, 0]))
# set_exchange!(sys, J1, Bond(1,2, [-1, 0, 0]))
# set_exchange!(sys, J1, Bond(1,2, [0, -1, 0]))

# Next-nearest J2: same-sublattice (A-A and B-B) bonds.
# Provide a minimal set of translations that generate the 6 second neighbors.
set_exchange!(sys, J2, Bond(1,1, [1, 0, 0]))
# set_exchange!(sys, J2, Bond(1,1, [0, 1, 0]))
# set_exchange!(sys, J2, Bond(1,1, [-1, 1, 0]))
# set_exchange!(sys, J2, Bond(2,2, [1, 0, 0]))
# set_exchange!(sys, J2, Bond(2,2, [0, 1, 0]))
# set_exchange!(sys, J2, Bond(2,2, [-1, 1, 0]))

# Third-neighbor J3: longer-range A<->B (across hexagon) — use the shifted translations
set_exchange!(sys, J3, Bond(1,2, [1, 0, 0]))
# set_exchange!(sys, J3, Bond(1,2, [0, 1, 0]))
# set_exchange!(sys, J3, Bond(1,2, [-1, 1, 0]))
#sys.interactions_union[1]
# Zeeman field (adjust so system is in desired state)
h = 6.0
set_field!(sys, [0, 0, h])
view_crystal(sys, ndims=2,refbonds=10)

# Prepare classical ground state and plot
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys; color=[S[3] for S in sys.dipoles], ndims=2)

println("Energy per site: ", energy_per_site(sys))
print_wrapped_intensities(sys)

# Suggest trivial magnetic supercell (adjust if you need ML ordering)
suggest_magnetic_supercell([[0,0,0]])

# Spin-wave spectrum
swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
qs = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
path = q_space_path(cryst, qs, 100)
res = intensities_bands(swt, path)
plot_intensities(res; units, title="Honeycomb J1-J2-J3 Bands")

# scatter plot data
fig = Figure()
ax1 = Axis(fig[1,1]; xlabel="Q point index", ylabel="Energy")
ylims!(ax1, 0, 15)
scatter!(ax1, 1:length(res.disp[1,:]), res.disp[1,:])
scatter!(ax1, 1:length(res.disp[1,:]), res.disp[2,:])
fig


fig = Figure()
ax1 = Axis(fig[1,1]; xlabel="Q point index", ylabel="Intensity")
scatter!(ax1, 1:length(res.data[1,:]), res.data[1,:])
scatter!(ax1, 1:length(res.data[1,:]), res.data[2,:])
fig
