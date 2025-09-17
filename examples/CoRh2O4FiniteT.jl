using Sunny, GLMakie

units = Units(:meV, :angstrom)
a = 8.5031 # (Å)
latvecs = lattice_vectors(a,a,a,90,90,90)
cryst = Crystal(latvecs,[[1/8,1/8,1/8]],227)

sys = System(cryst,[1 => Moment(s=3/2,g=2)],:dipole) # dipolar moments
J = 0.63 # (meV)
set_exchange!(sys, J, Bond(2,3,[0,0,0]))
randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys; color=[S[3] for S in sys.dipoles])

sys = repeat_periodically(sys, (10,10,10))
plot_spins(sys; color=[S[3] for S in sys.dipoles])

langevin = Langevin(; damping=0.2, kT=16*units.K)

suggest_timestep(sys, langevin; tol=1e-2)
langevin.dt = 0.025;

energies = [energy_per_site(sys)]
for _ in 1:1000
    step!(sys, langevin)
    push!(energies, energy_per_site(sys))
end

