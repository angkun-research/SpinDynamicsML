using Sunny, DataFrames, CSV
using GLMakie
#using Plots
using ProgressBars

units = Units(:meV, :angstrom)
a = 1 # (Å)
latvecs = lattice_vectors(a,a,100a,90,90,120) # angles ac, bc, ab
cryst = Crystal(latvecs,[[0,0,0]],162)
#cryst = Crystal(latvecs,[[0,0,0]])

gab, gcc = 3.13*1.09310, 1.01
gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
sys_template = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(1, 1, 1))

h = 20.0 # need to be large for more parameters J with longer range
qs = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
Nkpts = 100
path = q_space_path(cryst, qs, Nkpts)

# Units in meV
J1 = 0.551
Δ1  = 0.9
Jpmpm1 = 0.0J1
Jzpm1 = 0.2J1
J2 = 0.071 * J1
Δ2 = 0.071*Δ1
Jpmpm2 = 0.071*Jpmpm1
Jzpm2 = 0.071*Jzpm1
# Units in Tesla
B = 14.0

J1_ten  = [J1+2Jpmpm1 0 Jzpm1; 0 J1-2Jpmpm1 0; Jzpm1 0 J1*Δ1]
J2_ten = [J2+2Jpmpm2 0 Jzpm2; 0 J2-2Jpmpm2 0; Jzpm2 0 J2*Δ2]
set_exchange!(sys_template, J1_ten, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys_template, J2_ten, Bond(1, 1, [1, 2, 0]))
set_field!(sys_template, [B, 0, 0] * units.T)

randomize_spins!(sys_template)
minimize_energy!(sys_template)

swt = SpinWaveTheory(sys_template; measure=ssf_perp(sys_template))
res = intensities_bands(swt, path)


plot_intensities(res; units, title="Triangular Lattice")



