using Sunny
using CairoMakie
using LinearAlgebra

Js = [
 -0.08040006080575379
  0.6207173628700369
 -0.009398786694186256
  0.181951841692229
 -0.8781165565482123
]
J1 = Js[1]
Δ  = Js[2]
Jpmpm = Js[3] * J1
Jzpm = Js[4] * J1
J2 = Js[5]
# B = 1.1 * 0.8209228515625
B = 0.9545579467098153

units = Units(:meV, :angstrom)
latvecs = lattice_vectors(4, 4, 12, 90, 90, 120)
positions = [[0, 0, 0]]
cryst = Crystal(latvecs, positions, 162)

gab, gcc = 3.13*1.09310, 1.01

gtens = [gab 0 0; 0 gab 0; 0 0 gcc]
sys = System(cryst, [1 => Moment(s=1/2, g=gtens)], :dipole; dims=(1, 1, 1))

J1_ten  = [J1+2Jpmpm 0 Jzpm; 0 J1-2Jpmpm 0; Jzpm 0 J1*Δ]
set_exchange!(sys, J1_ten, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys, J2, Bond(1, 1, [1, 2, 0]))
set_field!(sys, [B, 0, 0] * units.T)

randomize_spins!(sys)
minimize_energy!(sys)

swt = SpinWaveTheory(sys; measure=ssf_perp(sys))

qs = [[0,0,0], [1/3, 1/3, 0], [1/2, 1/2, 0], [1, 1, 0]]
labels = ["Γ", "K", "M", "Γ"]
path = q_space_path(cryst, qs, 200; labels)
res = intensities_bands(swt, path)
plot_intensities(res)