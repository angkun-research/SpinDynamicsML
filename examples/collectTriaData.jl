using Sunny, GLMakie, DataFrames, CSV
using ProgressBars

units = Units(:meV, :angstrom)
a = 1 # (Å)
latvecs = lattice_vectors(a,a,100a,90,90,120) # angles ac, bc, ab
cryst = Crystal(latvecs,[[0,0,0]])

sys_template = System(cryst,[1 => Moment(s=1/2,g=2)],:dipole;dims=(1,1,1))
h = 6.0 # need to be above the saturation field to get the polarized state
qs = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
path = q_space_path(cryst, qs, 100)

Jvals = range(-1, 1; length=20)
results = DataFrame(J1=Float64[], J2=Float64[], J3=Float64[], data=Vector{Float64}[], disp=Vector{Float64}[], qs=Vector{Vector{Float64}}[])

for (J1, J2, J3) in ProgressBar(Iterators.product(Jvals, Jvals, Jvals))
    sys = clone_system(sys_template)
    set_exchange!(sys, J1, Bond(1,1,[1,0,0]))
    set_exchange!(sys, J2, Bond(1,1,[1,2,0]))
    set_exchange!(sys, J3, Bond(1,1,[2,0,0]))
    set_field!(sys, [0,0,h])
    randomize_spins!(sys)
    minimize_energy!(sys)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    res = intensities_bands(swt, path)
    push!(results, (
        J1, J2, J3,
        vec(res.data),
        vec(res.disp),
        res.qpts.qs
    ))
end

CSV.write("data/triangular_heisenberg_sweep.csv", results)