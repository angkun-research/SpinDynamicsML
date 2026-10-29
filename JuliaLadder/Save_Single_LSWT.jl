using Sunny
using DataFrames, CSV

# Reuse Hao's Lanczos and B_sat helpers
include("triangular_ladder.jl")

# -----------------------------
# User parameters
# -----------------------------
#Js = [ 1.008152, -0.874677,  0.864058, 0.0, -1.004507]   # [J1, Δ, Jpmpm, Jzpm, J2]
Js = [1.0, 0.7, 0.2, 0.0, 0.051] 
dir = [1.0, 0.0, 0.0]
B_sat, sys, _ = polarized_system(Js, [1,0,0])
println("B_sat = ", B_sat)
QZ = 0
qs = [[0, 0, QZ], [1, 1, QZ]]
Nk = 60
path = q_space_path(sys.crystal,qs,Nk)
# -----------------------------
# Build LSWT result 
# -----------------------------
swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
res = intensities_bands(swt, path)

results = DataFrame(Js=Vector{Float64}[], data=Vector{Float64}[], disp=Vector{Float64}[])

push!(results, (
    copy(Js[[1, 2, 3, 5]]), # only keep J1, Δ, J1pmpm, J2
    vec(res.data),
    vec(res.disp),
))
# print(res.data)
CSV.write("../data/LSWT_Js4_Bdep_Nk$(Nk)_test_unitless.csv", results)
println("saved results to data")
