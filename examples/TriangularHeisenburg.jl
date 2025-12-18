using Sunny, GLMakie
#using CairoMakie

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
# set_exchange!(sys, J4, Bond(1,1,[2,-1,0]))
# set_exchange!(sys, J5, Bond(1,1,[3,0,0]))
# set_exchange!(sys, J6, Bond(1,1,[2,4,0]))
# set_exchange!(sys, J7, Bond(1,1,[3,-1,0]))
# set_exchange!(sys, J8, Bond(1,1,[4,0,0]))
# set_exchange!(sys, J9, Bond(1,1,[3,-2,0]))
# set_exchange!(sys, J10, Bond(1,1,[4,-1,0]))
# set_exchange!(sys, J11, Bond(1,1,[5,0,0]))
# set_exchange!(sys, J12, Bond(1,1,[3,6,0]))
# set_exchange!(sys, J13, Bond(1,1,[4,-2,0]))
# set_exchange!(sys, J14, Bond(1,1,[5,-1,0]))
# set_exchange!(sys, J15, Bond(1,1,[6,0,0]))
h = 6.0 #6.0 # in J1 # need to be above the saturation field to get the polarized state
set_field!(sys, [0,0,h]) # add magnetic field in z direction
view_crystal(sys, ndims=2,refbonds=15)

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
res = intensities_bands(swt, path)#path.qs)
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



# function create_heatmap(disp_mat::Matrix{Float64}, data_mat::Matrix{Float64};
#                         Egrid = range(0.0, 10.0; length=300),
#                         sigma = 0.2,
#                         normalize = false)
#     Nbands, Nq = size(disp_mat)
#     # reshape so rows = q index, cols = mode index
#     E_mat = reshape(disp_mat, Nq, Nbands) # rows = k point, cols = bands
#     I_mat = reshape(data_mat, Nq, Nbands)

#     img = zeros(Float64, Nq, length(Egrid))  # rows = k point, cols = energy

#     # accumulate Gaussians
#     for j in 1:Nq
#         for b in 1:Nbands
#             E0 = disp_mat[b,j]
#             A  = data_mat[b,j]
#             img[j, :] .+= A .* exp.(-0.5 .* ((Egrid .- E0) ./ sigma).^2)
#         end
#     end

#     # optional normalization (global)
#     if normalize
#         img .-= minimum(img)
#         s = maximum(img)
#         if s > 0
#             img ./= s
#         end
#     end

#     return Egrid, img
# end



# # example: create and save heatmaps for each row in results
# Egrid, img = create_heatmap(res.disp, res.data;
#                                 Egrid = range(0.0, maximum(res.disp) + 1.0; length=400),
#                                 sigma = 0.1)

# fig = CairoMakie.Figure()
# ax = Axis(fig[1,1]; xlabel = "k (path index)", ylabel = "Energy (meV)")
#     # heatmap expects (ys, xs) dimensions -> img is (E x k)
# heatmap!(ax, 1:size(img,2),Egrid, img; colormap = :viridis)
# #Colorbar(fig[1,2], ax.plots[1]; label = "Intensity")
# fig
# #save("/Users/angkunwu/Desktop/fig1.pdf", fig)

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

qtest = sample_q_triangular(100)

# plot qtest
qxs = [q[1] for q in qtest]
qys = [q[2] for q in qtest]
scatter(qxs, qys)

test = intensities_bands(swt, qtest)