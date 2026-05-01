import numpy as np
import matplotlib.pyplot as plt

# We consider the Heisenberg model on a 2D honeycomb lattice up to the six nearest neighbors.
# The interactions are either between the same sublattice (A-A or B-B) or between different sublattices (A-B).
# For A-A and B-B, we denote it as AA, and for A-B, we denote it as AB.
######################################################
#   Interaction |   Type   |    Displacement Vectors |
#----------------------------------------------------#
#   J1          |   AB     |    (0, 0)               |
#   J1          |   AB     |    (0, 1)               |
#   J1          |   AB     |    (-1, 0)              |
#   J2          |   AA     |    (-1, -1)             |
#   J2          |   AA     |    (-1, 0)              |
#   J2          |   AA     |    (0, -1)              |
#   J3          |   AB     |    (1,  1)              |
#   J3          |   AB     |    (-1, 1)              |
#   J3          |   AB     |    (-1, -1)             |
#   J4          |   AB     |    (1, 2)               |
#   J4          |   AB     |    (1, 0)               |
#   J4          |   AB     |    (0, 2)               |
#   J4          |   AB     |    (0, -1)              |
#   J4          |   AB     |    (-2, 0)              |
#   J4          |   AB     |    (-2, -1)             |
#   J5          |   AA     |    (-2, -1)             |
#   J5          |   AA     |    (-1, -2)             |
#   J5          |   AA     |    (-1,  1)             |
#   J6          |   AA     |    (-2, -2)             |
#   J6          |   AA     |    (-2,  0)             |
#   J6          |   AA     |    (0,  -2)             |
class HoneycombLattice:
    def __init__(self, Js):
        # Js: dict like {"J1":..., "J2":..., ..., "J6":...}
        self.Js = dict(Js)

        # bonds["Jr"] = list of displacement vectors (dx, dy)
        self.bonds = {
            "J1": [(0, 0), (0, 1), (-1, 0)],
            "J2": [(-1, -1), (-1, 0), (0, -1)],
            "J3": [(1, 1), (-1, 1), (-1, -1)],
            "J4": [(1, 2), (1, 0), (0, 2), (0, -1), (-2, 0), (-2, -1)],
            "J5": [(-2, -1), (-1, -2), (-1, 1)],
            "J6": [(-2, -2), (-2, 0), (0, -2)],
        }

        # types["Jr"] = 0 for AA, 1 for AB
        self.types = {
            "J1": 1,  # AB
            "J2": 0,  # AA
            "J3": 1,  # AB
            "J4": 1,  # AB
            "J5": 0,  # AA
            "J6": 0,  # AA
        } 

    # S is the spin magnitude, h is the external magnetic field along the z-axis.
    # S default is 0.5 for spin-1/2, and h default is 0.0
    def omegas(self, qs, S=0.5, h=0.0):
        """
        qs: array-like of shape (N,2)
        phase convention: exp(i * 2π * dot(q, bond))
        Returns energies: shape (2, N) with [0]=omega_- and [1]=omega_+
        Sunny-matching conventions:
        - AA bond lists are "half shells" -> use full AA via 2*Re(...)
        """
        qs = np.asarray(qs, dtype=float)
        energies = np.zeros((2, len(qs)), dtype=float)

        # Sum z_r * J_r separately for AA(list) and AB
        JAA0_list, JAB0 = 0.0, 0.0
        for name, J in self.Js.items():
            z = len(self.bonds[name])
            if self.types[name] == 0:      # AA (half shell in your table)
                JAA0_list += z * J
            else:                           # AB
                JAB0 += z * J

        # Convert AA(list) -> AA(full)
        JAA0_full = 2.0 * JAA0_list

        for iq, q in enumerate(qs):
            JAAq = 0.0 + 0.0j
            JABq = 0.0 + 0.0j

            for name, J in self.Js.items():
                disp = np.asarray(self.bonds[name], dtype=float)
                phase = 2.0 * np.pi * (disp @ q)
                sf = np.sum(np.exp(1j * phase))

                if self.types[name] == 0:      # AA (half shell)
                    JAAq += J * sf
                else:                           # AB
                    JABq += J * sf

            # Full AA Fourier sum is 2*Re(JAAq)
            A = h - S*(JAA0_full + JAB0) + 2.0*S*np.real(JAAq)
            B = S * JABq

            energies[0, iq] = A - np.abs(B)
            energies[1, iq] = A + np.abs(B)

        return energies

    def omegas_intensities(self, qs, S=0.5, h=0.0):
        """
        qs: array-like of shape (N,2)
        phase convention: exp(i * 2π * dot(q, bond))

        Returns
        -------
        energies : shape (2, N)
            [0] = omega_+
            [1] = omega_-
        intensities : shape (2, N)
            [0] = intensity of omega_+ band
            [1] = intensity of omega_- band
        """

        qs = np.asarray(qs, dtype=float)
        energies = np.zeros((2, len(qs)), dtype=float)
        intensities = np.zeros((2, len(qs)), dtype=float)

        # Sum z_r * J_r separately for AA(list) and AB
        JAA0_list, JAB0 = 0.0, 0.0
        for name, J in self.Js.items():
            z = len(self.bonds[name])
            if self.types[name] == 0:      # AA (half shell in your table)
                JAA0_list += z * J
            else:                          # AB
                JAB0 += z * J

        # Convert AA(list) -> AA(full)
        JAA0_full = 2.0 * JAA0_list

        r_A = np.array([1/3,  2/3])
        r_B = np.array([2/3,  1/3])
        d   = r_B - r_A   # = (1/3, -1/3)
        for iq, q in enumerate(qs):
            JAAq = 0.0 + 0.0j
            JABq = 0.0 + 0.0j

            for name, J in self.Js.items():
                disp = np.asarray(self.bonds[name], dtype=float)
                phase = 2.0 * np.pi * (disp @ q)
                sf = np.sum(np.exp(1j * phase))

                if self.types[name] == 0:      # AA (half shell)
                    JAAq += J * sf
                else:                          # AB
                    JABq += J * sf

            # Full AA Fourier sum is 2*Re(JAAq)
            A = h - S * (JAA0_full + JAB0) + 2.0 * S * np.real(JAAq)
            B = S * JABq

            energies[0, iq] = A + np.abs(B)
            energies[1, iq] = A - np.abs(B)

            phi = np.angle(JABq)
            qd  = 2.0 * np.pi * (q @ d)
            intensities[0, iq] = 2.0 * S * (1 + np.cos(qd + phi))   # omega_+
            intensities[1, iq] = 2.0 * S * (1 - np.cos(qd + phi))   # omega_-

        return energies, 0.5*intensities

def make_bz_parallelogram_qs(N, return_mask=False):
    v1 = np.array([1/3,  1/3])
    v2 = np.array([2/3, -1/3])

    s_vals = np.linspace(0, 1, N, endpoint=True)
    t_vals = np.linspace(0, 1, N, endpoint=True)

    ss, tt = np.meshgrid(s_vals, t_vals, indexing='ij')
    qs = ss.ravel()[:, None] * v1 + tt.ravel()[:, None] * v2

    if not return_mask:
        return qs

    s, t = ss.ravel(), tt.ravel()
    mask_lower = (s + t) <= 1.0
    mask_upper = (s + t) >= 1.0

    return qs, mask_lower, mask_upper

def qs_to_cartesian(qs):
    """
    Fractional (h,k) -> Cartesian (qx, qy).
    a1=(1,0), a2=(-1/2, sqrt(3)/2)  =>  b1=(1, 1/sqrt(3)), b2=(0, 2/sqrt(3))
    """
    b1 = np.array([1.0,  1.0 / np.sqrt(3)])
    b2 = np.array([0.0,  2.0 / np.sqrt(3)])
    return qs[:, 0:1] * b1 + qs[:, 1:2] * b2


def plot_bz_parallelogram(N=50):
    qs, mask_l, mask_u = make_bz_parallelogram_qs(N, return_mask=True)
    qc = qs_to_cartesian(qs)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(*qc[mask_l].T, s=0.5, c='steelblue', label="Lower (primary) Γ→K→K'")
    ax.scatter(*qc[mask_u].T, s=0.5, c='tomato',    label="Upper (conjugate) K→Γ'→K'")
    ax.scatter(*qc[~mask_l & ~mask_u].T, s=2, c='k', label='Diagonal boundary')

    hsym = {'Γ': (0,0), 'K': (1/3,1/3), "Γ'": (1,0), "K'": (2/3,-1/3), 'M': (0.5,0)}
    for name, (hf, kf) in hsym.items():
        x, y = qs_to_cartesian(np.array([[hf, kf]]))[0]
        ax.plot(x, y, 'k+', ms=10, mew=1.5)
        ax.annotate(name, (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=12)

    ax.set_aspect('equal')
    ax.set_xlabel('$q_x$')
    ax.set_ylabel('$q_y$')
    ax.set_title('BZ parallelogram in Cartesian coordinates')
    ax.legend(markerscale=8, fontsize=9)
    plt.tight_layout()
    plt.show()

    print(f"Total points      : {len(qs)}")
    print(f"Lower (primary)   : {mask_l.sum()}  ({100*mask_l.mean():.1f}%)")
    print(f"Upper (conjugate) : {mask_u.sum()}  ({100*mask_u.mean():.1f}%)")
    print(f"Diagonal boundary : {(~mask_l & ~mask_u).sum()}")




#plot_bz_parallelogram(N=16)

N = 16
S = 0.5
h = 10.0
Js_vals = [-1, -0.9, 0.9, 0.0, 0.0, 0.0]
Js = {f"J{i+1}": val for i, val in enumerate(Js_vals)}
lat = HoneycombLattice(Js)
qs      = make_bz_parallelogram_qs(N)
qc      = qs_to_cartesian(qs)
qx, qy  = qc[:, 0], qc[:, 1]

energies, intensities = lat.omegas_intensities(qs, S=S, h=h)

hsym = {
    'Γ' : (0,    0   ),
    'K' : (1/3,  1/3 ),
    "Γ'": (1,    0   ),
    "K'": (2/3, -1/3 ),
    'M' : (0.5,  0   ),
}

titles = [r'$\omega_+$', r'$\omega_-$', r'$I_+$', r'$I_-$']
data   = [energies[0], energies[1], intensities[0], intensities[1]]

# fig, axes = plt.subplots(2, 2, figsize=(10, 9))
# axes = axes.ravel()

# for ax, d, title in zip(axes, data, titles):
#     sc = ax.scatter(qx, qy, c=d, s=600//N, cmap='viridis', rasterized=True)
#     plt.colorbar(sc, ax=ax, shrink=0.8)

#     for name, (hf, kf) in hsym.items():
#         x, y = qs_to_cartesian(np.array([[hf, kf]]))[0]
#         ax.plot(x, y, 'w+', ms=8, mew=1.5)
#         ax.annotate(name, (x, y), xytext=(4, 4),
#                     textcoords='offset points', fontsize=10, color='red')

#     ax.set_aspect('equal')
#     ax.set_title(title, fontsize=13)
#     ax.set_xlabel('$q_x$')
#     ax.set_ylabel('$q_y$')

# plt.suptitle(f'Dispersion and intensity in BZ parallelogram  (N={N})', fontsize=13)
# plt.tight_layout()
# plt.show()

qs, mask_l, mask_u = make_bz_parallelogram_qs(N, return_mask=True)
qc = qs_to_cartesian(qs)
qx = qc[:, 0].reshape(N, N)
qy = qc[:, 1].reshape(N, N)

energies, intensities = lat.omegas_intensities(qs, S=S, h=h)
Z = energies[0].reshape(N, N)   # choose band 0 (omega_+). use [1] for omega_-

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(qx, qy, Z, cmap='viridis', linewidth=0, antialiased=True)
ax.set_xlabel('$q_x$'); ax.set_ylabel('$q_y$'); ax.set_zlabel('Energy')
fig.colorbar(surf, ax=ax, shrink=0.6)
ax.view_init(elev=35, azim=-60)
plt.show()