import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from scipy.sparse.linalg import splu

import time

from scipy.sparse import (
    lil_matrix,
    coo_matrix
)

import gc
CHECKPOINT_DIR = (
    "316L_CPFEM_tensile_20percent_stepwise_checkpoints_finite_CP-FEM_v2"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# STEP 1: 3D Voronoi microstructure for 316L
# ============================================================

# -----------------------------
# 1. Model dimensions
# -----------------------------

Lx = 360.0       # micrometres
Ly = 360.0
Lz = 360.0

# Number of grains
N_grains = 216

# Voxel resolution
# 180 means ~2 um voxel size
Nx = 180
Ny = 180
Nz = 180

dx = Lx / Nx
dy = Ly / Ny
dz = Lz / Nz

print("Voxel size:")
print(f"dx = {dx:.2f} um")
print(f"dy = {dy:.2f} um")
print(f"dz = {dz:.2f} um")


# -----------------------------
# 2. Generate random grain seeds
# -----------------------------

np.random.seed(42)

seeds = np.column_stack([
    np.random.uniform(0, Lx, N_grains),
    np.random.uniform(0, Ly, N_grains),
    np.random.uniform(0, Lz, N_grains)
])

print("\nNumber of grains:", len(seeds))


# # -----------------------------
# # 3. Generate voxel coordinates
# # -----------------------------

# x = np.linspace(dx/2, Lx-dx/2, Nx)
# y = np.linspace(dy/2, Ly-dy/2, Ny)
# z = np.linspace(dz/2, Lz-dz/2, Nz)

# X, Y, Z = np.meshgrid(
#     x, y, z,
#     indexing="ij"
# )

# points = np.column_stack([
#     X.ravel(),
#     Y.ravel(),
#     Z.ravel()
# ])


# # -----------------------------
# # 4. Assign every voxel
# #    to nearest grain seed
# # -----------------------------

# tree = cKDTree(seeds)

# distances, grain_ids = tree.query(points)

# grain_ids = grain_ids.reshape((Nx, Ny, Nz))

# print("\nVoronoi microstructure generated.")
# print("Array shape:", grain_ids.shape)



# -----------------------------
# 3–4. Generate voxel coordinates
#     and assign every voxel
#     to nearest grain seed
#     MEMORY-SAFE VERSION
# -----------------------------

# Coordinate vectors only
x = np.linspace(
    dx / 2,
    Lx - dx / 2,
    Nx
)

y = np.linspace(
    dy / 2,
    Ly - dy / 2,
    Ny
)

z = np.linspace(
    dz / 2,
    Lz - dz / 2,
    Nz
)

# Total number of voxels
N_voxels = Nx * Ny * Nz

# KD-tree containing the 216 grain seeds
tree = cKDTree(seeds)

# Store only the final grain IDs.
# int32 is more than sufficient for 216 grains.
grain_ids_flat = np.empty(
    N_voxels,
    dtype=np.int32
)

# Number of voxels processed at one time
VOXEL_CHUNK_SIZE = 200_000

print("\nAssigning voxels to nearest grain seeds...")

# Size of one i-slice
YZ_SIZE = Ny * Nz

for start in range(
    0,
    N_voxels,
    VOXEL_CHUNK_SIZE
):

    end = min(
        start + VOXEL_CHUNK_SIZE,
        N_voxels
    )

    # Linear voxel indices
    idx = np.arange(
        start,
        end,
        dtype=np.int64
    )

    # Convert linear index to (i, j, k)
    i = idx // YZ_SIZE

    remainder = idx - i * YZ_SIZE

    j = remainder // Nz
    k = remainder - j * Nz

    # Construct coordinates only for this chunk
    points_chunk = np.column_stack((
        x[i],
        y[j],
        z[k]
    ))

    # Find nearest grain seed
    _, grain_ids_chunk = tree.query(
        points_chunk
    )

    grain_ids_flat[start:end] = (
        grain_ids_chunk.astype(np.int32)
    )

    print(
        f"  {end:,} / {N_voxels:,} voxels assigned",
        end="\r"
    )

# Restore 3-D voxel structure
grain_ids = grain_ids_flat.reshape(
    (Nx, Ny, Nz)
)

print("\nVoronoi microstructure generated.")
print("Array shape:", grain_ids.shape)



# ============================================================
# 5. Generate random crystallographic orientations
# ============================================================

# Random rotations
rotations = R.random(N_grains)

# Rotation matrices
orientation_matrices = rotations.as_matrix()

# Euler angles in degrees
euler_angles = rotations.as_euler(
    'ZXZ',
    degrees=True
)

print("\nOrientation matrices shape:",
      orientation_matrices.shape)

print("Euler angle array shape:",
      euler_angles.shape)


# ============================================================
# 6. Save microstructure
# ============================================================

np.savez(
    "316L_3D_Voronoi_microstructure.npz",

    grain_ids=grain_ids,

    grain_centers=seeds,

    orientation_matrices=orientation_matrices,

    euler_angles=euler_angles,

    dimensions=np.array([
        Lx, Ly, Lz
    ]),

    voxel_size=np.array([
        dx, dy, dz
    ])
)

print("\nMicrostructure saved as:")
print("316L_3D_Voronoi_microstructure.npz")


# ============================================================
# 7. Plot central cross-section
# ============================================================

mid_z = Nz // 2

plt.figure(figsize=(8, 7))

plt.imshow(
    grain_ids[:, :, mid_z].T,
    origin='lower',
    extent=[0, Lx, 0, Ly],
    interpolation='nearest'
)

plt.xlabel("x (μm)", fontsize=16)
plt.ylabel("y (μm)", fontsize=16)

plt.title(
    "316L 3D Voronoi Microstructure — Central Section",
    fontsize=16
)

plt.colorbar(label="Grain ID")

plt.tight_layout()
plt.show()


# ============================================================
# 8. Plot central x-z section
# ============================================================

mid_y = Ny // 2

plt.figure(figsize=(8, 7))

plt.imshow(
    grain_ids[:, mid_y, :].T,
    origin='lower',
    extent=[0, Lx, 0, Lz],
    interpolation='nearest'
)

plt.xlabel("x (μm)", fontsize=16)
plt.ylabel("z (μm)", fontsize=16)

plt.title(
    "316L 3D Voronoi Microstructure — x-z Section",
    fontsize=16
)

plt.colorbar(label="Grain ID")

plt.tight_layout()
plt.show()


# ============================================================
# 9. Plot grain seed locations
# ============================================================

fig = plt.figure(figsize=(9, 8))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    seeds[:, 0],
    seeds[:, 1],
    seeds[:, 2],
    s=15
)

ax.set_xlabel("x (μm)")
ax.set_ylabel("y (μm)")
ax.set_zlabel("z (μm)")

ax.set_title(
    "3D Grain Seed Distribution"
)

plt.tight_layout()
plt.show()


# ============================================================
# 10. Print basic information
# ============================================================

print("\n==============================")
print("MICROSTRUCTURE SUMMARY")
print("==============================")

print(f"Domain: {Lx} × {Ly} × {Lz} μm³")
print(f"Number of grains: {N_grains}")
print(f"Voxel resolution: {Nx} × {Ny} × {Nz}")
print(f"Voxel size: {dx:.2f} μm")
print("Crystal structure: FCC")
print("Material: 316L stainless steel")
print("Slip systems: 12 FCC {111}<110>")
print("Orientation: random")

# ============================================================
# 11. Calculate grain volume and equivalent diameter
# ============================================================

voxel_volume = dx * dy * dz

grain_volumes = np.zeros(N_grains)

for i in range(N_grains):

    number_of_voxels = np.sum(grain_ids == i)

    grain_volumes[i] = (
        number_of_voxels * voxel_volume
    )

# Equivalent spherical diameter
grain_diameters = (
    6.0 * grain_volumes / np.pi
) ** (1.0 / 3.0)


# ============================================================
# 12. Grain-size statistics
# ============================================================

print("\n==============================")
print("GRAIN SIZE STATISTICS")
print("==============================")

print(
    f"Minimum diameter = "
    f"{grain_diameters.min():.2f} μm"
)

print(
    f"Maximum diameter = "
    f"{grain_diameters.max():.2f} μm"
)

print(
    f"Mean diameter = "
    f"{grain_diameters.mean():.2f} μm"
)

print(
    f"Median diameter = "
    f"{np.median(grain_diameters):.2f} μm"
)

print(
    f"Std. deviation = "
    f"{grain_diameters.std():.2f} μm"
)


# ============================================================
# 13. Grain-size distribution
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    grain_diameters,
    bins=20
)

plt.xlabel(
    "Equivalent grain diameter (μm)",
    fontsize=16
)

plt.ylabel(
    "Number of grains",
    fontsize=16
)

plt.title(
    "Grain Size Distribution",
    fontsize=16
)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# MODULE 2
# 3D HEXAHEDRAL FE MESH
# ============================================================

# ------------------------------------------------------------
# 1. Load the microstructure from Module 1
# ------------------------------------------------------------

data = np.load(
    "316L_3D_Voronoi_microstructure.npz"
)

grain_ids = data["grain_ids"]
grain_centers = data["grain_centers"]
orientation_matrices = data["orientation_matrices"]
euler_angles = data["euler_angles"]

Lx, Ly, Lz = data["dimensions"]

print("Microstructure loaded.")

print("Domain:")
print(Lx, Ly, Lz, "μm")

print("Number of grains:",
      len(grain_centers))


# ============================================================
# 2. Define FE mesh resolution
# ============================================================

# ------------------------------------------------------------
# TEST MODEL
# ------------------------------------------------------------

Nx_FE = 20
Ny_FE = 20
Nz_FE = 20

dx_FE = Lx / Nx_FE
dy_FE = Ly / Ny_FE
dz_FE = Lz / Nz_FE

print("\nFE mesh:")
print(
    Nx_FE,
    "×",
    Ny_FE,
    "×",
    Nz_FE
)

print(
    "Element size:",
    dx_FE,
    "μm"
)

print(
    "Number of elements:",
    Nx_FE * Ny_FE * Nz_FE
)


# ============================================================
# 3. Generate FE nodes
# ============================================================

x_nodes = np.linspace(
    0,
    Lx,
    Nx_FE + 1
)

y_nodes = np.linspace(
    0,
    Ly,
    Ny_FE + 1
)

z_nodes = np.linspace(
    0,
    Lz,
    Nz_FE + 1
)


# Create node grid
Xn, Yn, Zn = np.meshgrid(
    x_nodes,
    y_nodes,
    z_nodes,
    indexing="ij"
)


# Node coordinates
nodes = np.column_stack([
    Xn.ravel(),
    Yn.ravel(),
    Zn.ravel()
])


N_nodes = len(nodes)

print(
    "\nNumber of nodes:",
    N_nodes
)


# ============================================================
# 4. Generate hexahedral element connectivity
# ============================================================

def node_id(i, j, k):

    return (
        i * (Ny_FE + 1) * (Nz_FE + 1)
        +
        j * (Nz_FE + 1)
        +
        k
    )


elements = []

for i in range(Nx_FE):

    for j in range(Ny_FE):

        for k in range(Nz_FE):

            n1 = node_id(i,     j,     k)
            n2 = node_id(i + 1, j,     k)
            n3 = node_id(i + 1, j + 1, k)
            n4 = node_id(i,     j + 1, k)

            n5 = node_id(i,     j,     k + 1)
            n6 = node_id(i + 1, j,     k + 1)
            n7 = node_id(i + 1, j + 1, k + 1)
            n8 = node_id(i,     j + 1, k + 1)

            elements.append([
                n1, n2, n3, n4,
                n5, n6, n7, n8
            ])


elements = np.asarray(
    elements,
    dtype=np.int32
)

N_elements = len(elements)

print(
    "Number of elements:",
    N_elements
)


# ============================================================
# 5. Calculate element centers
# ============================================================

element_centers = np.zeros(
    (N_elements, 3)
)

for e in range(N_elements):

    element_nodes = elements[e]

    element_centers[e] = np.mean(
        nodes[element_nodes],
        axis=0
    )


# ============================================================
# 6. Assign each FE element to a grain
# ============================================================

# ------------------------------------------------------------
# Use the original 2 μm microstructure.
# Each FE element center is mapped to the nearest
# microstructure voxel.
# ------------------------------------------------------------

voxel_size = data["voxel_size"]

dx_micro = voxel_size[0]
dy_micro = voxel_size[1]
dz_micro = voxel_size[2]


def position_to_voxel(point):

    x, y, z = point

    ix = int(x / dx_micro)
    iy = int(y / dy_micro)
    iz = int(z / dz_micro)

    # Prevent boundary overflow
    ix = min(max(ix, 0), grain_ids.shape[0] - 1)
    iy = min(max(iy, 0), grain_ids.shape[1] - 1)
    iz = min(max(iz, 0), grain_ids.shape[2] - 1)

    return ix, iy, iz


element_grain = np.zeros(
    N_elements,
    dtype=np.int32
)


for e in range(N_elements):

    ix, iy, iz = position_to_voxel(
        element_centers[e]
    )

    element_grain[e] = grain_ids[
        ix, iy, iz
    ]


print(
    "\nGrain assignment completed."
)


# ============================================================
# 7. Assign crystal orientation to each element
# ============================================================

element_orientation = (
    orientation_matrices[element_grain]
)


element_euler = (
    euler_angles[element_grain]
)


print(
    "Orientation assignment completed."
)

print(
    "Element orientation array:",
    element_orientation.shape
)


# ============================================================
# 8. Check how many grains are represented
# ============================================================

unique_grains = np.unique(
    element_grain
)

print(
    "\nNumber of grains represented in FE mesh:",
    len(unique_grains)
)


# ============================================================
# 9. Check element-grain distribution
# ============================================================

grain_element_count = np.bincount(
    element_grain,
    minlength=len(grain_centers)
)


print("\nFirst 10 grain element counts:")

print(
    grain_element_count[:10]
)


# ============================================================
# 10. Save FE mesh
# ============================================================

np.savez_compressed(
    "316L_FE_mesh_20cube.npz",

    nodes=nodes,

    elements=elements,

    element_centers=element_centers,

    element_grain=element_grain,

    element_orientation=element_orientation,

    element_euler=element_euler,

    grain_centers=grain_centers,

    grain_orientation=orientation_matrices,

    grain_euler=euler_angles,

    dimensions=np.array([
        Lx,
        Ly,
        Lz
    ])
)


print(
    "\nFE mesh saved as:"
)

print(
    "316L_FE_mesh_20cube.npz"
)


# ============================================================
# 11. Visualize FE element centers
# ============================================================

fig = plt.figure(
    figsize=(9, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

scatter = ax.scatter(
    element_centers[:, 0],
    element_centers[:, 1],
    element_centers[:, 2],
    c=element_grain,
    s=8
)

ax.set_xlabel(
    "x (μm)",
    fontsize=14
)

ax.set_ylabel(
    "y (μm)",
    fontsize=14
)

ax.set_zlabel(
    "z (μm)",
    fontsize=14
)

ax.set_title(
    "3D FE Element–Grain Assignment",
    fontsize=16
)

plt.tight_layout()

plt.show()


# ============================================================
# 12. Central cross-section
# ============================================================

# Select elements close to the middle of the cube

z_middle = Lz / 2

tolerance = dz_FE / 2

section = (
    np.abs(
        element_centers[:, 2] - z_middle
    )
    < tolerance
)


section_centers = (
    element_centers[section]
)

section_grains = (
    element_grain[section]
)


plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    section_centers[:, 0],
    section_centers[:, 1],
    c=section_grains,
    s=35,
    marker="s"
)

plt.xlabel(
    "x (μm)",
    fontsize=16
)

plt.ylabel(
    "y (μm)",
    fontsize=16
)

plt.title(
    "FE Mesh Grain Assignment — Central Section",
    fontsize=16
)

plt.axis("equal")

plt.tight_layout()

plt.show()


# ============================================================
# 13. Final summary
# ============================================================

print("\n")
print("==============================")
print("FE MESH SUMMARY")
print("==============================")

print(
    f"Domain: "
    f"{Lx:.1f} × {Ly:.1f} × {Lz:.1f} μm³"
)

print(
    f"Elements: "
    f"{Nx_FE} × {Ny_FE} × {Nz_FE}"
)

print(
    f"Total elements: "
    f"{N_elements}"
)

print(
    f"Total nodes: "
    f"{N_nodes}"
)

print(
    f"Element size: "
    f"{dx_FE:.2f} μm"
)

print(
    f"Represented grains: "
    f"{len(unique_grains)}"
)

print(
    "Element data:"
)

print(
    "  element_grain"
)

print(
    "  element_orientation"
)

print(
    "  element_euler"
)

print(
    "==============================")

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# MODULE 3
# 3D HEXAHEDRAL ELASTIC FINITE ELEMENT SOLVER
# ============================================================


# ------------------------------------------------------------
# 1. Load FE mesh
# ------------------------------------------------------------

data = np.load(
    "316L_FE_mesh_20cube.npz"
)

nodes = data["nodes"]
elements = data["elements"]

N_nodes = len(nodes)
N_elements = len(elements)

print("FE mesh loaded.")

print(
    "Nodes:",
    N_nodes
)

print(
    "Elements:",
    N_elements
)


# ============================================================
# 2. Material properties
# ============================================================

E = 193000.0       # MPa
NU = 0.29

print("\nMaterial:")
print("316L elastic modulus =", E, "MPa")
print("Poisson ratio =", NU)


# ============================================================
# 3. Elastic constitutive matrix
# ============================================================

# Engineering strain convention:
#
# eps =
# [eps_xx,
#  eps_yy,
#  eps_zz,
#  gamma_xy,
#  gamma_yz,
#  gamma_zx]
#
# stress =
# [sigma_xx,
#  sigma_yy,
#  sigma_zz,
#  tau_xy,
#  tau_yz,
#  tau_zx]


C = np.zeros((6, 6))

lam = (
    E * NU
    /
    ((1 + NU) * (1 - 2 * NU))
)

mu = (
    E
    /
    (2 * (1 + NU))
)


C[0, 0] = lam + 2 * mu
C[1, 1] = lam + 2 * mu
C[2, 2] = lam + 2 * mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


print("\nElastic constitutive matrix:")
print(C)


# ============================================================
# 4. Hexahedral shape functions
# ============================================================

def shape_functions_hex8(xi, eta, zeta):

    N = np.zeros(8)

    N[0] = (
        1/8
        * (1-xi)
        * (1-eta)
        * (1-zeta)
    )

    N[1] = (
        1/8
        * (1+xi)
        * (1-eta)
        * (1-zeta)
    )

    N[2] = (
        1/8
        * (1+xi)
        * (1+eta)
        * (1-zeta)
    )

    N[3] = (
        1/8
        * (1-xi)
        * (1+eta)
        * (1-zeta)
    )

    N[4] = (
        1/8
        * (1-xi)
        * (1-eta)
        * (1+zeta)
    )

    N[5] = (
        1/8
        * (1+xi)
        * (1-eta)
        * (1+zeta)
    )

    N[6] = (
        1/8
        * (1+xi)
        * (1+eta)
        * (1+zeta)
    )

    N[7] = (
        1/8
        * (1-xi)
        * (1+eta)
        * (1+zeta)
    )

    return N


# ============================================================
# 5. Derivatives of shape functions
# ============================================================

def shape_function_derivatives(
    xi,
    eta,
    zeta
):

    dN = np.zeros((8, 3))

    # dN/dxi
    dN[0, 0] = -1/8*(1-eta)*(1-zeta)
    dN[1, 0] =  1/8*(1-eta)*(1-zeta)
    dN[2, 0] =  1/8*(1+eta)*(1-zeta)
    dN[3, 0] = -1/8*(1+eta)*(1-zeta)

    dN[4, 0] = -1/8*(1-eta)*(1+zeta)
    dN[5, 0] =  1/8*(1-eta)*(1+zeta)
    dN[6, 0] =  1/8*(1+eta)*(1+zeta)
    dN[7, 0] = -1/8*(1+eta)*(1+zeta)

    # dN/deta
    dN[0, 1] = -1/8*(1-xi)*(1-zeta)
    dN[1, 1] = -1/8*(1+xi)*(1-zeta)
    dN[2, 1] =  1/8*(1+xi)*(1-zeta)
    dN[3, 1] =  1/8*(1-xi)*(1-zeta)

    dN[4, 1] = -1/8*(1-xi)*(1+zeta)
    dN[5, 1] = -1/8*(1+xi)*(1+zeta)
    dN[6, 1] =  1/8*(1+xi)*(1+zeta)
    dN[7, 1] =  1/8*(1-xi)*(1+zeta)

    # dN/dzeta
    dN[0, 2] = -1/8*(1-xi)*(1-eta)
    dN[1, 2] = -1/8*(1+xi)*(1-eta)
    dN[2, 2] = -1/8*(1+xi)*(1+eta)
    dN[3, 2] = -1/8*(1-xi)*(1+eta)

    dN[4, 2] =  1/8*(1-xi)*(1-eta)
    dN[5, 2] =  1/8*(1+xi)*(1-eta)
    dN[6, 2] =  1/8*(1+xi)*(1+eta)
    dN[7, 2] =  1/8*(1-xi)*(1+eta)

    return dN


# ============================================================
# 6. Gauss integration
# ============================================================

g = 1.0 / np.sqrt(3)

gauss_points = [

    (-g, -g, -g),
    ( g, -g, -g),
    ( g,  g, -g),
    (-g,  g, -g),

    (-g, -g,  g),
    ( g, -g,  g),
    ( g,  g,  g),
    (-g,  g,  g)
]

gauss_weights = np.ones(8)


# ============================================================
# 7. B matrix
# ============================================================

def calculate_B_matrix(
    element_coordinates,
    xi,
    eta,
    zeta
):

    dN_dxi = shape_function_derivatives(
        xi,
        eta,
        zeta
    )

    # Jacobian
    J = (
        element_coordinates.T
        @ dN_dxi
    )

    detJ = np.linalg.det(J)

    if detJ <= 0:

        raise ValueError(
            "Negative or zero Jacobian."
        )

    # Convert derivatives to physical coordinates
    dN_dx = (
        dN_dxi
        @ np.linalg.inv(J)
    )

    B = np.zeros((6, 24))

    for i in range(8):

        ix = 3 * i

        dNx = dN_dx[i, 0]
        dNy = dN_dx[i, 1]
        dNz = dN_dx[i, 2]

        B[0, ix]     = dNx
        B[1, ix + 1] = dNy
        B[2, ix + 2] = dNz

        B[3, ix]     = dNy
        B[3, ix + 1] = dNx

        B[4, ix + 1] = dNz
        B[4, ix + 2] = dNy

        B[5, ix]     = dNz
        B[5, ix + 2] = dNx

    return B, detJ


# ============================================================
# 8. Element stiffness matrix
# ============================================================

def element_stiffness(
    element_coordinates
):

    Ke = np.zeros((24, 24))

    for gp, weight in zip(
        gauss_points,
        gauss_weights
    ):

        xi, eta, zeta = gp

        B, detJ = calculate_B_matrix(
            element_coordinates,
            xi,
            eta,
            zeta
        )

        Ke += (
            B.T
            @ C
            @ B
            * detJ
            * weight
        )

    return Ke


# ============================================================
# 9. Test one element
# ============================================================

test_element = elements[0]

test_coordinates = (
    nodes[test_element]
)

Ke_test = element_stiffness(
    test_coordinates
)

print(
    "\nTest element stiffness matrix:"
)

print(
    "Shape:",
    Ke_test.shape
)

print(
    "Symmetry error:",
    np.max(
        np.abs(
            Ke_test - Ke_test.T
        )
    )
)








# ============================================================
# MODULE 3B - CORRECTED
# GLOBAL 3D ELASTIC FE SOLVER
#
# Direct displacement boundary-condition method
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# 1. Load FE mesh
# ============================================================

data = np.load(
    "316L_FE_mesh_20cube.npz"
)

nodes = data["nodes"]
elements = data["elements"]

N_nodes = len(nodes)
N_elements = len(elements)

print("FE mesh loaded.")

print("Nodes:", N_nodes)
print("Elements:", N_elements)


# ============================================================
# 2. Convert geometry μm -> mm
# ============================================================

nodes_mm = nodes / 1000.0

Lx = nodes_mm[:, 0].max()
Ly = nodes_mm[:, 1].max()
Lz = nodes_mm[:, 2].max()

print("\nModel dimensions:")
print(
    f"{Lx:.6f} × "
    f"{Ly:.6f} × "
    f"{Lz:.6f} mm"
)


# ============================================================
# 3. Material properties
# ============================================================

E = 193000.0      # MPa
NU = 0.29

print("\nMaterial:")
print("E  =", E, "MPa")
print("ν  =", NU)


# ============================================================
# 4. Rebuild elastic constitutive matrix
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * NU
    /
    ((1 + NU) * (1 - 2 * NU))
)

mu = (
    E
    /
    (2 * (1 + NU))
)

C[0, 0] = lam + 2*mu
C[1, 1] = lam + 2*mu
C[2, 2] = lam + 2*mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 5. Number of DOFs
# ============================================================

DOF_per_node = 3

N_DOF = (
    N_nodes
    *
    DOF_per_node
)

print(
    "\nTotal DOFs:",
    N_DOF
)


# ============================================================
# 6. Assemble global stiffness matrix
# ============================================================

K = lil_matrix(
    (N_DOF, N_DOF)
)

print(
    "\nAssembling global stiffness matrix..."
)

for e in range(N_elements):

    element_nodes = elements[e]

    element_coordinates = (
        nodes_mm[element_nodes]
    )

    Ke = element_stiffness(
        element_coordinates
    )

    # Global DOFs
    dofs = []

    for n in element_nodes:

        dofs.extend([
            3*n,
            3*n + 1,
            3*n + 2
        ])

    # Assembly
    for a in range(24):

        A = dofs[a]

        for b in range(24):

            B = dofs[b]

            K[A, B] += Ke[a, b]


K = K.tocsr()

print(
    "Global assembly completed."
)


# ============================================================
# 7. Identify bottom and top surfaces
# ============================================================

tol = 1e-12

bottom_nodes = np.where(
    np.abs(
        nodes_mm[:, 2]
    ) < tol
)[0]

top_nodes = np.where(
    np.abs(
        nodes_mm[:, 2] - Lz
    ) < tol
)[0]


print(
    "\nBottom nodes:",
    len(bottom_nodes)
)

print(
    "Top nodes:",
    len(top_nodes)
)


# ============================================================
# 8. Applied displacement
# ============================================================

strain_target = 0.0036

u_top = (
    strain_target
    * Lz
)

print(
    "\nTarget strain:",
    strain_target
)

print(
    "Applied top displacement:",
    u_top,
    "mm"
)


# ============================================================
# 9. Construct prescribed displacement vector
# ============================================================

u_prescribed = np.zeros(
    N_DOF
)

prescribed_dofs = []


# ------------------------------------------------------------
# Bottom surface
# ------------------------------------------------------------

for n in bottom_nodes:

    # Fix z displacement
    dof_z = 3*n + 2

    prescribed_dofs.append(
        dof_z
    )


# ------------------------------------------------------------
# Remove rigid-body x translation
# ------------------------------------------------------------

# Select one bottom node
node_x_fix = bottom_nodes[0]

prescribed_dofs.append(
    3*node_x_fix
)


# ------------------------------------------------------------
# Remove rigid-body y translation
# ------------------------------------------------------------

# Select a different bottom node
node_y_fix = bottom_nodes[-1]

prescribed_dofs.append(
    3*node_y_fix + 1
)


# ------------------------------------------------------------
# Top surface z displacement
# ------------------------------------------------------------

for n in top_nodes:

    dof_z = 3*n + 2

    prescribed_dofs.append(
        dof_z
    )

    u_prescribed[dof_z] = u_top


prescribed_dofs = np.unique(
    prescribed_dofs
)


print(
    "\nNumber of prescribed DOFs:",
    len(prescribed_dofs)
)


# ============================================================
# 10. Free DOFs
# ============================================================

all_dofs = np.arange(
    N_DOF
)

free_dofs = np.setdiff1d(
    all_dofs,
    prescribed_dofs
)


print(
    "Number of free DOFs:",
    len(free_dofs)
)


# ============================================================
# 11. Partition global system
# ============================================================

# Original equation:
#
# K u = F
#
# Partition:
#
# Kff uf + Kfp up = Ff
#
# Therefore:
#
# Kff uf = Ff - Kfp up
#


K_ff = K[
    free_dofs[:, None],
    free_dofs
]

K_fp = K[
    free_dofs[:, None],
    prescribed_dofs
]


F = np.zeros(
    N_DOF
)


F_free = (
    F[free_dofs]
    -
    K_fp @
    u_prescribed[
        prescribed_dofs
    ]
)


# # ============================================================
# # 12. Solve free DOFs
# # ============================================================

# print(
#     "\nSolving reduced FE system..."
# )

# u_free = spsolve(
#     K_ff,
#     F_free
# )


# # ============================================================
# # 13. Construct complete displacement vector
# # ============================================================

# u = np.zeros(
#     N_DOF
# )

# u[free_dofs] = u_free

# u[prescribed_dofs] = (
#     u_prescribed[
#         prescribed_dofs
#     ]
# )



# # ============================================================
# # 12–13. Skip standalone elastic direct solve
# # ============================================================
# #
# # The standalone elastic reduced-system solve is not used by
# # the subsequent FE–CPFEM calculation.
# #
# # Solving the 26,899-DOF reduced system with spsolve() invokes
# # a sparse direct factorization and causes a large SuperLU
# # memory requirement.
# #
# # The actual FE–CPFEM equilibrium is solved later using the
# # algorithmic tangent K_solver_alg.
# # ============================================================

# print(
#     "\nStandalone elastic reduced-system solve skipped."
# )

# displacements = u.reshape(
#     (-1, 3)
# )


# print(
#     "FE solution completed."
# )


# # ============================================================
# # 14. Check measured strain
# # ============================================================

# bottom_average = np.mean(
#     displacements[
#         bottom_nodes,
#         2
#     ]
# )

# top_average = np.mean(
#     displacements[
#         top_nodes,
#         2
#     ]
# )

# measured_strain = (
#     top_average
#     -
#     bottom_average
# ) / Lz


# print("\n==============================")
# print("DISPLACEMENT RESULTS")
# print("==============================")

# print(
#     "Target strain:",
#     strain_target
# )

# print(
#     "Measured strain:",
#     measured_strain
# )

# print(
#     "Bottom displacement:",
#     bottom_average,
#     "mm"
# )

# print(
#     "Top displacement:",
#     top_average,
#     "mm"
# )

# ============================================================
# 12–13. Skip standalone elastic direct solve
# ============================================================
#
# The standalone elastic reduced-system solve is intentionally
# skipped.
#
# spsolve(K_ff, F_free) performs a sparse direct factorization
# of the 26,899 free-DOF system and creates a large temporary
# SuperLU memory requirement.
#
# This elastic solution is NOT used by the subsequent
# FE–CPFEM calculation.
#
# The actual FE–CPFEM equilibrium is solved later using
# K_solver_alg.solve(-residual).
#
# For this preliminary displacement check, the prescribed
# boundary displacements are sufficient.
# ============================================================

print(
    "\nStandalone elastic reduced-system solve skipped."
)

# Construct displacement vector using the prescribed
# boundary conditions only.
#
# Free DOFs are intentionally not solved here because this
# displacement field is used only for the boundary-condition
# / macroscopic-strain sanity check.

u = np.zeros(
    N_DOF,
    dtype=np.float64
)

u[prescribed_dofs] = (
    u_prescribed[
        prescribed_dofs
    ]
)

displacements = u.reshape(
    (-1, 3)
)

print(
    "Prescribed displacement field constructed."
)


# ============================================================
# 14. Check measured strain
# ============================================================

bottom_average = np.mean(
    displacements[
        bottom_nodes,
        2
    ]
)

top_average = np.mean(
    displacements[
        top_nodes,
        2
    ]
)

measured_strain = (
    top_average
    -
    bottom_average
) / Lz


print("\n==============================")
print("DISPLACEMENT RESULTS")
print("==============================")

print(
    "Target strain:",
    strain_target
)

print(
    "Measured strain:",
    measured_strain
)







# ============================================================
# 15. Skip standalone elastic reaction-force calculation
# ============================================================
#
# The standalone elastic equilibrium solve was skipped because
# spsolve(K_ff, F_free) creates a large SuperLU memory demand.
#
# Therefore the current displacement vector u contains only
# prescribed boundary displacements and is NOT an equilibrium
# elastic solution.
#
# Consequently, K @ u must NOT be used to calculate reactions.
#
# The actual reaction force and axial stress will be obtained
# from the converged FE–CPFEM solution later.
# ============================================================

print(
    "\nStandalone elastic reaction-force calculation skipped."
)


# ============================================================
# 16. Cross-sectional area
# ============================================================

A = Lx * Ly

print(
    f"Cross-sectional area = {A:.6f} mm^2"
)


# ============================================================
# 17–18. Skip standalone elastic FE stress verification
# ============================================================
#
# No standalone elastic stress is calculated because the
# corresponding equilibrium displacement solution was skipped.
#
# The actual FE–CPFEM stress response is calculated later
# during the nonlinear solution.
# ============================================================

sigma_FE = np.nan
sigma_theoretical = np.nan
stress_error = np.nan

print(
    "Standalone elastic FE stress verification skipped."
)

# # ============================================================
# # 19. Check Poisson contraction
# # ============================================================

# # Average x and y displacement at top

# ux_top = np.mean(
#     displacements[
#         top_nodes,
#         0
#     ]
# )

# uy_top = np.mean(
#     displacements[
#         top_nodes,
#         1
#     ]
# )


# strain_x = (
#     ux_top
#     /
#     Lx
# )

# strain_y = (
#     uy_top
#     /
#     Ly
# )


# print(
#     "\nLateral strains:"
# )

# print(
#     "epsilon_x =",
#     strain_x
# )

# print(
#     "epsilon_y =",
#     strain_y
# )

# print(
#     "Expected approximately:",
#     -nu * strain_target
# )


# # ============================================================
# # 20. Stress-strain verification
# # ============================================================

# print("\n==============================")
# print("ELASTIC FE VERIFICATION")
# print("==============================")

# print(
#     f"Target strain      = "
#     f"{strain_target:.8f}"
# )

# print(
#     f"FE strain          = "
#     f"{measured_strain:.8f}"
# )

# print(
#     "Theoretical stress = "
#     "SKIPPED"
# )

# print(
#     "FE stress          = "
#     "SKIPPED"
# )

# print(
#     "Stress error       = "
#     "SKIPPED"
# )

# print(
#     "=============================="
# )

# ============================================================
# 19–20. Skip standalone elastic Poisson/stress verification
# ============================================================
#
# The preliminary elastic equilibrium solve was intentionally
# removed because the 26,899-DOF sparse direct solve creates
# excessive SuperLU memory usage.
#
# Therefore the displacement vector available here contains
# only prescribed boundary displacements and cannot be used
# to calculate:
#
#   - Poisson contraction
#   - elastic reaction force
#   - elastic FE stress
#   - stress error
#
# These quantities are not required by the subsequent
# FE–CPFEM calculation.
#
# The actual stress, strain, plastic deformation, and reaction
# response are obtained from the converged FE–CPFEM solution.
# ============================================================

print(
    "\nStandalone elastic FE verification skipped."
)

print(
    "Proceeding directly to FE–CPFEM calculation."
)



# ============================================================
# MODULE 4A
# FCC CRYSTAL PLASTICITY
#
# Part 1:
# Generate the 12 FCC {111}<110> slip systems
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


# ============================================================
# 1. Generate FCC slip systems
# ============================================================

def generate_fcc_slip_systems():

    # --------------------------------------------------------
    # FCC slip planes
    #
    # {111}
    # --------------------------------------------------------

    plane_normals = np.array([
        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [-1,  1,  1]
    ], dtype=float)


    # --------------------------------------------------------
    # <110> directions
    # --------------------------------------------------------

    directions = np.array([

        [ 1,  1,  0],
        [ 1, -1,  0],
        [ 1,  0,  1],
        [ 1,  0, -1],
        [ 0,  1,  1],
        [ 0,  1, -1]

    ], dtype=float)


    # Normalize plane normals
    plane_normals /= np.linalg.norm(
        plane_normals,
        axis=1
    )[:, None]


    # Normalize directions
    directions /= np.linalg.norm(
        directions,
        axis=1
    )[:, None]


    slip_systems = []


    # --------------------------------------------------------
    # Select only directions lying in the plane
    #
    # n · s = 0
    # --------------------------------------------------------

    for n in plane_normals:

        for s in directions:

            if abs(
                np.dot(n, s)
            ) < 1e-12:

                slip_systems.append(
                    (
                        n.copy(),
                        s.copy()
                    )
                )


    return slip_systems


slip_systems = (
    generate_fcc_slip_systems()
)


# ============================================================
# 2. Check number of slip systems
# ============================================================

print("==============================")
print("FCC SLIP SYSTEM CHECK")
print("==============================")

print(
    "Number of FCC slip systems:",
    len(slip_systems)
)


# ============================================================
# 3. Print slip systems
# ============================================================

for i, (n, s) in enumerate(
    slip_systems
):

    print(
        f"{i+1:2d}: "
        f"n = {n}, "
        f"s = {s}"
    )


# ============================================================
# 4. Mathematical validation
# ============================================================

max_orthogonality_error = 0.0

for n, s in slip_systems:

    error = abs(
        np.dot(n, s)
    )

    max_orthogonality_error = max(
        max_orthogonality_error,
        error
    )


print(
    "\nMaximum n·s error:",
    max_orthogonality_error
)


# ============================================================
# 5. Verify unit vectors
# ============================================================

normal_errors = []
direction_errors = []

for n, s in slip_systems:

    normal_errors.append(
        abs(np.linalg.norm(n) - 1)
    )

    direction_errors.append(
        abs(np.linalg.norm(s) - 1)
    )


print(
    "Maximum normal normalization error:",
    max(normal_errors)
)

print(
    "Maximum direction normalization error:",
    max(direction_errors)
)


# ============================================================
# 6. Schmid tensor for each slip system
# ============================================================

schmid_tensors = []

for n, s in slip_systems:

    P = 0.5 * (
        np.outer(s, n)
        +
        np.outer(n, s)
    )

    schmid_tensors.append(P)


schmid_tensors = np.array(
    schmid_tensors
)


print(
    "\nSchmid tensor array:",
    schmid_tensors.shape
)

# ============================================================
# MODULE 4B
# CRYSTAL ORIENTATION TRANSFORMATION
# ============================================================


# ============================================================
# 1. Load grain orientations
# ============================================================

micro_data = np.load(
    "316L_3D_Voronoi_microstructure.npz"
)

grain_orientation = (
    micro_data[
        "orientation_matrices"
    ]
)

grain_euler = (
    micro_data[
        "euler_angles"
    ]
)


print(
    "Number of grain orientations:",
    len(grain_orientation)
)


# ============================================================
# 2. Select one example grain
# ============================================================

grain_id = 0

R_grain = (
    grain_orientation[grain_id]
)


print(
    "\nSelected grain:",
    grain_id
)

print(
    "Orientation matrix:"
)

print(
    R_grain
)


# ============================================================
# 3. Rotate slip systems
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n_sample = (
        R_grain
        @ n_crystal
    )

    s_sample = (
        R_grain
        @ s_crystal
    )

    # Re-normalize to remove numerical error

    n_sample /= np.linalg.norm(
        n_sample
    )

    s_sample /= np.linalg.norm(
        s_sample
    )

    rotated_slip_systems.append(
        (
            n_sample,
            s_sample
        )
    )


# ============================================================
# 4. Verify transformed systems
# ============================================================

print(
    "\nTransformed FCC slip systems:"
)

for i, (n, s) in enumerate(
    rotated_slip_systems
):

    print(
        f"{i+1:2d}: "
        f"n = {n}, "
        f"s = {s}, "
        f"n·s = {np.dot(n,s):.3e}"
    )

# ============================================================
# MODULE 4C
# SCHMID FACTOR CALCULATION
# ============================================================


sigma_test = 1.0   # arbitrary unit stress


stress_tensor = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, sigma_test]
])


schmid_factors = []


for n, s in rotated_slip_systems:

    tau = (
        s
        @ stress_tensor
        @ n
    )

    schmid_factors.append(
        tau / sigma_test
    )


schmid_factors = np.array(
    schmid_factors
)


print("==============================")
print("SCHMID FACTORS")
print("==============================")


for i, m in enumerate(
    schmid_factors
):

    print(
        f"Slip system {i+1:2d}: "
        f"m = {m:.6f}"
    )


print(
    "\nMaximum absolute Schmid factor:",
    np.max(
        np.abs(schmid_factors)
    )
)


critical_system = np.argmax(
    np.abs(schmid_factors)
)


print(
    "Critical slip system:",
    critical_system + 1
)

# ============================================================
# MODULE 4D
# SINGLE-CRYSTAL CRYSTAL-PLASTICITY MATERIAL POINT
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CP material parameters
# ============================================================

tau0 = 80.0          # MPa
gamma_dot_0 = 0.001   # 1/s
rate_exponent = 20.0

print("==============================")
print("CRYSTAL PLASTICITY PARAMETERS")
print("==============================")

print("Initial CRSS:", tau0, "MPa")
print("Reference slip rate:", gamma_dot_0, "1/s")
print("Rate exponent:", rate_exponent)


# ============================================================
# 2. Use grain 0 orientation
# ============================================================

R_grain = (
    grain_orientation[0]
)


# ============================================================
# 3. Transform slip systems
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n = R_grain @ n_crystal
    s = R_grain @ s_crystal

    n /= np.linalg.norm(n)
    s /= np.linalg.norm(s)

    rotated_slip_systems.append(
        (n, s)
    )


# ============================================================
# 4. Calculate slip rates
# ============================================================

def calculate_slip_rates(
    stress,
    slip_systems,
    resistance,
    gamma_dot_0,
    rate_exponent
):

    slip_rates = np.zeros(
        len(slip_systems)
    )

    resolved_shear = np.zeros(
        len(slip_systems)
    )


    for a, (n, s) in enumerate(
        slip_systems
    ):

        # Resolved shear stress
        tau = (
            s
            @ stress
            @ n
        )

        resolved_shear[a] = tau


        # Power-law slip kinetics

        ratio = (
            abs(tau)
            /
            resistance[a]
        )

        slip_rates[a] = (
            gamma_dot_0
            *
            ratio**rate_exponent
            *
            np.sign(tau)
        )


    return (
        slip_rates,
        resolved_shear
    )


# ============================================================
# 5. Test at several stresses
# ============================================================

test_stresses = np.linspace(
    0,
    800,
    81
)

max_slip_rates = []

max_resolved_shear = []


for sigma in test_stresses:

    stress = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, sigma]
    ])


    resistance = np.full(
        12,
        tau0
    )


    rates, taus = (
        calculate_slip_rates(
            stress,
            rotated_slip_systems,
            resistance,
            gamma_dot_0,
            rate_exponent
        )
    )


    max_slip_rates.append(
        np.max(np.abs(rates))
    )

    max_resolved_shear.append(
        np.max(np.abs(taus))
    )


max_slip_rates = np.array(
    max_slip_rates
)

max_resolved_shear = np.array(
    max_resolved_shear
)


# ============================================================
# 6. Plot slip activity
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.semilogy(
    test_stresses,
    max_slip_rates
)

plt.xlabel(
    "Applied tensile stress (MPa)",
    fontsize=14
)

plt.ylabel(
    "Maximum slip rate (1/s)",
    fontsize=14
)

plt.title(
    "FCC crystal slip activity",
    fontsize=16
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 7. Print selected results
# ============================================================

print("\n==============================")
print("SLIP ACTIVITY CHECK")
print("==============================")

for sigma in [
    100,
    150,
    200,
    300,
    500,
    700
]:

    stress = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, sigma]
    ])


    resistance = np.full(
        12,
        tau0
    )


    rates, taus = (
        calculate_slip_rates(
            stress,
            rotated_slip_systems,
            resistance,
            gamma_dot_0,
            rate_exponent
        )
    )


    print(
        f"\nApplied stress = {sigma} MPa"
    )

    print(
        "Maximum RSS =",
        np.max(np.abs(taus)),
        "MPa"
    )

    print(
        "Maximum slip rate =",
        np.max(np.abs(rates)),
        "1/s"
    )

# ============================================================
# MODULE 4E
# SINGLE-CRYSTAL CRYSTAL PLASTICITY
#
# Incremental small-strain constitutive model
#
# FCC {111}<110>
# Power-law slip
# Isotropic hardening
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Material parameters
# ============================================================

E = 193000.0          # MPa
NU = 0.29             # Poisson ratio

tau0 = 80.0          # Initial CRSS, MPa
tau_sat = 350.0       # Saturation resistance, MPa
h0 = 3000.0           # Hardening modulus, MPa

gamma_dot_0 = 0.001   # Reference slip rate, 1/s
rate_exponent = 20.0

strain_rate = 0.001   # Applied strain rate, 1/s

total_strain = 0.2

n_steps = 1000

dt = (
    total_strain
    /
    strain_rate
    /
    n_steps
)

delta_strain = (
    strain_rate
    * dt
)


print("==============================")
print("SINGLE-CRYSTAL CP PARAMETERS")
print("==============================")

print("E =", E, "MPa")
print("nu =", NU)
print("Initial CRSS =", tau0, "MPa")
print("Saturation CRSS =", tau_sat, "MPa")
print("Hardening modulus =", h0, "MPa")
print("Reference slip rate =", gamma_dot_0, "1/s")
print("Rate exponent =", rate_exponent)
print("Applied strain rate =", strain_rate, "1/s")
print("Total strain =", total_strain)
print("Number of increments =", n_steps)
print("Strain increment =", delta_strain)


# ============================================================
# 2. Elastic stiffness matrix
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * NU
    /
    ((1 + NU) * (1 - 2 * NU))
)

mu = (
    E
    /
    (2 * (1 + NU))
)

C[0, 0] = lam + 2*mu
C[1, 1] = lam + 2*mu
C[2, 2] = lam + 2*mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 3. Select grain 0
# ============================================================

R_grain = (
    grain_orientation[0]
)


# ============================================================
# 4. Rotate slip systems
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n = R_grain @ n_crystal
    s = R_grain @ s_crystal

    n /= np.linalg.norm(n)
    s /= np.linalg.norm(s)

    rotated_slip_systems.append(
        (n, s)
    )


# ============================================================
# 5. Schmid tensors
# ============================================================

P = []

for n, s in rotated_slip_systems:

    P_alpha = 0.5 * (
        np.outer(s, n)
        +
        np.outer(n, s)
    )

    P.append(
        P_alpha
    )

P = np.array(P)


# ============================================================
# 6. Tensor <-> Voigt conversion
# ============================================================

def tensor_to_voigt(A):

    return np.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        2*A[0, 1],
        2*A[1, 2],
        2*A[0, 2]
    ])


def voigt_to_tensor(v):

    A = np.zeros((3, 3))

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = v[3] / 2
    A[1, 0] = v[3] / 2

    A[1, 2] = v[4] / 2
    A[2, 1] = v[4] / 2

    A[0, 2] = v[5] / 2
    A[2, 0] = v[5] / 2

    return A


# ============================================================
# 7. State variables
# ============================================================

plastic_strain = np.zeros(
    (3, 3)
)

total_strain_tensor = np.zeros(
    (3, 3)
)

stress = np.zeros(
    (3, 3)
)

slip_resistance = np.full(
    12,
    tau0
)

accumulated_slip = np.zeros(
    12
)


# ============================================================
# 8. Storage
# ============================================================

strain_history = []
stress_history = []

plastic_strain_history = []
slip_history = []

crss_history = []

rss_history = []


# ============================================================
# 9. Local constitutive calculation
# ============================================================

def calculate_slip_rates_from_stress(
    stress,
    slip_resistance
):

    tau = np.zeros(12)
    gamma_dot = np.zeros(12)

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        tau[a] = (
            s
            @ stress
            @ n
        )

        ratio = (
            abs(tau[a])
            /
            slip_resistance[a]
        )

        gamma_dot[a] = (
            gamma_dot_0
            *
            ratio**rate_exponent
            *
            np.sign(tau[a])
        )

    return (
        tau,
        gamma_dot
    )


# ============================================================
# 10. Incremental strain-controlled simulation
# ============================================================

for step in range(
    n_steps
):

    # --------------------------------------------------------
    # Applied total strain increment
    #
    # Uniaxial strain in z
    # --------------------------------------------------------

    total_strain_tensor[2, 2] += (
        delta_strain
    )


    # --------------------------------------------------------
    # Elastic trial stress
    # --------------------------------------------------------

    elastic_strain_trial = (
        total_strain_tensor
        -
        plastic_strain
    )

    stress_trial = voigt_to_tensor(
        C
        @
        tensor_to_voigt(
            elastic_strain_trial
        )
    )


    # --------------------------------------------------------
    # Calculate resolved shear stresses
    # --------------------------------------------------------

    tau_trial = np.zeros(12)

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        tau_trial[a] = (
            s
            @ stress_trial
            @ n
        )


    # --------------------------------------------------------
    # Determine slip rates
    # --------------------------------------------------------

    gamma_dot = np.zeros(12)

    for a in range(12):

        ratio = (
            abs(tau_trial[a])
            /
            slip_resistance[a]
        )

        gamma_dot[a] = (
            gamma_dot_0
            *
            ratio**rate_exponent
            *
            np.sign(
                tau_trial[a]
            )
        )


    # --------------------------------------------------------
    # Limit numerical instability
    #
    # This is a prototype material-point model.
    # We limit the maximum slip increment per step.
    # --------------------------------------------------------

    max_gamma_increment = 0.005

    gamma_increment = (
        gamma_dot
        *
        dt
    )

    gamma_increment = np.clip(
        gamma_increment,
        -max_gamma_increment,
        max_gamma_increment
    )


    # --------------------------------------------------------
    # Plastic strain increment
    # --------------------------------------------------------

    plastic_increment = np.zeros(
        (3, 3)
    )

    for a in range(12):

        plastic_increment += (
            gamma_increment[a]
            *
            P[a]
        )


    plastic_strain += (
        plastic_increment
    )


    # --------------------------------------------------------
    # Accumulated slip
    # --------------------------------------------------------

    accumulated_slip += np.abs(
        gamma_increment
    )


    # --------------------------------------------------------
    # Isotropic hardening
    # --------------------------------------------------------

    total_slip_increment = np.sum(
        np.abs(
            gamma_increment
        )
    )


    slip_resistance += (
        h0
        *
        (
            1
            -
            slip_resistance
            /
            tau_sat
        )
        *
        total_slip_increment
    )


    slip_resistance = np.maximum(
        slip_resistance,
        tau0
    )


    # --------------------------------------------------------
    # Recalculate stress
    # --------------------------------------------------------

    elastic_strain = (
        total_strain_tensor
        -
        plastic_strain
    )

    stress = voigt_to_tensor(
        C
        @
        tensor_to_voigt(
            elastic_strain
        )
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    strain_history.append(
        total_strain_tensor[2, 2]
    )

    stress_history.append(
        stress[2, 2]
    )

    plastic_strain_history.append(
        plastic_strain[2, 2]
    )

    slip_history.append(
        np.sum(
            accumulated_slip
        )
    )

    crss_history.append(
        np.mean(
            slip_resistance
        )
    )

    rss_history.append(
        np.max(
            np.abs(
                tau_trial
            )
        )
    )


# Convert arrays

strain_history = np.array(
    strain_history
)

stress_history = np.array(
    stress_history
)

plastic_strain_history = np.array(
    plastic_strain_history
)

slip_history = np.array(
    slip_history
)

crss_history = np.array(
    crss_history
)

rss_history = np.array(
    rss_history
)


# ============================================================
# 11. Results
# ============================================================

print("\n==============================")
print("SINGLE-CRYSTAL CP RESULTS")
print("==============================")

print(
    "Final total strain:",
    strain_history[-1]
)

print(
    "Final stress:",
    stress_history[-1],
    "MPa"
)

print(
    "Final plastic strain:",
    plastic_strain_history[-1]
)

print(
    "Final accumulated slip:",
    slip_history[-1]
)

print(
    "Final mean CRSS:",
    crss_history[-1],
    "MPa"
)

print(
    "Maximum RSS:",
    np.max(rss_history),
    "MPa"
)


# ============================================================
# 12. Stress-strain curve
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    stress_history,
    linewidth=2
)

plt.xlabel(
    "Engineering strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial stress (MPa)",
    fontsize=14
)

plt.title(
    "Single-crystal FCC crystal-plasticity response",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 13. Plastic strain evolution
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    plastic_strain_history,
    linewidth=2
)

plt.xlabel(
    "Engineering strain (%)",
    fontsize=14
)

plt.ylabel(
    "Plastic strain",
    fontsize=14
)

plt.title(
    "Accumulation of crystal plastic strain",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 14. CRSS evolution
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    crss_history,
    linewidth=2
)

plt.xlabel(
    "Engineering strain (%)",
    fontsize=14
)

plt.ylabel(
    "Mean CRSS (MPa)",
    fontsize=14
)

plt.title(
    "Crystal-plasticity hardening",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

# ============================================================
# MODULE 4F-CORRECTED
# IMPLICIT SINGLE-CRYSTAL CRYSTAL PLASTICITY
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# ============================================================
# 1. Material parameters
# ============================================================

E = 193000.0
NU = 0.29

tau0 = 80.0
tau_sat = 350.0
h0 = 3000.0

gamma_dot_0 = 0.001
rate_exponent = 20.0

strain_rate = 0.001

total_strain = 0.2
n_steps = 1000

dt = (
    total_strain
    /
    strain_rate
    /
    n_steps
)

delta_strain = (
    strain_rate * dt
)


print("==============================")
print("CORRECTED IMPLICIT CP")
print("==============================")

print("E =", E, "MPa")
print("nu =", NU)
print("Initial CRSS =", tau0, "MPa")
print("Saturation CRSS =", tau_sat, "MPa")
print("Hardening modulus =", h0, "MPa")
print("Reference slip rate =", gamma_dot_0, "1/s")
print("Rate exponent =", rate_exponent)
print("Strain rate =", strain_rate, "1/s")
print("Total strain =", total_strain)
print("Number of increments =", n_steps)
print("dt =", dt, "s")
print("dε =", delta_strain)


# ============================================================
# 2. Elastic stiffness
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * NU
    /
    ((1 + NU) * (1 - 2 * NU))
)

mu = (
    E
    /
    (2 * (1 + NU))
)

C[0, 0] = lam + 2 * mu
C[1, 1] = lam + 2 * mu
C[2, 2] = lam + 2 * mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 3. Tensor / Voigt conversion
# ============================================================

def tensor_to_voigt(A):

    return np.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        2.0 * A[0, 1],
        2.0 * A[1, 2],
        2.0 * A[0, 2]
    ])


def voigt_to_tensor(v):

    A = np.zeros((3, 3))

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = v[3] / 2
    A[1, 0] = v[3] / 2

    A[1, 2] = v[4] / 2
    A[2, 1] = v[4] / 2

    A[0, 2] = v[5] / 2
    A[2, 0] = v[5] / 2

    return A


# ============================================================
# 4. Grain orientation
# ============================================================

R_grain = grain_orientation[0]


# ============================================================
# 5. Rotate FCC slip systems
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n = R_grain @ n_crystal
    s = R_grain @ s_crystal

    n /= np.linalg.norm(n)
    s /= np.linalg.norm(s)

    rotated_slip_systems.append(
        (n, s)
    )


# ============================================================
# 6. Schmid tensors
# ============================================================

P = np.zeros(
    (12, 3, 3)
)

for a in range(12):

    n = rotated_slip_systems[a][0]
    s = rotated_slip_systems[a][1]

    P[a] = 0.5 * (
        np.outer(s, n)
        +
        np.outer(n, s)
    )


# ============================================================
# 7. Initial state
# ============================================================

plastic_strain = np.zeros(
    (3, 3)
)

slip_resistance = np.full(
    12,
    tau0
)

accumulated_slip = np.zeros(
    12
)


# ============================================================
# 8. Functions
# ============================================================

def calculate_resolved_shear(stress):

    tau = np.zeros(12)

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        tau[a] = (
            s
            @ stress
            @ n
        )

    return tau


def calculate_slip_rate(
    tau,
    resistance
):

    ratio = (
        np.abs(tau)
        /
        resistance
    )

    gamma_dot = (
        gamma_dot_0
        *
        ratio**rate_exponent
        *
        np.sign(tau)
    )

    return gamma_dot


# ============================================================
# 9. Local residual
# ============================================================

def local_residual(
    x,
    eps_zz_new,
    plastic_old,
    resistance_old
):

    # --------------------------------------------------------
    # Unknowns
    # --------------------------------------------------------

    eps_xx = x[0]
    eps_yy = x[1]
    eps_xy = x[2]
    eps_yz = x[3]
    eps_xz = x[4]

    dg = x[5:17]


    # --------------------------------------------------------
    # Total strain
    # --------------------------------------------------------

    total_strain = np.array([
        [eps_xx, eps_xy, eps_xz],
        [eps_xy, eps_yy, eps_yz],
        [eps_xz, eps_yz, eps_zz_new]
    ])


    # --------------------------------------------------------
    # Plastic increment
    # --------------------------------------------------------

    plastic_increment = np.zeros(
        (3, 3)
    )

    for a in range(12):

        plastic_increment += (
            dg[a] * P[a]
        )


    plastic_new = (
        plastic_old
        +
        plastic_increment
    )


    # --------------------------------------------------------
    # Elastic strain
    # --------------------------------------------------------

    elastic_strain = (
        total_strain
        -
        plastic_new
    )


    # --------------------------------------------------------
    # Stress
    # --------------------------------------------------------

    stress_new = voigt_to_tensor(
        C
        @
        tensor_to_voigt(
            elastic_strain
        )
    )


    # --------------------------------------------------------
    # RSS
    # --------------------------------------------------------

    tau = calculate_resolved_shear(
        stress_new
    )


    # --------------------------------------------------------
    # Hardening
    # --------------------------------------------------------

    total_slip_increment = np.sum(
        np.abs(dg)
    )

    resistance_new = (
        resistance_old
        +
        h0
        *
        (
            1
            -
            resistance_old
            /
            tau_sat
        )
        *
        total_slip_increment
    )

    resistance_new = np.maximum(
        resistance_new,
        tau0
    )


    # --------------------------------------------------------
    # Slip-rate equation
    # --------------------------------------------------------

    gamma_dot = calculate_slip_rate(
        tau,
        resistance_new
    )

    flow_residual = (
        dg
        -
        dt * gamma_dot
    )


    # --------------------------------------------------------
    # Stress-free conditions
    # --------------------------------------------------------

    stress_residual = np.array([

        stress_new[0, 0],
        stress_new[1, 1],
        stress_new[0, 1],
        stress_new[1, 2],
        stress_new[0, 2]

    ])


    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    stress_scale = E

    slip_scale = max(
        gamma_dot_0 * dt,
        1e-8
    )


    return np.concatenate([

        stress_residual / stress_scale,

        flow_residual / slip_scale

    ])


# ============================================================
# 10. History
# ============================================================

strain_history = []
stress_history = []

plastic_history = []
crss_history = []

slip_history = []
rss_history = []

lateral_x_history = []
lateral_y_history = []

shear_xy_history = []
shear_xz_history = []
shear_yz_history = []

residual_history = []


# ============================================================
# 11. Initial solution
# ============================================================

x_previous = np.zeros(17)

x_previous[0] = 0.0
x_previous[1] = 0.0


# ============================================================
# 12. Increment loop
# ============================================================

converged_steps = 0


for step in range(n_steps):

    eps_zz_new = (
        (step + 1)
        *
        delta_strain
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Use the previous converged solution directly.
    #
    # Do NOT reset lateral strain to elastic Poisson values
    # at every increment.
    # --------------------------------------------------------

    x0 = x_previous.copy()


    # First increment only:
    if step == 0:

        x0[0] = -NU * eps_zz_new
        x0[1] = -NU * eps_zz_new


    # --------------------------------------------------------
    # Nonlinear solve
    # --------------------------------------------------------

    solution = least_squares(
        local_residual,
        x0,
        args=(
            eps_zz_new,
            plastic_strain,
            slip_resistance
        ),
        max_nfev=2000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        verbose=0
    )


    if not solution.success:

        print(
            "\nWARNING: Local solver failed"
        )

        print(
            "Step:",
            step + 1
        )

        print(
            "Strain:",
            eps_zz_new
        )

        print(
            solution.message
        )

        print(
            "Residual norm:",
            np.linalg.norm(
                solution.fun
            )
        )

        break


    # --------------------------------------------------------
    # Save converged solution
    # --------------------------------------------------------

    x = solution.x

    x_previous = x.copy()


    eps_xx = x[0]
    eps_yy = x[1]
    eps_xy = x[2]
    eps_yz = x[3]
    eps_xz = x[4]

    dg = x[5:17]


    # --------------------------------------------------------
    # Total strain tensor
    # --------------------------------------------------------

    total_strain = np.array([
        [eps_xx, eps_xy, eps_xz],
        [eps_xy, eps_yy, eps_yz],
        [eps_xz, eps_yz, eps_zz_new]
    ])


    # --------------------------------------------------------
    # Plastic increment
    # --------------------------------------------------------

    plastic_increment = np.zeros(
        (3, 3)
    )

    for a in range(12):

        plastic_increment += (
            dg[a] * P[a]
        )


    plastic_strain += (
        plastic_increment
    )


    # --------------------------------------------------------
    # Accumulated slip
    # --------------------------------------------------------

    accumulated_slip += np.abs(dg)


    # --------------------------------------------------------
    # Hardening
    # --------------------------------------------------------

    total_slip_increment = np.sum(
        np.abs(dg)
    )

    slip_resistance += (
        h0
        *
        (
            1
            -
            slip_resistance
            /
            tau_sat
        )
        *
        total_slip_increment
    )

    slip_resistance = np.maximum(
        slip_resistance,
        tau0
    )


    # --------------------------------------------------------
    # Final stress
    # --------------------------------------------------------

    elastic_strain = (
        total_strain
        -
        plastic_strain
    )

    stress = voigt_to_tensor(
        C
        @
        tensor_to_voigt(
            elastic_strain
        )
    )


    # --------------------------------------------------------
    # RSS
    # --------------------------------------------------------

    tau = calculate_resolved_shear(
        stress
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    strain_history.append(
        eps_zz_new
    )

    stress_history.append(
        stress[2, 2]
    )

    plastic_history.append(
        plastic_strain[2, 2]
    )

    crss_history.append(
        np.mean(
            slip_resistance
        )
    )

    slip_history.append(
        np.sum(
            accumulated_slip
        )
    )

    rss_history.append(
        np.max(
            np.abs(tau)
        )
    )

    lateral_x_history.append(
        eps_xx
    )

    lateral_y_history.append(
        eps_yy
    )

    shear_xy_history.append(
        eps_xy
    )

    shear_xz_history.append(
        eps_xz
    )

    shear_yz_history.append(
        eps_yz
    )

    residual_history.append(
        np.linalg.norm(
            solution.fun
        )
    )

    converged_steps += 1


# ============================================================
# 13. Convert history
# ============================================================

strain_history = np.array(
    strain_history
)

stress_history = np.array(
    stress_history
)

plastic_history = np.array(
    plastic_history
)

crss_history = np.array(
    crss_history
)

slip_history = np.array(
    slip_history
)

rss_history = np.array(
    rss_history
)

lateral_x_history = np.array(
    lateral_x_history
)

lateral_y_history = np.array(
    lateral_y_history
)

shear_xy_history = np.array(
    shear_xy_history
)

shear_xz_history = np.array(
    shear_xz_history
)

shear_yz_history = np.array(
    shear_yz_history
)

residual_history = np.array(
    residual_history
)


# ============================================================
# 14. Final verification
# ============================================================

print("\n==============================")
print("CORRECTED IMPLICIT CP RESULTS")
print("==============================")

print(
    "Converged increments:",
    converged_steps,
    "/",
    n_steps
)

print(
    "Final strain:",
    strain_history[-1]
)

print(
    "Final axial stress:",
    stress_history[-1],
    "MPa"
)

print(
    "Final plastic strain:",
    plastic_history[-1]
)

print(
    "Final accumulated slip:",
    slip_history[-1]
)

print(
    "Final mean CRSS:",
    crss_history[-1],
    "MPa"
)

print(
    "Maximum RSS:",
    np.max(rss_history),
    "MPa"
)


# ============================================================
# 15. Reconstruct actual final strain tensor
# ============================================================

eps_final = np.array([

    [
        lateral_x_history[-1],
        shear_xy_history[-1],
        shear_xz_history[-1]
    ],

    [
        shear_xy_history[-1],
        lateral_y_history[-1],
        shear_yz_history[-1]
    ],

    [
        shear_xz_history[-1],
        shear_yz_history[-1],
        strain_history[-1]
    ]

])


# ============================================================
# 16. Final stress
# ============================================================

elastic_final = (
    eps_final
    -
    plastic_strain
)

stress_final = voigt_to_tensor(
    C
    @
    tensor_to_voigt(
        elastic_final
    )
)


print("\n==============================")
print("FINAL STRESS STATE")
print("==============================")

print(
    "sigma_xx =",
    stress_final[0, 0],
    "MPa"
)

print(
    "sigma_yy =",
    stress_final[1, 1],
    "MPa"
)

print(
    "sigma_zz =",
    stress_final[2, 2],
    "MPa"
)

print(
    "sigma_xy =",
    stress_final[0, 1],
    "MPa"
)

print(
    "sigma_xz =",
    stress_final[0, 2],
    "MPa"
)

print(
    "sigma_yz =",
    stress_final[1, 2],
    "MPa"
)

print(
    "Final nonlinear residual norm =",
    residual_history[-1]
)


# ============================================================
# 17. Stress-strain curve
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    stress_history,
    linewidth=2
)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial stress (MPa)",
    fontsize=14
)

plt.title(
    "Implicit single-crystal CP response",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 18. Lateral strain
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    lateral_x_history,
    label="epsilon_x"
)

plt.plot(
    strain_history * 100,
    lateral_y_history,
    label="epsilon_y"
)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Lateral strain",
    fontsize=14
)

plt.title(
    "Crystal lateral deformation",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. CRSS evolution
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    strain_history * 100,
    crss_history,
    linewidth=2
)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Mean CRSS (MPa)",
    fontsize=14
)

plt.title(
    "Crystal hardening",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 20. Solver residual
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.semilogy(
    strain_history * 100,
    residual_history,
    linewidth=2
)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Nonlinear residual norm",
    fontsize=14
)

plt.title(
    "Local CP solver convergence",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

# ============================================================
# MODULE 4G
# THREE-GRAIN ORIENTATION VERIFICATION
#
# Purpose:
# Verify that different grain orientations produce
# different crystal-plasticity responses.
#
# Uses:
#   grain_orientation
#   grain_euler
#   slip_systems
#
# from the previously generated microstructure.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CHECK INPUT DATA
# ============================================================

print("==============================")
print("THREE-GRAIN CP VERIFICATION")
print("==============================")

print(
    "Number of available grains:",
    len(grain_orientation)
)

print(
    "Orientation array shape:",
    np.array(grain_orientation).shape
)


# ============================================================
# 2. MATERIAL PARAMETERS
# ============================================================

E = 193000.0
NU = 0.29

tau0 = 80.0
tau_sat = 350.0
h0 = 3000.0

gamma_dot_0 = 0.001
rate_exponent = 20.0

strain_rate = 0.001

total_strain = 0.2
n_steps = 1000

dt = (
    total_strain
    /
    strain_rate
    /
    n_steps
)

delta_strain = (
    strain_rate * dt
)


# ============================================================
# 3. ELASTIC STIFFNESS MATRIX
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * NU
    /
    ((1 + NU) * (1 - 2 * NU))
)

mu = (
    E
    /
    (2 * (1 + NU))
)

C[0, 0] = lam + 2 * mu
C[1, 1] = lam + 2 * mu
C[2, 2] = lam + 2 * mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 4. VOIGT FUNCTIONS
# ============================================================

def tensor_to_voigt(A):

    return np.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        2.0 * A[0, 1],
        2.0 * A[1, 2],
        2.0 * A[0, 2]
    ])


def voigt_to_tensor(v):

    A = np.zeros((3, 3))

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = v[3] / 2
    A[1, 0] = v[3] / 2

    A[1, 2] = v[4] / 2
    A[2, 1] = v[4] / 2

    A[0, 2] = v[5] / 2
    A[2, 0] = v[5] / 2

    return A


# ============================================================
# 5. RUN CP FOR ONE GRAIN
# ============================================================

def run_single_grain_cp(grain_id):

    print("\n--------------------------------")
    print("Running grain", grain_id)
    print("--------------------------------")


    # --------------------------------------------------------
    # Grain orientation
    # --------------------------------------------------------

    R_grain = np.array(
        grain_orientation[grain_id]
    )


    # --------------------------------------------------------
    # Rotate slip systems
    # --------------------------------------------------------

    rotated_slip_systems = []

    for n_crystal, s_crystal in slip_systems:

        n = R_grain @ n_crystal
        s = R_grain @ s_crystal

        n /= np.linalg.norm(n)
        s /= np.linalg.norm(s)

        rotated_slip_systems.append(
            (n, s)
        )


    # --------------------------------------------------------
    # Schmid factors for uniaxial z loading
    # --------------------------------------------------------

    schmid = np.zeros(12)

    loading_direction = np.array([
        0.0,
        0.0,
        1.0
    ])

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        schmid[a] = (
            np.dot(
                loading_direction,
                s
            )
            *
            np.dot(
                loading_direction,
                n
            )
        )


    critical_slip_system = np.argmax(
        np.abs(schmid)
    )

    maximum_schmid = np.max(
        np.abs(schmid)
    )


    print(
        "Maximum Schmid factor =",
        maximum_schmid
    )

    print(
        "Critical slip system =",
        critical_slip_system + 1
    )


    # --------------------------------------------------------
    # Schmid tensors
    # --------------------------------------------------------

    P = np.zeros(
        (12, 3, 3)
    )

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        P[a] = 0.5 * (
            np.outer(s, n)
            +
            np.outer(n, s)
        )


    # --------------------------------------------------------
    # State variables
    # --------------------------------------------------------

    plastic_strain = np.zeros(
        (3, 3)
    )

    slip_resistance = np.full(
        12,
        tau0
    )

    accumulated_slip = np.zeros(
        12
    )


    # --------------------------------------------------------
    # Functions specific to this grain
    # --------------------------------------------------------

    def calculate_tau(stress):

        tau = np.zeros(12)

        for a in range(12):

            n = rotated_slip_systems[a][0]
            s = rotated_slip_systems[a][1]

            tau[a] = (
                s
                @
                stress
                @
                n
            )

        return tau


    def calculate_slip_rate(
        tau,
        resistance
    ):

        ratio = (
            np.abs(tau)
            /
            resistance
        )

        return (
            gamma_dot_0
            *
            ratio**rate_exponent
            *
            np.sign(tau)
        )


    # --------------------------------------------------------
    # Local residual
    # --------------------------------------------------------

    def residual(
        x,
        eps_zz_new,
        plastic_old,
        resistance_old
    ):

        eps_xx = x[0]
        eps_yy = x[1]
        eps_xy = x[2]
        eps_yz = x[3]
        eps_xz = x[4]

        dg = x[5:17]


        total_eps = np.array([

            [eps_xx, eps_xy, eps_xz],

            [eps_xy, eps_yy, eps_yz],

            [eps_xz, eps_yz, eps_zz_new]

        ])


        plastic_increment = np.zeros(
            (3, 3)
        )

        for a in range(12):

            plastic_increment += (
                dg[a] * P[a]
            )


        plastic_new = (
            plastic_old
            +
            plastic_increment
        )


        elastic_eps = (
            total_eps
            -
            plastic_new
        )


        stress = voigt_to_tensor(
            C
            @
            tensor_to_voigt(
                elastic_eps
            )
        )


        tau = calculate_tau(
            stress
        )


        total_slip_increment = np.sum(
            np.abs(dg)
        )


        resistance_new = (
            resistance_old
            +
            h0
            *
            (
                1.0
                -
                resistance_old
                /
                tau_sat
            )
            *
            total_slip_increment
        )


        resistance_new = np.maximum(
            resistance_new,
            tau0
        )


        gamma_dot = calculate_slip_rate(
            tau,
            resistance_new
        )


        flow_residual = (
            dg
            -
            dt * gamma_dot
        )


        stress_residual = np.array([

            stress[0, 0],
            stress[1, 1],
            stress[0, 1],
            stress[1, 2],
            stress[0, 2]

        ])


        stress_scale = E

        slip_scale = max(
            gamma_dot_0 * dt,
            1e-8
        )


        return np.concatenate([

            stress_residual / stress_scale,

            flow_residual / slip_scale

        ])


    # --------------------------------------------------------
    # Histories
    # --------------------------------------------------------

    strain_hist = []
    stress_hist = []
    plastic_hist = []
    crss_hist = []
    slip_hist = []
    rss_hist = []


    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    x_previous = np.zeros(17)


    # --------------------------------------------------------
    # Increment loop
    # --------------------------------------------------------

    from scipy.optimize import least_squares


    for step in range(n_steps):

        eps_zz_new = (
            (step + 1)
            *
            delta_strain
        )


        x0 = x_previous.copy()


        if step == 0:

            x0[0] = (
                -NU
                *
                eps_zz_new
            )

            x0[1] = (
                -NU
                *
                eps_zz_new
            )


        solution = least_squares(

            residual,
            x0,

            args=(
                eps_zz_new,
                plastic_strain,
                slip_resistance
            ),

            max_nfev=2000,

            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11

        )


        if not solution.success:

            print(
                "Solver failed at step:",
                step + 1
            )

            print(
                solution.message
            )

            break


        x = solution.x

        x_previous = x.copy()


        dg = x[5:17]


        total_eps = np.array([

            [x[0], x[2], x[4]],

            [x[2], x[1], x[3]],

            [x[4], x[3], eps_zz_new]

        ])


        # ----------------------------------------------------
        # Plastic update
        # ----------------------------------------------------

        plastic_increment = np.zeros(
            (3, 3)
        )

        for a in range(12):

            plastic_increment += (
                dg[a] * P[a]
            )


        plastic_strain += (
            plastic_increment
        )


        # ----------------------------------------------------
        # Slip update
        # ----------------------------------------------------

        accumulated_slip += np.abs(dg)


        # ----------------------------------------------------
        # Hardening
        # ----------------------------------------------------

        total_slip_increment = np.sum(
            np.abs(dg)
        )

        slip_resistance += (

            h0
            *
            (
                1.0
                -
                slip_resistance
                /
                tau_sat
            )
            *
            total_slip_increment

        )


        slip_resistance = np.maximum(
            slip_resistance,
            tau0
        )


        # ----------------------------------------------------
        # Stress
        # ----------------------------------------------------

        elastic_eps = (
            total_eps
            -
            plastic_strain
        )


        stress = voigt_to_tensor(

            C
            @
            tensor_to_voigt(
                elastic_eps
            )

        )


        # ----------------------------------------------------
        # RSS
        # ----------------------------------------------------

        tau = calculate_tau(
            stress
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        strain_hist.append(
            eps_zz_new
        )

        stress_hist.append(
            stress[2, 2]
        )

        plastic_hist.append(
            plastic_strain[2, 2]
        )

        crss_hist.append(
            np.mean(
                slip_resistance
            )
        )

        slip_hist.append(
            np.sum(
                accumulated_slip
            )
        )

        rss_hist.append(
            np.max(
                np.abs(tau)
            )
        )


    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    strain_hist = np.array(
        strain_hist
    )

    stress_hist = np.array(
        stress_hist
    )

    plastic_hist = np.array(
        plastic_hist
    )

    crss_hist = np.array(
        crss_hist
    )

    slip_hist = np.array(
        slip_hist
    )

    rss_hist = np.array(
        rss_hist
    )


    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    result = {

        "grain_id":
            grain_id,

        "schmid":
            schmid,

        "max_schmid":
            maximum_schmid,

        "critical_system":
            critical_slip_system,

        "strain":
            strain_hist,

        "stress":
            stress_hist,

        "plastic_strain":
            plastic_hist,

        "crss":
            crss_hist,

        "accumulated_slip":
            slip_hist,

        "rss":
            rss_hist,

        "orientation":
            R_grain

    }


    print(
        "Final stress =",
        stress_hist[-1],
        "MPa"
    )

    print(
        "Final plastic strain =",
        plastic_hist[-1]
    )

    print(
        "Final accumulated slip =",
        slip_hist[-1]
    )

    print(
        "Final mean CRSS =",
        crss_hist[-1],
        "MPa"
    )

    print(
        "Maximum RSS =",
        np.max(rss_hist),
        "MPa"
    )


    return result


# ============================================================
# 6. Select three grains
# ============================================================

test_grains = [
    0,
    1,
    2
]


# ============================================================
# 7. Run the three grains
# ============================================================

results_3grain = []

for grain_id in test_grains:

    results_3grain.append(
        run_single_grain_cp(
            grain_id
        )
    )


# ============================================================
# 8. Summary table
# ============================================================

print("\n")
print("==============================")
print("THREE-GRAIN SUMMARY")
print("==============================")

print(
    "Grain | Max Schmid | Final Stress | "
    "Plastic Strain | Accumulated Slip | Mean CRSS"
)

print("-" * 85)

for r in results_3grain:

    print(
        f"{r['grain_id']:5d} | "
        f"{r['max_schmid']:.6f} | "
        f"{r['stress'][-1]:.3f} MPa | "
        f"{r['plastic_strain'][-1]:.6f} | "
        f"{r['accumulated_slip'][-1]:.6f} | "
        f"{r['crss'][-1]:.3f} MPa"
    )


# ============================================================
# 9. Plot stress-strain responses
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for r in results_3grain:

    plt.plot(

        r["strain"] * 100,

        r["stress"],

        linewidth=2,

        label=(
            "Grain "
            + str(r["grain_id"])
            + "  "
            + "m="
            + f"{r['max_schmid']:.3f}"
        )

    )


plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial stress (MPa)",
    fontsize=14
)

plt.title(
    "Orientation-dependent crystal-plasticity response",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 10. Plot plastic strain
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for r in results_3grain:

    plt.plot(

        r["strain"] * 100,

        r["plastic_strain"],

        linewidth=2,

        label=(
            "Grain "
            + str(r["grain_id"])
        )

    )


plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial plastic strain",
    fontsize=14
)

plt.title(
    "Orientation-dependent plastic deformation",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 11. Plot accumulated slip
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for r in results_3grain:

    plt.plot(

        r["strain"] * 100,

        r["accumulated_slip"],

        linewidth=2,

        label=(
            "Grain "
            + str(r["grain_id"])
        )

    )


plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Accumulated slip",
    fontsize=14
)

plt.title(
    "Orientation-dependent slip accumulation",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

# ============================================================
# MODULE 5A-C
# CORRECTED ONE-ELEMENT FE–CPFEM COUPLING
#
# IMPORTANT:
# This uses the SAME 17-variable implicit CP formulation
# that produced the verified grain-0 result:
#
# Final stress ≈ 395.906 MPa
#
# The FE element is used to generate the strain.
# The CP integration algorithm is kept identical.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


print("==============================")
print("MODULE 5A-C")
print("CORRECTED ONE-ELEMENT FE–CPFEM")
print("==============================")


# ============================================================
# 1. MATERIAL PARAMETERS
# ============================================================

E = 193000.0
NU = 0.29

tau0 = 80.0
tau_sat = 350.0
h0 = 3000.0

gamma_dot_0 = 0.001
rate_exponent = 20.0

strain_rate = 0.001

total_strain = 0.2
n_steps = 1000

dt = 0.01
delta_strain = 1e-5


# ============================================================
# 2. ELASTIC MATRIX
# ============================================================

C = np.zeros((6, 6))

lam = (
    E * NU
    /
    ((1 + NU) * (1 - 2 * NU))
)

mu = (
    E
    /
    (2 * (1 + NU))
)

C[0, 0] = lam + 2 * mu
C[1, 1] = lam + 2 * mu
C[2, 2] = lam + 2 * mu

C[0, 1] = lam
C[0, 2] = lam
C[1, 0] = lam
C[1, 2] = lam
C[2, 0] = lam
C[2, 1] = lam

C[3, 3] = mu
C[4, 4] = mu
C[5, 5] = mu


# ============================================================
# 3. VOIGT CONVERSION
# ============================================================

def tensor_to_voigt(A):

    return np.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        2*A[0, 1],
        2*A[1, 2],
        2*A[0, 2]
    ])


def voigt_to_tensor(v):

    A = np.zeros((3, 3))

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = v[3] / 2
    A[1, 0] = v[3] / 2

    A[1, 2] = v[4] / 2
    A[2, 1] = v[4] / 2

    A[0, 2] = v[5] / 2
    A[2, 0] = v[5] / 2

    return A


# ============================================================
# 4. HEX8 ELEMENT
# ============================================================

L = 0.018

nodes = np.array([

    [0.0, 0.0, 0.0],
    [L,   0.0, 0.0],
    [L,   L,   0.0],
    [0.0, L,   0.0],

    [0.0, 0.0, L],
    [L,   0.0, L],
    [L,   L,   L],
    [0.0, L,   L]

])


# ============================================================
# 5. HEX8 SHAPE FUNCTIONS
# ============================================================

def shape_function_derivatives(
    xi,
    eta,
    zeta
):

    dN = np.zeros((8, 3))

    dN[0] = [
        -0.125*(1-eta)*(1-zeta),
        -0.125*(1-xi)*(1-zeta),
        -0.125*(1-xi)*(1-eta)
    ]

    dN[1] = [
         0.125*(1-eta)*(1-zeta),
        -0.125*(1+xi)*(1-zeta),
        -0.125*(1+xi)*(1-eta)
    ]

    dN[2] = [
         0.125*(1+eta)*(1-zeta),
         0.125*(1+xi)*(1-zeta),
        -0.125*(1+xi)*(1+eta)
    ]

    dN[3] = [
        -0.125*(1+eta)*(1-zeta),
         0.125*(1-xi)*(1-zeta),
        -0.125*(1-xi)*(1+eta)
    ]

    dN[4] = [
        -0.125*(1-eta)*(1+zeta),
        -0.125*(1-xi)*(1+zeta),
         0.125*(1-xi)*(1-eta)
    ]

    dN[5] = [
         0.125*(1-eta)*(1+zeta),
        -0.125*(1+xi)*(1+zeta),
         0.125*(1+xi)*(1-eta)
    ]

    dN[6] = [
         0.125*(1+eta)*(1+zeta),
         0.125*(1+xi)*(1+zeta),
         0.125*(1+xi)*(1+eta)
    ]

    dN[7] = [
        -0.125*(1+eta)*(1+zeta),
         0.125*(1-xi)*(1+zeta),
         0.125*(1-xi)*(1+eta)
    ]

    return dN


# ============================================================
# 6. FE B MATRIX
# ============================================================

def calculate_B(node_coordinates):

    dN = shape_function_derivatives(
        0.0,
        0.0,
        0.0
    )

    J = dN.T @ node_coordinates

    detJ = np.linalg.det(J)

    if detJ <= 0:

        raise ValueError(
            "Negative Jacobian."
        )

    dN_dx = dN @ np.linalg.inv(J)

    B = np.zeros((6, 24))

    for a in range(8):

        i = 3*a

        dNx = dN_dx[a, 0]
        dNy = dN_dx[a, 1]
        dNz = dN_dx[a, 2]

        B[0, i] = dNx

        B[1, i+1] = dNy

        B[2, i+2] = dNz

        B[3, i] = dNy
        B[3, i+1] = dNx

        B[4, i+1] = dNz
        B[4, i+2] = dNy

        B[5, i] = dNz
        B[5, i+2] = dNx

    return B, J, detJ


B, J, detJ = calculate_B(nodes)


print("\n==============================")
print("FE ELEMENT CHECK")
print("==============================")

print("B shape =", B.shape)
print("detJ =", detJ)


# ============================================================
# 7. GRAIN 0 ORIENTATION
# ============================================================

grain_id = 0

R_grain = np.array(
    grain_orientation[grain_id]
)


# ============================================================
# 8. ROTATE SLIP SYSTEMS
# ============================================================

rotated_slip_systems = []

for n_crystal, s_crystal in slip_systems:

    n = R_grain @ n_crystal

    s = R_grain @ s_crystal

    n /= np.linalg.norm(n)

    s /= np.linalg.norm(s)

    rotated_slip_systems.append(
        (n, s)
    )


# ============================================================
# 9. SCHMID TENSORS
# ============================================================

P = np.zeros(
    (12, 3, 3)
)

for a in range(12):

    n = rotated_slip_systems[a][0]
    s = rotated_slip_systems[a][1]

    P[a] = 0.5 * (

        np.outer(s, n)
        +
        np.outer(n, s)

    )


# ============================================================
# 10. RSS
# ============================================================

def calculate_tau(stress):

    tau = np.zeros(12)

    for a in range(12):

        n = rotated_slip_systems[a][0]
        s = rotated_slip_systems[a][1]

        tau[a] = s @ stress @ n

    return tau


# ============================================================
# 11. SLIP RATE
# ============================================================

def calculate_slip_rate(
    tau,
    resistance
):

    ratio = (
        np.abs(tau)
        /
        resistance
    )

    return (

        gamma_dot_0
        *
        ratio**rate_exponent
        *
        np.sign(tau)

    )


# ============================================================
# 12. STATE VARIABLES
# ============================================================

plastic_strain = np.zeros(
    (3, 3)
)

slip_resistance = np.full(
    12,
    tau0
)

accumulated_slip = np.zeros(
    12
)


# ============================================================
# 13. HISTORY
# ============================================================

strain_history = []

stress_history = []

plastic_history = []

slip_history = []

crss_history = []

rss_history = []

residual_history = []


# ============================================================
# 14. PREVIOUS SOLUTION VECTOR
#
# THIS IS IMPORTANT.
#
# We retain the complete previous nonlinear solution as the
# initial guess for the next increment.
# ============================================================

x_previous = np.zeros(17)


# ============================================================
# 15. LOAD LOOP
# ============================================================

for step in range(n_steps):


    eps_zz = (
        (step + 1)
        *
        delta_strain
    )


    # --------------------------------------------------------
    # Initial guess from previous increment
    # --------------------------------------------------------

    x0 = x_previous.copy()


    if step == 0:

        x0[0] = -NU * eps_zz

        x0[1] = -NU * eps_zz


    # ========================================================
    # LOCAL RESIDUAL
    # ========================================================

    def local_residual(x):

        eps_xx = x[0]
        eps_yy = x[1]

        eps_xy = x[2]
        eps_yz = x[3]
        eps_xz = x[4]

        dg = x[5:17]


        # ----------------------------------------------------
        # Total strain tensor
        # ----------------------------------------------------

        total_eps = np.array([

            [
                eps_xx,
                eps_xy,
                eps_xz
            ],

            [
                eps_xy,
                eps_yy,
                eps_yz
            ],

            [
                eps_xz,
                eps_yz,
                eps_zz
            ]

        ])


        # ----------------------------------------------------
        # Plastic increment
        # ----------------------------------------------------

        plastic_increment = np.zeros(
            (3, 3)
        )

        for a in range(12):

            plastic_increment += (
                dg[a] * P[a]
            )


        plastic_new = (
            plastic_strain
            +
            plastic_increment
        )


        # ----------------------------------------------------
        # Elastic strain
        # ----------------------------------------------------

        elastic_eps = (
            total_eps
            -
            plastic_new
        )


        # ----------------------------------------------------
        # Stress
        # ----------------------------------------------------

        stress = voigt_to_tensor(

            C
            @
            tensor_to_voigt(
                elastic_eps
            )

        )


        # ----------------------------------------------------
        # RSS
        # ----------------------------------------------------

        tau = calculate_tau(
            stress
        )


        # ----------------------------------------------------
        # Hardening
        # ----------------------------------------------------

        total_slip_increment = np.sum(
            np.abs(dg)
        )


        resistance_new = (

            slip_resistance

            +

            h0
            *
            (
                1.0
                -
                slip_resistance
                /
                tau_sat
            )
            *
            total_slip_increment

        )


        resistance_new = np.maximum(
            resistance_new,
            tau0
        )


        # ----------------------------------------------------
        # Slip rate
        # ----------------------------------------------------

        gamma_dot = calculate_slip_rate(

            tau,
            resistance_new

        )


        # ----------------------------------------------------
        # Flow residual
        # ----------------------------------------------------

        flow_residual = (

            dg
            -
            dt * gamma_dot

        )


        # ----------------------------------------------------
        # Stress-free lateral/shear conditions
        # ----------------------------------------------------

        stress_residual = np.array([

            stress[0, 0],
            stress[1, 1],
            stress[0, 1],
            stress[1, 2],
            stress[0, 2]

        ])


        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        stress_scale = E

        slip_scale = max(
            gamma_dot_0 * dt,
            1e-8
        )


        return np.concatenate([

            stress_residual
            /
            stress_scale,

            flow_residual
            /
            slip_scale

        ])


    # ========================================================
    # SOLVE LOCAL CP PROBLEM
    # ========================================================

    solution = least_squares(

        local_residual,

        x0,

        max_nfev=2000,

        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11

    )


    if not solution.success:

        raise RuntimeError(

            "Local CP solver failed at "
            f"step {step+1}: "
            +
            solution.message

        )


    # --------------------------------------------------------
    # Save complete solution
    # --------------------------------------------------------

    x_previous = solution.x.copy()


    # --------------------------------------------------------
    # Extract slip increments
    # --------------------------------------------------------

    dg = solution.x[5:17]


    # --------------------------------------------------------
    # Reconstruct total strain
    # --------------------------------------------------------

    total_eps = np.array([

        [
            solution.x[0],
            solution.x[2],
            solution.x[4]
        ],

        [
            solution.x[2],
            solution.x[1],
            solution.x[3]
        ],

        [
            solution.x[4],
            solution.x[3],
            eps_zz
        ]

    ])


    # --------------------------------------------------------
    # Plastic increment
    # --------------------------------------------------------

    plastic_increment = np.zeros(
        (3, 3)
    )

    for a in range(12):

        plastic_increment += (
            dg[a] * P[a]
        )


    # --------------------------------------------------------
    # Update plastic strain
    # --------------------------------------------------------

    plastic_strain += (
        plastic_increment
    )


    # --------------------------------------------------------
    # Update accumulated slip
    # --------------------------------------------------------

    accumulated_slip += np.abs(dg)


    # --------------------------------------------------------
    # Update CRSS
    # --------------------------------------------------------

    total_slip_increment = np.sum(
        np.abs(dg)
    )


    slip_resistance += (

        h0
        *
        (
            1.0
            -
            slip_resistance
            /
            tau_sat
        )
        *
        total_slip_increment

    )


    slip_resistance = np.maximum(
        slip_resistance,
        tau0
    )


    # --------------------------------------------------------
    # Calculate final stress
    # --------------------------------------------------------

    elastic_eps = (
        total_eps
        -
        plastic_strain
    )


    stress = voigt_to_tensor(

        C
        @
        tensor_to_voigt(
            elastic_eps
        )

    )


    tau = calculate_tau(
        stress
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    strain_history.append(
        eps_zz
    )

    stress_history.append(
        stress[2, 2]
    )

    plastic_history.append(
        plastic_strain[2, 2]
    )

    slip_history.append(
        np.sum(
            accumulated_slip
        )
    )

    crss_history.append(
        np.mean(
            slip_resistance
        )
    )

    rss_history.append(
        np.max(
            np.abs(tau)
        )
    )

    residual_history.append(
        np.linalg.norm(
            solution.fun
        )
    )


# ============================================================
# 16. CONVERT ARRAYS
# ============================================================

strain_history = np.array(
    strain_history
)

stress_history = np.array(
    stress_history
)

plastic_history = np.array(
    plastic_history
)

slip_history = np.array(
    slip_history
)

crss_history = np.array(
    crss_history
)

rss_history = np.array(
    rss_history
)

residual_history = np.array(
    residual_history
)


# ============================================================
# 17. FINAL RESULT
# ============================================================

print("\n==============================")
print("CORRECTED FE–CP RESULTS")
print("==============================")

print(
    "Converged increments:",
    len(strain_history),
    "/",
    n_steps
)

print(
    "Final strain:",
    strain_history[-1]
)

print(
    "Final axial stress:",
    stress_history[-1],
    "MPa"
)

print(
    "Final plastic strain:",
    plastic_history[-1]
)

print(
    "Final accumulated slip:",
    slip_history[-1]
)

print(
    "Final mean CRSS:",
    crss_history[-1],
    "MPa"
)

print(
    "Maximum RSS:",
    np.max(rss_history),
    "MPa"
)

print(
    "Maximum residual:",
    np.max(residual_history)
)


# ============================================================
# 18. FINAL STRESS STATE
# ============================================================

print("\n==============================")
print("FINAL STRESS STATE")
print("==============================")

print(
    "sigma_xx =",
    stress[0, 0],
    "MPa"
)

print(
    "sigma_yy =",
    stress[1, 1],
    "MPa"
)

print(
    "sigma_zz =",
    stress[2, 2],
    "MPa"
)

print(
    "sigma_xy =",
    stress[0, 1],
    "MPa"
)

print(
    "sigma_xz =",
    stress[0, 2],
    "MPa"
)

print(
    "sigma_yz =",
    stress[1, 2],
    "MPa"
)


# ============================================================
# 19. COMPARE AGAINST VERIFIED RESULT
# ============================================================

reference_stress = 395.90632217011205

difference = (
    stress_history[-1]
    -
    reference_stress
)

error = (
    abs(difference)
    /
    reference_stress
    *
    100
)

print("\n==============================")
print("VERIFICATION")
print("==============================")

print(
    "Reference grain-0 CP stress =",
    reference_stress,
    "MPa"
)

print(
    "FE–CP stress =",
    stress_history[-1],
    "MPa"
)

print(
    "Difference =",
    difference,
    "MPa"
)

print(
    "Relative difference =",
    error,
    "%"
)


# ============================================================
# 20. STRESS-STRAIN CURVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(

    strain_history * 100,

    stress_history,

    linewidth=2,

    label="Corrected FE–CP"

)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Axial stress (MPa)",
    fontsize=14
)

plt.title(
    "One-element FE–CPFEM verification",
    fontsize=15
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# 21. CONVERGENCE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.semilogy(

    strain_history * 100,

    residual_history,

    linewidth=2

)

plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)

plt.ylabel(
    "Local CP residual norm",
    fontsize=14
)

plt.title(
    "Local CP convergence",
    fontsize=15
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

# ================================================================
# MODULE 5B-OPTIMIZED
# SELF-CONTAINED 3D FE–CPFEM SOLVER FOR 316L
#
# Version: v02 corrected / optimized
#
# IMPORTANT:
# This module is self-contained.
#
# It DOES NOT require:
#   B_all
#   element_volume
#   crystal_slip_systems
#   slip_systems
#   K_global
#
# It loads the FE mesh and microstructure information directly
# from:
#
#     316L_FE_mesh_20cube.npz
#
# Main optimizations:
#
# 1. Build HEX8 B matrix once
# 2. Use identical B for all regular-mesh elements
# 3. Precompute element DOFs
# 4. Precompute grain-level Schmid tensors
# 5. Map grain Schmid tensors to elements
# 6. Precompute CP matrices
# 7. Analytical 12 x 12 local CP Jacobian
# 8. Local Newton solver instead of least_squares
# 9. Factorize global K_ff only once
# 10. Preallocated trial arrays
# 11. No dictionaries in the hot element loop
#
# Constitutive model:
#
#   sigma = C : (epsilon - epsilon_p)
#
#   epsilon_p = sum_a gamma_a * S_a
#
#   tau_a = S_a : sigma
#
#   gamma_dot_a =
#       GAMMA0 * sign(tau_a)
#       * (abs(tau_a)/g_a)^RATE_EXPONENT
#
#   g_a(new) =
#       g_a(old)
#       + H * (1-g_a(old)/TAU_SAT) * abs(dgamma_a)
#
# This follows the constitutive formulation used in the existing
# v02 Module 5B.
# ================================================================


        
import numpy as np
import time

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu




# ================================================================
# RUNTIME PROFILING
# ================================================================

runtime_stats = {
    "cp_total": 0.0,
    "strain_total": 0.0,
    "internal_force_total": 0.0,
    "tangent_assembly_total": 0.0,
    "factorization_total": 0.0,
    "linear_solve_total": 0.0,
    "global_iterations": 0,
    "element_cp_calls": 0,
}







# ================================================================
# 0. USER SETTINGS
# ================================================================

E = 193000.0
NU = 0.29

TAU0 = 80.0
TAU_SAT = 350.0
HARDENING_MODULUS = 3000.0

GAMMA0 = 1.0e-3
RATE_EXPONENT = 20.0
STRAIN_RATE = 1.0e-3



# INITIAL_STRAIN_STEP = 5.0e-4
# MIN_STRAIN_STEP = 5.0e-5
# MAX_STRAIN_STEP = 1.0e-3






MAX_GLOBAL_ITER = 30
GLOBAL_TOL = 1.0e-6
GLOBAL_DAMPING = 1.0

ADAPTIVE_TANGENT_RATIO = 0.30
MAX_TANGENT_AGE = 2
# ============================================================
# GLOBAL NEWTON STEP SAFETY
# ============================================================
GLOBAL_DAMPING_MIN = 0.125
GLOBAL_DAMPING_REDUCTION = 0.5
MIN_DET_F = 0.20



# ============================================================
# MODIFIED NEWTON TANGENT REBUILD FREQUENCY
# ============================================================

TANGENT_REBUILD_FREQUENCY = 1

# ============================================================
# MONOTONIC TENSILE STRAIN CONTROL
# ============================================================

TOTAL_STRAIN = 0.200          # 30.00% total tensile strain

# Adaptive strain increment
INITIAL_STRAIN_STEP = 6.25e-5   # 0.05%
MIN_STRAIN_STEP = 5.0e-7       # 0.005%
MAX_STRAIN_STEP = 1.0e-3       # 0.10%

# ------------------------------------------------
# Local CP Newton parameters
# ------------------------------------------------

CP_MAX_ITER = 30
CP_TOL = 1.0e-7

CP_LINESEARCH_MIN = 1.0e-4

# Maximum allowed local slip increment
CP_DGAMMA_MAX = 1.0

# ------------------------------------------------
# Debug
# ------------------------------------------------

DEBUG_GLOBAL = True
DEBUG_CP = False


# ================================================================
# 1. START
# ================================================================

print()
print("=" * 70)
print("MODULE 5B-OPTIMIZED")
print("SELF-CONTAINED FE–CPFEM SOLVER")
print("=" * 70)


total_start = time.time()


# ================================================================
# 2. LOAD FE MESH
# ================================================================

print()
print("Loading FE mesh...")

data = np.load(
    "316L_FE_mesh_20cube.npz"
)

nodes_um = np.asarray(
    data["nodes"],
    dtype=np.float64
)

elements = np.asarray(
    data["elements"],
    dtype=np.int32
)

element_grain = np.asarray(
    data["element_grain"],
    dtype=np.int32
)

grain_orientation = np.asarray(
    data["grain_orientation"],
    dtype=np.float64
)

dimensions_um = np.asarray(
    data["dimensions"],
    dtype=np.float64
)

# ------------------------------------------------
# Geometry
# ------------------------------------------------

# Convert μm -> mm
nodes = nodes_um / 1000.0

Lx, Ly, Lz = dimensions_um / 1000.0

n_nodes = nodes.shape[0]
n_elements = elements.shape[0]
n_grains = grain_orientation.shape[0]

NDOF = 3 * n_nodes

print(
    "Nodes    =",
    n_nodes
)

print(
    "Elements =",
    n_elements
)

print(
    "DOFs     =",
    NDOF
)

print(
    "Grains   =",
    n_grains
)

print(
    f"Dimensions = "
    f"{Lx:.6f} × {Ly:.6f} × {Lz:.6f} mm"
)


# ================================================================
# 3. CHECK FE MESH
# ================================================================

if elements.ndim != 2 or elements.shape[1] != 8:

    raise RuntimeError(
        "Expected HEX8 element connectivity "
        "with shape (N,8)."
    )


if element_grain.shape[0] != n_elements:

    raise RuntimeError(
        "element_grain size does not match "
        "number of elements."
    )


if np.min(element_grain) < 0:

    raise RuntimeError(
        "Invalid negative grain ID."
    )


if np.max(element_grain) >= n_grains:

    raise RuntimeError(
        "element_grain contains an invalid grain ID."
    )


# ================================================================
# 4. ELASTIC CONSTITUTIVE MATRIX
# ================================================================

lam = (
    E * NU
    /
    ((1.0 + NU) * (1.0 - 2.0 * NU))
)

mu = (
    E
    /
    (2.0 * (1.0 + NU))
)


C_ELASTIC = np.zeros(
    (6, 6),
    dtype=np.float64
)


C_ELASTIC[0, 0] = lam + 2.0 * mu
C_ELASTIC[1, 1] = lam + 2.0 * mu
C_ELASTIC[2, 2] = lam + 2.0 * mu

C_ELASTIC[0, 1] = lam
C_ELASTIC[0, 2] = lam
C_ELASTIC[1, 0] = lam
C_ELASTIC[1, 2] = lam
C_ELASTIC[2, 0] = lam
C_ELASTIC[2, 1] = lam

C_ELASTIC[3, 3] = mu
C_ELASTIC[4, 4] = mu
C_ELASTIC[5, 5] = mu


print()
print("Elastic constants:")
print("E  =", E, "MPa")
print("ν  =", NU)
print("λ  =", lam, "MPa")
print("μ  =", mu, "MPa")


# ================================================================
# 5. VOIGT / TENSOR CONVERSION
#
# Convention:
#
# [xx, yy, zz, xy, yz, xz]
#
# Engineering shear strain:
#
# gamma_xy = 2 epsilon_xy
# gamma_yz = 2 epsilon_yz
# gamma_xz = 2 epsilon_xz
# ================================================================

def tensor_to_voigt(A):

    return np.array(
        [
            A[0, 0],
            A[1, 1],
            A[2, 2],
            2.0 * A[0, 1],
            2.0 * A[1, 2],
            2.0 * A[0, 2]
        ],
        dtype=np.float64
    )


def voigt_to_tensor(v):

    A = np.zeros(
        (3, 3),
        dtype=np.float64
    )

    A[0, 0] = v[0]
    A[1, 1] = v[1]
    A[2, 2] = v[2]

    A[0, 1] = 0.5 * v[3]
    A[1, 0] = 0.5 * v[3]

    A[1, 2] = 0.5 * v[4]
    A[2, 1] = 0.5 * v[4]

    A[0, 2] = 0.5 * v[5]
    A[2, 0] = 0.5 * v[5]

    return A


# ================================================================
# 6. HEX8 SHAPE FUNCTION DERIVATIVES
#
# Same node ordering as v02:
#
# 1 ---- 2
# |      |
# 4 ---- 3
#
# lower surface
#
# and 5-8 on upper surface.
# ================================================================

def shape_function_derivatives(
    xi,
    eta,
    zeta
):

    dN = np.zeros(
        (8, 3),
        dtype=np.float64
    )

    # dN/dxi

    dN[0, 0] = (
        -0.125
        * (1.0 - eta)
        * (1.0 - zeta)
    )

    dN[1, 0] = (
         0.125
        * (1.0 - eta)
        * (1.0 - zeta)
    )

    dN[2, 0] = (
         0.125
        * (1.0 + eta)
        * (1.0 - zeta)
    )

    dN[3, 0] = (
        -0.125
        * (1.0 + eta)
        * (1.0 - zeta)
    )

    dN[4, 0] = (
        -0.125
        * (1.0 - eta)
        * (1.0 + zeta)
    )

    dN[5, 0] = (
         0.125
        * (1.0 - eta)
        * (1.0 + zeta)
    )

    dN[6, 0] = (
         0.125
        * (1.0 + eta)
        * (1.0 + zeta)
    )

    dN[7, 0] = (
        -0.125
        * (1.0 + eta)
        * (1.0 + zeta)
    )


    # dN/deta

    dN[0, 1] = (
        -0.125
        * (1.0 - xi)
        * (1.0 - zeta)
    )

    dN[1, 1] = (
        -0.125
        * (1.0 + xi)
        * (1.0 - zeta)
    )

    dN[2, 1] = (
         0.125
        * (1.0 + xi)
        * (1.0 - zeta)
    )

    dN[3, 1] = (
         0.125
        * (1.0 - xi)
        * (1.0 - zeta)
    )

    dN[4, 1] = (
        -0.125
        * (1.0 - xi)
        * (1.0 + zeta)
    )

    dN[5, 1] = (
        -0.125
        * (1.0 + xi)
        * (1.0 + zeta)
    )

    dN[6, 1] = (
         0.125
        * (1.0 + xi)
        * (1.0 + zeta)
    )

    dN[7, 1] = (
         0.125
        * (1.0 - xi)
        * (1.0 + zeta)
    )


    # dN/dzeta

    dN[0, 2] = (
        -0.125
        * (1.0 - xi)
        * (1.0 - eta)
    )

    dN[1, 2] = (
        -0.125
        * (1.0 + xi)
        * (1.0 - eta)
    )

    dN[2, 2] = (
        -0.125
        * (1.0 + xi)
        * (1.0 + eta)
    )

    dN[3, 2] = (
        -0.125
        * (1.0 - xi)
        * (1.0 + eta)
    )

    dN[4, 2] = (
         0.125
        * (1.0 - xi)
        * (1.0 - eta)
    )

    dN[5, 2] = (
         0.125
        * (1.0 + xi)
        * (1.0 - eta)
    )

    dN[6, 2] = (
         0.125
        * (1.0 + xi)
        * (1.0 + eta)
    )

    dN[7, 2] = (
         0.125
        * (1.0 - xi)
        * (1.0 + eta)
    )

    return dN


# ================================================================
# 7. BUILD HEX8 B MATRIX
#
# For this regular structured mesh:
#
# all elements have identical geometry.
#
# Therefore only one B matrix is needed.
# ================================================================

print()
print("Building HEX8 B matrix...")

t0 = time.time()

element0_nodes = elements[0]

element0_coordinates = (
    nodes[element0_nodes]
)

# Element center
xi = 0.0
eta = 0.0
zeta = 0.0

dN_dxi = shape_function_derivatives(
    xi,
    eta,
    zeta
)

# Jacobian
#
# x = N_a X_a
#
# J_ij = d x_i / d xi_j
#
J = (
    element0_coordinates.T
    @
    dN_dxi
)

detJ = np.linalg.det(J)

if detJ <= 0.0:

    raise RuntimeError(
        f"Invalid HEX8 Jacobian: detJ={detJ}"
    )


dN_dx = (
    dN_dxi
    @
    np.linalg.inv(J)
)


B = np.zeros(
    (6, 24),
    dtype=np.float64
)


for a in range(8):

    i = 3 * a

    dNx = dN_dx[a, 0]
    dNy = dN_dx[a, 1]
    dNz = dN_dx[a, 2]

    B[0, i] = dNx

    B[1, i + 1] = dNy

    B[2, i + 2] = dNz

    B[3, i] = dNy
    B[3, i + 1] = dNx

    B[4, i + 1] = dNz
    B[4, i + 2] = dNy

    B[5, i] = dNz
    B[5, i + 2] = dNx


# ------------------------------------------------
# Element volume
#
# 8 Gauss points
# ------------------------------------------------

g = 1.0 / np.sqrt(3.0)

gauss_points = [
    (-g, -g, -g),
    ( g, -g, -g),
    ( g,  g, -g),
    (-g,  g, -g),

    (-g, -g,  g),
    ( g, -g,  g),
    ( g,  g,  g),
    (-g,  g,  g)
]


element_volume = 0.0


for gp in gauss_points:

    xi_gp, eta_gp, zeta_gp = gp

    dN_gp = shape_function_derivatives(
        xi_gp,
        eta_gp,
        zeta_gp
    )

    J_gp = (
        element0_coordinates.T
        @
        dN_gp
    )

    detJ_gp = np.linalg.det(J_gp)

    if detJ_gp <= 0.0:

        raise RuntimeError(
            "Invalid Jacobian at Gauss point."
        )

    element_volume += detJ_gp


print(
    f"B construction = "
    f"{time.time() - t0:.3f} s"
)

print(
    "B shape =",
    B.shape
)

print(
    "detJ =",
    detJ
)

print(
    "Element volume =",
    element_volume,
    "mm^3"
)


# ================================================================
# 8. PRECOMPUTE ELEMENT DOFS
# ================================================================

print()
print("Precomputing element DOFs...")

t0 = time.time()

conn = elements.astype(np.int64, copy=False)

element_dofs = np.empty(
    (n_elements, 24),
    dtype=np.int32
)

element_dofs[:, 0::3] = 3 * conn
element_dofs[:, 1::3] = 3 * conn + 1
element_dofs[:, 2::3] = 3 * conn + 2

print(
    f"Element DOF construction = "
    f"{time.time() - t0:.3f} s"
)

# ------------------------------------------------
# Precompute global stiffness COO indices
#
# Every HEX8 element contributes a 24 x 24 block.
# The connectivity never changes.
# ------------------------------------------------

print("Precomputing global tangent assembly indices...")

t0 = time.time()

row_idx = np.repeat(
    element_dofs,
    24,
    axis=1
).reshape(-1)

col_idx = np.tile(
    element_dofs,
    (1, 24)
).reshape(-1)

print(
    f"Global tangent index construction = "
    f"{time.time() - t0:.3f} s"
)

print(
    "Tangent entries per assembly =",
    len(row_idx)
)




# ================================================================
# 9. FCC SLIP SYSTEMS
#
# Define the 12 FCC {111}<110> systems directly.
#
# For each system:
#
#   s = unit slip direction
#   n = unit slip-plane normal
#   P = s ⊗ n
#   S = 1/2 (P + P.T)
#
# IMPORTANT:
# Keep crystal_s, crystal_n and crystal_S in exactly the
# same 12-system ordering.
# ================================================================

def build_fcc_slip_systems():

    # ------------------------------------------------------------
    # Four {111} plane normals
    # ------------------------------------------------------------

    planes = [
        np.array([ 1.0,  1.0,  1.0]),
        np.array([ 1.0,  1.0, -1.0]),
        np.array([ 1.0, -1.0,  1.0]),
        np.array([-1.0,  1.0,  1.0])
    ]

    # ------------------------------------------------------------
    # Three <110> directions lying in each plane
    #
    # The ordering here defines the permanent FCC slip-system
    # numbering used throughout the CPFEM calculation.
    # ------------------------------------------------------------

    directions = [
        [
            np.array([ 1.0, -1.0,  0.0]),
            np.array([ 1.0,  0.0, -1.0]),
            np.array([ 0.0,  1.0, -1.0])
        ],

        [
            np.array([ 1.0, -1.0,  0.0]),
            np.array([ 1.0,  0.0,  1.0]),
            np.array([ 0.0,  1.0,  1.0])
        ],

        [
            np.array([ 1.0,  1.0,  0.0]),
            np.array([ 1.0,  0.0, -1.0]),
            np.array([ 0.0,  1.0,  1.0])
        ],

        [
            np.array([ 1.0,  1.0,  0.0]),
            np.array([ 1.0,  0.0,  1.0]),
            np.array([ 0.0,  1.0, -1.0])
        ]
    ]

    crystal_s = []
    crystal_n = []
    crystal_S = []

    # ------------------------------------------------------------
    # Build all 12 systems
    # ------------------------------------------------------------

    for p, dirs in zip(
        planes,
        directions
    ):

        # Unit plane normal
        n = p / np.linalg.norm(p)

        for s in dirs:

            # Unit slip direction
            s = s / np.linalg.norm(s)

            # ----------------------------------------------------
            # Orthogonality check
            # ----------------------------------------------------

            if abs(np.dot(n, s)) > 1.0e-12:

                raise RuntimeError(
                    "FCC slip direction and plane normal "
                    "are not orthogonal."
                )

            # ----------------------------------------------------
            # Store the unsymmetrized plastic-flow dyad
            #
            # P = s ⊗ n
            #
            # This is the tensor required for:
            #
            #   Lp = Σ gamma_dot * (s ⊗ n)
            # ----------------------------------------------------

            P = np.outer(
                s,
                n
            )

            # ----------------------------------------------------
            # Symmetric Schmid tensor
            #
            # S = 1/2 (s⊗n + n⊗s)
            #
            # Used for resolved shear stress / diagnostics and
            # compatibility with the existing small-strain code.
            # ----------------------------------------------------

            S = 0.5 * (
                P + P.T
            )

            crystal_s.append(s)
            crystal_n.append(n)
            crystal_S.append(S)

    return (
        np.asarray(crystal_s, dtype=np.float64),
        np.asarray(crystal_n, dtype=np.float64),
        np.asarray(crystal_S, dtype=np.float64)
    )


(
    crystal_s,
    crystal_n,
    crystal_S
) = build_fcc_slip_systems()


# ------------------------------------------------
# Validate dimensions
# ------------------------------------------------

if crystal_s.shape != (12, 3):

    raise RuntimeError(
        "FCC slip-direction construction failed."
    )


if crystal_n.shape != (12, 3):

    raise RuntimeError(
        "FCC slip-plane construction failed."
    )


if crystal_S.shape != (12, 3, 3):

    raise RuntimeError(
        "FCC Schmid-tensor construction failed."
    )


# ------------------------------------------------
# Validate every slip system
# ------------------------------------------------

for a in range(12):

    # Unit-vector checks
    if not np.isclose(
        np.linalg.norm(crystal_s[a]),
        1.0,
        atol=1.0e-12
    ):

        raise RuntimeError(
            f"Slip direction {a} is not unit length."
        )

    if not np.isclose(
        np.linalg.norm(crystal_n[a]),
        1.0,
        atol=1.0e-12
    ):

        raise RuntimeError(
            f"Slip normal {a} is not unit length."
        )

    # Orthogonality
    if not np.isclose(
        np.dot(
            crystal_s[a],
            crystal_n[a]
        ),
        0.0,
        atol=1.0e-12
    ):

        raise RuntimeError(
            f"Slip system {a} is not orthogonal."
        )

    # Check that crystal_S is exactly the symmetric part
    # of the corresponding slip dyad.
    P = np.outer(
        crystal_s[a],
        crystal_n[a]
    )

    S_check = 0.5 * (
        P + P.T
    )

    if not np.allclose(
        crystal_S[a],
        S_check,
        atol=1.0e-12
    ):

        raise RuntimeError(
            f"Schmid tensor mismatch for slip system {a}."
        )


print()
print(
    "FCC slip systems =",
    crystal_s.shape[0]
)

print(
    "crystal_s shape =",
    crystal_s.shape
)

print(
    "crystal_n shape =",
    crystal_n.shape
)

print(
    "crystal_S shape =",
    crystal_S.shape
)



# ================================================================
# 10. INITIAL GRAIN-LEVEL FCC SLIP SYSTEMS
# ================================================================

print()
print("Building initial grain-level FCC slip systems...")

t0 = time.time()

# ------------------------------------------------
# IMPORTANT:
# crystal_s and crystal_n must correspond to the
# same 12 FCC slip systems and ordering used by
# crystal_S in the original code.
#
# Shapes:
#
# crystal_s : (12, 3)
# crystal_n : (12, 3)
#
# crystal_S[a] is the symmetric Schmid tensor:
#
#     S = 0.5 * (s ⊗ n + n ⊗ s)
#
# The unsymmetric slip-flow tensor used in the
# finite-strain formulation is:
#
#     P = s ⊗ n

grain_s_initial = np.empty(
    (n_grains, 12, 3),
    dtype=np.float64
)

grain_n_initial = np.empty(
    (n_grains, 12, 3),
    dtype=np.float64
)

grain_S_initial = np.empty(
    (n_grains, 12, 3, 3),
    dtype=np.float64
)


for g_id in range(n_grains):

    Q = grain_orientation[g_id]

    for a in range(12):

        # ------------------------------------------------
        # Rotate crystal slip direction and plane normal
        # into the initial sample/material frame
        # ------------------------------------------------

        s0 = Q @ crystal_s[a]
        n0 = Q @ crystal_n[a]

        # Numerical normalization
        s0 /= np.linalg.norm(s0)
        n0 /= np.linalg.norm(n0)

        grain_s_initial[
            g_id,
            a
        ] = s0

        grain_n_initial[
            g_id,
            a
        ] = n0

        # ------------------------------------------------
        # Initial unsymmetrized slip dyad
        # ------------------------------------------------

        P0 = np.outer(
            s0,
            n0
        )

        # Symmetric Schmid tensor retained for
        # diagnostics / small-strain comparison
        S0 = 0.5 * (
            P0 + P0.T
        )

        grain_S_initial[
            g_id,
            a
        ] = S0


print(
    f"Initial slip-system construction = "
    f"{time.time() - t0:.3f} s"
)


# ------------------------------------------------
# Element -> grain
# ------------------------------------------------

element_s_initial = grain_s_initial[
    element_grain
]

element_n_initial = grain_n_initial[
    element_grain
]

element_S_initial = grain_S_initial[
    element_grain
]


print(
    "element_s_initial shape =",
    element_s_initial.shape
)

print(
    "element_n_initial shape =",
    element_n_initial.shape
)

print(
    "element_S_initial shape =",
    element_S_initial.shape
)





    
# ================================================================
# 11. PRECOMPUTE CP MATRICES
#
# For each element/slip system:
#
# T[a,:]  = tensor form of Schmid tensor
# S6[a,:] = engineering-strain Voigt form
#
# Stress:
#
# sigma = C (epsilon - S6^T gamma)
#
# RSS:
#
# tau_a = T_a · sigma
#
# Therefore:
#
# tau = tau_trial - A gamma
#
# where:
#
# A[e,a,b] =
#     T[e,a,:] · C · S6[e,b,:]
#
# ================================================================

print()
print("Precomputing CP matrices...")

t0 = time.time()


# ------------------------------------------------
# Tensor-form Schmid tensors in stress Voigt form
#
# T[a] =
# [Sxx, Syy, Szz, Sxy, Syz, Sxz]
#
# because:
#
# tau = S : sigma
#
# ------------------------------------------------

element_T = np.empty(
    (n_elements, 12, 6),
    dtype=np.float64
)


# ------------------------------------------------
# Engineering-strain Voigt representation
#
# S6[a] =
# [Sxx, Syy, Szz, 2Sxy, 2Syz, 2Sxz]
#
# because:
#
# epsilon_p_voigt =
# sum(dgamma_a * S6_a)
#
# ------------------------------------------------

element_S6 = np.empty(
    (n_elements, 12, 6),
    dtype=np.float64
)


for a in range(12):

    # S = element_S[:, a]

    S = element_S_initial[:, a]

    

    # Stress contraction form
    element_T[:, a, 0] = S[:, 0, 0]
    element_T[:, a, 1] = S[:, 1, 1]
    element_T[:, a, 2] = S[:, 2, 2]
    element_T[:, a, 3] = S[:, 0, 1]
    element_T[:, a, 4] = S[:, 1, 2]
    element_T[:, a, 5] = S[:, 0, 2]

    # Engineering strain form
    element_S6[:, a, 0] = S[:, 0, 0]
    element_S6[:, a, 1] = S[:, 1, 1]
    element_S6[:, a, 2] = S[:, 2, 2]

    element_S6[:, a, 3] = (
        2.0 * S[:, 0, 1]
    )

    element_S6[:, a, 4] = (
        2.0 * S[:, 1, 2]
    )

    element_S6[:, a, 5] = (
        2.0 * S[:, 0, 2]
    )


# ------------------------------------------------
# P[e,b,:] = C · S6[e,b,:]
#
# Shape:
#   C       = (6,6)
#   S6      = (8000,12,6)
#   P       = (8000,12,6)
# ------------------------------------------------

element_P = np.einsum(
    "ij,ebj->ebi",
    C_ELASTIC,
    element_S6,
    optimize=True
)


# ------------------------------------------------
# A[e,a,b] =
#
# T[e,a,:] · P[e,b,:]
#
# Shape:
#
#   T = (8000,12,6)
#   P = (8000,12,6)
#   A = (8000,12,12)
#
# ------------------------------------------------

element_A = np.einsum(
    "eai,ebi->eab",
    element_T,
    element_P,
    optimize=True
)

# ------------------------------------------------
# Precompute M = T @ C
#
# This is independent of:
#   - displacement
#   - strain
#   - slip increment
#   - CRSS
#
# Therefore it never needs to be recalculated
# inside solve_cp_local().
# ------------------------------------------------

element_M = np.einsum(
    "eai,ij->eaj",
    element_T,
    C_ELASTIC,
    optimize=True
)

print(
    "element_M shape =",
    element_M.shape
)









print(
    f"CP matrix construction = "
    f"{time.time() - t0:.3f} s"
)

print(
    "element_T shape =",
    element_T.shape
)

print(
    "element_S6 shape =",
    element_S6.shape
)

print(
    "element_P shape =",
    element_P.shape
)

print(
    "element_A shape =",
    element_A.shape
)


# ------------------------------------------------
# Numerical sanity checks
# ------------------------------------------------

if not np.all(
    np.isfinite(element_A)
):

    raise RuntimeError(
        "element_A contains NaN or Inf."
    )


print(
    "element_A min =",
    np.min(element_A)
)

print(
    "element_A max =",
    np.max(element_A)
)

# ================================================================
# 11A. FINITE-STRAIN DISPLACEMENT-GRADIENT OPERATOR
# ================================================================
#
# G maps element nodal displacements to the full displacement
# gradient:
#
#     grad_u = G @ Ue
#
# Ordering:
#
#     0 = ux,X
#     1 = ux,Y
#     2 = ux,Z
#     3 = uy,X
#     4 = uy,Y
#     5 = uy,Z
#     6 = uz,X
#     7 = uz,Y
#     8 = uz,Z
#
# G shape = (9, 24)
# ================================================================

G = np.zeros(
    (9, 24),
    dtype=np.float64
)

for a in range(8):

    dNx = dN_dx[a, 0]
    dNy = dN_dx[a, 1]
    dNz = dN_dx[a, 2]

    dof = 3 * a

    # ------------------------------------------------
    # ux derivatives
    # ------------------------------------------------

    G[0, dof + 0] = dNx
    G[1, dof + 0] = dNy
    G[2, dof + 0] = dNz

    # ------------------------------------------------
    # uy derivatives
    # ------------------------------------------------

    G[3, dof + 1] = dNx
    G[4, dof + 1] = dNy
    G[5, dof + 1] = dNz

    # ------------------------------------------------
    # uz derivatives
    # ------------------------------------------------

    G[6, dof + 2] = dNx
    G[7, dof + 2] = dNy
    G[8, dof + 2] = dNz


print()
print("Finite-strain displacement-gradient matrix:")
print("G shape =", G.shape)

if G.shape != (9, 24):
    raise RuntimeError(
        "Unexpected G shape."
    )

# ================================================================
# 12. BOUNDARY CONDITIONS
# ================================================================

print()
print("Building boundary conditions...")

tol_geom = 1.0e-12

z_min = np.min(nodes[:, 2])
z_max = np.max(nodes[:, 2])


# ------------------------------------------------
# Identify bottom and top surface nodes
# ------------------------------------------------

bottom_nodes = np.where(
    np.abs(
        nodes[:, 2] - z_min
    )
    <
    tol_geom
)[0]


top_nodes = np.where(
    np.abs(
        nodes[:, 2] - z_max
    )
    <
    tol_geom
)[0]


# ================================================================
# TOP Z DISPLACEMENT
# ================================================================

top_z_dofs = (
    3 * top_nodes + 2
).astype(np.int32)


# ================================================================
# BOTTOM Z FIXED
# ================================================================

bottom_z_dofs = (
    3 * bottom_nodes + 2
).astype(np.int32)


# ================================================================
# REMOVE RIGID-BODY X MOTION
# ================================================================

reference_x_node = bottom_nodes[0]

fix_x_dof = np.int32(
    3 * reference_x_node
)


# ================================================================
# REMOVE RIGID-BODY Y MOTION
# ================================================================

reference_y_node = bottom_nodes[
    np.argmax(
        nodes[
            bottom_nodes,
            0
        ]
    )
]

fix_y_dof = np.int32(
    3 * reference_y_node + 1
)


# ================================================================
# COMPLETE SET OF BOTTOM FIXED DOFs
# ================================================================

bottom_fixed = np.concatenate(
    [
        bottom_z_dofs,
        np.array(
            [
                fix_x_dof,
                fix_y_dof
            ],
            dtype=np.int32
        )
    ]
)


bottom_fixed = np.asarray(
    sorted(
        set(
            bottom_fixed.tolist()
        )
    ),
    dtype=np.int32
)


# ================================================================
# PRESCRIBED DOFs
# ================================================================

prescribed_dofs = np.unique(
    np.concatenate(
        [
            bottom_fixed,
            top_z_dofs
        ]
    )
).astype(np.int32)


# ================================================================
# FREE DOFs
# ================================================================

free_mask = np.ones(
    NDOF,
    dtype=bool
)

free_mask[
    prescribed_dofs
] = False


free_dofs = np.flatnonzero(
    free_mask
).astype(np.int32)


# ================================================================
# REPORT
# ================================================================

print(
    "Bottom nodes  =",
    len(bottom_nodes)
)

print(
    "Top nodes     =",
    len(top_nodes)
)

print(
    "Fixed DOFs    =",
    len(bottom_fixed)
)

print(
    "Prescribed    =",
    len(prescribed_dofs)
)

print(
    "Free DOFs     =",
    len(free_dofs)
)

print(
    "Reference X node =",
    reference_x_node,
    "DOF =",
    fix_x_dof
)

print(
    "Reference Y node =",
    reference_y_node,
    "DOF =",
    fix_y_dof
)




print(
    "Preparing initial elastic FE tangent..."
)
t0 = time.time()


Ke = (
    B.T
    @
    C_ELASTIC
    @
    B
    *
    element_volume
)


K_global = lil_matrix(
    (NDOF, NDOF),
    dtype=np.float64
)


for e in range(n_elements):

    dofs = element_dofs[e]

    K_global[
        np.ix_(
            dofs,
            dofs
        )
    ] += Ke


K_global = K_global.tocsc()


print(
    f"Global stiffness assembly = "
    f"{time.time() - t0:.3f} s"
)


# ================================================================
# 14. REDUCED GLOBAL STIFFNESS
# ================================================================

print()
print("Preparing initial elastic tangent...")

t0 = time.time()

K_ff = (
    K_global
    .tocsr()[free_dofs][:, free_dofs]
    .tocsc()
)

print(f"K_ff shape = {K_ff.shape}")
print(f"K_ff nnz = {K_ff.nnz:,}")
print(
    f"K_ff memory = "
    f"{(K_ff.data.nbytes + K_ff.indices.nbytes + K_ff.indptr.nbytes)/1024**2:.1f} MB"
)

print(
    f"Initial elastic tangent preparation = "
    f"{time.time() - t0:.3f} s"
)




# ================================================================
# 15. FINITE-STRAIN CP STATE ARRAYS
# ================================================================

U = np.zeros(
    NDOF,
    dtype=np.float64
)

Fp_committed = np.tile(
    np.eye(3, dtype=np.float64),
    (n_elements, 1, 1)
)

plastic_strain_committed = np.zeros(
    (n_elements, 3, 3),
    dtype=np.float64
)

resistance_committed = np.full(
    (n_elements, 12),
    TAU0,
    dtype=np.float64
)

accumulated_slip_committed = np.zeros(
    n_elements,
    dtype=np.float64
)

accumulated_slip_system_committed = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)

cp_x_committed = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)

# # ============================================================
# # SECTION 16 — LOCAL CRYSTAL PLASTICITY SOLVER
# # ============================================================


#     """
#     Local constitutive update for one FE element.

#     Inputs
#     ------
#     total_eps       : (3,3)
#         Total strain tensor at the current FE iteration.

#     plastic_old    : (3,3)
#         Committed plastic strain tensor.

#     resistance_old : (12,)
#         Committed CRSS for the 12 slip systems.

#     systems         : (12,3,3)
#         Symmetric Schmid tensors.

#     T               : (12,6)
#         Stress -> resolved shear stress operator.

#     S6              : (12,6)
#         Slip-system strain tensors in engineering-Voigt form.

#     P               : (12,6,6)
#         Precomputed tensor used in consistent tangent.

#     A               : (12,12)
#         Coupling matrix between slip increments and RSS.

#     M               : (12,6)
#         Stress sensitivity of RSS to total strain.

#     x_initial       : (12,)
#         Initial guess for slip increments.

#     dt              : float
#         Crystal-plasticity time increment.



def solve_cp_local_finite(
    F,
    Fp_old,
    resistance_old,
    s_initial,
    n_initial,
    x_initial,
    dt
):

    # ============================================================
    # FINITE-STRAIN LOCAL CPFEM SOLVER
    #
    # Unknowns:
    #     x[a] = Delta-gamma[a]
    #
    # State update:
    #     Fp_new = exp(sum(Delta-gamma[a] P[a])) @ Fp_old
    #
    # Multiplicative decomposition:
    #     F = Fe @ Fp
    #
    # Elasticity:
    #     St. Venant-Kirchhoff
    #
    # Stress returned:
    #     Cauchy stress
    #
    # IMPORTANT:
    #     J_e = det(Fe)
    #     This is NOT the total J = det(F)
    # ============================================================


    # ============================================================
    # BASIC DEFINITIONS
    # ============================================================

    I3 = np.eye(3)

    E = 193000.0
    NU_local = NU

    lam = (
        E * NU_local
        / (
            (1.0 + NU_local)
            * (1.0 - 2.0 * NU_local)
        )
    )

    mu = (
        E
        / (2.0 * (1.0 + NU_local))
    )


    # ============================================================
    # INITIAL NEWTON GUESS
    # ============================================================

    x = np.asarray(
        x_initial,
        dtype=np.float64
    ).copy()


    # ============================================================
    # INITIAL SLIP SYSTEMS
    #
    # P^alpha = s^alpha \otimes n^alpha
    #
    # UNSYMMETRIC slip-flow tensor
    # ============================================================

    P_slip = np.einsum(
        "ai,aj->aij",
        s_initial,
        n_initial
    )


    # ============================================================
    # LOCAL SOLVER PARAMETERS
    # ============================================================

    LOCAL_NEWTON_TOL = CP_TOL

    LOCAL_NEWTON_MAX_ITER = CP_MAX_ITER

    LOCAL_DAMPING = 1.0

    JAC_EPS = 1.0e-8

    NSLIP = 12


    # ============================================================
    # CONSTITUTIVE STATE EVALUATION
    # ============================================================

    def evaluate_finite_state(dgamma):

        # --------------------------------------------------------
        # Plastic deformation increment
        # --------------------------------------------------------

        Lp_increment = np.einsum(
            "a,aij->ij",
            dgamma,
            P_slip
        )

        Fp_increment = scipy.linalg.expm(
            Lp_increment
        )


        # --------------------------------------------------------
        # Updated plastic deformation gradient
        # --------------------------------------------------------

        Fp_new_local = (
            Fp_increment
            @ Fp_old
        )


        # --------------------------------------------------------
        # Elastic deformation gradient
        #
        # F = Fe Fp
        #
        # therefore:
        #
        # Fe = F Fp^{-1}
        # --------------------------------------------------------

        Fp_inv = np.linalg.inv(
            Fp_new_local
        )

        Fe_local = (
            F
            @ Fp_inv
        )


        # --------------------------------------------------------
        # Elastic Jacobian
        # --------------------------------------------------------

        J_e_local = np.linalg.det(
            Fe_local
        )

        if J_e_local <= 0.0:

            raise RuntimeError(
                "Invalid elastic deformation gradient: "
                f"det(Fe) = {J_e_local:.6e}"
            )


        # --------------------------------------------------------
        # Elastic right Cauchy-Green tensor
        # --------------------------------------------------------

        Ce_local = (
            Fe_local.T
            @ Fe_local
        )


        # --------------------------------------------------------
        # Elastic Green-Lagrange strain
        # --------------------------------------------------------

        Ee_local = 0.5 * (
            Ce_local - I3
        )


        # --------------------------------------------------------
        # Second Piola-Kirchhoff stress
        #
        # St. Venant-Kirchhoff elasticity
        # --------------------------------------------------------

        Se_local = (
            lam
            * np.trace(Ee_local)
            * I3
            +
            2.0
            * mu
            * Ee_local
        )


        # --------------------------------------------------------
        # Cauchy stress
        #
        # sigma =
        #     Fe Se Fe^T / J_e
        # --------------------------------------------------------

        sigma_local = (
            Fe_local
            @ Se_local
            @ Fe_local.T
        ) / J_e_local


        # --------------------------------------------------------
        # Numerical symmetrization
        # --------------------------------------------------------

        sigma_local = 0.5 * (
            sigma_local
            + sigma_local.T
        )


        # ========================================================
        # ELASTIC POLAR DECOMPOSITION
        #
        # Fe = Re Ue
        # ========================================================

        U_svd, singular_values, Vt = np.linalg.svd(
            Fe_local
        )

        Re_local = (
            U_svd
            @ Vt
        )


        # --------------------------------------------------------
        # Enforce proper rotation
        # --------------------------------------------------------

        if np.linalg.det(Re_local) < 0.0:

            U_svd[:, -1] *= -1.0

            Re_local = (
                U_svd
                @ Vt
            )


        # ========================================================
        # CURRENT SLIP SYSTEMS
        # ========================================================

        s_current_local = (
            Re_local
            @ s_initial.T
        ).T

        n_current_local = (
            Re_local
            @ n_initial.T
        ).T


        # --------------------------------------------------------
        # Normalize
        # --------------------------------------------------------

        s_norm = np.linalg.norm(
            s_current_local,
            axis=1
        )

        n_norm = np.linalg.norm(
            n_current_local,
            axis=1
        )

        s_current_local /= (
            s_norm[:, None]
        )

        n_current_local /= (
            n_norm[:, None]
        )


        # ========================================================
        # CURRENT SLIP FLOW TENSORS
        # ========================================================

        P_current_local = np.einsum(
            "ai,aj->aij",
            s_current_local,
            n_current_local
        )


        # ========================================================
        # RESOLVED SHEAR STRESS
        #
        # tau^alpha =
        #     sigma : (s^alpha \otimes n^alpha)
        # ========================================================

        rss_local = np.einsum(
            "aij,ij->a",
            P_current_local,
            sigma_local
        )


        return (
            Fp_new_local,
            Fe_local,
            sigma_local,
            rss_local,
            Re_local,
            J_e_local,
            s_current_local,
            n_current_local,
            P_current_local
        )


    # ============================================================
    # HARDENING FUNCTION
    # ============================================================

    def calculate_resistance(
        dgamma,
        resistance_old_local
    ):

        slip_increment = np.abs(
            dgamma
        )

        total_slip_increment = np.sum(
            slip_increment
        )


        # --------------------------------------------------------
        # Saturating isotropic hardening
        #
        # dg/dGamma =
        #
        #     H (1 - g/g_sat)
        # --------------------------------------------------------

        resistance_new_local = (
            resistance_old_local
            +
            HARDENING_MODULUS
            * total_slip_increment
            *
            (
                1.0
                -
                resistance_old_local / TAU_SAT
            )
        )


        resistance_new_local = np.clip(
            resistance_new_local,
            TAU0,
            TAU_SAT
        )


        return resistance_new_local


    # ============================================================
    # SLIP-RATE RESIDUAL
    # ============================================================

    def calculate_flow_residual(
        dgamma,
        rss_local,
        resistance_local
    ):

        resistance_safe = np.maximum(
            resistance_local,
            1.0e-12
        )


        # --------------------------------------------------------
        # |tau| / g
        # --------------------------------------------------------

        tau_ratio = (
            np.abs(rss_local)
            / resistance_safe
        )


        # --------------------------------------------------------
        # Power-law slip rate
        # --------------------------------------------------------

        # gamma_dot = (
        #     GAMMA0
        #     * np.sign(rss_local)
        #     *
        #     tau_ratio ** RATE_EXPONENT
        # )
        
        # --------------------------------------------------------
        # Power-law slip rate
        #
        # Evaluate tau_ratio^RATE_EXPONENT safely in log space.
        # This prevents floating-point overflow during Newton
        # trial evaluations.
        # --------------------------------------------------------
        
        tau_ratio_safe = np.maximum(
            tau_ratio,
            1.0e-300
        )
        
        log_slip_rate = (
            np.log(GAMMA0)
            +
            RATE_EXPONENT * np.log(tau_ratio_safe)
        )
        
        MAX_LOG = np.log(np.finfo(np.float64).max)
        
        log_slip_rate = np.minimum(
            log_slip_rate,
            MAX_LOG
        )
        
        gamma_dot = (
            np.sign(rss_local)
            *
            np.exp(log_slip_rate)
        )


        

        # --------------------------------------------------------
        # Implicit slip increment residual
        #
        # R =
        #     Delta-gamma
        #     -
        #     dt * gamma_dot
        # --------------------------------------------------------

        residual_local = (
            dgamma
            -
            dt * gamma_dot
        )


        return residual_local


    # ============================================================
    # INITIAL VALUES
    # ============================================================

    cp_residual_norm = np.inf

    cp_converged = False

    cp_iterations = 0

    # ============================================================
    # BROYDEN INITIALIZATION
    # ============================================================
    
    x_previous = None
    residual_previous = None
    J_local = None





    # ============================================================
    # LOCAL NEWTON ITERATION
    # ============================================================

    for local_iter in range(
        1,
        LOCAL_NEWTON_MAX_ITER + 1
    ):

        cp_iterations = local_iter


        # ========================================================
        # 1. CURRENT CONSTITUTIVE STATE
        # ========================================================

        (
            Fp_trial,
            Fe_trial,
            sigma_trial,
            rss_trial,
            Re_trial,
            J_e_trial,
            s_current_trial,
            n_current_trial,
            P_current_trial
        ) = evaluate_finite_state(
            x
        )


        # ========================================================
        # 2. CURRENT RESISTANCE
        # ========================================================

        resistance_trial = calculate_resistance(
            x,
            resistance_old
        )


        # ========================================================
        # 3. CURRENT FLOW RESIDUAL
        # ========================================================

        residual = calculate_flow_residual(
            x,
            rss_trial,
            resistance_trial
        )


        cp_residual_norm = np.linalg.norm(
            residual
        )


        # ========================================================
        # 4. CONVERGENCE CHECK
        # ========================================================

        if cp_residual_norm < LOCAL_NEWTON_TOL:

            cp_converged = True

            break


        # # ========================================================
        # # 5. NUMERICAL LOCAL JACOBIAN
        # # ========================================================

        # J_local = np.zeros(
        #     (NSLIP, NSLIP),
        #     dtype=np.float64
        # )


        # for beta in range(NSLIP):

        #     # x_perturbed = x.copy()


        #     # perturbation = (
        #     #     JAC_EPS
        #     #     *
        #     #     max(
        #     #         1.0,
        #     #         abs(x[beta])
        #     #     )
        #     # )


        #     # x_perturbed[beta] += (
        #     #     perturbation
        #     # )

        #     x_perturbed = x.copy()

        #     perturbation = (
        #         JAC_EPS
        #         *
        #         max(
        #             1.0,
        #             abs(x[beta])
        #         )
        #     )

        #     x_perturbed[beta] += (
        #         perturbation
        #     )

        #     # ----------------------------------------------------
        #     # Keep finite-difference trial state inside the same
        #     # numerical bounds used by the local Newton state.
        #     # ----------------------------------------------------

        #     x_perturbed = np.clip(
        #         x_perturbed,
        #         -1.0,
        #         1.0
        #     )


            

        #     # ----------------------------------------------------
        #     # Perturbed constitutive state
        #     # ----------------------------------------------------

        #     (
        #         Fp_p,
        #         Fe_p,
        #         sigma_p,
        #         rss_p,
        #         Re_p,
        #         J_e_p,
        #         s_current_p,
        #         n_current_p,
        #         P_current_p
        #     ) = evaluate_finite_state(
        #         x_perturbed
        #     )


        #     # ----------------------------------------------------
        #     # Perturbed resistance
        #     # ----------------------------------------------------

        #     resistance_p = calculate_resistance(
        #         x_perturbed,
        #         resistance_old
        #     )


        #     # ----------------------------------------------------
        #     # Perturbed residual
        #     # ----------------------------------------------------

        #     residual_p = calculate_flow_residual(
        #         x_perturbed,
        #         rss_p,
        #         resistance_p
        #     )


        #     # ----------------------------------------------------
        #     # Numerical derivative
        #     # ----------------------------------------------------

        #     J_local[:, beta] = (
        #         residual_p
        #         -
        #         residual
        #     ) / perturbation


        # # ========================================================
        # # 6. NEWTON CORRECTION
        # # ========================================================

        # try:

        #     delta_x = np.linalg.solve(
        #         J_local,
        #         -residual
        #     )

        # except np.linalg.LinAlgError:

        #     delta_x = np.linalg.lstsq(
        #         J_local,
        #         -residual,
        #         rcond=None
        #     )[0]


        # # ========================================================
        # # 7. DAMPED UPDATE
        # # ========================================================

        # x += (
        #     LOCAL_DAMPING
        #     * delta_x
        # )


        # # --------------------------------------------------------
        # # Development safeguard
        # # --------------------------------------------------------

        # x = np.clip(
        #     x,
        #     -1.0,
        #     1.0
        # )

        # ========================================================
        # 5. LOCAL JACOBIAN
        #
        # First local iteration:
        #     full finite-difference Jacobian
        #
        # Later iterations:
        #     Broyden rank-one update
        #
        # This avoids 12 expensive constitutive evaluations
        # at every local Newton iteration.
        # ========================================================

        if local_iter == 1:

            J_local = np.zeros(
                (NSLIP, NSLIP),
                dtype=np.float64
            )

            for beta in range(NSLIP):

                x_perturbed = x.copy()

                perturbation = (
                    JAC_EPS
                    * max(
                        1.0,
                        abs(x[beta])
                    )
                )

                x_perturbed[beta] += (
                    perturbation
                )

                x_perturbed = np.clip(
                    x_perturbed,
                    -1.0,
                    1.0
                )

                # ------------------------------------------------
                # Perturbed constitutive state
                # ------------------------------------------------

                (
                    Fp_p,
                    Fe_p,
                    sigma_p,
                    rss_p,
                    Re_p,
                    J_e_p,
                    s_current_p,
                    n_current_p,
                    P_current_p
                ) = evaluate_finite_state(
                    x_perturbed
                )

                # ------------------------------------------------
                # Perturbed resistance
                # ------------------------------------------------

                resistance_p = calculate_resistance(
                    x_perturbed,
                    resistance_old
                )

                # ------------------------------------------------
                # Perturbed residual
                # ------------------------------------------------

                residual_p = calculate_flow_residual(
                    x_perturbed,
                    rss_p,
                    resistance_p
                )

                # ------------------------------------------------
                # Numerical derivative
                # ------------------------------------------------

                J_local[:, beta] = (
                    residual_p
                    - residual
                ) / perturbation

        else:

            # ----------------------------------------------------
            # Broyden update
            #
            # Previous state:
            #     x_previous
            #     residual_previous
            #
            # Current state:
            #     x
            #     residual
            #
            # J_new = J_old +
            #   [(ΔR - J_old Δx) ⊗ Δx] / (Δx · Δx)
            # ----------------------------------------------------

            delta_x_secant = (
                x
                - x_previous
            )

            delta_r_secant = (
                residual
                - residual_previous
            )

            denominator = np.dot(
                delta_x_secant,
                delta_x_secant
            )

            if denominator > 1.0e-30:

                J_local += np.outer(
                    (
                        delta_r_secant
                        - J_local
                        @ delta_x_secant
                    ),
                    delta_x_secant
                ) / denominator

            else:

                # ------------------------------------------------
                # Extremely small step.
                # Keep previous Jacobian.
                # ------------------------------------------------
                pass


        # ========================================================
        # 6. NEWTON CORRECTION
        # ========================================================

        try:

            delta_x = np.linalg.solve(
                J_local,
                -residual
            )

        except np.linalg.LinAlgError:

            # ----------------------------------------------------
            # If Broyden Jacobian becomes singular, rebuild it
            # numerically once.
            #
            # This safeguard preserves robustness.
            # ----------------------------------------------------

            J_local = np.zeros(
                (NSLIP, NSLIP),
                dtype=np.float64
            )

            for beta in range(NSLIP):

                x_perturbed = x.copy()

                perturbation = (
                    JAC_EPS
                    * max(
                        1.0,
                        abs(x[beta])
                    )
                )

                x_perturbed[beta] += (
                    perturbation
                )

                x_perturbed = np.clip(
                    x_perturbed,
                    -1.0,
                    1.0
                )

                (
                    Fp_p,
                    Fe_p,
                    sigma_p,
                    rss_p,
                    Re_p,
                    J_e_p,
                    s_current_p,
                    n_current_p,
                    P_current_p
                ) = evaluate_finite_state(
                    x_perturbed
                )

                resistance_p = calculate_resistance(
                    x_perturbed,
                    resistance_old
                )

                residual_p = calculate_flow_residual(
                    x_perturbed,
                    rss_p,
                    resistance_p
                )

                J_local[:, beta] = (
                    residual_p
                    - residual
                ) / perturbation

            try:

                delta_x = np.linalg.solve(
                    J_local,
                    -residual
                )

            except np.linalg.LinAlgError:

                delta_x = np.linalg.lstsq(
                    J_local,
                    -residual,
                    rcond=None
                )[0]


        # ========================================================
        # 7. SAVE CURRENT STATE FOR NEXT BROYDEN UPDATE
        # ========================================================

        x_previous = x.copy()

        residual_previous = residual.copy()


        # ========================================================
        # 8. DAMPED UPDATE
        # ========================================================

        x += (
            LOCAL_DAMPING
            * delta_x
        )

        x = np.clip(
            x,
            -1.0,
            1.0
        )





    # ============================================================
    # FINAL CONSTITUTIVE EVALUATION
    # ============================================================

    (
        Fp_new,
        Fe,
        sigma,
        rss,
        Re,
        J_e,
        s_current,
        n_current,
        P_current
    ) = evaluate_finite_state(
        x
    )


    # ============================================================
    # FINAL HARDENING
    # ============================================================

    resistance_new = calculate_resistance(
        x,
        resistance_old
    )


    # ============================================================
    # FINAL RESIDUAL
    # ============================================================

    residual = calculate_flow_residual(
        x,
        rss,
        resistance_new
    )


    cp_residual_norm = np.linalg.norm(
        residual
    )


    # ============================================================
    # FINAL CONVERGENCE CHECK
    # ============================================================

    if cp_residual_norm < LOCAL_NEWTON_TOL:

        cp_converged = True

    else:

        cp_converged = False


    # ============================================================
    # STRESS VOIGT VECTOR
    #
    # Ordering:
    #
    # [Sxx, Syy, Szz, Sxy, Syz, Sxz]
    # ============================================================

    stress_voigt = np.array(
        [
            sigma[0, 0],
            sigma[1, 1],
            sigma[2, 2],
            sigma[0, 1],
            sigma[1, 2],
            sigma[0, 2]
        ],
        dtype=np.float64
    )


    # ============================================================
    # TEMPORARY ALGORITHMIC TANGENT
    #
    # DO NOT use this zero matrix for the global Newton solve.
    # ============================================================

    C_alg = np.zeros(
        (6, 6),
        dtype=np.float64
    )


    # ============================================================
    # RETURN
    # ============================================================

    return (
        x,                    # Delta-gamma
        Fp_new,               # updated Fp
        Fe,                   # elastic F
        sigma,                # Cauchy stress
        stress_voigt,         # stress vector
        resistance_new,       # CRSS
        rss,                  # RSS
        Re,                   # lattice rotation
        J_e,                  # det(Fe)
        cp_residual_norm,
        cp_iterations,
        cp_converged,
        C_alg
    )





# ================================================================
# 17. LOCAL DEFORMATION / FATIGUE DIAGNOSTICS
# ================================================================

# ------------------------------------------------
# Local total strain
#
# Voigt order:
# [xx, yy, zz, xy, yz, xz]
#
# Engineering shear convention
# ------------------------------------------------

trial_total_strain = np.zeros(
    (n_elements, 6),
    dtype=np.float64
)


# ------------------------------------------------
# J2 equivalent total strain
# ------------------------------------------------

trial_equiv_total_strain = np.zeros(
    n_elements,
    dtype=np.float64
)


# ------------------------------------------------
# J2 equivalent plastic strain
# ------------------------------------------------

trial_equiv_plastic_strain = np.zeros(
    n_elements,
    dtype=np.float64
)


# ------------------------------------------------
# Von Mises equivalent stress
# ------------------------------------------------

trial_equiv_stress = np.zeros(
    n_elements,
    dtype=np.float64
)


# ------------------------------------------------
# Maximum absolute RSS among 12 slip systems
# ------------------------------------------------

trial_max_abs_rss = np.zeros(
    n_elements,
    dtype=np.float64
)


# ------------------------------------------------
# Slip-system index giving maximum |RSS|
# ------------------------------------------------

trial_max_rss_system = np.zeros(
    n_elements,
    dtype=np.int32
)


# ================================================================
# ACCUMULATED SLIP PER SLIP SYSTEM
# ================================================================
#
# trial_slip[e,a]
#     = slip increment Δgamma during current load step
#
# accumulated_slip_system_committed[e,a]
#     = cumulative sum of |Δgamma| for slip system a
#
# This is different from accumulated_slip_committed[e],
# which is the scalar sum over all 12 systems.
# ================================================================

# accumulated_slip_system_committed = np.zeros(
#     (n_elements, 12),
#     dtype=np.float64
# )

trial_accumulated_slip_system = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)


# ================================================================
# 18. GLOBAL HISTORY
# ================================================================

# current_strain = 0.0

# strain_step = INITIAL_STRAIN_STEP

# step = 0

# global_strain_history = []

# global_stress_history = []

# step_history = []

# cp_state_history = []


# ================================================================
# 18. GLOBAL HISTORY — CYCLIC LOADING
# ================================================================

current_strain = 0.0

strain_step = INITIAL_STRAIN_STEP

step = 0


# ------------------------------------------------
# Monotonic tensile loading
# ------------------------------------------------

# current_strain = 0.0
# strain_step = INITIAL_STRAIN_STEP
# step = 0

global_strain_history = []
global_stress_history = []
step_history = []

cp_state_history = []

# Monotonic loading:
#
# 0.00% -> 2.00%
#
target_strain = TOTAL_STRAIN


# ================================================================
# FINITE-STRAIN LOCAL NUMERICAL TANGENT
# ================================================================
#
# Computes dP/dF while HOLDING Fp fixed.
#
# This is a finite-strain elastic/quasi-Newton tangent.
# It is NOT yet the fully consistent CPFEM tangent because
# the derivative of Fp with respect to F is not included.it is finite-strain frozen-Fp quasi-Newton tangent

# ================================================================

def calculate_dP_dF_frozen_Fp(
    F,
    Fp
    # element_s_initial,
    # element_n_initial
):

    I3 = np.eye(3)

    # ------------------------------------------------------------
    # Material constants
    # ------------------------------------------------------------

    E_mod = E
    nu_mod = NU

    mu = (
        E_mod
        /
        (2.0 * (1.0 + nu_mod))
    )

    lam = (
        E_mod * nu_mod
        /
        (
            (1.0 + nu_mod)
            *
            (1.0 - 2.0 * nu_mod)
        )
    )

    # ------------------------------------------------------------
    # Calculate PK1 for a given F while Fp is frozen
    # ------------------------------------------------------------

    def calculate_PK1(F_local):

        Fp_inv = np.linalg.inv(Fp)

        Fe = (
            F_local
            @ Fp_inv
        )

        J_e = np.linalg.det(
            Fe
        )

        if J_e <= 0.0:

            raise RuntimeError(
                "Non-positive elastic Jacobian "
                "while calculating dP/dF."
            )

        # --------------------------------------------------------
        # Green-Lagrange elastic strain
        # --------------------------------------------------------

        C_e = (
            Fe.T
            @ Fe
        )

        E_e = (
            0.5
            * (C_e - I3)
        )

        # --------------------------------------------------------
        # Second Piola-Kirchhoff stress
        # --------------------------------------------------------

        S_e = (
            lam
            * np.trace(E_e)
            * I3
            +
            2.0 * mu * E_e
        )

        # --------------------------------------------------------
        # Cauchy stress
        # --------------------------------------------------------

        sigma = (
            Fe
            @ S_e
            @ Fe.T
            / J_e
        )

        sigma = 0.5 * (
            sigma + sigma.T
        )

        # --------------------------------------------------------
        # First Piola-Kirchhoff stress
        #
        # P = J * sigma * F^-T
        # --------------------------------------------------------

        J_total = np.linalg.det(
            F_local
        )

        F_inv_T = np.linalg.inv(
            F_local
        ).T

        P = (
            J_total
            * (
                sigma
                @ F_inv_T
            )
        )

        return P

    # ============================================================
    # Numerical dP/dF
    # ============================================================

    # P0 = calculate_PK1(F)

    A_F = np.zeros(
        (9, 9),
        dtype=np.float64
    )

    F_flat = F.reshape(9)

    for beta in range(9):

        h = (
            1.0e-7
            *
            max(
                1.0,
                abs(F_flat[beta])
            )
        )

        F_plus = F_flat.copy()
        F_minus = F_flat.copy()

        F_plus[beta] += h
        F_minus[beta] -= h

        P_plus = calculate_PK1(
            F_plus.reshape(3, 3)
        )

        P_minus = calculate_PK1(
            F_minus.reshape(3, 3)
        )

        dP = (
            P_plus
            - P_minus
        ) / (
            2.0 * h
        )

        # --------------------------------------------------------
        # Flatten P in the SAME ordering as F
        # --------------------------------------------------------

        A_F[:, beta] = (
            dP.reshape(9)
        )

    return A_F



# ================================================================
# LOCAL FIELD HISTORY
# ================================================================
#
# Only CONVERGED load steps are stored.
#
# This is important because Newton iterations represent trial
# states and should not be interpreted as physical load history.
# ================================================================

local_field_history = []

local_field_step_history = []

local_field_strain_history = []


# ================================================================
# 18B. TRIAL FE–CPFEM STATE ARRAYS
# ================================================================

print()
print("Allocating trial FE–CPFEM state arrays...")


# trial_stress = np.zeros(
#     (n_elements, 3, 3),
#     dtype=np.float64
# )


# trial_stress_voigt = np.zeros(
#     (n_elements, 6),
#     dtype=np.float64
# )


# trial_C_alg = np.zeros(
#     (n_elements, 6, 6),
#     dtype=np.float64
# )


# trial_plastic = np.zeros(
#     (n_elements, 3, 3),
#     dtype=np.float64
# )


# trial_resistance = np.zeros(
#     (n_elements, 12),
#     dtype=np.float64
# )

trial_stress = np.zeros(
    (n_elements, 3, 3)
)

trial_stress_voigt = np.zeros(
    (n_elements, 6)
)

trial_C_alg = np.zeros(
    (n_elements, 6, 6)
)
# ============================================================
# FINITE-STRAIN TRIAL ARRAYS
# ============================================================

trial_F = np.zeros(
    (n_elements, 3, 3)
)

trial_Fp = np.zeros(
    (n_elements, 3, 3)
)

trial_Fe = np.zeros(
    (n_elements, 3, 3)
)

trial_J_total = np.ones(
    n_elements
)

trial_J_e = np.ones(
    n_elements
)

trial_PK1 = np.zeros(
    (n_elements, 3, 3)
)

# ============================================================
# FINITE-STRAIN TRIAL STATE
# ============================================================

trial_lattice_rotation = np.tile(
    np.eye(3),
    (n_elements, 1, 1)
)

# ============================================================
# DIAGNOSTIC / CP ARRAYS
# ============================================================

trial_plastic = np.zeros(
    (n_elements, 3, 3)
)

trial_resistance = np.zeros(
    (n_elements, 12)
)






trial_accumulated = np.zeros(
    n_elements,
    dtype=np.float64
)


trial_accumulated_slip_system = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)

trial_slip = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)


trial_rss = np.zeros(
    (n_elements, 12),
    dtype=np.float64
)


trial_cp_residual = np.zeros(
    n_elements,
    dtype=np.float64
)


trial_cp_iterations = np.zeros(
    n_elements,
    dtype=np.int32
)


trial_cp_converged = np.zeros(
    n_elements,
    dtype=bool
)


print(
    "Trial state allocation complete."
)

# ================================================================
# GLOBAL INTERNAL FORCE VECTOR
# ================================================================

F_internal = np.zeros(
    NDOF,
    dtype=np.float64
)

print(
    f"Global internal force vector allocated: "
    f"shape = {F_internal.shape}"
)


MAX_GLOBAL_ITER = 30

# ================================================================
# SECTION 19 — GLOBAL FE–CPFEM SOLVER
# MONOTONIC TENSILE STRAIN CONTROL
# ================================================================
K_solver_alg = None
while current_strain < TOTAL_STRAIN - 1.0e-14:

    # ------------------------------------------------------------
    # Target strain for this load step
    # ------------------------------------------------------------

    target_strain = min(
        current_strain + strain_step,
        TOTAL_STRAIN
    )

    actual_increment = (
        target_strain - current_strain
    )

    # ------------------------------------------------------------
    # CP time increment
    # ------------------------------------------------------------

    cp_dt = (
        actual_increment
        / STRAIN_RATE
    )

    step += 1

    # ------------------------------------------------------------
    # New physical load increment
    # Never reuse LU factorization from previous load increment
    # ------------------------------------------------------------
    
    if K_solver_alg is not None:
    
        del K_solver_alg
    
        K_solver_alg = None
    
        gc.collect()
    

    

    # --------------------------------------------------------
    # Backup committed state
    #
    # These states are restored if the global step fails.
    # --------------------------------------------------------

    U_backup = U.copy()

    # plastic_backup = (
    #     plastic_strain_committed.copy()
    # )
    Fp_backup = (
        Fp_committed.copy()
    )
    resistance_backup = (
        resistance_committed.copy()
    )

    accumulated_backup = (
        accumulated_slip_committed.copy()
    )

    accumulated_slip_system_backup = (
        accumulated_slip_system_committed.copy()
    )

    cp_x_backup = (
        cp_x_committed.copy()
    )

    # --------------------------------------------------------
    # Trial local CPFEM Newton guess
    #
    # This is updated after every global Newton iteration.
    # It is ONLY an initial guess; committed state remains
    # cp_x_backup / Fp_backup / resistance_backup.
    # --------------------------------------------------------
    cp_x_trial_guess = (
        cp_x_backup.copy()
    )





    # --------------------------------------------------------
    # Initial guess for global Newton iteration
    # --------------------------------------------------------

    U_iter = U.copy()

    # --------------------------------------------------------
    # Apply prescribed displacement to top surface
    # --------------------------------------------------------

    U_iter[top_z_dofs] = (
        target_strain * Lz
    )

    # --------------------------------------------------------
    # Reapply fixed boundary conditions
    # --------------------------------------------------------

    U_iter[bottom_z_dofs] = 0.0

    # One x and one y DOF fixed to remove rigid-body motion
    U_iter[fix_x_dof] = 0.0
    U_iter[fix_y_dof] = 0.0

    # global_converged = False

   

    global_converged = False

    # ------------------------------------------------------------
    # Adaptive tangent history
    # ------------------------------------------------------------
    
    previous_residual_norm = None




    # # ========================================================
    # # GLOBAL NEWTON ITERATION
    # # ========================================================

    # for global_iter in range(
    #     1,
    #     MAX_GLOBAL_ITER + 1
    # ):

    # --------------------------------------------------------
    # Reference residual for this physical load increment
    #
    # This must be reset ONCE per load step,
    # before entering the Newton iterations.
    # --------------------------------------------------------
    
    residual_reference = None
    
    # ========================================================
    # GLOBAL NEWTON ITERATION
    # ========================================================
    
    for global_iter in range(
        1,
        MAX_GLOBAL_ITER + 1
    ):

        runtime_stats["global_iterations"] += 1

        global_iter_start = time.time()

        # ----------------------------------------------------
        # Reset trial arrays
        # ----------------------------------------------------




        trial_cp_converged.fill(False)

        trial_stress.fill(0.0)
        trial_stress_voigt.fill(0.0)
        trial_C_alg.fill(0.0)
        trial_plastic.fill(0.0)
        trial_resistance.fill(0.0)
        trial_accumulated.fill(0.0)
        trial_accumulated_slip_system.fill(0.0)
        trial_slip.fill(0.0)
        trial_rss.fill(0.0)
        trial_cp_residual.fill(0.0)
        trial_cp_iterations.fill(0)

        # trial_F[:] = np.eye(3)
        
        # trial_Fp[:] = np.eye(3)
        
        # trial_Fe[:] = np.eye(3)
        
        # trial_J.fill(1.0)
        
        # trial_lattice_rotation[:] = np.eye(3)
        trial_F[:] = np.eye(3)
        
        trial_Fp[:] = np.eye(3)
        
        trial_Fe[:] = np.eye(3)
        
        trial_J_total.fill(1.0)
        
        trial_J_e.fill(1.0)
        
        trial_PK1.fill(0.0)
        
        trial_lattice_rotation[:] = np.eye(3)




        # ----------------------------------------------------
        # Element displacement vectors
        #
        # Shape:
        #     (n_elements, 24)
        # ----------------------------------------------------


        # ============================================================
        # FINITE-STRAIN ELEMENT KINEMATICS
        # ============================================================
        
        Ue_all = U_iter[
            element_dofs
        ]
        
        
        # ------------------------------------------------------------
        # Full displacement gradient
        # ------------------------------------------------------------
        
        grad_u_all = np.einsum(
            "ij,ej->ei",
            G,
            Ue_all,
            optimize=True
        )

        # ------------------------------------------------------------
        # Deformation gradient
        #
        # F = I + grad_u
        # ------------------------------------------------------------


        # F_all = I + grad_u_all

        # F_all = (
        #     I_flat
        #     + grad_u_all
        # ).reshape(
        #     n_elements,
        #     3,
        #     3
        # )
        
        # J_total_all = np.linalg.det(
        #     F_all
        # )


        # ============================================================
        # TOTAL DEFORMATION GRADIENT
        # ============================================================
        
        # J_total_all = np.linalg.det(
        #     F_all.reshape(n_elements, 3, 3)
        # )
                
        F_all = np.empty(
            (n_elements, 3, 3),
            dtype=np.float64
        )
        
        
        F_all[:, 0, 0] = (
            1.0 + grad_u_all[:, 0]
        )
        
        F_all[:, 0, 1] = (
            grad_u_all[:, 1]
        )
        
        F_all[:, 0, 2] = (
            grad_u_all[:, 2]
        )
        
        
        F_all[:, 1, 0] = (
            grad_u_all[:, 3]
        )
        
        F_all[:, 1, 1] = (
            1.0 + grad_u_all[:, 4]
        )
        
        F_all[:, 1, 2] = (
            grad_u_all[:, 5]
        )
        
        
        F_all[:, 2, 0] = (
            grad_u_all[:, 6]
        )
        
        F_all[:, 2, 1] = (
            grad_u_all[:, 7]
        )
        
        F_all[:, 2, 2] = (
            1.0 + grad_u_all[:, 8]
        )




        
        
        
        # # ------------------------------------------------------------
        # # Basic deformation-gradient sanity check
        # # ------------------------------------------------------------
        
        # # J_all = np.linalg.det(
        # #     F_all
        # # )
        
        # if np.any(J_all <= 0.0):
        
        #     bad = np.where(
        #         J_all <= 0.0
        #     )[0]
        
        #     raise RuntimeError(
        #         "Non-positive deformation-gradient determinant "
        #         f"det(F) encountered in elements: "
        #         f"{bad[:10]}"
        #     )

        # ============================================================
        # TOTAL DEFORMATION GRADIENT DETERMINANT
        # ============================================================
        
        J_total_all = np.linalg.det(
            F_all
        )
        
        # ============================================================
        # STORE TOTAL DEFORMATION GRADIENT
        # ============================================================
        
        trial_F[:] = F_all
        trial_J_total[:] = J_total_all

        # ============================================================
        # FINITE-STRAIN TOTAL STRAIN
        # GREEN-LAGRANGE STRAIN
        #
        # E = 1/2 * (F^T F - I)
        # ============================================================
        
        C_all = np.einsum(
            "eji,ejk->eik",
            F_all,
            F_all,
            optimize=True
        )
        
        E_all = 0.5 * (
            C_all
            - np.eye(3)
        )
        
        trial_total_strain[:, 0] = E_all[:, 0, 0]
        trial_total_strain[:, 1] = E_all[:, 1, 1]
        trial_total_strain[:, 2] = E_all[:, 2, 2]
        
        trial_total_strain[:, 3] = E_all[:, 0, 1]
        trial_total_strain[:, 4] = E_all[:, 1, 2]
        trial_total_strain[:, 5] = E_all[:, 0, 2]

        # Equivalent Green-Lagrange strain
        E_mean = (
            E_all[:, 0, 0]
            + E_all[:, 1, 1]
            + E_all[:, 2, 2]
        ) / 3.0
        
        E_dev = E_all.copy()
        
        E_dev[:, 0, 0] -= E_mean
        E_dev[:, 1, 1] -= E_mean
        E_dev[:, 2, 2] -= E_mean
        
        trial_equiv_total_strain[:] = np.sqrt(
            (2.0 / 3.0)
            * (
                E_dev[:, 0, 0]**2
                + E_dev[:, 1, 1]**2
                + E_dev[:, 2, 2]**2
                + 2.0 * (
                    E_dev[:, 0, 1]**2
                    + E_dev[:, 1, 2]**2
                    + E_dev[:, 0, 2]**2
                )
            )
        )



        
        
        del C_all
        del E_all
        del E_dev



        
        
        # ============================================================
        # DEFORMATION-GRADIENT SANITY CHECK
        # ============================================================
        
        if np.any(J_total_all <= 0.0):
        
            bad = np.where(
                J_total_all <= 0.0
            )[0]
        
            raise RuntimeError(
                "Invalid total deformation gradient: "
                "det(F) <= 0 in elements: "
                f"{bad[:10]}"
            )

        
     

       

        # ====================================================
        # LOCAL CPFEM SOLVES
        # ====================================================
        
        cp_start = time.time()
        
        cp_calls_this_iteration = 0
        local_cp_failed = False
        failed_element = None
        failed_cp_residual = None
        failed_cp_iterations = None        
        for e in range(n_elements):
        
            # ------------------------------------------------
            # Current deformation gradient
            # ------------------------------------------------
        
            F = F_all[e]
        
            # ------------------------------------------------
            # IMPORTANT:
            # Always use the committed Fp from the beginning
            # of the current global load increment.
            #
            # Do NOT use trial_Fp from a previous global
            # Newton iteration.
            # ------------------------------------------------
        
            Fp_old = Fp_backup[e]
        
            # ------------------------------------------------
            # Finite-strain local CPFEM constitutive update
            # ------------------------------------------------
        
            # if e == 19:
        
            #     print()
            #     print("================================================")
            #     print("DEBUG ELEMENT 19")
            #     print("Global load step =", step)
            #     print("Global iteration =", global_iter)
        
            #     print("F =")
            #     print(F)
        
            #     print("det(F) =", np.linalg.det(F))
        
            #     print("Fp_old =")
            #     print(Fp_old)
        
            #     print("det(Fp_old) =", np.linalg.det(Fp_old))
        
            #     print("resistance_old =")
            #     print(resistance_backup[e])
        
            #     print("x_initial =")
            #     print(cp_x_backup[e])
        
            #     print("dt =", cp_dt)
        
            #     print("================================================")
        
            # ------------------------------------------------
            # LOCAL FINITE-STRAIN CPFEM SOLVE
            # ------------------------------------------------
        
            (
                dgamma,
                Fp_new,
                Fe_new,
                stress,
                stress_voigt,
                resistance_new,
                rss,
                lattice_rotation,
                J_e,
                cp_residual,
                cp_iterations,
                cp_ok,
                C_alg
            ) = solve_cp_local_finite(
                F,
                Fp_old,
                resistance_backup[e],
                element_s_initial[e],
                element_n_initial[e],
                # cp_x_backup[e],
                cp_x_trial_guess[e],
                cp_dt
            )
        
            # ============================================================
            # STORE FINITE-STRAIN TRIAL STATE
            # ============================================================
        
            trial_F[e] = F
        
            trial_Fp[e] = Fp_new
        
            trial_Fe[e] = Fe_new
        
            # Elastic Jacobian from the LOCAL constitutive calculation
            trial_J_e[e] = J_e
        
            # Total deformation Jacobian was already calculated
            # before the element loop.
            trial_J_total[e] = J_total_all[e]
        
            trial_lattice_rotation[e] = (
                lattice_rotation
            )
        
            # ------------------------------------------------
            # Store stress
            # ------------------------------------------------
        
            trial_stress[e] = stress
        
            trial_stress_voigt[e] = stress_voigt
        
            # ------------------------------------------------
            # Store CP state
            # ------------------------------------------------
        
            trial_resistance[e] = resistance_new
        
            trial_rss[e] = rss
        
            # trial_slip[e] = dgamma
        
            # trial_cp_residual[e] = cp_residual

            trial_slip[e] = dgamma
            
            # --------------------------------------------------------
            # Warm-start local CPFEM on the next GLOBAL Newton
            # iteration.
            #
            # This is only a numerical initial guess.
            # It is NOT committed history.
            # --------------------------------------------------------
            cp_x_trial_guess[e] = dgamma
            
            trial_cp_residual[e] = cp_residual


            
        
            trial_cp_iterations[e] = cp_iterations
        
            trial_cp_converged[e] = cp_ok
        
            trial_C_alg[e] = C_alg
        
            # ============================================================
            # ACCUMULATED SLIP
            # ============================================================
        
            # trial_accumulated_slip_system[e] = (
            #     accumulated_slip_system_backup[e]
            #     + np.abs(dgamma)
            # )
            # trial_accumulated_slip_system[e] = (
            #     accumulated_slip_system_backup[e]
            #     + np.abs(dgamma)
            # )


            
        
            # trial_accumulated[e] = (
            #     # accumulated_slip_backup[e]
            #     accumulated_slip_system_backup[e]
            #     + np.sum(np.abs(dgamma))
            # )
            trial_accumulated_slip_system[e] = (
                accumulated_slip_system_backup[e]
                + np.abs(dgamma)
            )
            
            trial_accumulated[e] = (
                accumulated_backup[e]
                + np.sum(np.abs(dgamma))
            )



            
            # ============================================================
            # FIRST PIOLA-KIRCHHOFF STRESS
            #
            # P = J * sigma * F^{-T}
            # ============================================================
        
            F_inv_T = np.linalg.inv(F).T
        
            # trial_PK1[e] = (
            #     J_total_all[e]
            #     * stress
            #     @ F_inv_T
            # )

            trial_PK1[e] = (
                J_total_all[e]
                * (stress @ F_inv_T)
            )
            
        
            cp_calls_this_iteration += 1
        
            # ------------------------------------------------
            # Check local constitutive convergence
            # ------------------------------------------------





            if not cp_ok:
            
                local_cp_failed = True
                failed_element = e
                failed_cp_residual = cp_residual
                failed_cp_iterations = cp_iterations
            
                print()
                print(
                    "------------------------------------------------------"
                )
                print(
                    "LOCAL CPFEM FAILURE"
                )
                print(
                    f"Element              : {e}"
                )
                print(
                    f"Load step            : {step}"
                )
                print(
                    f"Global iteration     : {global_iter}"
                )
                print(
                    f"CP iterations        : {cp_iterations}"
                )
                print(
                    f"CP residual           : {cp_residual:.6e}"
                )
                print(
                    "Global strain step will be reduced."
                )
                print(
                    "------------------------------------------------------"
                )
            
                break


   
            



        # ----------------------------------------------------
        # Runtime accounting for local CPFEM
        # ----------------------------------------------------

        cp_elapsed = (
            time.time()
            - cp_start
        )

        runtime_stats["cp_total"] += (
            cp_elapsed
        )

        runtime_stats["element_cp_calls"] += (
            cp_calls_this_iteration
        )


        # ----------------------------------------------------
        # Abort this global Newton iteration if local CPFEM failed
        # ----------------------------------------------------
        
        if local_cp_failed:
        
            global_converged = False
        
            if DEBUG_GLOBAL:
        
                print(
                    f"      Global iteration aborted because "
                    f"local CPFEM failed in element "
                    f"{failed_element}."
                )
        
                print(
                    f"      CP residual = "
                    f"{failed_cp_residual:.6e}"
                )
        
                print(
                    f"      CP iterations = "
                    f"{failed_cp_iterations}"
                )
        
            break






        # ====================================================
        # INTERNAL FORCE
        # FINITE-STRAIN PK1 FORMULATION
        # ====================================================
        
        # fint_start = time.time()
        
        # F_internal.fill(0.0)
        
        # for e in range(n_elements):
        
        #     # ------------------------------------------------
        #     # First Piola-Kirchhoff stress
        #     # ------------------------------------------------
        
        #     P = trial_PK1[e]
        
        #     # ------------------------------------------------
        #     # Element internal force vector
        #     # ------------------------------------------------
        
        #     fint_e = np.zeros(
        #         24,
        #         dtype=np.float64
        #     )
        
        #     # ------------------------------------------------
        #     # HEX8 nodes
        #     # ------------------------------------------------
        
        #     for a in range(8):
        
        #         gradN = dN_dx[a]
        
        #         fint_e[
        #             3*a : 3*a + 3
        #         ] = P @ gradN
        
        #     # ------------------------------------------------
        #     # Reference element volume
        #     # ------------------------------------------------
        
        #     fint_e *= element_volume
        
        #     # ------------------------------------------------
        #     # Global assembly
        #     # ------------------------------------------------
        
        #     np.add.at(
        #         F_internal,
        #         element_dofs[e],
        #         fint_e
        #     )


        # ====================================================
        # INTERNAL FORCE
        # FINITE-STRAIN PK1 FORMULATION
        # VECTORIZED ELEMENT CALCULATION
        # ====================================================
        
        fint_start = time.time()
        
        # ----------------------------------------------------
        # P @ gradN for every element and every HEX8 node
        #
        # trial_PK1 : (n_elements, 3, 3)
        # dN_dx     : (8, 3)
        #
        # Result:
        # fint_nodes : (n_elements, 8, 3)
        # ----------------------------------------------------
        
        fint_nodes = np.einsum(
            "eij,aj->eai",
            trial_PK1,
            dN_dx,
            optimize=True
        )
        
        # ----------------------------------------------------
        # Reference element volume
        # ----------------------------------------------------
        
        fint_nodes *= element_volume
        
        # ----------------------------------------------------
        # Flatten to element DOF ordering
        #
        # (e, node, xyz)
        # ->
        # (e, 24)
        # ----------------------------------------------------
        
        fint_all = (
            fint_nodes
            .reshape(
                n_elements,
                24
            )
        )
        
        # ----------------------------------------------------
        # Global assembly
        # ----------------------------------------------------
        
        F_internal.fill(0.0)
        
        np.add.at(
            F_internal,
            element_dofs,
            fint_all
        )

        del fint_nodes
        del fint_all

        runtime_stats["internal_force_total"] += (
            time.time()
            - fint_start
        )
            


 
        # ====================================================
        # GLOBAL RESIDUAL
        # ====================================================
        
        residual = (
            F_internal[free_dofs]
        )
        
        # ----------------------------------------------------
        # Residual norm on FREE DOFs only
        # ----------------------------------------------------
        
        residual_norm = (
            np.linalg.norm(residual)
        )

        # ------------------------------------------------------------
        # Residual reduction ratio
        # ------------------------------------------------------------
        
        if previous_residual_norm is None:
        
            residual_ratio = np.nan
        
        else:
        
            residual_ratio = (
                residual_norm
                / max(
                    previous_residual_norm,
                    1.0e-30
                )
            )


        
        # ----------------------------------------------------
        # Establish reference residual only on the FIRST
        # Newton iteration of this physical load increment.
        # ----------------------------------------------------
        
        if residual_reference is None:
        
            residual_reference = max(
                residual_norm,
                1.0
            )
        
        # ----------------------------------------------------
        # Relative residual
        # ----------------------------------------------------
        
        relative_residual = (
            residual_norm
            / residual_reference
        )





        # ----------------------------------------------------
        # Debug output
        # ----------------------------------------------------

        if DEBUG_GLOBAL:

            print(
                f"  Global iteration {global_iter:2d}"
                f" | Residual = {residual_norm:.6e}"
                f" | Relative = {relative_residual:.6e}"
                f" | CP time = {cp_elapsed:.2f} s"
            )

        # ====================================================
        # GLOBAL CONVERGENCE CHECK
        # ====================================================

        if (
            relative_residual
            < GLOBAL_TOL
        ):

            global_converged = True

            if DEBUG_GLOBAL:

                print(
                    f"  --> GLOBAL CONVERGED "
                    f"in {global_iter} iterations"
                )

            break


        # # ----------------------------------------------------
        # # Release temporary tangent-assembly arrays
        # # They are no longer needed after K_ff_alg is formed.
        # # This frees memory for SuperLU factorization.
        # # ----------------------------------------------------
        
        # del Ke_alg_all
        # del K_global_alg



        # ============================================================
        # FINITE-STRAIN GLOBAL TANGENT / MODIFIED NEWTON
        # ============================================================
        #
        # The tangent and LU factorization are rebuilt periodically.
        # Between rebuilds, the existing factorization is reused.
        #
        # This avoids repeating the expensive:
        #
        #     tangent assembly
        #     sparse conversion
        #     LU factorization
        #
        # at every Newton iteration.
        # ============================================================
        
        # rebuild_tangent = (
        #     K_solver_alg is None
        #     or global_iter == 1
        #     or (
        #         (global_iter - 1)
        #         % TANGENT_REBUILD_FREQUENCY
        #         == 0
        #     )
        # )

        # ============================================================
        # ADAPTIVE GLOBAL TANGENT REBUILD
        # ============================================================
        #
        # Rebuild tangent when:
        #   1. This is the first Newton iteration.
        #   2. No valid LU factorization exists.
        #   3. Previous residual reduction was poor.
        #   4. Previous residual increased.
        #
        # Otherwise reuse the existing tangent/LU factorization.
        #
        # This avoids the expensive tangent assembly + LU factorization
        # when the current tangent is still giving good Newton steps.
        # ============================================================
        
        if global_iter == 1 or K_solver_alg is None:
        
            rebuild_tangent = True
        
        elif previous_residual_norm is None:
        
            rebuild_tangent = True
        
        else:
        
            residual_ratio = (
                residual_norm
                / max(previous_residual_norm, 1.0e-30)
            )
        
            # Conservative threshold initially.
            # If residual decreases by >70%, reuse tangent.
            # If reduction becomes weak or residual increases,
            # rebuild the tangent.
            ADAPTIVE_TANGENT_RATIO = 0.30
        
            rebuild_tangent = (
                residual_ratio > ADAPTIVE_TANGENT_RATIO
            )



        
        # ============================================================
        # REBUILD GLOBAL TANGENT
        # ============================================================
        
        if rebuild_tangent:
        
            # --------------------------------------------------------
            # If an old factorization exists, release it first
            # --------------------------------------------------------
        
            if K_solver_alg is not None:
        
                del K_solver_alg
        
                K_solver_alg = None
        
                gc.collect()
        
            # --------------------------------------------------------
            # Tangent assembly
            # --------------------------------------------------------
        
            tangent_start = time.time()
        
            K_global_alg = lil_matrix(
                (
                    NDOF,
                    NDOF
                ),
                dtype=np.float64
            )
        
            for e in range(n_elements):
        
                # ----------------------------------------------------
                # Current deformation gradient
                # ----------------------------------------------------
        
                F = trial_F[e]
        
                # ----------------------------------------------------
                # Current plastic deformation gradient
                #
                # Frozen-Fp tangent
                # ----------------------------------------------------
        
                Fp = trial_Fp[e]
        
                # ----------------------------------------------------
                # Numerical dP/dF
                # ----------------------------------------------------
        
                A_F = calculate_dP_dF_frozen_Fp(
                    F,
                    Fp
                )
        
                # ----------------------------------------------------
                # Element tangent
                # ----------------------------------------------------
        
                Ke = (
                    G.T
                    @ A_F
                    @ G
                )
        
                Ke *= element_volume
        
                # ----------------------------------------------------
                # Global assembly
                # ----------------------------------------------------
        
                dofs = element_dofs[e]
        
                for i in range(24):
        
                    I = dofs[i]
        
                    for j in range(24):
        
                        J = dofs[j]
        
                        K_global_alg[I, J] += Ke[i, j]
        
            # --------------------------------------------------------
            # Convert to CSC
            # --------------------------------------------------------
        
            K_global_alg = K_global_alg.tocsc()
        
            # --------------------------------------------------------
            # Free-free system
            # --------------------------------------------------------
        
            K_ff_alg = (
                K_global_alg[free_dofs, :]
                [:, free_dofs]
            )
        
            # --------------------------------------------------------
            # Release full global matrix
            # --------------------------------------------------------
        
            del K_global_alg
        
            gc.collect()
        
            # --------------------------------------------------------
            # Tangent timing
            # --------------------------------------------------------
        
            runtime_stats["tangent_assembly_total"] += (
                time.time()
                - tangent_start
            )
        
            # ========================================================
            # GLOBAL FACTORIZATION
            # ========================================================
        
            factor_start = time.time()
        
            try:
        
                K_solver_alg = splu(
                    K_ff_alg,
                    permc_spec="COLAMD",
                    diag_pivot_thresh=0.0
                )
        
            except RuntimeError as err:
        
                print()
                print(
                    "------------------------------------------------"
                )
                print(
                    "GLOBAL SPARSE FACTORIZATION FAILED"
                )
                print(
                    f"Load step       : {step}"
                )
                print(
                    f"Global iteration: {global_iter}"
                )
                print(
                    f"Error           : {err}"
                )
                print(
                    "------------------------------------------------"
                )
        
                del K_ff_alg
                gc.collect()
        
                K_solver_alg = None
        
                global_converged = False
        
                break
        
            # --------------------------------------------------------
            # Release sparse matrix.
            # The LU factorization is retained.
            # --------------------------------------------------------
        
            del K_ff_alg
        
            gc.collect()
        
            runtime_stats["factorization_total"] += (
                time.time()
                - factor_start
            )
        
            if DEBUG_GLOBAL:
        
                print(
                    "      GLOBAL TANGENT REBUILT"
                )
        
        else:
        
            if DEBUG_GLOBAL:
        
                print(
                    "      GLOBAL TANGENT REUSED"
                )
        
        
        # ============================================================
        # GLOBAL LINEAR SOLVE
        # ============================================================
        
        solve_start = time.time()
        
        try:
        
            delta_U_free = (
                K_solver_alg.solve(
                    -residual
                )
            )
        
        except Exception as err:
        
            print()
            print(
                "------------------------------------------------"
            )
            print(
                "GLOBAL LINEAR SOLVE FAILED"
            )
            print(
                f"Load step       : {step}"
            )
            print(
                f"Global iteration: {global_iter}"
            )
            print(
                f"Error           : {err}"
            )
            print(
                "------------------------------------------------"
            )
        
            if K_solver_alg is not None:
        
                del K_solver_alg
        
                K_solver_alg = None
        
            gc.collect()
        
            global_converged = False
        
            break
        
        runtime_stats["linear_solve_total"] += (
            time.time()
            - solve_start
        )
    
    
    

        # ====================================================
        # GLOBAL NEWTON UPDATE
        # WITH CHEAP DET(F) SAFEGUARD
        # ====================================================

        alpha = GLOBAL_DAMPING
        global_step_rejected = False             
        # Current displacement state
        U_base = U_iter.copy()
        
        while True:
        
            # ------------------------------------------------
            # Trial global displacement
            # ------------------------------------------------
            U_trial = U_base.copy()
        
            U_trial[free_dofs] += (
                alpha * delta_U_free
            )
        
            # ------------------------------------------------
            # Re-enforce prescribed boundary conditions
            # ------------------------------------------------
            U_trial[top_z_dofs] = (
                target_strain * Lz
            )
        
            U_trial[bottom_z_dofs] = 0.0
            U_trial[fix_x_dof] = 0.0
            U_trial[fix_y_dof] = 0.0
        
            # ------------------------------------------------
            # Calculate trial deformation gradient
            # ------------------------------------------------
            Ue_trial = U_trial[
                element_dofs
            ]
        
            grad_u_trial = np.einsum(
                "ij,ej->ei",
                G,
                Ue_trial,
                optimize=True
            )
        
            F_trial = np.empty(
                (n_elements, 3, 3),
                dtype=np.float64
            )
        
            F_trial[:, 0, 0] = (
                1.0 + grad_u_trial[:, 0]
            )
        
            F_trial[:, 0, 1] = (
                grad_u_trial[:, 1]
            )
        
            F_trial[:, 0, 2] = (
                grad_u_trial[:, 2]
            )
        
            F_trial[:, 1, 0] = (
                grad_u_trial[:, 3]
            )
        
            F_trial[:, 1, 1] = (
                1.0 + grad_u_trial[:, 4]
            )
        
            F_trial[:, 1, 2] = (
                grad_u_trial[:, 5]
            )
        
            F_trial[:, 2, 0] = (
                grad_u_trial[:, 6]
            )
        
            F_trial[:, 2, 1] = (
                grad_u_trial[:, 7]
            )
        
            F_trial[:, 2, 2] = (
                1.0 + grad_u_trial[:, 8]
            )
        
            # ------------------------------------------------
            # Check deformation-gradient determinant
            # ------------------------------------------------
            J_trial = np.linalg.det(
                F_trial
            )
        
            min_J_trial = np.min(
                J_trial
            )
        
            # ------------------------------------------------
            # Accept if deformation remains reasonable
            # ------------------------------------------------
            if (
                np.all(np.isfinite(J_trial))
                and min_J_trial > MIN_DET_F
            ):
                U_iter = U_trial
        
                if DEBUG_GLOBAL and alpha < GLOBAL_DAMPING:
                    print(
                        f"      GLOBAL STEP DAMPED: "
                        f"alpha = {alpha:.4f}, "
                        f"min det(F) = {min_J_trial:.6e}"
                    )
        
                break
        
            # ------------------------------------------------
            # Reduce Newton correction
            # ------------------------------------------------
            alpha *= GLOBAL_DAMPING_REDUCTION
        
            # if alpha < GLOBAL_DAMPING_MIN:
        
            #     print()
            #     print(
            #         "------------------------------------------------------"
            #     )
            #     print(
            #         "GLOBAL NEWTON STEP REJECTED"
            #     )
            #     print(
            #         f"Minimum alpha reached: {alpha:.6e}"
            #     )
            #     print(
            #         f"Minimum det(F): {min_J_trial:.6e}"
            #     )
            #     print(
            #         "------------------------------------------------------"
            #     )
        
            #     global_converged = False

            if alpha < GLOBAL_DAMPING_MIN:

                print()
                print(
                    "------------------------------------------------------"
                )
                print(
                    "GLOBAL NEWTON STEP REJECTED"
                )
                print(
                    f"Minimum alpha reached: {alpha:.6e}"
                )
                print(
                    f"Minimum det(F): {min_J_trial:.6e}"
                )
                print(
                    "------------------------------------------------------"
                )

                global_converged = False
                global_step_rejected = True

                break            

            if DEBUG_GLOBAL:
                print(
                    f"      Reducing global Newton step: "
                    f"alpha = {alpha:.4f}"
                )

        # ----------------------------------------------------
        # Abort global Newton loop
        # ----------------------------------------------------
        

        
        # ----------------------------------------------------
        # Clean temporary arrays
        # ----------------------------------------------------
        del U_trial
        del Ue_trial
        del grad_u_trial
        del F_trial
        del J_trial
        
        if global_step_rejected:
        
            global_converged = False
            break




        
        # ----------------------------------------------------
        # Re-enforce prescribed boundary conditions
        # ----------------------------------------------------
    
        U_iter[top_z_dofs] = (
            target_strain * Lz
        )
    
        U_iter[bottom_z_dofs] = 0.0
        U_iter[fix_x_dof] = 0.0
        U_iter[fix_y_dof] = 0.0
    
        global_iter_elapsed = (
            time.time()
            - global_iter_start
        )
    
        if DEBUG_GLOBAL:
    
            print(
                f"      iteration time = "
                f"{global_iter_elapsed:.3f} s"
            )

    # ========================================================
    # END GLOBAL NEWTON ITERATION
    # ========================================================

    # --------------------------------------------------------
    # If global convergence failed, reduce load increment
    # --------------------------------------------------------

    if not global_converged:

        # ----------------------------------------------------
        # Restore committed state
        # ----------------------------------------------------

        U = U_backup.copy()

        # plastic_strain_committed = (
        #     plastic_backup.copy()
        # )
        Fp_committed = (
            Fp_backup.copy()
        )




        resistance_committed = (
            resistance_backup.copy()
        )

        accumulated_slip_committed = (
            accumulated_backup.copy()
        )

        accumulated_slip_system_committed = (
            accumulated_slip_system_backup.copy()
        )



        cp_x_committed = (
            cp_x_backup.copy()
        )

        # ----------------------------------------------------
        # Reduce strain increment
        # ----------------------------------------------------

        old_strain_step = strain_step

        strain_step = max(
            strain_step * 0.5,
            MIN_STRAIN_STEP
        )

        print()
        print(
            "======================================================"
        )
        print(
            f"LOAD STEP {step} FAILED"
        )
        print(
            f"Strain increment reduced:"
            f" {old_strain_step:.6e}"
            f" -> {strain_step:.6e}"
        )
        print(
            "======================================================"
        )

        # ----------------------------------------------------
        # If already at minimum increment, stop.
        # ----------------------------------------------------

        if (
            old_strain_step
            <= MIN_STRAIN_STEP
            + 1.0e-15
        ):

            raise RuntimeError(
                "Global FE–CPFEM solver failed "
                "at the minimum strain increment."
            )

        # Do not advance current_strain.
        # Retry the same load target.
       
        # ----------------------------------------------------
        # Discard LU factorization before retrying
        # ----------------------------------------------------

        if K_solver_alg is not None:

            del K_solver_alg

            K_solver_alg = None

            gc.collect()

        step -= 1

        continue        
        




    # ========================================================
    # COMMIT CONVERGED STATE
    # ========================================================
    
   
    


    # ============================================================
    # FINALIZE CONVERGED TOTAL STRAIN FIELD
    #
    # trial_total_strain has already been calculated from the
    # converged trial_F during the final Newton iteration.
    # ============================================================

    U = U_iter.copy()
    
    Fp_committed = (
        trial_Fp.copy()
    )
    
    resistance_committed = (
        trial_resistance.copy()
    )
    
    accumulated_slip_committed = (
        trial_accumulated.copy()
    )
    
    accumulated_slip_system_committed = (
        trial_accumulated_slip_system.copy()
    )
    
    cp_x_committed = (
        trial_slip.copy()
    )

    # ------------------------------------------------------------
    # Advance the physical load state
    # ------------------------------------------------------------
    
    current_strain = target_strain



    
    # ============================================================
    # FINALIZE CONVERGED LOCAL DEFORMATION / FATIGUE DIAGNOSTICS
    # ============================================================
    
    # # ------------------------------------------------------------
    # # 1. J2 equivalent total strain
    # # ------------------------------------------------------------
    
    # eps_xx = trial_total_strain[:, 0]
    # eps_yy = trial_total_strain[:, 1]
    # eps_zz = trial_total_strain[:, 2]
    
    # gamma_xy = trial_total_strain[:, 3]
    # gamma_yz = trial_total_strain[:, 4]
    # gamma_xz = trial_total_strain[:, 5]
    
    # eps_mean = (
    #     eps_xx
    #     + eps_yy
    #     + eps_zz
    # ) / 3.0
    
    # dev_xx = eps_xx - eps_mean
    # dev_yy = eps_yy - eps_mean
    # dev_zz = eps_zz - eps_mean
    
    # trial_equiv_total_strain[:] = np.sqrt(
    #     np.maximum(
    #         0.0,
    #         (2.0 / 3.0) * (
    #             dev_xx**2
    #             + dev_yy**2
    #             + dev_zz**2
    #             + 0.5 * gamma_xy**2
    #             + 0.5 * gamma_yz**2
    #             + 0.5 * gamma_xz**2
    #         )
    #     )
    # )
    
    
    # ------------------------------------------------------------
    # # 2. J2 equivalent plastic strain
    # # ------------------------------------------------------------
    
    # plastic_trace = (
    #     trial_plastic[:, 0, 0]
    #     + trial_plastic[:, 1, 1]
    #     + trial_plastic[:, 2, 2]
    # ) / 3.0
    
    # plastic_dev_xx = (
    #     trial_plastic[:, 0, 0]
    #     - plastic_trace
    # )
    
    # plastic_dev_yy = (
    #     trial_plastic[:, 1, 1]
    #     - plastic_trace
    # )
    
    # plastic_dev_zz = (
    #     trial_plastic[:, 2, 2]
    #     - plastic_trace
    # )
    
    # trial_equiv_plastic_strain[:] = np.sqrt(
    #     np.maximum(
    #         0.0,
    #         (2.0 / 3.0) * (
    #             plastic_dev_xx**2
    #             + plastic_dev_yy**2
    #             + plastic_dev_zz**2
    #             + 2.0 * trial_plastic[:, 0, 1]**2
    #             + 2.0 * trial_plastic[:, 1, 2]**2
    #             + 2.0 * trial_plastic[:, 0, 2]**2
    #         )
    #     )
    # )

    # ------------------------------------------------------------
    # 2. FINITE-STRAIN J2 EQUIVALENT PLASTIC STRAIN
    # ------------------------------------------------------------
    #
    # Multiplicative decomposition:
    #
    #     F = Fe @ Fp
    #
    # The authoritative plastic state is Fp.
    #
    # Define the plastic right Cauchy-Green tensor:
    #
    #     Cp = Fp.T @ Fp
    #
    # and the corresponding plastic Green-Lagrange strain:
    #
    #     Ep = 0.5 * (Cp - I)
    #
    # Store Ep in trial_plastic using:
    #
    #     [Ep11, Ep22, Ep33, Ep12, Ep23, Ep13]
    #
    # All shear components are TENSORIAL components,
    # not engineering shear strains.
    #
    # ------------------------------------------------------------

    Cp_plastic = np.einsum(
        "eji,ejk->eik",
        trial_Fp,
        trial_Fp,
        optimize=True
    )

    Ep_plastic = 0.5 * (
        Cp_plastic - np.eye(3)
    )

    # ------------------------------------------------------------
    # Store finite-strain plastic tensor
    # ------------------------------------------------------------

    trial_plastic[:, 0, 0] = Ep_plastic[:, 0, 0]
    trial_plastic[:, 1, 1] = Ep_plastic[:, 1, 1]
    trial_plastic[:, 2, 2] = Ep_plastic[:, 2, 2]

    trial_plastic[:, 0, 1] = Ep_plastic[:, 0, 1]
    trial_plastic[:, 1, 2] = Ep_plastic[:, 1, 2]
    trial_plastic[:, 0, 2] = Ep_plastic[:, 0, 2]

    # Enforce symmetry explicitly
    trial_plastic[:, 1, 0] = Ep_plastic[:, 0, 1]
    trial_plastic[:, 2, 1] = Ep_plastic[:, 1, 2]
    trial_plastic[:, 2, 0] = Ep_plastic[:, 0, 2]

    # ------------------------------------------------------------
    # Deviatoric plastic Green-Lagrange strain
    # ------------------------------------------------------------

    plastic_mean = (
        Ep_plastic[:, 0, 0]
        + Ep_plastic[:, 1, 1]
        + Ep_plastic[:, 2, 2]
    ) / 3.0

    plastic_dev = Ep_plastic.copy()

    plastic_dev[:, 0, 0] -= plastic_mean
    plastic_dev[:, 1, 1] -= plastic_mean
    plastic_dev[:, 2, 2] -= plastic_mean

    # ------------------------------------------------------------
    # J2 equivalent plastic strain
    #
    #     Ep_eq = sqrt(2/3 * Ep_dev:Ep_dev)
    #
    # Since Ep contains tensorial shear components:
    #
    #     Ep_dev:Ep_dev
    #
    # includes 2*E12^2 + 2*E23^2 + 2*E13^2.
    # ------------------------------------------------------------

    trial_equiv_plastic_strain[:] = np.sqrt(
        np.maximum(
            0.0,
            (2.0 / 3.0) * (
                plastic_dev[:, 0, 0]**2
                + plastic_dev[:, 1, 1]**2
                + plastic_dev[:, 2, 2]**2
                + 2.0 * (
                    plastic_dev[:, 0, 1]**2
                    + plastic_dev[:, 1, 2]**2
                    + plastic_dev[:, 0, 2]**2
                )
            )
        )
    )

    del Cp_plastic
    del Ep_plastic
    del plastic_dev


    
    
    
    # ------------------------------------------------------------
    # 3. Von Mises equivalent stress
    # ------------------------------------------------------------
    
    stress_mean = (
        trial_stress[:, 0, 0]
        + trial_stress[:, 1, 1]
        + trial_stress[:, 2, 2]
    ) / 3.0
    
    stress_dev_xx = (
        trial_stress[:, 0, 0]
        - stress_mean
    )
    
    stress_dev_yy = (
        trial_stress[:, 1, 1]
        - stress_mean
    )
    
    stress_dev_zz = (
        trial_stress[:, 2, 2]
        - stress_mean
    )
    
    trial_equiv_stress[:] = np.sqrt(
        np.maximum(
            0.0,
            1.5 * (
                stress_dev_xx**2
                + stress_dev_yy**2
                + stress_dev_zz**2
                + 2.0 * trial_stress[:, 0, 1]**2
                + 2.0 * trial_stress[:, 1, 2]**2
                + 2.0 * trial_stress[:, 0, 2]**2
            )
        )
    )
    
    
    # ------------------------------------------------------------
    # 4. Maximum absolute RSS among the 12 slip systems
    # ------------------------------------------------------------
    
    abs_rss = np.abs(trial_rss)
    
    trial_max_abs_rss[:] = np.max(
        abs_rss,
        axis=1
    )
    
    
    # ------------------------------------------------------------
    # 5. Slip-system index corresponding to maximum |RSS|
    # ------------------------------------------------------------
    
    trial_max_rss_system[:] = np.argmax(
        abs_rss,
        axis=1
    )
    
    del abs_rss








    # ============================================================
    # FINALIZE CONVERGED TOTAL STRAIN FIELD
    #
    # trial_total_strain has already been calculated from the
    # converged trial_F during the final Newton iteration.
    # ============================================================

    # ============================================================
    # MACROSCOPIC STRESS FOR CONVERGED CHECKPOINT
    # ============================================================

    sigma_tensor_average = np.mean(
        trial_stress,
        axis=0
    )

    sigma_macro = sigma_tensor_average[2, 2]


    # ============================================================
    # 5. SAVE CONVERGED STEPWISE CHECKPOINT
    # ============================================================

    np.savez_compressed(
        os.path.join(
            CHECKPOINT_DIR,
            f"step_{step:04d}.npz"
        ),

        step=step,

        strain=current_strain,

        stress=sigma_macro,

        global_iterations=global_iter,
    
        residual_norm=residual_norm,
    
        relative_residual=relative_residual,



        # ----------------------------------------------------
        # Equivalent local diagnostics
        # ----------------------------------------------------
    
        equiv_total_strain=(
            trial_equiv_total_strain.copy()
        ),
    
        equiv_plastic_strain=(
            trial_equiv_plastic_strain.copy()
        ),
    
        equiv_stress=(
            trial_equiv_stress.copy()
        ),
    
        max_abs_rss=(
            trial_max_abs_rss.copy()
        ),
    
        max_rss_system=(
            trial_max_rss_system.copy()
        ),




        element_grain=element_grain.copy(),

        element_centers=element_centers.copy(),

        total_strain=trial_total_strain.copy(),

        stress_tensor=trial_stress.copy(),

        stress_voigt=trial_stress_voigt.copy(),

        plastic_strain=trial_plastic.copy(),

        resistance=trial_resistance.copy(),

        accumulated_slip=trial_accumulated.copy(),

        accumulated_slip_system=(
            trial_accumulated_slip_system.copy()
        ),

        slip=trial_slip.copy(),

        rss=trial_rss.copy(),

        cp_residual=trial_cp_residual.copy(),

        cp_iterations=trial_cp_iterations.copy(),

        F=trial_F.copy(),
        Fp=trial_Fp.copy(),
        Fe=trial_Fe.copy(),
        # J=trial_J.copy(),
        J_total=trial_J_total.copy(),
        J_e=trial_J_e.copy(),




        lattice_rotation=trial_lattice_rotation.copy(),

        cp_converged=trial_cp_converged.copy()
    )

    print(
        f"Checkpoint saved: "
        f"{CHECKPOINT_DIR}\\step_{step:04d}.npz"
    )

    
    
    # ========================================================
    # STORE CONVERGED LOCAL FIELDS
    # ========================================================
    #
    # Only the converged physical state is stored.
    # Newton trial states are NOT stored.
    # ========================================================
    
    # trial_total_strain = strain_all.copy()

    
    # local_field_history.append(
    #     {
    #         "total_strain_voigt": trial_total_strain.copy(),
    #         "stress_voigt": trial_stress_voigt.copy(),
    #         "stress": trial_stress.copy(),
    #         "plastic_strain": trial_plastic.copy(),
    #         "resistance": trial_resistance.copy(),
    #         # "accumulated_slip": trial_accumulated.copy(),
    #         "accumulated_slip": trial_accumulated.copy(),
            
    #         "accumulated_slip_system":
    #             trial_accumulated_slip_system.copy(),



            
    #         "slip": trial_slip.copy(),
    #         "rss": trial_rss.copy(),
    #         "cp_residual": trial_cp_residual.copy(),
    #         "cp_iterations": trial_cp_iterations.copy(),
    #         "cp_converged": trial_cp_converged.copy()
    #     }
    # )

    local_field_history.append(
        {
            # ====================================================
            # FINITE-STRAIN STATE
            # ====================================================
    
            "F": trial_F.copy(),
    
            "Fp": trial_Fp.copy(),
    
            "Fe": trial_Fe.copy(),
    
            # "J": trial_J.copy(),
            "J_total": trial_J_total.copy(),

            "J_e": trial_J_e.copy(),
    
            "lattice_rotation":
                trial_lattice_rotation.copy(),
    
    
            # ====================================================
            # STRAIN / STRESS
            # ====================================================
    
            "total_strain_voigt":
                trial_total_strain.copy(),
    
            "stress_voigt":
                trial_stress_voigt.copy(),
    
            "stress":
                trial_stress.copy(),
    
            "plastic_strain":
                trial_plastic.copy(),
    
            "resistance":
                trial_resistance.copy(),
    
            "accumulated_slip":
                trial_accumulated.copy(),
    
            "accumulated_slip_system":
                trial_accumulated_slip_system.copy(),
    
            "slip":
                trial_slip.copy(),
    
            "rss":
                trial_rss.copy(),
    
            "cp_residual":
                trial_cp_residual.copy(),
    
            "cp_iterations":
                trial_cp_iterations.copy(),
    
            "cp_converged":
                trial_cp_converged.copy()
        }
    )




    
    local_field_step_history.append(
        step
    )
    
    local_field_strain_history.append(
        current_strain
    )
    
    
    # ========================================================
    # MACROSCOPIC RESPONSE
    # ========================================================








    # ========================================================
    # MACROSCOPIC RESPONSE
    # ========================================================


    # --------------------------------------------------------
    # Maximum RSS in the microstructure
    # --------------------------------------------------------

    max_rss = np.max(
        np.abs(trial_rss)
    )

    # --------------------------------------------------------
    # Mean CRSS
    # --------------------------------------------------------

    # mean_crss = np.mean(
    #     resistance_committed
    # )
    mean_crss = np.mean(
        trial_resistance
    )






    # --------------------------------------------------------
    # Mean accumulated slip
    #
    # This remains a primary CP plastic-deformation measure
    # for the finite-strain formulation.
    # --------------------------------------------------------
    
    mean_accumulated_slip = np.mean(
        trial_accumulated
    )
    
    
    # --------------------------------------------------------
    # Finite-strain plasticity diagnostic
    #
    # Do NOT use the old infinitesimal plastic-strain tensor
    # here as the primary finite-strain measure.
    #
    # Fp is the physical plastic state.
    # A dedicated Fp-based equivalent measure will be
    # calculated in post-processing.
    # --------------------------------------------------------
    
    mean_plastic_strain = np.nan


    # --------------------------------------------------------
    # Mean accumulated slip
    # --------------------------------------------------------

    # mean_accumulated_slip = np.mean(
    #     accumulated_slip_committed
    # )

    # --------------------------------------------------------
    # CP iteration statistics
    # --------------------------------------------------------

    mean_cp_iterations = np.mean(
        trial_cp_iterations
    )

    max_cp_iterations = np.max(
        trial_cp_iterations
    )

    max_cp_residual = np.max(
        trial_cp_residual
    )

    # ========================================================
    # STORE GLOBAL HISTORY
    # ========================================================

    global_strain_history.append(
        current_strain
    )

    global_stress_history.append(
        sigma_macro
    )

    step_history.append(
        step
    )


  

    # --------------------------------------------------------
    # Store useful global CP statistics
    # --------------------------------------------------------

    # cp_state_history.append(
    #     {
    #         "strain": current_strain,
    #         "stress": sigma_macro,
    #         "mean_crss": mean_crss,
    #         "max_rss": max_rss,
    #         "mean_plastic_strain": mean_plastic_strain,
    #         "mean_accumulated_slip": mean_accumulated_slip,
    #         "mean_cp_iterations": mean_cp_iterations,
    #         "max_cp_iterations": max_cp_iterations,
    #         "max_cp_residual": max_cp_residual,
    #         "global_iterations": global_iter,
    #         "relative_residual": relative_residual,
    #         "residual_norm": residual_norm
    #     }
    # )

    
    cp_state_history.append(
        {
            "strain": current_strain,
            "stress": sigma_macro,
            "mean_crss": mean_crss,
            "max_rss": max_rss,
            "mean_plastic_strain": mean_plastic_strain,
            "mean_accumulated_slip": mean_accumulated_slip,
            "mean_cp_iterations": mean_cp_iterations,
            "max_cp_iterations": max_cp_iterations,
            "max_cp_residual": max_cp_residual,
            "global_iterations": global_iter,
            "relative_residual": relative_residual,
            "residual_norm": residual_norm,
    
            # Cyclic information
            # "cyclic_segment": cyclic_segment + 1,
            # "cyclic_direction": cyclic_direction,
            # "segment_target": segment_target
        }
    )






    

    # ========================================================
    # LOAD STEP OUTPUT
    # ========================================================

    print()
    print(
        "======================================================"
    )

    print(
        f"LOAD STEP {step} CONVERGED"
    )

    print(
        f"Current strain          = "
        f"{current_strain:.8e}"
    )

    # print(
    #     f"Cyclic segment          = "
    #     f"{cyclic_segment + 1}"
    # )
    
    # print(
    #     f"Loading direction       = "
    #     f"{'LOADING' if cyclic_direction > 0 else 'UNLOADING'}"
    # )
    
    # print(
    #     f"Segment target          = "
    #     f"{segment_target:.8e}"
    # )
    
    print(
        f"Strain increment        = "
        f"{actual_increment:.8e}"
    )









    print(
        f"Axial stress            = "
        f"{sigma_macro:.6f} MPa"
    )

    print(
        f"Mean CRSS               = "
        f"{mean_crss:.6f} MPa"
    )

    print(
        f"Maximum |RSS|           = "
        f"{max_rss:.6f} MPa"
    )

    print(
        f"Mean plastic strain     = "
        # f"{mean_plastic_strain:.8e}"

        "not yet defined for finite-strain CPFEM"

    )

    print(
        f"Mean accumulated slip   = "
        f"{mean_accumulated_slip:.8e}"
    )

    print(
        f"Mean CP iterations      = "
        f"{mean_cp_iterations:.2f}"
    )

    print(
        f"Maximum CP iterations   = "
        f"{int(max_cp_iterations)}"
    )

    print(
        f"Maximum CP residual     = "
        f"{max_cp_residual:.6e}"
    )

    print(
        f"Global iterations       = "
        f"{global_iter}"
    )

    print(
        "======================================================"
    )

    # ========================================================
    # ADAPTIVE STRAIN-INCREMENT CONTROL
    # ========================================================

    if global_iter <= 4:

        strain_step = min(
            strain_step * 1.50,
            MAX_STRAIN_STEP
        )

    elif global_iter <= 8:

        strain_step = min(
            strain_step * 1.25,
            MAX_STRAIN_STEP
        )

    elif global_iter >= 15:

        strain_step = max(
            strain_step * 0.75,
            MIN_STRAIN_STEP
        )

    # --------------------------------------------------------
    # Make sure the next step does not exceed TOTAL_STRAIN
    # --------------------------------------------------------

    # strain_step = min(
    #     strain_step,
    #     TOTAL_STRAIN - current_strain
    # )

    # # --------------------------------------------------------
    # # Safety against zero/negative next increment
    # # --------------------------------------------------------

    # if strain_step <= 0.0:
    #     break


    #     # ========================================================
    # # ADAPTIVE STRAIN-INCREMENT CONTROL
    # # ========================================================
    
    # if global_iter <= 4:
    
    #     strain_step = min(
    #         strain_step * 1.50,
    #         MAX_STRAIN_STEP
    #     )
    
    # elif global_iter <= 8:
    
    #     strain_step = min(
    #         strain_step * 1.25,
    #         MAX_STRAIN_STEP
    #     )
    
    # elif global_iter >= 15:
    
    #     strain_step = max(
    #         strain_step * 0.75,
    #         MIN_STRAIN_STEP
    #     )
    
    # # --------------------------------------------------------
    # # Limit next increment to the current segment target
    # # --------------------------------------------------------
    
    # distance_to_target = abs(
    #     segment_target - current_strain
    # )
    
    # strain_step = min(
    #     strain_step,
    #     distance_to_target
    # )
    
    # # --------------------------------------------------------
    # # Segment completed
    # # --------------------------------------------------------
    
    # if distance_to_target <= 1.0e-14:
    
    #     cyclic_segment += 1
    
    #     if cyclic_segment >= len(cyclic_targets):
    #         break
    
    #     # Reset increment at reversal point
    #     strain_step = INITIAL_STRAIN_STEP
    
    # --------------------------------------------------------
    # Limit next increment to TOTAL_STRAIN
    # --------------------------------------------------------
    
    distance_to_target = (
        TOTAL_STRAIN - current_strain
    )
    
    if distance_to_target > 0.0:
    
        strain_step = min(
            strain_step,
            distance_to_target
        )
    
    else:
    
        strain_step = 0.0    




# ================================================================
# 20. CONVERT HISTORY
# ================================================================

global_strain_history = np.asarray(
    global_strain_history
)

global_stress_history = np.asarray(
    global_stress_history
)


local_field_step_history = np.asarray(
    local_field_step_history,
    dtype=np.int32
)

local_field_strain_history = np.asarray(
    local_field_strain_history,
    dtype=np.float64
)


# ================================================================
# CONVERT LOCAL FIELD HISTORY
# ================================================================

local_total_strain_history = np.asarray(
    [
        x["total_strain_voigt"]
        for x in local_field_history
    ],
    dtype=np.float64
)

# ============================================================
# FINITE-STRAIN HISTORY
# ============================================================

local_F_history = np.asarray(
    [
        x["F"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_Fp_history = np.asarray(
    [
        x["Fp"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_Fe_history = np.asarray(
    [
        x["Fe"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_J_history = np.asarray(
    [
        x["J"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_lattice_rotation_history = np.asarray(
    [
        x["lattice_rotation"]
        for x in local_field_history
    ],
    dtype=np.float64
)




local_stress_voigt_history = np.asarray(
    [
        x["stress_voigt"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_stress_history = np.asarray(
    [
        x["stress"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_plastic_strain_history = np.asarray(
    [
        x["plastic_strain"]
        for x in local_field_history
    ],
    dtype=np.float64
)


# ========================================================
# EQUIVALENT PLASTIC STRAIN HISTORY
# ========================================================

# eps_p = local_plastic_strain_history

# eps_p_dev = (
#     eps_p
#     - np.trace(
#         eps_p,
#         axis1=-2,
#         axis2=-1
#     )[..., None, None] / 3.0
# )

# local_equiv_plastic_strain_history = np.sqrt(
#     (2.0 / 3.0) *
#     np.einsum(
#         "...ij,...ij->...",
#         eps_p_dev,
#         eps_p_dev
#     )
# )


# ========================================================
# OTHER LOCAL HISTORY ARRAYS
# ========================================================

local_resistance_history = np.asarray(
    [
        x["resistance"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_accumulated_slip_history = np.asarray(
    [
        x["accumulated_slip"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_slip_history = np.asarray(
    [
        x["slip"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_rss_history = np.asarray(
    [
        x["rss"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_cp_residual_history = np.asarray(
    [
        x["cp_residual"]
        for x in local_field_history
    ],
    dtype=np.float64
)

local_cp_iterations_history = np.asarray(
    [
        x["cp_iterations"]
        for x in local_field_history
    ],
    dtype=np.int32
)

local_cp_converged_history = np.asarray(
    [
        x["cp_converged"]
        for x in local_field_history
    ],
    dtype=bool
)


# ========================================================
# PER-SLIP-SYSTEM ACCUMULATED SLIP HISTORY
# ========================================================

local_accumulated_slip_system_history = np.asarray(
    [
        x["accumulated_slip_system"]
        for x in local_field_history
    ],
    dtype=np.float64
)


# ========================================================
# DERIVED LOCAL HISTORY QUANTITIES
# ========================================================

# Slip increment history
local_slip_increment_history = (
    local_slip_history.copy()
)

# CRSS / resistance history
local_crss_history = (
    local_resistance_history.copy()
)

# Maximum absolute RSS for each element
# at each load step
local_max_abs_rss_history = np.max(
    np.abs(local_rss_history),
    axis=-1
)



# ========================================================
# LOCAL YIELD / SLIP UTILIZATION RATIO
# ========================================================

# yield_ratio_history = np.zeros_like(
#     local_rss_history
# )

# ========================================================
# LOCAL SLIP-SYSTEM UTILIZATION RATIO
# ========================================================
#
# utilization = |RSS| / CRSS
#
# This is a rate-dependent slip-utilization measure.
# It is not a strict rate-independent yield criterion.
# ========================================================

slip_utilization_history = np.zeros_like(
    local_rss_history
)




valid_crss = (
    local_crss_history > 1.0e-14
)

yield_ratio_history[valid_crss] = (
    np.abs(local_rss_history[valid_crss])
    /
    local_crss_history[valid_crss]
)

# local_max_yield_ratio_history = np.max(
#     yield_ratio_history,
#     axis=-1
# )
local_max_slip_utilization_history = np.max(
    slip_utilization_history,
    axis=-1
)



# local_active_slip_count_history = np.sum(
#     yield_ratio_history >= 1.0,
#     axis=-1
# )

local_active_slip_count_history = np.sum(
    slip_utilization_history >= 1.0,
    axis=-1
)




# local_yielded_element_count_history = np.sum(
#     local_max_yield_ratio_history >= 1.0,
#     axis=-1
# )

local_high_utilization_element_count_history = np.sum(
    local_max_slip_utilization_history >= 1.0,
    axis=-1
)



# local_yielded_element_fraction_history = (
#     local_yielded_element_count_history
#     / n_elements
# )

local_high_utilization_element_fraction_history = (
    local_high_utilization_element_count_history
    / n_elements
)




# Slip-system index corresponding to maximum |RSS|
local_max_rss_system_history = np.argmax(
    np.abs(local_rss_history),
    axis=-1
)


# ========================================================
# EQUIVALENT VON-MISES STRESS HISTORY
# ========================================================

sigma_hist = local_stress_history

sigma_dev_hist = (
    sigma_hist
    - np.trace(
        sigma_hist,
        axis1=-2,
        axis2=-1
    )[..., None, None] / 3.0
)

local_equiv_stress_history = np.sqrt(
    1.5 *
    np.einsum(
        "...ij,...ij->...",
        sigma_dev_hist,
        sigma_dev_hist
    )
)

# ================================================================
# 21. RUNTIME SUMMARY
# ================================================================

total_runtime = (
    time.time()
    -
    total_start
)


print()
print("=" * 70)
print("RUNTIME SUMMARY")
print("=" * 70)


print(
    f"Total runtime              : "
    f"{total_runtime:.2f} s "
    f"({total_runtime / 3600.0:.2f} h)"
)


print(
    f"CPFEM local solver         : "
    f"{runtime_stats['cp_total']:.2f} s "
    f"({runtime_stats['cp_total']/3600.0:.2f} h)"
)


print(
    f"Global tangent assembly    : "
    f"{runtime_stats['tangent_assembly_total']:.2f} s "
    f"({runtime_stats['tangent_assembly_total']/3600.0:.2f} h)"
)


print(
    f"Global factorization       : "
    f"{runtime_stats['factorization_total']:.2f} s "
    f"({runtime_stats['factorization_total']/3600.0:.2f} h)"
)


print(
    f"Global linear solve        : "
    f"{runtime_stats['linear_solve_total']:.2f} s "
    f"({runtime_stats['linear_solve_total']/3600.0:.2f} h)"
)


print(
    f"Global iterations          : "
    f"{runtime_stats['global_iterations']}"
)


print(
    f"Local CP calls             : "
    f"{runtime_stats['element_cp_calls']:,}"
)


print("=" * 70)


# ================================================================
# 22. FINAL RESULTS
# ================================================================

total_time = (
    time.time()
    -
    total_start
)


final_equiv_plastic_strain = (
    trial_equiv_plastic_strain.copy()
)


final_equiv_stress = (
    trial_equiv_stress.copy()
)


final_max_abs_rss = (
    trial_max_abs_rss.copy()
)


final_max_rss_system = (
    trial_max_rss_system.copy()
)


# ================================================================
# ELEMENT CENTERS
# ================================================================

element_centers = np.mean(
    nodes_um[
        elements
    ],
    axis=1
)


# ================================================================
# GRAIN-LEVEL STATISTICS
# ================================================================

grain_mean_equiv_plastic_strain = np.zeros(
    n_grains,
    dtype=np.float64
)

grain_max_equiv_plastic_strain = np.zeros(
    n_grains,
    dtype=np.float64
)

grain_mean_equiv_stress = np.zeros(
    n_grains,
    dtype=np.float64
)

grain_max_equiv_stress = np.zeros(
    n_grains,
    dtype=np.float64
)

grain_max_abs_rss = np.zeros(
    n_grains,
    dtype=np.float64
)

grain_mean_accumulated_slip = np.zeros(
    n_grains,
    dtype=np.float64
)

grain_max_accumulated_slip = np.zeros(
    n_grains,
    dtype=np.float64
)


for g in range(
    n_grains
):

    mask = (
        element_grain
        ==
        g
    )


    if not np.any(
        mask
    ):

        continue


    grain_mean_equiv_plastic_strain[g] = (
        np.mean(
            final_equiv_plastic_strain[
                mask
            ]
        )
    )


    grain_max_equiv_plastic_strain[g] = (
        np.max(
            final_equiv_plastic_strain[
                mask
            ]
        )
    )


    grain_mean_equiv_stress[g] = (
        np.mean(
            final_equiv_stress[
                mask
            ]
        )
    )


    grain_max_equiv_stress[g] = (
        np.max(
            final_equiv_stress[
                mask
            ]
        )
    )


    grain_max_abs_rss[g] = (
        np.max(
            final_max_abs_rss[
                mask
            ]
        )
    )


    grain_mean_accumulated_slip[g] = (
        np.mean(
            accumulated_slip_committed[
                mask
            ]
        )
    )


    grain_max_accumulated_slip[g] = (
        np.max(
            accumulated_slip_committed[
                mask
            ]
        )
    )


# ================================================================
# CRITICAL GRAIN
# ================================================================

critical_grain_id = np.argmax(
    grain_max_equiv_plastic_strain
)


critical_element_id = np.argmax(
    final_equiv_plastic_strain
)


critical_rss_element_id = np.argmax(
    final_max_abs_rss
)


critical_rss_grain_id = (
    element_grain[
        critical_rss_element_id
    ]
)


critical_rss_slip_system = (
    final_max_rss_system[
        critical_rss_element_id
    ]
)


# ================================================================
# FINAL PRINT
# ================================================================

print()
print("=" * 70)
print("FINAL FE–CPFEM RESULTS")
print("=" * 70)


# print(
#     "Final strain =",
#     global_strain_history[-1]
# )


# print(
#     "Final axial stress =",
#     global_stress_history[-1],
#     "MPa"
# )


# print(
#     "Final mean CRSS =",
#     np.mean(
#         resistance_committed
#     ),
#     "MPa"
# )


# print(
#     "Final maximum RSS =",
#     np.max(
#         final_max_abs_rss
#     ),
#     "MPa"
# )


# print(
#     "Final mean J2 plastic strain =",
#     np.mean(
#         final_equiv_plastic_strain
#     )
# )


# print(
#     "Final mean accumulated slip =",
#     np.mean(
#         accumulated_slip_committed
#     )
# )


# print(
#     "Maximum local J2 plastic strain =",
#     np.max(
#         final_equiv_plastic_strain
#     )
# )


# print(
#     "Maximum local von-Mises stress =",
#     np.max(
#         final_equiv_stress
#     ),
#     "MPa"
# )


# print(
#     "Critical element ID =",
#     critical_element_id
# )


# print(
#     "Critical grain ID =",
#     critical_grain_id
# )


# print(
#     "Critical RSS element ID =",
#     critical_rss_element_id
# )


# print(
#     "Critical RSS grain ID =",
#     critical_rss_grain_id
# )


# print(
#     "Critical RSS slip system =",
#     critical_rss_slip_system
# )


# print(
#     "Total runtime =",
#     total_time,
#     "s"
# )


# ================================================================
# 23. SAVE RESULTS
# ================================================================

np.savez_compressed(

    "316L_FE_CPFEM_optimized_results_local_fields_20%_v2.npz",


    # ------------------------------------------------------------
    # GLOBAL RESPONSE
    # ------------------------------------------------------------

    strain_history=
        global_strain_history,

    stress_history=
        global_stress_history,

    step_history=
        np.asarray(
            step_history
        ),


    # ------------------------------------------------------------
    # FINAL LOCAL FIELDS
    # ------------------------------------------------------------

    element_total_strain=
        trial_total_strain.copy(),

    element_equiv_total_strain=
        trial_equiv_total_strain.copy(),

    element_stress=
        trial_stress.copy(),

    element_stress_voigt=
        trial_stress_voigt.copy(),

    element_equiv_stress=
        trial_equiv_stress.copy(),

    element_plastic_strain=
        plastic_strain_committed.copy(),

    element_equiv_plastic_strain=
        final_equiv_plastic_strain.copy(),

    element_resistance=
        resistance_committed.copy(),

    element_accumulated_slip=
        accumulated_slip_committed.copy(),

    element_accumulated_slip_system=
        accumulated_slip_system_committed.copy(),

    element_rss=
        trial_rss.copy(),

    element_max_abs_rss=
        final_max_abs_rss.copy(),

    element_max_rss_system=
        final_max_rss_system.copy(),

    element_slip_increment=
        cp_x_committed.copy(),


    # ------------------------------------------------------------
    # ELEMENT / GRAIN INFORMATION
    # ------------------------------------------------------------

    element_grain=
        element_grain.copy(),

    element_centers=
        element_centers.copy(),

    nodes=
        nodes_um.copy(),

    elements=
        elements.copy(),

    grain_orientation=
        grain_orientation.copy(),


    # ------------------------------------------------------------
    # GRAIN-LEVEL FINAL STATISTICS
    # ------------------------------------------------------------

    grain_mean_equiv_plastic_strain=
        grain_mean_equiv_plastic_strain,

    grain_max_equiv_plastic_strain=
        grain_max_equiv_plastic_strain,

    grain_mean_equiv_stress=
        grain_mean_equiv_stress,

    grain_max_equiv_stress=
        grain_max_equiv_stress,

    grain_max_abs_rss=
        grain_max_abs_rss,

    grain_mean_accumulated_slip=
        grain_mean_accumulated_slip,

    grain_max_accumulated_slip=
        grain_max_accumulated_slip,


    # ------------------------------------------------------------
    # CRITICAL LOCATIONS
    # ------------------------------------------------------------

    critical_element_id=
        np.int32(
            critical_element_id
        ),

    critical_grain_id=
        np.int32(
            critical_grain_id
        ),

    critical_rss_element_id=
        np.int32(
            critical_rss_element_id
        ),

    critical_rss_grain_id=
        np.int32(
            critical_rss_grain_id
        ),

    critical_rss_slip_system=
        np.int32(
            critical_rss_slip_system
        ),


    # ------------------------------------------------------------
    # COMPLETE CONVERGED LOCAL HISTORY
    # ------------------------------------------------------------

    local_step_history=
        local_field_step_history,

    local_strain_history=
        local_field_strain_history,

    local_total_strain_history=
        local_total_strain_history,

    # local_equiv_total_strain_history=
    #     local_equiv_total_strain_history,

    local_plastic_strain_history=
        local_plastic_strain_history,

    # local_equiv_plastic_strain_history=
    #     local_equiv_plastic_strain_history,

    local_stress_history=
        local_stress_history,

    local_stress_voigt_history=
        local_stress_voigt_history,

    local_equiv_stress_history=
        local_equiv_stress_history,

    local_rss_history=
        local_rss_history,

    local_max_abs_rss_history=
        local_max_abs_rss_history,

    local_max_rss_system_history=
        local_max_rss_system_history,

    local_crss_history=
        local_crss_history,

    local_slip_increment_history=
        local_slip_increment_history,

    local_accumulated_slip_history=
        local_accumulated_slip_history,

    local_accumulated_slip_system_history=
        local_accumulated_slip_system_history
)


# ============================================================
# DERIVED LOCAL HISTORY QUANTITIES
# ============================================================

# ------------------------------------------------------------
# Equivalent von-Mises stress
# ------------------------------------------------------------

sigma_hist = local_stress_history

sigma_dev_hist = (
    sigma_hist
    - np.trace(
        sigma_hist,
        axis1=-2,
        axis2=-1
    )[..., None, None] / 3.0
)

local_equiv_stress_history = np.sqrt(
    1.5 * np.einsum(
        "...ij,...ij->...",
        sigma_dev_hist,
        sigma_dev_hist
    )
)


# ------------------------------------------------------------
# Maximum absolute RSS for each element at each load step
# ------------------------------------------------------------

local_max_abs_rss_history = np.max(
    np.abs(local_rss_history),
    axis=-1
)


# ------------------------------------------------------------
# Slip-system index producing maximum absolute RSS
# ------------------------------------------------------------

local_max_rss_system_history = np.argmax(
    np.abs(local_rss_history),
    axis=-1
)


# ------------------------------------------------------------
# CRSS history
#
# "resistance" is the current CRSS/resistance for
# each slip system.
# ------------------------------------------------------------

local_crss_history = (
    local_resistance_history.copy()
)


# ------------------------------------------------------------
# Slip increment history
#
# "slip" is the current-load-step Δγ for each
# slip system.
# ------------------------------------------------------------

local_slip_increment_history = (
    local_slip_history.copy()
)


print()
print(
    "Results saved as:"
)

print(
    "316L_FE_CPFEM_optimized_results_v03_local_fields.npz"
)


# ================================================================
# 24. STRESS–STRAIN CURVE
# ================================================================

import matplotlib.pyplot as plt


plt.figure(
    figsize=(9, 6)
)


plt.plot(
    global_strain_history * 100.0,
    global_stress_history,
    linewidth=2,
    label="Optimized FE–CPFEM"
)


plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)


plt.ylabel(
    "Axial stress (MPa)",
    fontsize=14
)


plt.title(
    "316L FE–CPFEM Stress–Strain Response",
    fontsize=15
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()

plt.show()



print("\nAvailable cp_state_history keys:")

if len(cp_state_history) > 0:
    print(cp_state_history[0].keys())
    print("\nFirst history entry:")
    print(cp_state_history[0])
else:
    print("cp_state_history is EMPTY")







# ================================================================
# 25. GLOBAL CONVERGENCE
# ================================================================

global_iteration_history = np.asarray(
    [
        x["global_iterations"]
        for x in cp_state_history
    ]
)


# relative_residual_history = np.asarray(
#     [
#         x["relative_residual"]
#         for x in cp_state_history
#     ]
# )


relative_residual_history = np.asarray(
    [
        x.get("relative_residual", np.nan)
        for x in cp_state_history
    ],
    dtype=float
)







plt.figure(
    figsize=(9, 6)
)


plt.semilogy(
    global_strain_history * 100.0,
    relative_residual_history,
    linewidth=2
)


plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)


plt.ylabel(
    "Global relative residual",
    fontsize=14
)


plt.title(
    "Global FE–CPFEM Convergence",
    fontsize=15
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()

plt.show()


# ================================================================
# 26. GLOBAL ITERATION HISTORY
# ================================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    global_strain_history * 100.0,
    global_iteration_history,
    marker="o",
    linewidth=1.5
)


plt.xlabel(
    "Axial strain (%)",
    fontsize=14
)


plt.ylabel(
    "Global Newton iterations",
    fontsize=14
)


plt.title(
    "Global Iteration Count",
    fontsize=15
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()

plt.show()




# ================================================================
# FINAL LOCAL FIELD CALCULATION
# ================================================================

print()
print("=" * 70)
print("FINAL LOCAL FIELD CALCULATION")
print("=" * 70)

# ------------------------------------------------
# Final total strain
# ------------------------------------------------

# element_total_strain_voigt = trial_total_strain.copy()


# ------------------------------------------------
# Retrieve final converged local fields
# ------------------------------------------------

final_local_state = local_field_history[-1]

element_total_strain_voigt = (
    final_local_state["total_strain_voigt"].copy()
)

element_stress_voigt = (
    final_local_state["stress_voigt"].copy()
)

element_stress = (
    final_local_state["stress"].copy()
)

element_plastic_strain = (
    final_local_state["plastic_strain"].copy()
)

element_rss = (
    final_local_state["rss"].copy()
)

element_accumulated_slip = (
    final_local_state["accumulated_slip"].copy()
)


# # ================================================================
# # DEBUG: INSPECT FINAL LOCAL STRESS TENSORS
# # ================================================================

# print()
# print("=" * 70)
# print("DEBUG — FINAL LOCAL STRESS TENSORS")
# print("=" * 70)

# for test_e in [0, 832, 1623, 6371]:

#     print()
#     print(f"Element {test_e}")
#     print(f"Grain   {element_grain[test_e]}")

#     print("Stress tensor [MPa]:")
#     print(
#         element_stress[test_e]
#     )

#     print("Stress Voigt [MPa]:")
#     print(
#         element_stress_voigt[test_e]
#     )

#     print(
#         f"Von-Mises = "
#         f"{element_von_mises[test_e]:.6f} MPa"
#     )

#     print(
#         f"Max |RSS| = "
#         f"{np.max(np.abs(element_rss[test_e])):.6f} MPa"
#     )

sigma = element_stress

sigma_dev = (
    sigma
    - np.trace(
        sigma,
        axis1=1,
        axis2=2
    )[:, None, None] / 3.0
)

element_von_mises = np.sqrt(
    1.5 *
    np.einsum(
        "eij,eij->e",
        sigma_dev,
        sigma_dev
    )
)






element_total_strain = np.zeros(
    (n_elements, 3, 3),
    dtype=np.float64
)

for e in range(n_elements):
    element_total_strain[e] = voigt_to_tensor(
        element_total_strain_voigt[e]
    )


# ------------------------------------------------
# Final plastic strain
# ------------------------------------------------

# element_plastic_strain = (
#     plastic_strain_committed.copy()
# )


# ------------------------------------------------
# J2 equivalent plastic strain
# ------------------------------------------------

eps_p = element_plastic_strain

eps_p_dev = (
    eps_p
    - np.trace(
        eps_p,
        axis1=1,
        axis2=2
    )[:, None, None] / 3.0
)

element_equivalent_plastic_strain = np.sqrt(
    (2.0 / 3.0)
    * np.einsum(
        "eij,eij->e",
        eps_p_dev,
        eps_p_dev
    )
)


# ------------------------------------------------
# Final stress
# ------------------------------------------------

# element_stress = trial_stress.copy()

# element_stress_voigt = trial_stress_voigt.copy()


# ------------------------------------------------
# Von-Mises stress
# ------------------------------------------------

sigma = element_stress

sigma_dev = (
    sigma
    - np.trace(
        sigma,
        axis1=1,
        axis2=2
    )[:, None, None] / 3.0
)

element_von_mises = np.sqrt(
    1.5
    * np.einsum(
        "eij,eij->e",
        sigma_dev,
        sigma_dev
    )
)


# ------------------------------------------------
# Final RSS
# ------------------------------------------------

# element_rss = trial_rss.copy()

element_max_abs_rss = np.max(
    np.abs(element_rss),
    axis=1
)

element_max_rss_system = np.argmax(
    np.abs(element_rss),
    axis=1
)


# ------------------------------------------------
# Critical elements
# ------------------------------------------------

critical_element_id = int(
    np.argmax(
        element_equivalent_plastic_strain
    )
)

critical_grain_id = int(
    element_grain[critical_element_id]
)

critical_rss_element_id = int(
    np.argmax(element_max_abs_rss)
)

critical_rss_grain_id = int(
    element_grain[critical_rss_element_id]
)

critical_rss_slip_system = int(
    element_max_rss_system[
        critical_rss_element_id
    ]
)


# ================================================================
# DEBUG: CRITICAL RSS ELEMENT
# ================================================================

print()
print("=" * 70)
print("CRITICAL RSS ELEMENT")
print("=" * 70)

e = critical_rss_element_id

print(f"Element ID = {e}")
print(f"Grain ID   = {element_grain[e]}")

print("Stress tensor [MPa]:")
print(element_stress[e])

print("Stress Voigt [MPa]:")
print(element_stress_voigt[e])

print("RSS for all 12 slip systems [MPa]:")
print(element_rss[e])

print(
    f"Maximum |RSS| = "
    f"{np.max(np.abs(element_rss[e])):.6f} MPa"
)

print(
    f"Critical slip system = "
    f"{np.argmax(np.abs(element_rss[e]))}"
)

print(
    f"Von-Mises stress = "
    f"{element_von_mises[e]:.6f} MPa"
)





# # ------------------------------------------------
# # Global local-field statistics
# # ------------------------------------------------

# final_mean_j2_plastic_strain = float(
#     np.mean(element_equivalent_plastic_strain)
# )

# final_max_j2_plastic_strain = float(
#     np.max(element_equivalent_plastic_strain)
# )

# final_mean_von_mises = float(
#     np.mean(element_von_mises)
# )

# final_max_von_mises = float(
#     np.max(element_von_mises)
# )

# final_max_rss = float(
#     np.max(np.abs(element_rss))
# )

# final_mean_accumulated_slip = float(
#     np.mean(accumulated_slip_committed)
# )

# print(
#     f"Mean J2 plastic strain      = "
#     f"{final_mean_j2_plastic_strain:.8e}"
# )

# print(
#     f"Maximum J2 plastic strain   = "
#     f"{final_max_j2_plastic_strain:.8e}"
# )

# print(
#     f"Mean von-Mises stress       = "
#     f"{final_mean_von_mises:.6f} MPa"
# )

# print(
#     f"Maximum von-Mises stress    = "
#     f"{final_max_von_mises:.6f} MPa"
# )

# print(
#     f"Maximum |RSS|               = "
#     f"{final_max_rss:.6f} MPa"
# )

# print(
#     f"Critical element            = "
#     f"{critical_element_id}"
# )

# print(
#     f"Critical grain              = "
#     f"{critical_grain_id}"
# )

# print(
#     f"Critical RSS element        = "
#     f"{critical_rss_element_id}"
# )

# print(
#     f"Critical RSS grain          = "
#     f"{critical_rss_grain_id}"
# )

# print(
#     f"Critical RSS slip system    = "
#     f"{critical_rss_slip_system}"
# )









# # ================================================================
# # 27. FINAL SUMMARY
# # ================================================================

# print()
# print("=" * 70)
# print("MODULE 5B-OPTIMIZED SUMMARY")
# print("=" * 70)


# print(
#     f"Elements             = {n_elements}"
# )


# print(
#     f"Nodes                = {n_nodes}"
# )


# print(
#     f"DOFs                 = {NDOF}"
# )


# print(
#     f"Grains               = {n_grains}"
# )


# print(
#     f"Final strain         = "
#     f"{global_strain_history[-1]:.8e}"
# )


# print(
#     f"Final axial stress   = "
#     f"{global_stress_history[-1]:.6f} MPa"
# )


# print(
#     f"Total load steps     = "
#     f"{len(global_strain_history)}"
# )


# print(
#     f"Maximum local J2 plastic strain = "
#     f"{np.max(final_equiv_plastic_strain):.8e}"
# )


# print(
#     f"Maximum local von-Mises stress = "
#     f"{np.max(final_equiv_stress):.6f} MPa"
# )


# print(
#     f"Critical grain      = "
#     f"{critical_grain_id}"
# )


# print(
#     f"Critical element    = "
#     f"{critical_element_id}"
# )


# print(
#     f"Critical RSS grain  = "
#     f"{critical_rss_grain_id}"
# )


# print(
#     f"Critical slip system = "
#     f"{critical_rss_slip_system}"
# )


# print(
#     f"Total runtime        = "
#     f"{total_time:.3f} s"
# )


# print("=" * 70)





# ------------------------------------------------
# Global local-field statistics
# ------------------------------------------------

final_mean_j2_plastic_strain = float(
    np.mean(element_equivalent_plastic_strain)
)

final_max_j2_plastic_strain = float(
    np.max(element_equivalent_plastic_strain)
)

final_mean_von_mises = float(
    np.mean(element_von_mises)
)

final_max_von_mises = float(
    np.max(element_von_mises)
)

final_max_rss = float(
    np.max(np.abs(element_rss))
)

final_mean_accumulated_slip = float(
    np.mean(element_accumulated_slip)
)

final_max_accumulated_slip = float(
    np.max(element_accumulated_slip)
)


# ------------------------------------------------
# Critical locations
# ------------------------------------------------

critical_element_id = int(
    np.argmax(element_equivalent_plastic_strain)
)

critical_grain_id = int(
    element_grain[critical_element_id]
)

critical_von_mises_element_id = int(
    np.argmax(element_von_mises)
)

critical_von_mises_grain_id = int(
    element_grain[critical_von_mises_element_id]
)

critical_rss_element_id = int(
    np.argmax(np.max(np.abs(element_rss), axis=1))
)

critical_rss_grain_id = int(
    element_grain[critical_rss_element_id]
)

critical_rss_slip_system = int(
    np.argmax(
        np.abs(element_rss[critical_rss_element_id])
    )
)


# ------------------------------------------------
# Print local-field statistics
# ------------------------------------------------

print(
    f"Mean J2 plastic strain      = "
    f"{final_mean_j2_plastic_strain:.8e}"
)

print(
    f"Maximum J2 plastic strain   = "
    f"{final_max_j2_plastic_strain:.8e}"
)

print(
    f"Mean von-Mises stress       = "
    f"{final_mean_von_mises:.6f} MPa"
)

print(
    f"Maximum von-Mises stress    = "
    f"{final_max_von_mises:.6f} MPa"
)

print(
    f"Maximum |RSS|               = "
    f"{final_max_rss:.6f} MPa"
)

print(
    f"Mean accumulated slip       = "
    f"{final_mean_accumulated_slip:.8e}"
)

print(
    f"Maximum accumulated slip    = "
    f"{final_max_accumulated_slip:.8e}"
)

print(
    f"Critical plastic element    = "
    f"{critical_element_id}"
)

print(
    f"Critical plastic grain      = "
    f"{critical_grain_id}"
)

print(
    f"Critical von-Mises element  = "
    f"{critical_von_mises_element_id}"
)

print(
    f"Critical von-Mises grain    = "
    f"{critical_von_mises_grain_id}"
)

print(
    f"Critical RSS element        = "
    f"{critical_rss_element_id}"
)

print(
    f"Critical RSS grain          = "
    f"{critical_rss_grain_id}"
)

print(
    f"Critical RSS slip system    = "
    f"{critical_rss_slip_system}"
)


# ================================================================
# 27. FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("MODULE 5B-OPTIMIZED SUMMARY")
print("=" * 70)

print(
    f"Elements             = {n_elements}"
)

print(
    f"Nodes                = {n_nodes}"
)

print(
    f"DOFs                 = {NDOF}"
)

print(
    f"Grains               = {n_grains}"
)

print(
    f"Final strain         = "
    f"{global_strain_history[-1]:.8e}"
)

print(
    f"Final axial stress   = "
    f"{global_stress_history[-1]:.6f} MPa"
)

print(
    f"Total load steps     = "
    f"{len(global_strain_history)}"
)

print(
    f"Maximum local J2 plastic strain = "
    f"{final_max_j2_plastic_strain:.8e}"
)

print(
    f"Maximum local von-Mises stress = "
    f"{final_max_von_mises:.6f} MPa"
)

print(
    f"Maximum local |RSS|            = "
    f"{final_max_rss:.6f} MPa"
)

print(
    f"Maximum local accumulated slip = "
    f"{final_max_accumulated_slip:.8e}"
)

print(
    f"Critical plastic grain         = "
    f"{critical_grain_id}"
)

print(
    f"Critical plastic element       = "
    f"{critical_element_id}"
)

print(
    f"Critical von-Mises grain       = "
    f"{critical_von_mises_grain_id}"
)

print(
    f"Critical von-Mises element     = "
    f"{critical_von_mises_element_id}"
)

print(
    f"Critical RSS grain             = "
    f"{critical_rss_grain_id}"
)

print(
    f"Critical RSS slip system       = "
    f"{critical_rss_slip_system}"
)

print(
    f"Total runtime                  = "
    f"{total_time:.3f} s"
)

print("=" * 70)


