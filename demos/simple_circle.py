from firedrake import (
    RectangleMesh,
    FunctionSpace,
    Function,
    SpatialCoordinate,
    conditional,
    File,
)

import numpy as np
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from mpi4py import MPI
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import spyro

# 创建文件夹
import os
os.makedirs('shots', exist_ok=True)

model = {}

# Choose method and parameters
model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV", # Equi or KMV
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML.
model["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

# Specify a 250-m PML on the three sides of the domain.
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  # None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "source_pos":  [(0.02, 0.5)],
    "frequency": 8.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect((0.98, 0.0), (0.98, 1.0), 101),
}

# Simulate for 2.0 seconds.
model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 1.00,  # Final time for event
    "dt": 0.00004082,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}

# Create a simple mesh of a rectangle ∈ [1 x 1] km
mesh = RectangleMesh(100, 100, 1.0, 1.0)
# mesh = UnitSquareMesh(200, 200)
comm = spyro.utils.mpi_init(model)
element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
V = FunctionSpace(mesh, element)

# Create a simple two-layer seismic velocity model `vp`.
x, y = SpatialCoordinate(mesh)
radius = 0.15
center_x, center_y = 0.5, 0.5
condition = (x - center_x)**2 + (y - center_y)**2 < radius**2
velocity = conditional(condition, 3.0, 2.5)
vp = Function(V, name="velocity").interpolate(velocity)
File("ex1_tria_mesh.pvd").write(vp)

# Use the smoothed model for FWI
# Now we instantiate both the receivers and source objects.
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)

# dt=model["timeaxis"]["dt"]
# tf=model["timeaxis"]["tf"]
# Create a wavelet to force the simulation
wavelet = spyro.full_ricker_wavelet(model["timeaxis"]["dt"], tf=1, freq=10.0)

# 模拟波场
# Calculate running time
start_time = time.time()
p_field, p_at_recv = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers)
end_time = time.time()
running_time = end_time - start_time
print("Running Time: {:.2f} seconds".format(running_time))

# # Visualize the shot record
# # 在地震波模拟中生成接收器记录（shot records）的可视化图像
# spyro.plots.plot_shots(model, comm, p_at_recv, file_name="shot_record")

# # Save the shot (a Numpy array) as a pickle for other use.
# spyro.io.save_shots(model, comm, p_at_recv, file_name="rec_")
# spyro.io.save_shots(model, comm, p_field, file_name="p_field_")

# # Load the shot
# my_shot = spyro.io.load_shots(model, comm)