## runnig by mpiexec -n 8 python m2d_spyro.py 

from firedrake import File
import spyro
import time

number_sou = 3
number_rec = 101
sou = spyro.create_transect((0, 0), (0, 15.4), 3) # in km
rev = spyro.create_transect((0, 0), (0, 15.4), 101)  # in km

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 5,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "spatial",
}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 10.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "/Users/yw11823/ACSE/irp/spyro/FWI_2D_DATA/meshes/marmousi_exact.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "/Users/yw11823/ACSE/irp/spyro/FWI_2D_DATA/velocity_models/marmousi_exact.hdf5",
}
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  # None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": number_sou,
    "source_pos": sou,
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": number_rec,
    "receiver_locations": rev,
}
model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 2.00,  # Final time for event
    "dt": 0.00011799,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 10,  # how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}

comm = spyro.utils.mpi_init(model)
mesh, V = spyro.io.read_mesh(model, comm)
vp = spyro.io.interpolate(model, mesh, V, guess=False)
if comm.ensemble_comm.rank == 0:
    File("true_velocity.pvd", comm=comm.comm).write(vp)
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)

d0=model["timeaxis"]["t0"]
dt=model["timeaxis"]["dt"]
tf=model["timeaxis"]["tf"]
freq=model["acquisition"]["frequency"]
wavelet = spyro.full_ricker_wavelet(dt=dt,tf=tf,freq=freq,)

start_time = time.time()
p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers, source_num = 2)
end_time = time.time()
running_time = end_time - start_time
print("Running Time: {:.2f} seconds".format(running_time))

# Plot the shot record using the function written in Spyro
spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-5, vmax=1e-5, file_name="M2d_true")
# spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)
spyro.io.save_shots(model, comm, p_r, file_name="M2d_true_")


spyro.io.save_shots(model, comm, p, file_name="M2d_true_pressure_")