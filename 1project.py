import sys
sys.path.append("/Users/yw11823/ACSE/irp/spyro")

import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from mpi4py import MPI
import psutil

import spyro

sou_pos = [(0, 0.5)]
rec_pos = spyro.create_transect((1.0, 0.0), (1.0, 1.0), 980)

# t0 =  0.0, 
# tf = 2.0, 
# dt = 5e-4,
# freq = 8.0, 

model = {}
model["opts"] = {
    "method": "KMV", 
    "quadrature": "KMV", 
    "degree": 1, 
    "dimension": 2, 
}
model["parallelism"] = {
    "type": "spatial", 
}
model["mesh"] = {
    "Lz": 1.0, 
    "Lx": 1.0, 
    "Ly": 0.0, 
    "meshfile": "not_used.msh", 
    "initmodel": "not_used.hdf5", 
    "truemodel": "not_used.hdf5", 
}
model["BCs"] = {
    "status": True, 
    "outer_bc": "non-reflective", 
    "damping_type": "polynomial", 
    "exponent": 2, 
    "cmax": 4.5, 
    "R": 1e-6, 
    "lz": 0.25, 
    "lx": 0.25,  
    "ly": 0.0, 
}
model["acquisition"] = {
    "source_type": "Ricker", 
    "source_pos": sou_pos, 
    "frequency": 8.0, 
    "delay": 0.1, 
    "receiver_locations": rec_pos, 
}
model["timeaxis"] = {
    "t0": 0.0, 
    "tf": 2.0, 
    "dt": 5e-4, 
    "amplitude": 1, 
    "nspool": 100, 
    "fspool": 100, 
}

# mesh = UnitSquareMesh(100, 100, 1.0, 1.0)
mesh = RectangleMesh(100, 100, 1.0, 1.0)
# mesh.coordinates.dat.data[:, 0] -= 0.25 将x坐标向左平移0.25个单位
# mesh.coordinates.dat.data[:, 1] -= 1.25 将y坐标向下平移1.25个单位

comm = spyro.utils.mpi_init(model)
element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
V = FunctionSpace(mesh, element)

# Create a simple two-layer seismic velocity model `vp`.
x, y = SpatialCoordinate(mesh)
radius = 0.3
center_x, center_y = 0.5, 0.5
condition = (x - center_x)**2 + (y - center_y)**2 < radius**2
velocity = conditional(condition, 3.0, 2.5)
vp = Function(V, name="velocity").interpolate(velocity)
File("true_velocity_model_circle.pvd").write(vp)

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)

dt=model["timeaxis"]["dt"]
tf=model["timeaxis"]["tf"]
freq=model["acquisition"]["frequency"]
wavelet = spyro.full_ricker_wavelet( dt= dt, tf = tf, freq= freq)

p_field, p_at_recv = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers)
spyro.plots.plot_shots(model, comm, p_at_recv)
spyro.io.save_shots(model, comm, p_at_recv)

outdir = "out/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Create a simple two-layer seismic velocity model `vp`.
x, y = SpatialCoordinate(mesh)
radius = 0.3
center_x, center_y = 0.5, 0.5
condition = (x - center_x)**2 + (y - center_y)**2 < radius**2
velocity = conditional(condition, 2.5, 2.5)
vp = Function(V, name="velocity").interpolate(velocity)
File("initial_velocity_model_circle.pvd").write(vp)

# Set up file output for control and gradient fields if on the master process, and define a lumped quadrature rule for integration.
if comm.ensemble_comm.rank == 0:
    control_file = File(outdir + "control.pvd", comm=comm.comm)
    grad_file = File(outdir + "grad.pvd", comm=comm.comm)
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
dxlump = dx(scheme=quad_rule)

class L2Inner(object):
    def __init__(self):
        self.A = assemble(TrialFunction(V) * TestFunction(V) * dxlump, mat_type="matfree")
        self.Ap = as_backend_type(self.A).mat()

    def eval(self, _u, _v):
        upet = as_backend_type(_u).vec()
        vpet = as_backend_type(_v).vec()
        A_u = self.Ap.createVecLeft()
        self.Ap.mult(upet, A_u)
        return vpet.dot(A_u)

def regularize_gradient(vp, dJ):
    m_u = TrialFunction(V)
    m_v = TestFunction(V)
    mgrad = m_u * m_v * dx(scheme=quad_rule)
    ffG = dot(grad(vp), grad(m_v)) * dx(scheme=quad_rule)
    G = mgrad - ffG
    lhsG, rhsG = lhs(G), rhs(G)
    gradreg = Function(V)
    grad_prob = LinearVariationalProblem(lhsG, rhsG, gradreg)
    grad_solver = LinearVariationalSolver(
        grad_prob,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "jacobi",
            "mat_type": "matfree",
        },
    )
    grad_solver.solve()
    dJ += gradreg
    return dJ

class Objective(ROL.Objective):
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.p_guess = None
        self.misfit = 0.0
        self.p_exact_recv = spyro.io.load_shots(model, comm)

    def value(self, x, tol):
        J_total = np.zeros((1))
        self.p_guess, p_guess_recv = spyro.solvers.forward(
            model,
            mesh,
            comm,
            vp,
            sources,
            wavelet,
            receivers,
        )
        self.misfit = spyro.utils.evaluate_misfit(model, p_guess_recv, self.p_exact_recv)
        J_total[0] += spyro.utils.compute_functional(model, self.misfit, velocity=vp)
        J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
        J_total[0] /= comm.ensemble_comm.size
        if comm.comm.size > 1:
            J_total[0] /= comm.comm.size
        return J_total[0]

    def gradient(self, g, x, tol):
        dJ = Function(V, name="gradient")
        dJ_local = spyro.solvers.gradient(
            model,
            mesh,
            comm,
            vp,
            receivers,
            self.p_guess,
            self.misfit,
        )
        if comm.ensemble_comm.size > 1:
            comm.allreduce(dJ_local, dJ)
        else:
            dJ = dJ_local
        dJ /= comm.ensemble_comm.size
        if comm.comm.size > 1:
            dJ /= comm.comm.size
        if "regularization" in model["opts"] and model["opts"]["regularization"]:
            dJ = regularize_gradient(vp, dJ)
        if comm.ensemble_comm.rank == 0:
            grad_file.write(dJ)
        g.scale(0)
        g.vec += dJ

    def update(self, x, flag, iteration):
        vp.assign(Function(V, x.vec, name="velocity"))
        if iteration >= 0:
            if comm.ensemble_comm.rank == 0:
                control_file.write(vp)

paramsDict = {
    "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
    "Step": {
        "Type": "Augmented Lagrangian",
        "Augmented Lagrangian": {
            "Subproblem Step Type": "Line Search",
            "Subproblem Iteration Limit": 5,
        },
        "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
    },
    "Status Test": {
        "Gradient Tolerance": 1e-15,
        "Iteration Limit": 100,
        "Step Tolerance": 1e-15,
    },
}

params = ROL.ParameterList(paramsDict, "Parameters")

inner_product = L2Inner()

obj = Objective(inner_product)

u = Function(V, name="velocity").assign(vp)

opt = FeVector(u.vector(), inner_product)

# algo = ROL.Algorithm("Line Search", params)

algo = ROL.LinMoreAlgorithm(params)

algo.run(opt, obj)

if comm.ensemble_comm.rank == 0:
    File("res.pvd", comm=comm.comm).write(vp)