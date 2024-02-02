import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from firedrake import File
import firedrake as fire
import spyro


def check_with_non_cell_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm, plot=False):
    steps = [1e-3, 1e-4, 1e-5]  # step length

    errors = []
    V_c = Wave_obj_guess.function_space
    dm = fire.Function(V_c)
    size, = np.shape(dm.dat.data[:])
    dm_data = np.random.rand(size)
    dm.dat.data[:] = dm_data
    # dm.assign(dJ)

    for step in steps:

        Wave_obj_guess.reset_pressure()
        c_guess = fire.Constant(2.0) + step*dm
        Wave_obj_guess.initial_velocity_model = c_guess
        Wave_obj_guess.forward_solve()
        misfit_plusdm = rec_out_exact - Wave_obj_guess.receivers_output
        J_plusdm = compute_functional(Wave_obj_guess, misfit_plusdm)

        grad_fd = (J_plusdm - Jm) / (step)
        projnorm = fire.assemble(dJ * dm * fire.dx(scheme=Wave_obj_guess.quadrature_rule))

        error = 100 * ((grad_fd - projnorm) / projnorm)

        errors.append(error)
        # print(f"Error : {error}")
        # step /= 2

    # all errors less than 1 %
    errors = np.array(errors)
    # Checking if error is first order in step
    theory = [t for t in steps]
    theory = [errors[0] * th / theory[0] for th in theory]
    if plot:
        plt.close()
        plt.plot(steps, errors, label="Error")
        plt.plot(steps, theory, "--",label="first order")
        plt.legend()
        plt.title(" Adjoint gradient versus finite difference gradient")
        plt.xlabel("Step")
        plt.ylabel("Error \%")
        plt.savefig("gradient_error_verification.png")
        plt.close()

    assert math.isclose(np.log(theory[-1]), np.log(errors[-1]), rel_tol=1e-1)


def compute_functional(Wave_object, residual):
    """Compute the functional to be optimized.
    Accepts the velocity optionally and uses
    it if regularization is enabled
    """
    num_receivers = Wave_object.number_of_receivers
    dt = Wave_object.dt
    tf = Wave_object.final_time
    # nt = int(tf / dt) + 1  # number of timesteps

    # J = np.zeros((num_receivers))
    # for ti in range(nt):
    #     for rn in range(num_receivers):
    #         first_integral[ti] += residual[ti][rn] ** 2
    J = 0
    for rn in range(num_receivers):
        J += np.trapz(residual[:, rn] ** 2, dx=dt)

    J *= 0.5
    return J


def compare_with_devito(
        spyro_receivers,
        devito_receivers_filename,
        output_filename="test",
        devito_dx=5,
        title="Receiver",
):

    plt.close()

    devito_true_rec = np.load(devito_receivers_filename)
    nt_devito, nreceivers_devito = np.shape(devito_true_rec)
    devito_timevector = np.linspace(0.0, 1.6, nt_devito)
    nt_spyro, nreceivers_spyro = np.shape(spyro_receivers)
    spyro_timevector = np.linspace(0.0, 1.6, nt_spyro)

    amp_factor = devito_dx**2

    if nreceivers_spyro != nreceivers_devito:
        return ValueError("Receiver count does not match between spyro and devito.")

    for i in range(nreceivers_spyro):
        plot_title = title + str(i)
        filename = output_filename + str(i) + ".png"
        plt.plot(devito_timevector, devito_true_rec[:, i], label="devito")
        plt.plot(spyro_timevector, amp_factor*spyro_receivers[:, i], label="spyro")
        plt.legend()
        plt.title(plot_title)
        plt.savefig(filename)
        plt.close()


final_time = 1.0

dictionary = {}
dictionary["options"] = {
    "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
dictionary["mesh"] = {
    "Lz": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 3.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-1.1, 1.5)],
    "frequency": 5.0,
    # "delay": 1.2227264394269568,
    # "delay_type": "time",
    "delay": 1.5,
    "delay_type": "multiples_of_minimun",
    "receiver_locations": spyro.create_transect((-1.8, 1.2), (-1.8, 1.8), 10),
    # "receiver_locations": [(-2.0, 2.5) , (-2.3, 2.5), (-3.0, 2.5), (-3.5, 2.5)],
}

# Simulate for 2.0 seconds.
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
}
dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": True,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": True,
}


def get_forward_model(load_true=False):
    if load_true is False:
        Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
        Wave_obj_exact.set_mesh(mesh_parameters={"dx": 0.1})
        # Wave_obj_exact.set_initial_velocity_model(constant=3.0)
        cond = fire.conditional(Wave_obj_exact.mesh_z > -2.5, 1.5, 3.5)
        Wave_obj_exact.set_initial_velocity_model(
            conditional=cond,
            # output=True
        )
        spyro.plots.plot_model(Wave_obj_exact, abc_points=[(-1, 1), (-2, 1), (-2, 4), (-1, 2)])
        Wave_obj_exact.forward_solve()
        # forward_solution_exact = Wave_obj_exact.forward_solution
        rec_out_exact = Wave_obj_exact.receivers_output
        np.save("rec_out_exact", rec_out_exact)

    else:
        rec_out_exact = np.load("rec_out_exact.npy")

    # compare_with_devito(rec_out_exact, "devito_true_rec.npy")

    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(mesh_parameters={"dx": 0.1})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)
    Wave_obj_guess.forward_solve()
    # forward_solution = Wave_obj_guess.forward_solution
    # forward_solution_guess = deepcopy(forward_solution)
    rec_out_guess = Wave_obj_guess.receivers_output

    return rec_out_exact, rec_out_guess, Wave_obj_guess


def test_gradient():
    # beginning of debugging variables
    num_recvs = 100
    dt = 0.0005
    tf = final_time
    show = True
    vabs = 1e-2
    timevector = np.linspace(0.0, tf, 2001)
    # end of debugging variables

    rec_out_exact, rec_out_guess, Wave_obj_guess = get_forward_model(load_true=False)
    forward_solution = Wave_obj_guess.forward_solution
    forward_solution_guess = deepcopy(forward_solution)

    misfit = rec_out_exact - rec_out_guess

    Jm = compute_functional(Wave_obj_guess, misfit)
    print(f"Cost functional : {Jm}")

    # compute the gradient of the control (to be verified)
    dJ = Wave_obj_guess.gradient_solve(misfit=misfit, forward_solution=forward_solution_guess)
    File("gradient.pvd").write(dJ)
    gradient = dJ.dat.data[:]

    check_with_non_cell_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm)

    print("END")


if __name__ == "__main__":
    test_gradient()
