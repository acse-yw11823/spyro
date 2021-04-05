import math

import numpy as np
from firedrake import *
from scipy.signal import butter, filtfilt


class Sources:
    """Methods that inject a wavelet into a mesh"""

    def __init__(self, model, mesh, V, comm):
        """Create injection operator(s) to excite source(s).

        Parameters
        ----------
        model: `dictionary`
            Contains simulation parameters and options.
        mesh: a Firedrake.mesh
            2D/3D simplicial mesh read in by Firedrake.Mesh
        V: Firedrake.FunctionSpace object
            The space of the finite elements
        comm: Firedrake.ensemble_communicator
            An ensemble communicator


        Returns
        -------
        Sources: :class:`Source` object
            Contains a list of excitations

        """

        self.source_type = model["acquisition"]["source_type"]
        self.num_sources = model["acquisition"]["num_sources"]
        self.pos = model["acquisition"]["source_pos"]
        self.dimension = model["opts"]["dimension"]
        self.mesh = mesh
        self.V = V
        self.my_ensemble = comm

        @property
        def num_sources(self):
            return self.__num_sources

        @num_sources.setter
        def num_sources(self, value):
            assert value > 0, "Number of sources must be > 0"
            self.__num_sources = value

        @property
        def pos(self):
            return self.__pos

        @pos.setter
        def pos(self, value):
            self.__pos = value

    def create(self):
        """ Create injection operator(s) to excite source(s)."""

        V = self.V

        if self.dimension == 2:
            z, x = SpatialCoordinate(self.mesh)
        elif self.dimension == 3:
            z, x, y = SpatialCoordinate(self.mesh)

        if self.dimension == 2:
            source = Constant([0, 0])
            if self.source_type == "Ricker":
                delta = Interpolator(delta_expr(source, z, x), V)
            elif self.source_type == "MMS":
                delta = Interpolator(MMS_space(source, z, x), V)
            else:
                raise ValueError("Unrecognized source type")
        elif self.dimension == 3:
            source = Constant([0, 0, 0])
            if self.source_type == "Ricker":
                delta = Interpolator(delta_expr_3d(source, z, x, y), V)
            elif self.source_type == "MMS":
                delta = Interpolator(MMS_space_3d(source, z, x, y), V)
        else:
            raise ValueError("Incorrect dimension")

        excitations = []
        for x0 in self.pos:
            source.assign(x0)
            excitations.append(Function(delta.interpolate()))

        return excitations


def timedependentSource(model, t, freq=None, amp=1, delay=1.5):
    if model["acquisition"]["source_type"] == "Ricker":
        return RickerWavelet(t, freq, amp, delay=delay)
    elif model["acquisition"]["source_type"] == "MMS":
        return MMS_time(t)
    else:
        raise ValueError("source not implemented")


def RickerWavelet(t, freq, amp=1.0, delay=1.5):
    """Creates a Ricker source function with a
    delay in term of multiples of the distance
    between the minimums.
    """
    t = t - delay * math.sqrt(6.0) / (math.pi * freq)
    return (
        amp
        * (1.0 - (1.0 / 2.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t)
        * math.exp(
            (-1.0 / 4.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t
        )
    )


def MMS_time(t):
    return 2 * t + 2 * math.pi ** 2 * t ** 3 / 3.0


def MMS_space(x0, z, x):
    """ Mesh variable part of the MMS """
    return sin(pi * z) * sin(pi * x) * Constant(1.0)


def MMS_space_3d(x0, z, x, y):
    """ Mesh variable part of the MMS """
    return sin(pi * z) * sin(pi * x) * sin(pi * y) * Constant(1.0)


def FullRickerWavelet(dt, tf, freq, amp=1.0, cutoff=None):
    """Compute the full Ricker wavelet and apply low-pass filtering
    using cutoff frequency in hz
    """
    nt = int(tf / dt)  # number of timesteps
    time = 0.0
    FullWavelet = np.zeros((nt,))
    for t in range(nt):
        FullWavelet[t] = RickerWavelet(time, freq, amp)
        time += dt
    if cutoff is not None:
        fs = 1.0 / dt
        order = 2
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        FullWavelet = filtfilt(b, a, FullWavelet)
    return FullWavelet

def source_dof_finder(space,  model):

    # getting 1 source position
    source_positions = model["acquisition"]['source_pos']
    if len(source_positions) != 1:
        raise ValueError('Not yet implemented for more then 1 source.')
    
    mesh = space.mesh()
    source_z , source_x = source_positions[0]

    # Getting mesh coordinates
    z, x = SpatialCoordinate(mesh)
    ux = Function(space).interpolate(x)
    uz = Function(space).interpolate(z)
    datax = ux.dat.data_ro_with_halos[:]
    dataz = uz.dat.data_ro_with_halos[:]
    node_locations = np.zeros((len(datax), 2))
    node_locations[:, 0] = dataz
    node_locations[:, 1] = datax

    # generating cell node map
    fdrake_cell_node_map = space.cell_node_map()
    cell_node_map = fdrake_cell_node_map.values_with_halo

    # finding cell where the source is located
    cell_id = mesh.locate_cell( [source_z, source_x], tolerance = 0.01 )

    # finding dof where the source is located
    for dof in cell_node_map[cell_id]:
        if np.isclose(dataz[dof], source_z, rtol = 1e-8) and np.isclose(datax[dof], source_x, rtol = 1e-8):
            model['acquisition']['source_point_dof'] = dof

    if model['acquisition']['source_point_dof'] == False:
        raise ValueError('Finding source dof failed')
    print("test")
    return False



def delta_expr(x0, z, x, sigma_x=500.0):
    sigma_x = Constant(sigma_x)
    return exp(-sigma_x * ((z - x0[0]) ** 2 + (x - x0[1]) ** 2))


def delta_expr_3d(x0, z, x, y, sigma_x=2000.0):
    sigma_x = Constant(sigma_x)
    return exp(-sigma_x * ((z - x0[0]) ** 2 + (x - x0[1]) ** 2 + (y - x0[2]) ** 2))
