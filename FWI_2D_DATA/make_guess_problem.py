import meshio
from SeismicMesh import *
import segyio

fname = "vp_guess4.segy"
bbox = (-3500.0, 0.0, 0.0, 17000.0)
cpwl = 3.08 #2.22 #1.69 #$20 #7.02 #3.96 #2.67 #2.028
freq = 5.0
grade = 0.15
hmin = 1500 / (cpwl * freq)
pad = 4500.0 / freq  # cmax / freq
###########
rectangle = Rectangle(bbox)
ef = get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    wl=cpwl,
    freq=freq,
    grade=grade,
    domain_pad=pad,
    pad_style="edge",
    units="m-s",
)

write_velocity_model(
    fname,
    ofname="velocity_models/marmousi_guess",
    bbox=bbox,
    domain_pad=pad,
    pad_style="edge",
    units="m-s",
)
points, cells = generate_mesh(domain=rectangle, edge_length=ef, verbose=2)
meshio.write_points_cells("meshes/marmousi_guess.vtk", points / 1000.0, [("triangle", cells)])
meshio.write_points_cells(
    "meshes/marmousi_guess.msh",
    points / 1000.0,
    [("triangle", cells)],
    file_format="gmsh22",
    binary=False,
)
