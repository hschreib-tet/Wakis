"""
Conformal (cut-cell) geometry: accuracy against analytic cavity modes,
stability of the explicit update with cut cells, and grid persistence.
"""

import warnings

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros, y0, y1

from wakis import GridFIT3D, SolverFIT3D, WakeSolver
from wakis.intersection import (
    edge_diagnostic_masks,
    face_polygons,
    geometry_array_to_field_component,
)

# PEC pillbox: radius R, length L (closed by the PEC domain boundaries in z).
R = 0.1
L = 0.1

# Coaxial dielectric rod over the full cavity length: radius ROD_R, eps_r ROD_EPS.
ROD_R = 0.5 * R
ROD_EPS = 2.0


def _f_tm01p(p, rod=False):
    """Exact frequency of the TM01p mode of the pillbox.

    Fields vary as cos(p pi z / L). Without the rod this is
    f = c / (2 pi) sqrt((j01 / R)^2 + (p pi / L)^2). With the full-length
    coaxial rod, azimuthally symmetric TM modes stay decoupled from TE modes:
    Ez = J0(q1 r) in the rod and J0(q2 r) Y0(q2 R) - Y0(q2 r) J0(q2 R) in
    vacuum, with q1^2 = eps_r k^2 - b^2, q2^2 = k^2 - b^2, b = p pi / L.
    Ez and H_phi = j omega eps / q^2 dEz/dr are continuous at r = ROD_R.
    """
    b = p * np.pi / L
    f_empty = np.hypot(jn_zeros(0, 1)[0] / R, b) * c / (2 * np.pi)
    if not rod:
        return f_empty

    def mismatch(f):
        k = 2 * np.pi * f / c
        q1, q2 = np.sqrt(ROD_EPS * k**2 - b**2), np.sqrt(k**2 - b**2)
        outer = j0(q2 * ROD_R) * y0(q2 * R) - y0(q2 * ROD_R) * j0(q2 * R)
        outer_slope = -q2 * (j1(q2 * ROD_R) * y0(q2 * R) - y1(q2 * ROD_R) * j0(q2 * R))
        return (ROD_EPS / q1) * j1(q1 * ROD_R) * outer + outer_slope * j0(
            q1 * ROD_R
        ) / q2**2

    # The rod lowers the frequency, at most to the fully filled value; q2
    # must stay real (above the vacuum cutoff of the z standing wave).
    f_low = max(f_empty / np.sqrt(ROD_EPS), 1.01 * b * c / (2 * np.pi))
    return brentq(mismatch, f_low, f_empty)


@pytest.fixture(scope="module")
def pillbox_stls(tmp_path_factory):
    folder = tmp_path_factory.mktemp("stl")
    # Longer than the domain, so the PEC domain walls form the flat end plates
    # and the rod spans the full cavity length.
    for name, radius in (("cavity", R), ("rod", ROD_R)):
        cylinder = pv.Cylinder(
            radius=radius, height=1.2 * L, direction=(0, 0, 1), resolution=200
        )
        cylinder.triangulate().save(folder / f"{name}.stl")
    return {name: str(folder / f"{name}.stl") for name in ("cavity", "rod")}


def _pillbox_grid(stls, geometry_mode, rod=False, n_cells=20):
    """Pillbox on a uniform grid; the curved wall cuts through the cells.

    With ``rod=True`` a dielectric rod is added inside the vacuum, so the
    grid also has curved dielectric/vacuum interfaces.
    """
    solids = ("cavity", "rod") if rod else ("cavity",)
    materials = {"cavity": [1.0, 1.0, 0.0], "rod": [ROD_EPS, 1.0, 0.0]}
    h = 1.1 * R
    conformal = {}
    if geometry_mode == "conformal":
        conformal = dict(
            stl_material_types={key: "normal" for key in solids},
            bg_material_type="pec",
            conformal_stabilization="area_floor",
        )
    return GridFIT3D(
        -h,
        h,
        -h,
        h,
        -L / 2,
        L / 2,
        n_cells + 1,
        n_cells + 1,
        int(n_cells * L / (2 * h)) + 1,
        stl_solids={key: stls[key] for key in solids},
        stl_materials={key: materials[key] for key in solids},
        stl_method="legacy",
        geometry_mode=geometry_mode,
        verbose=0,
        **conformal,
    )


_mode_error_cache = {}


def _mode_errors(stls, tmp_dir, geometry_mode, rod=False, n_cells=20):
    """Relative errors of the simulated TM010 and TM011 frequencies against
    the analytic ones.

    The frequencies are the peaks of the longitudinal impedance of an
    on-axis beam. PEC walls make the modes undamped, so the truncated wake
    gives symmetric peaks centred on the mode frequencies.
    """
    key = (geometry_mode, rod, n_cells)
    if key in _mode_error_cache:
        return _mode_error_cache[key]

    grid = _pillbox_grid(stls, geometry_mode, rod, n_cells)
    wake = WakeSolver(
        q=1e-9,
        sigmaz=0.03,
        beta=1.0,
        save=False,
        results_folder=f"{tmp_dir}/",
        Ez_file=f"{tmp_dir}/Ez_{geometry_mode}_{rod}_{n_cells}.h5",
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        wake,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        use_stl=True,
        bg="pec",
        dtype=np.float64,
        verbose=0,
    )
    solver.wakesolve(wakelength=10.0, plot=False)
    wake.calc_long_Z(samples=20001, fmax=2.5e9)

    f, Z = np.asarray(wake.f), np.abs(wake.Z)
    errors = []
    for p in (0, 1):
        f_exact = _f_tm01p(p, rod)
        window = np.abs(f - f_exact) < 0.05 * f_exact
        errors.append(f[window][np.argmax(Z[window])] / f_exact - 1)

    _mode_error_cache[key] = errors
    return errors


def test_pillbox_frequency(pillbox_stls, tmp_path):
    """TM010 of the empty PEC pillbox.

    With R spanning only about 9.5 cells, the conformal error is about
    0.25 %; the staircase error is about 3.9 %.
    """
    tm010, _ = _mode_errors(pillbox_stls, tmp_path, "conformal")
    assert abs(tm010) < 5e-3


def test_dielectric_rod_frequency(pillbox_stls, tmp_path):
    """TM010 and TM011 of the pillbox with a coaxial dielectric rod.

    Checks the conformal materials at a curved dielectric/vacuum interface.
    TM010 has only Ez, parallel to the rod surface (measured error 0.14 %).
    The radial E field of TM011 crosses the rod surface, so it also tests
    the edges cut by the interface whose dual face is not cut (measured
    0.71 %; 0.93 % when those edges kept staircase materials; 2.7 % with
    the staircase geometry).
    """
    tm010, tm011 = _mode_errors(pillbox_stls, tmp_path, "conformal", rod=True)
    assert abs(tm010) < 5e-3
    assert abs(tm011) < 8e-3


def test_conformal_beats_staircase(pillbox_stls, tmp_path):
    """On the same coarse mesh, conformal cuts reduce the TM010 error at
    least threefold compared with the staircase geometry
    (measured: about sixteenfold)."""
    conformal, _ = _mode_errors(pillbox_stls, tmp_path, "conformal")
    staircase, _ = _mode_errors(pillbox_stls, tmp_path, "staircase")
    assert abs(conformal) < 0.3 * abs(staircase)


@pytest.mark.slow
def test_pillbox_convergence(pillbox_stls, tmp_path):
    """Halving the cell size reduces the conformal TM010 error with an order
    above 1.5 (measured: about 2.2, second-order accurate boundary)."""
    coarse, _ = _mode_errors(pillbox_stls, tmp_path, "conformal", n_cells=20)
    fine, _ = _mode_errors(pillbox_stls, tmp_path, "conformal", n_cells=40)
    assert np.log2(abs(coarse) / abs(fine)) > 1.5


# ----------------------------------------------------------------------
# Stability of the explicit update with cut cells
# ----------------------------------------------------------------------


def _slanted_box_grid(stl_file, **kwargs):
    """PEC-bounded box cavity, rotated so its walls cut through the cells."""
    box = pv.Box(bounds=(-0.15, 0.15, -0.11, 0.11, -0.085, 0.085)).triangulate()
    box.rotate_z(20).rotate_x(10).save(stl_file)
    return GridFIT3D(
        -0.2,
        0.2,
        -0.2,
        0.2,
        -0.2,
        0.2,
        13,
        13,
        13,
        stl_solids={"cav": str(stl_file)},
        stl_materials={"cav": "vacuum"},
        stl_material_types={"cav": "normal"},
        bg_material_type="pec",
        geometry_mode="conformal",
        stl_method="legacy",
        verbose=0,
        **kwargs,
    )


def _pec_solver(grid):
    return SolverFIT3D(
        grid, bg="pec", use_stl=True, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
    )


def _field_growth(solver, dt, n_steps=3000):
    """Growth of a random initial E field after n_steps leapfrog steps."""
    rng = np.random.default_rng(0)
    active = solver.ieps.toarray() != 0  # PEC edges are not updated
    solver.E.fromarray(rng.standard_normal(3 * solver.N) * active)
    solver.H.fromarray(np.zeros(3 * solver.N))
    solver.dt = dt
    norm0 = np.linalg.norm(solver.E.toarray())
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            solver.one_step()
    return np.linalg.norm(solver.E.toarray()) / norm0


def test_area_method_requires_backend_faces(tmp_path):
    """The ray-tracing backend leaves the face split to the shared
    straight-line method, so asking for its own area fractions fails."""
    with pytest.raises(ValueError, match="provides no area fractions"):
        _slanted_box_grid(tmp_path / "cav.stl", conformal_area_method="backend")


def test_stable_timestep(tmp_path):
    """compute_stable_timestep() gives the exact leapfrog limit.

    On an uncut uniform grid it equals the analytic CFL limit. On a grid with
    conductor cut cells, time stepping stays bounded 2 % below dt_max and
    diverges 2 % above it.
    """
    uncut = SolverFIT3D(GridFIT3D(0, 1, 0, 2, 0, 3, 9, 11, 13, verbose=0), verbose=0)
    result = uncut.compute_stable_timestep(verbose=False)
    assert result["dt_max"] == pytest.approx(result["dt_cfl"], rel=1e-6)

    solver = _pec_solver(_slanted_box_grid(tmp_path / "cav.stl"))
    dt_max = solver.compute_stable_timestep(verbose=False)["dt_max"]
    assert _field_growth(solver, 0.98 * dt_max) < 10.0
    growth = _field_growth(solver, 1.02 * dt_max)
    assert not np.isfinite(growth) or growth > 1e3


def test_area_floor_stabilizes_cut_cells(tmp_path):
    """Tiny cut faces make the exact cut-cell update unstable at the default
    cfln = 0.5; raising them to the area floor restores stability."""
    stl_file = tmp_path / "cav.stl"
    exact = _pec_solver(_slanted_box_grid(stl_file))
    floored = _pec_solver(
        _slanted_box_grid(stl_file, conformal_stabilization="area_floor")
    )
    assert not exact.compute_stable_timestep(verbose=False)["stable"]
    assert floored.compute_stable_timestep(verbose=False)["stable"]


def test_stabilization_options():
    """Valid stabilization settings are stored; invalid ones are rejected."""

    def unit_grid(**kwargs):
        return GridFIT3D(0, 1, 0, 1, 0, 1, 3, 3, 3, verbose=0, **kwargs)

    grid = unit_grid(conformal_stabilization="area_floor")
    assert grid.conformal_area_floor == pytest.approx(0.1)
    assert grid.conformal_min_edge_fraction == pytest.approx(0.1)
    assert unit_grid().conformal_stabilization == "none"

    # Conformal settings are logged for reproducibility, staircase logs unchanged.
    conformal = unit_grid(
        geometry_mode="conformal", conformal_stabilization="area_floor"
    )
    assert conformal.logger.grid["geometry_mode"] == "conformal"
    assert conformal.logger.grid["conformal_area_floor"] == pytest.approx(0.1)
    assert conformal.logger.grid["conformal_method"] == "raytracing"
    assert "geometry_mode" not in grid.logger.grid

    for invalid in (
        dict(conformal_stabilization="snapping"),
        dict(conformal_stabilization="area_floor", conformal_stabilization_fraction=0),
        dict(conformal_stabilization="area_floor", conformal_min_edge_fraction=0.5),
        dict(conformal_method="unknown"),
        dict(conformal_area_method="unknown"),
    ):
        with pytest.raises(ValueError):
            unit_grid(**invalid)


def test_coincident_wall(tmp_path):
    """A dielectric slab sharing one wall with the box cavity, both oblique
    and triangulated differently. At that wall the interface is PEC|slab, so
    it is cut by the slab surface alone: the cut edges and faces there are
    the same as for a cavity ending inside the slab, where only the slab
    surface exists. Mixing in the coincident cavity surface would leave
    faces uncut."""

    def build(cavity_x_max, folder):
        folder.mkdir()
        for name, bounds, level in (
            ("cav", (-0.15, cavity_x_max, -0.11, 0.11, -0.085, 0.085), 1),
            ("slab", (0.05, 0.15, -0.11, 0.11, -0.085, 0.085), 3),
        ):
            box = pv.Box(bounds=bounds, level=level).triangulate()
            box.rotate_z(20).rotate_x(10).save(folder / f"{name}.stl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # thin features at the slab corners
            return GridFIT3D(
                *(-0.2, 0.2) * 3,
                13,
                13,
                13,
                stl_solids={key: str(folder / f"{key}.stl") for key in ("cav", "slab")},
                stl_materials={"cav": "vacuum", "slab": [4, 1, 0]},
                bg_material_type="pec",
                geometry_mode="conformal",
                stl_method="legacy",
                verbose=0,
            )

    grid = build(0.15, tmp_path / "coincident")
    reference = build(0.10, tmp_path / "reference")
    slab = grid.region_key_to_id["slab"]
    np.testing.assert_array_equal(
        grid.region_ids["primal"], reference.region_ids["primal"]
    )

    def wall(g):
        # Cut edges between PEC (0) and the slab, and the faces split there.
        edges = []
        regions = g.region_ids["primal"]
        for direction, cuts in g.edge_intersections["primal"].items():
            step = np.eye(3, dtype=int)["xyz".index(direction)]
            start = regions[tuple(cuts.index.T)]
            end = regions[tuple((cuts.index + step).T)]
            pair = (np.minimum(start, end) == 0) & (np.maximum(start, end) == slab)
            flat = g._flat_field_index("primal", "edge", direction, cuts.index[pair])
            edges.append(np.column_stack((flat, cuts.alpha[pair])))
        edges = np.concatenate(edges)
        faces = g.face_intersections["primal"]
        split = (faces.region_a == 0) & (faces.region_b == slab)
        return (
            edges[:, 0].astype(int),
            edges[:, 1],
            faces.face[split],
            faces.fraction_a[split],
        )

    edges, alpha, faces, fraction = wall(grid)
    ref_edges, ref_alpha, ref_faces, ref_fraction = wall(reference)
    assert len(faces) > 0
    np.testing.assert_array_equal(np.sort(edges), np.sort(ref_edges))
    np.testing.assert_allclose(
        alpha[np.argsort(edges)], ref_alpha[np.argsort(ref_edges)], atol=1e-6
    )
    np.testing.assert_array_equal(np.sort(faces), np.sort(ref_faces))
    np.testing.assert_allclose(
        fraction[np.argsort(faces)], ref_fraction[np.argsort(ref_faces)], atol=1e-6
    )


def test_surface_on_grid_planes(tmp_path):
    """A coated pillbox with beam pipes, the domain set to the STL bounds and
    the end walls on node planes. Many grid lines then touch surfaces instead
    of crossing them: they run along the end caps, graze the outer cylinder
    at the domain boundary, or meet the surfaces at grid nodes. The node
    classification decides the side there, so no edge is reported as
    inconsistent."""
    pipe = pv.Cylinder(radius=0.025, height=0.3, direction=(0, 0, 1), resolution=100)
    inner = pv.Cylinder(radius=0.0925, height=0.1, direction=(0, 0, 1), resolution=200)
    (pipe.triangulate() | inner.triangulate()).save(tmp_path / "vacuum.stl")
    pv.Cylinder(radius=0.1, height=0.1, direction=(0, 0, 1), resolution=200).save(
        tmp_path / "coating.stl"
    )
    grid = GridFIT3D(
        *(-0.1, 0.1) * 2,
        -0.15,
        0.15,
        41,
        41,
        61,
        stl_solids={key: str(tmp_path / f"{key}.stl") for key in ("coating", "vacuum")},
        stl_materials={"coating": [3.0, 1.0, 0.0], "vacuum": "vacuum"},
        bg_material_type="pec",
        geometry_mode="conformal",
        stl_method="voxelize_rectilinear",
        verbose=0,
    )
    for grid_type in ("primal", "dual"):
        for direction in "xyz":
            _, inconsistent = edge_diagnostic_masks(grid, grid_type, direction)
            assert not inconsistent.any(), (grid_type, direction)


def test_flat_field_index():
    """The flat field index of every primal/dual edge and face is the
    field entry the solver assigns it (including the shift of the dual
    grid), and every DOF with a geometric counterpart is reached once."""
    grid = GridFIT3D(0, 1, 0, 1, 0, 1, 4, 5, 6, geometry_mode="conformal", verbose=0)
    solver = SolverFIT3D(grid, verbose=0)
    nodes = np.array([grid.Nx, grid.Ny, grid.Nz])
    for grid_type, kind, shift in (
        ("primal", "edge", [-1, 0]),
        ("primal", "face", [0, -1]),
        ("dual", "edge", [0, 1]),
        ("dual", "face", [1, 0]),
    ):
        for axis, direction in enumerate("xyz"):
            # Geometry shape: (along axis, transverse) offset from the nodes.
            shape = nodes + shift[1]
            shape[axis] = nodes[axis] + shift[0]
            geometry_id = np.arange(np.prod(shape)).reshape(shape)
            expected = geometry_array_to_field_component(
                geometry_id, f"{grid_type}_{kind}", direction, nodes, fill_value=-1
            ).ravel(order="F")

            index = np.indices(shape).reshape(3, -1).T
            flat = grid._flat_field_index(grid_type, kind, direction, index)
            valid = flat >= 0
            local = flat[valid] - axis * solver.N
            assert np.array_equal(expected[local], geometry_id.ravel()[valid])
            assert np.array_equal(np.sort(local), np.flatnonzero(expected >= 0))


def test_face_split_tilted_plane(tmp_path):
    """A half-space bounded by a tilted plane cuts every face along a
    straight line, so the split is exact: on both grids, all faces with
    corners on both sides are split and listed, with the area fractions of
    the face clipped by the plane, and the returned polygons have these
    areas."""
    center = np.array([0.513, 0.47, 0.5])
    normal = np.array([np.cos(np.radians(25)), np.sin(np.radians(25)), 0.0])
    # Box whose +x face is the plane, reaching beyond the domain elsewhere.
    box = pv.Box(bounds=(-3, 0, -3, 3, -3, 3)).triangulate()
    box.rotate_z(25).translate(center).save(tmp_path / "half.stl")
    grid = GridFIT3D(
        *(0, 1) * 3,
        9,
        9,
        9,
        stl_solids={"half": str(tmp_path / "half.stl")},
        stl_materials={"half": [2, 1, 0]},
        geometry_mode="conformal",
        stl_method="legacy",
        verbose=0,
    )
    solid = grid.region_key_to_id["half"]

    def area(polygon):
        # Area of a planar polygon in 3D; repeated vertices add nothing.
        return 0.5 * np.linalg.norm(np.cross(polygon, np.roll(polygon, -1, 0)).sum(0))

    def outside_fraction(corners):
        # Face clipped to the background side of the plane (one-line
        # Sutherland-Hodgman).
        kept = []
        for p, q in zip(corners, np.roll(corners, -1, axis=0)):
            fp, fq = (p - center) @ normal, (q - center) @ normal
            if fp > 0:
                kept.append(p)
            if (fp > 0) != (fq > 0):
                kept.append(p + fp / (fp - fq) * (q - p))
        return area(np.array(kept)) / area(corners)

    for grid_type in ("primal", "dual"):
        coordinates = np.array(grid._get_grid_coordinates(grid_type))
        n_listed = 0
        for axis, direction in enumerate("xyz"):
            a, b = (axis + 1) % 3, (axis + 2) % 3
            shape = np.full(3, len(coordinates[0]) - 1)
            shape[axis] += 1
            face = np.indices(shape).reshape(3, -1).T
            unit = np.eye(3, dtype=int)
            corner_index = np.stack(
                [face, face + unit[a], face + unit[a] + unit[b], face + unit[b]], axis=1
            )
            corners = np.stack(
                [coordinates[d][corner_index[:, :, d]] for d in range(3)], axis=-1
            )
            side = (corners - center) @ normal > 0
            crossed = np.flatnonzero(side.any(axis=1) & ~side.all(axis=1))

            split = face_polygons(
                grid, grid_type, direction, face, grid.edge_intersections[grid_type]
            )
            np.testing.assert_array_equal(np.sort(split["position"]), crossed)
            assert np.all(split["regions"] == [0, solid])
            for k, position in enumerate(split["position"]):
                face_area = area(corners[position])
                expected = outside_fraction(corners[position])
                # STL vertices are stored in single precision.
                assert split["fractions"][k, 0] == pytest.approx(expected, abs=1e-6)
                for polygon, fraction in zip(
                    split["vertices"][k], split["fractions"][k]
                ):
                    assert area(polygon) == pytest.approx(fraction * face_area)
            flat = grid._flat_field_index(grid_type, "face", direction, face[crossed])
            n_listed += np.count_nonzero(flat >= 0)
        assert len(grid.face_intersections[grid_type].face) == n_listed


def test_conductor_background_type():
    """A conformal grid without bg_material_type traces its cuts with a
    normal background, so a PEC solver background is rejected."""
    grid = GridFIT3D(0, 1, 0, 1, 0, 1, 3, 3, 3, geometry_mode="conformal", verbose=0)
    with pytest.raises(ValueError, match="bg_material_type='pec'"):
        SolverFIT3D(grid, bg="pec", verbose=0)


def test_conformal_grid_h5_roundtrip(pillbox_stls, tmp_path):
    """A conformal grid saved to HDF5 and loaded again gives a solver with
    identical cut-cell metrics and materials."""
    grid = _pillbox_grid(pillbox_stls, "conformal", rod=True, n_cells=10)
    grid.save_to_h5(str(tmp_path / "grid.h5"))
    loaded = GridFIT3D(load_from_h5=str(tmp_path / "grid.h5"), verbose=0)

    original, reloaded = _pec_solver(grid), _pec_solver(loaded)
    for name in ("L", "iA", "ieps", "imu"):
        np.testing.assert_array_equal(
            getattr(original, name).toarray(), getattr(reloaded, name).toarray()
        )


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


@pytest.mark.parametrize("geometry_mode", ["conformal", "staircase"])
@pytest.mark.parametrize("grid_type, quantity", [("primal", "eps"), ("dual", "sigma")])
def test_plot_mesh_materials(pillbox_stls, geometry_mode, grid_type, quantity):
    """Smoke test of the mesh/material plane view: every normal direction,
    and the interactive widgets switching normal, plane index (the last
    plane lies entirely in PEC) and material, which also switches between
    the primal and the dual grid."""
    from matplotlib.figure import Figure

    grid = _pillbox_grid(pillbox_stls, geometry_mode, rod=True, n_cells=8)
    solver = _pec_solver(grid)
    # A bare Figure needs no GUI backend, so nothing is displayed.
    for normal in "xyz":
        solver.plot_mesh_materials(
            grid_type=grid_type,
            normal=normal,
            quantity=quantity,
            ax=Figure().add_subplot(),
        )

    fig, _ = solver.plot_mesh_materials(
        grid_type=grid_type, interactive=True, ax=Figure().add_subplot()
    )
    slider, normal_buttons, material_buttons = fig._wakis_mesh_widgets
    normal_buttons.set_active(0)
    slider.set_val(slider.valmax)
    for material in (0, 2, 1):  # mu (primal), sigma and eps (dual)
        material_buttons.set_active(material)
        slider.set_val(slider.valmin)


def test_plot_mesh_materials_diagnostics(tmp_path):
    """Diagnostic markers: a dielectric slab thinner than a cell (thin
    feature on the z edges through it) and an open box surface (inconsistent
    x edges at the missing face). The markers appear in the planes holding
    the flagged edges, on the dual grid also where the flagged primal edges
    pierce the faces, and the legend counts them. The open box is also
    reported as not closed (the same check runs for staircase grids)."""
    from matplotlib.figure import Figure

    n = 10
    d = 1.0 / n
    pv.Cylinder(
        center=(0, 0, 0.5), direction=(0, 0, 1), radius=0.42, height=1.2
    ).triangulate().save(tmp_path / "cav.stl")
    # Slab between the node plane z = 0.5 and the next edge midpoint.
    pv.Box(
        bounds=(-0.25, 0.25, -0.25, 0.25, 0.5 + 0.1 * d, 0.5 + 0.4 * d)
    ).triangulate().save(tmp_path / "slab.stl")
    box = pv.Box(bounds=(0.03, 0.27, -0.17, 0.17, 0.13, 0.37)).triangulate()
    keep = np.nonzero(box.cell_centers().points[:, 0] < 0.26)[0]  # drop +x face
    box.extract_cells(keep).extract_surface(algorithm="dataset_surface").save(
        tmp_path / "box.stl"
    )

    with pytest.warns(UserWarning) as record:
        grid = GridFIT3D(
            -0.5,
            0.5,
            -0.5,
            0.5,
            0,
            1,
            n + 1,
            n + 1,
            n + 1,
            stl_solids={
                key: str(tmp_path / f"{key}.stl") for key in ("cav", "slab", "box")
            },
            stl_materials={"cav": "vacuum", "slab": [3, 1, 0], "box": [2, 1, 0]},
            bg_material_type="pec",
            geometry_mode="conformal",
            verbose=0,
        )
    messages = [str(warning.message) for warning in record]
    assert any("Conformal geometry diagnostics" in m for m in messages)
    assert any("'box' is not a closed surface" in m for m in messages)
    solver = SolverFIT3D(grid, use_stl=True, bg="pec", verbose=0)

    for grid_type, normal, index, expected in (
        ("primal", "y", 5, {"thin_feature": 5, "inconsistent": 2}),
        ("dual", "z", 6, {"thin_feature": 25, "inconsistent": 0}),
        ("primal", "z", 2, {"thin_feature": 0, "inconsistent": 3}),
    ):
        data = solver._mesh_slice_data(grid_type, normal, index, "eps", [])
        counts = {key: len(points) for key, points in data["diagnostics"].items()}
        assert counts == expected

        _, ax = solver.plot_mesh_materials(
            grid_type=grid_type, normal=normal, index=index, ax=Figure().add_subplot()
        )
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        for key, count in expected.items():
            entry = f"{key.replace('_', ' ')} ({count})"
            assert (entry in labels) == (count > 0)

    _, ax = solver.plot_mesh_materials(
        "dual", normal="z", index=6, show_diagnostics=False, ax=Figure().add_subplot()
    )
    assert not any("thin feature" in t.get_text() for t in ax.get_legend().get_texts())

    # Without use_stl the solver has no cut-cell metrics: plain mesh, no markers.
    plain = SolverFIT3D(grid, bg="pec", verbose=0)
    data = plain._mesh_slice_data("dual", "z", 6, "eps", [])
    assert all(len(points) == 0 for points in data["diagnostics"].values())
    plain.plot_mesh_materials("primal", normal="z", ax=Figure().add_subplot())
