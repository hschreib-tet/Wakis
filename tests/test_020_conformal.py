"""
Conformal (cut-cell) geometry: accuracy against analytic cavity modes,
stability of the explicit update with cut cells, and grid persistence.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros, y0, y1

from wakis import GridFIT3D, SolverFIT3D, WakeSolver

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
        stl_method="interior_points",
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
    0.25-0.3 %; the staircase error is about 1.7 %.
    """
    tm010, _ = _mode_errors(pillbox_stls, tmp_path, "conformal")
    assert abs(tm010) < 5e-3


def test_dielectric_rod_frequency(pillbox_stls, tmp_path):
    """TM010 and TM011 of the pillbox with a coaxial dielectric rod.

    Checks the conformal materials at a curved dielectric/vacuum interface.
    TM010 has only Ez, parallel to the rod surface (measured error 0.14 %).
    The radial E field of TM011 crosses the rod surface, so it also tests
    the edges cut by the interface whose dual face is not cut (measured
    0.71 %; 0.93 % when those edges kept staircase materials; 4.8 % with
    the staircase geometry).
    """
    tm010, tm011 = _mode_errors(pillbox_stls, tmp_path, "conformal", rod=True)
    assert abs(tm010) < 5e-3
    assert abs(tm011) < 8e-3


def test_conformal_beats_staircase(pillbox_stls, tmp_path):
    """On the same coarse mesh, conformal cuts reduce the TM010 error at
    least threefold compared with the staircase geometry
    (measured: about sevenfold)."""
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
        stl_method="interior_points",
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
    assert "geometry_mode" not in grid.logger.grid

    for invalid in (
        dict(conformal_stabilization="snapping"),
        dict(conformal_stabilization="area_floor", conformal_stabilization_fraction=0),
        dict(conformal_stabilization="area_floor", conformal_min_edge_fraction=0.5),
    ):
        with pytest.raises(ValueError):
            unit_grid(**invalid)


def test_conformal_grid_h5_roundtrip(pillbox_stls, tmp_path):
    """A conformal grid saved to HDF5 and loaded again gives a solver with
    identical cut-cell metrics and materials."""
    grid = _pillbox_grid(pillbox_stls, "conformal", rod=True, n_cells=10)
    grid.save_to_h5(str(tmp_path / "grid.h5"))
    loaded = GridFIT3D(load_from_h5=str(tmp_path / "grid.h5"), verbose=0)

    original, reloaded = _pec_solver(grid), _pec_solver(loaded)
    for name in ("L", "A_fraction", "ieps", "imu"):
        np.testing.assert_array_equal(
            getattr(original, name).toarray(), getattr(reloaded, name).toarray()
        )


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


@pytest.mark.parametrize("geometry_mode", ["conformal", "staircase"])
@pytest.mark.parametrize("grid_type, quantity", [("primal", "eps"), ("dual", "sigma")])
def test_plot_conformal_mesh(pillbox_stls, geometry_mode, grid_type, quantity):
    """Smoke test of the mesh/material plane view: every normal direction,
    and the interactive widgets switching normal and plane index (the last
    plane lies entirely in PEC)."""
    from matplotlib.figure import Figure

    grid = _pillbox_grid(pillbox_stls, geometry_mode, rod=True, n_cells=8)
    solver = _pec_solver(grid)
    # A bare Figure needs no GUI backend, so nothing is displayed.
    for normal in "xyz":
        solver.plot_conformal_mesh(
            grid_type=grid_type,
            normal=normal,
            quantity=quantity,
            ax=Figure().add_subplot(),
        )

    fig, _ = solver.plot_conformal_mesh(
        grid_type=grid_type, interactive=True, ax=Figure().add_subplot()
    )
    slider, normal_buttons = fig._wakis_conformal_widgets[:2]
    normal_buttons.set_active(0)
    slider.set_val(slider.valmax)
