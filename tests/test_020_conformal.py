"""
Conformal (cut-cell) geometry: accuracy against analytic cavity modes,
stability of the explicit update with cut cells, and grid persistence.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c
from scipy.special import jn_zeros

from wakis import GridFIT3D, SolverFIT3D, WakeSolver

# PEC pillbox: radius R, length L (closed by the PEC domain boundaries in z).
R = 0.1
L = 0.1
F_TM010 = jn_zeros(0, 1)[0] * c / (2 * np.pi * R)


@pytest.fixture(scope="module")
def cylinder_stl(tmp_path_factory):
    path = tmp_path_factory.mktemp("stl") / "cylinder.stl"
    # Longer than the domain, so the PEC domain walls form the flat end plates.
    cylinder = pv.Cylinder(
        radius=R, height=1.2 * L, direction=(0, 0, 1), resolution=200
    )
    cylinder.triangulate().save(path)
    return str(path)


def _pillbox_grid(stl_file, geometry_mode, eps_r=1.0, n_cells=20):
    """Pillbox on a uniform grid; the curved wall cuts through the cells."""
    h = 1.1 * R
    conformal = {}
    if geometry_mode == "conformal":
        conformal = dict(
            stl_material_types={"cavity": "normal"},
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
        stl_solids={"cavity": stl_file},
        stl_materials={"cavity": [eps_r, 1.0, 0.0]},
        stl_method="interior_points",
        geometry_mode=geometry_mode,
        verbose=0,
        **conformal,
    )


_tm010_cache = {}


def _tm010_error(stl_file, tmp_dir, geometry_mode, eps_r=1.0, n_cells=20):
    """Relative error of the simulated TM010 frequency against the analytic one.

    The frequency is the peak of the longitudinal impedance of an on-axis
    beam. PEC walls make the mode undamped, so the truncated wake gives a
    symmetric peak centred on the mode frequency.
    """
    key = (geometry_mode, eps_r, n_cells)
    if key in _tm010_cache:
        return _tm010_cache[key]

    grid = _pillbox_grid(stl_file, geometry_mode, eps_r, n_cells)
    wake = WakeSolver(
        q=1e-9,
        sigmaz=0.03,
        beta=1.0,
        save=False,
        results_folder=f"{tmp_dir}/",
        Ez_file=f"{tmp_dir}/Ez_{geometry_mode}_{eps_r}_{n_cells}.h5",
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
    wake.calc_long_Z(samples=20001, fmax=2e9)

    f_exact = F_TM010 / np.sqrt(eps_r)
    f, Z = np.asarray(wake.f), np.abs(wake.Z)
    window = np.abs(f - f_exact) < 0.1 * f_exact
    f_peak = f[window][np.argmax(Z[window])]

    _tm010_cache[key] = f_peak / f_exact - 1
    return _tm010_cache[key]


@pytest.mark.parametrize("eps_r", [1.0, 2.0])
def test_pillbox_frequency(cylinder_stl, tmp_path, eps_r):
    """TM010 of a PEC pillbox, empty and filled with a dielectric.

    The analytic frequency is 2.405 c / (2 pi R sqrt(eps_r)). With R spanning
    only about 9.5 cells, the conformal error is about 0.25-0.3 %; the
    staircase error is about 1.7 %. The filled case also checks the conformal
    material assignment at a curved dielectric/PEC interface.
    """
    error = _tm010_error(cylinder_stl, tmp_path, "conformal", eps_r)
    assert abs(error) < 5e-3


def test_conformal_beats_staircase(cylinder_stl, tmp_path):
    """On the same coarse mesh, conformal cuts reduce the TM010 error at
    least threefold compared with the staircase (legacy) geometry
    (measured: about sevenfold)."""
    conformal = _tm010_error(cylinder_stl, tmp_path, "conformal")
    staircase = _tm010_error(cylinder_stl, tmp_path, "legacy")
    assert abs(conformal) < 0.3 * abs(staircase)


@pytest.mark.slow
def test_pillbox_convergence(cylinder_stl, tmp_path):
    """Halving the cell size reduces the conformal TM010 error with an order
    above 1.5 (measured: about 2.2, second-order accurate boundary)."""
    coarse = _tm010_error(cylinder_stl, tmp_path, "conformal", n_cells=20)
    fine = _tm010_error(cylinder_stl, tmp_path, "conformal", n_cells=40)
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

    for invalid in (
        dict(conformal_stabilization="snapping"),
        dict(conformal_stabilization="area_floor", conformal_stabilization_fraction=0),
        dict(conformal_stabilization="area_floor", conformal_min_edge_fraction=0.5),
    ):
        with pytest.raises(ValueError):
            unit_grid(**invalid)


def test_conformal_grid_h5_roundtrip(cylinder_stl, tmp_path):
    """A conformal grid saved to HDF5 and loaded again gives a solver with
    identical cut-cell metrics and materials."""
    grid = _pillbox_grid(cylinder_stl, "conformal", eps_r=2.0, n_cells=10)
    grid.save_to_h5(str(tmp_path / "grid.h5"))
    loaded = GridFIT3D(load_from_h5=str(tmp_path / "grid.h5"), verbose=0)

    original, reloaded = _pec_solver(grid), _pec_solver(loaded)
    for name in ("L", "A_fraction", "ieps", "imu"):
        np.testing.assert_array_equal(
            getattr(original, name).toarray(), getattr(reloaded, name).toarray()
        )
