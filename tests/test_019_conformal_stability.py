"""
Stability check of the explicit FIT update; conformal conductor cuts
affect only the primal metrics.
"""

import sys

import numpy as np
import pytest
import pyvista as pv

sys.path.append("../wakis")

from wakis import GridFIT3D, SolverFIT3D  # noqa: E402


def test_stable_timestep_equals_cfl_on_uncut_grid():
    """On a uniform periodic grid dt_max is exactly the CFL limit."""
    grid = GridFIT3D(0, 1, 0, 2, 0, 3, 8, 10, 12, verbose=0)
    solver = SolverFIT3D(grid, use_stl=False, verbose=0)

    result = solver.compute_stable_timestep(verbose=False)

    assert result["dt_max"] == pytest.approx(result["dt_cfl"], rel=1e-6)
    assert result["stable"]
    assert result["dt_ratio"] == pytest.approx(solver.cfln, rel=1e-6)


@pytest.fixture(scope="module")
def slanted_cavity_grid(tmp_path_factory):
    stl_file = tmp_path_factory.mktemp("stl") / "slanted_cavity.stl"
    box = pv.Box(bounds=(-0.15, 0.15, -0.11, 0.11, -0.085, 0.085)).triangulate()
    box.rotate_z(20).rotate_x(10).save(stl_file)

    return GridFIT3D(
        -0.2,
        0.2,
        -0.2,
        0.2,
        -0.2,
        0.2,
        10,
        10,
        10,
        stl_solids={"cav": str(stl_file)},
        stl_materials={"cav": "vacuum"},
        stl_material_types={"cav": "normal"},
        bg_material_type="pec",
        geometry_mode="conformal",
        stl_method="interior_points",
        verbose=0,
    )


def _solver(grid):
    return SolverFIT3D(
        grid,
        bg="pec",
        use_stl=True,
        verbose=0,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
    )


def test_conductor_cuts_only_primal_metrics(slanted_cavity_grid):
    """PEC walls cut L and A only; tL and itA keep their grid values."""
    grid = slanted_cavity_grid
    solver = _solver(grid)

    assert np.any(solver.L_fraction.toarray() < 1.0)
    assert np.any(solver.A_fraction.toarray() < 1.0)

    np.testing.assert_array_equal(solver.tL.toarray(), grid.tL.toarray())
    np.testing.assert_array_equal(solver.itA.toarray(), grid.itA.toarray())

    # The dual grid is not traced at a pure conductor wall.
    for d in ("x", "y", "z"):
        assert not np.any(grid.interface_intersections["dual"][d]["has_crossing"])
        assert not np.any(grid.interface_face_data["dual"][d]["resolved"])


def test_stable_timestep_matches_time_stepping(slanted_cavity_grid):
    """Leapfrog stays bounded just below dt_max and diverges just above."""
    solver = _solver(slanted_cavity_grid)
    dt_max = solver.compute_stable_timestep(verbose=False)["dt_max"]

    A = solver.tDsiDmuiDaC
    B = solver.itDaiDepsDstC
    active = solver.iDeps.diagonal() != 0

    growth = {}
    for factor in (0.98, 1.02):
        dt = factor * dt_max
        e = np.random.default_rng(0).standard_normal(3 * solver.N) * active
        h = np.zeros(3 * solver.N)
        norm0 = np.linalg.norm(e)
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(3000):
                h = h - dt * (A @ e)
                e = e + dt * (B @ h)
        growth[factor] = np.linalg.norm(e) / norm0

    assert growth[0.98] < 10.0
    assert not np.isfinite(growth[1.02]) or growth[1.02] > 1e3


def test_dielectric_interface_keeps_dual_geometry(tmp_path):
    """Normal-normal interfaces still use the dual geometry for materials."""
    stl_file = tmp_path / "dielectric.stl"
    block = pv.Box(bounds=(-0.07, 0.07, -0.05, 0.05, -0.04, 0.04)).triangulate()
    block.rotate_z(20).rotate_x(10).save(stl_file)

    grid = GridFIT3D(
        -0.2,
        0.2,
        -0.2,
        0.2,
        -0.2,
        0.2,
        12,
        12,
        12,
        stl_solids={"block": str(stl_file)},
        stl_materials={"block": [4.0, 2.0, 0.0]},
        stl_material_types={"block": "normal"},
        bg_material_type="normal",
        geometry_mode="conformal",
        stl_method="interior_points",
        verbose=0,
    )
    solver = SolverFIT3D(grid, bg="vacuum", use_stl=True, verbose=0)

    assert any(
        np.any(grid.interface_intersections["dual"][d]["has_crossing"])
        for d in ("x", "y", "z")
    )
    assert any(
        np.any(np.isfinite(solver.conformal_diagonal_materials["eps_r"][d]))
        for d in ("x", "y", "z")
    )
    assert any(
        np.any(np.isfinite(solver.conformal_diagonal_materials["mu_r"][d]))
        for d in ("x", "y", "z")
    )
