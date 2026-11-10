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
    grid = GridFIT3D(0, 1, 0, 2, 0, 3, 9, 11, 13, verbose=0)
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
        11,
        11,
        11,
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
        13,
        13,
        13,
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


# ----------------------------------------------------------------------
# Small-cut-cell treatment: area floor
# ----------------------------------------------------------------------


def _unit_grid(**kwargs):
    return GridFIT3D(0, 1, 0, 1, 0, 1, 3, 3, 3, verbose=0, **kwargs)


def test_stabilization_selector_validation():
    grid = _unit_grid(conformal_stabilization="area_floor")
    assert grid.conformal_stabilization == "area_floor"
    assert grid.conformal_area_floor == pytest.approx(0.1)
    # The edge threshold defaults to the floor value.
    assert grid.conformal_min_edge_fraction == pytest.approx(0.1)

    assert _unit_grid().conformal_stabilization == "none"
    assert _unit_grid().conformal_area_floor == 0.0

    with pytest.raises(ValueError):
        _unit_grid(conformal_stabilization="merging")
    with pytest.raises(ValueError):
        # PEC snapping was removed in favor of the area floor
        _unit_grid(conformal_stabilization="snapping")
    with pytest.raises(ValueError):
        _unit_grid(
            conformal_stabilization="area_floor", conformal_stabilization_fraction=0.0
        )
    with pytest.raises(ValueError):
        _unit_grid(
            conformal_stabilization="area_floor", conformal_min_edge_fraction=0.5
        )


def _slanted_grid(stl_file, **kwargs):
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


def _localized_low_modes(solver, f_max, n_modes=30, max_participation=30):
    """Count eigenmodes below f_max carried by only a few DOFs."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    inv_meps = solver.Ds.diagonal() * solver.iDeps.diagonal() * solver.itDa.diagonal()
    active = np.flatnonzero(inv_meps > 0)
    m_nu = solver.tDs.diagonal() * solver.iDmu.diagonal() * solver.iDa.diagonal()
    C = solver.C.tocsr().astype(float)[:, active]
    K = (C.T @ diags(m_nu) @ C).tocsc()
    M = diags(1.0 / inv_meps[active]).tocsc()

    lam, vectors = eigsh(K, k=n_modes, M=M, sigma=(2 * np.pi * 0.3 * f_max) ** 2)
    count = 0
    for value, v in zip(lam, vectors.T):
        f = np.sqrt(max(value, 0.0)) / (2 * np.pi)
        if f < 0.1 * f_max or f > f_max:  # static null space / above band
            continue
        w = np.abs((M @ v) * v)
        if w.sum() ** 2 / np.sum(w**2) < max_participation:
            count += 1
    return count


def test_area_floor_is_stable_without_spurious_modes(tmp_path):
    stl_file = tmp_path / "cav.stl"
    box = pv.Box(bounds=(-0.15, 0.15, -0.11, 0.11, -0.085, 0.085)).triangulate()
    box.rotate_z(20).rotate_x(10).save(stl_file)

    exact = _slanted_grid(stl_file)
    floored = _slanted_grid(stl_file, conformal_stabilization="area_floor")

    s_exact = _solver(exact)
    s_floor = _solver(floored)

    # Exact cuts are unstable at the default cfln = 0.5, the floor is not.
    assert not s_exact.compute_stable_timestep(verbose=False)["stable"]
    assert s_floor.compute_stable_timestep(verbose=False)["stable"]

    # The geometry and the cut edge lengths are untouched ...
    for d in ("x", "y", "z"):
        np.testing.assert_array_equal(
            exact.interface_intersections["primal"][d]["alpha"],
            floored.interface_intersections["primal"][d]["alpha"],
        )
    np.testing.assert_array_equal(s_exact.L.toarray(), s_floor.L.toarray())
    np.testing.assert_array_equal(
        s_exact.A_fraction.toarray(), s_floor.A_fraction_geometric.toarray()
    )

    # ... only small conductor-cut areas are raised.
    raised = s_floor.A_fraction.toarray() != s_floor.A_fraction_geometric.toarray()
    assert s_floor.n_area_floored == np.count_nonzero(raised) > 0
    np.testing.assert_allclose(s_floor.A_fraction.toarray()[raised], 0.1)
    assert s_floor.n_edges_eliminated > 0

    # Raised areas next to nearly covered edges would create localized
    # low-frequency modes; the edge elimination must keep them out of the
    # band the mesh resolves (here: at least 8 cells per wavelength).
    from scipy.constants import c

    f_max = c / (8 * np.max(floored.dx))
    assert _localized_low_modes(s_floor, f_max) == 0

    # Without the edge elimination they do appear in that band.
    no_elimination = _slanted_grid(
        stl_file,
        conformal_stabilization="area_floor",
        conformal_min_edge_fraction=0.0,
    )
    assert _localized_low_modes(_solver(no_elimination), f_max) > 0
