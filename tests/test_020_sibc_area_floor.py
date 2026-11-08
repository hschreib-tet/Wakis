"""
The SIBC term and the reluctance matrix must convert b -> H with the same
face area, also when conformal_stabilization="area_floor" raises small areas.
"""

import numpy as np
import pytest
import pyvista as pv

from wakis import GridFIT3D
from wakis.solverFIT3D_FD import SolverFIT3D_FD


def _sibc_solver(stl_path, **stabilization):
    grid = GridFIT3D(
        xmin=-1.0,
        xmax=1.0,
        ymin=-1.0,
        ymax=1.0,
        zmin=-1.0,
        zmax=1.0,
        Nx=12,
        Ny=12,
        Nz=12,
        stl_solids={"wall": str(stl_path)},
        stl_materials={"wall": [1.0, 1.0, 5.8e7]},
        stl_material_types={"wall": "sibc"},
        geometry_mode="conformal",
        stl_method="interior_points",
        verbose=0,
        **stabilization,
    )
    return SolverFIT3D_FD(
        grid=grid,
        use_stl=True,
        use_sibc=False,
        bc_low=["PEC", "PEC", "PEC"],
        bc_high=["PEC", "PEC", "PEC"],
        verbose=0,
    )


@pytest.fixture(scope="module")
def stl_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("sibc_area_floor") / "conductor.stl"
    solid = pv.Cube(
        center=(0.08, -0.04, 0.03), x_length=0.97, y_length=0.85, z_length=0.92
    ).triangulate()
    solid.rotate_z(21.0, inplace=True)
    solid.rotate_y(13.0, inplace=True)
    solid.save(str(path))
    return path


def test_sibc_uses_exact_area_without_floor(stl_path):
    solver = _sibc_solver(stl_path)
    active = solver.sibc_g > 0
    assert np.any(active)

    # Without floor, the area in Mnu is the exact normal-region face area.
    np.testing.assert_allclose(
        solver.A_fraction.toarray(), solver.A_fraction_geometric.toarray()
    )


def test_sibc_uses_floored_area(stl_path):
    exact = _sibc_solver(stl_path)
    floored = _sibc_solver(
        stl_path,
        conformal_stabilization="area_floor",
        conformal_stabilization_fraction=0.1,
    )

    active = (exact.sibc_g > 0) & (floored.sibc_g > 0)
    assert np.count_nonzero(active) > 0

    # g ~ 1 / A, and A is exactly the area of Mnu (1 / iA).
    ratio_g = floored.sibc_g[active] / exact.sibc_g[active]
    ratio_area = floored.iA.toarray()[active] / exact.iA.toarray()[active]
    np.testing.assert_allclose(ratio_g, ratio_area, rtol=1e-10)

    # Some SIBC faces are actually raised by the floor.
    assert np.any(ratio_g < 1.0 - 1e-12)
