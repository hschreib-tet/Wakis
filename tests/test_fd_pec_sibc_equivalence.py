"""Regression test: before adding D_Zs, SIBC must assemble like PEC.

Run in the WAKIS repository with:
    python -m pytest -v -s tests/test_fd_pec_sibc_equivalence.py

Requires the conformal STL pipeline and the FD conductor-mask change.
An expected failure identifies a missing PEC-equivalent SIBC bulk treatment.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.sparse import issparse

from wakis import GridFIT3D
from wakis.solverFIT3D_FD import SolverFIT3D_FD


def _solver(stl_path, material_type):
    # Identical geometry, but physically correct distinct material definitions.
    if material_type == "pec":
        material = [np.inf, 1.0, 0.0]
    else:
        material = [1.0, 1.0, 5.8e7]

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
        stl_materials={"wall": material},
        stl_material_types={"wall": material_type},
        geometry_mode="conformal",
        stl_method="interior_points",
        verbose=0,
    )
    return SolverFIT3D_FD(
        grid=grid,
        use_stl=True,
        use_sibc=False,  # Legacy TD SIBC switch must stay disabled.
        bc_low=["PEC", "PEC", "PEC"],
        bc_high=["PEC", "PEC", "PEC"],
        verbose=0,
    )


def _max_sparse_diff(left, right):
    assert issparse(left) and issparse(right)
    assert left.shape == right.shape
    diff = (left - right).tocsr()
    diff.eliminate_zeros()
    return float(np.max(np.abs(diff.data))) if diff.nnz else 0.0


@pytest.fixture(scope="module")
def solvers(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sibc_equivalence")
    path = folder / "conductor.stl"
    # Oblique, closed conductor: cuts primal/dual edges and faces.
    solid = pv.Cube(
        center=(0.08, -0.04, 0.03), x_length=0.97, y_length=0.85, z_length=0.92
    ).triangulate()
    solid.rotate_z(21.0, inplace=True)
    solid.rotate_y(13.0, inplace=True)
    solid.save(str(path))
    return _solver(path, "pec"), _solver(path, "sibc")


def test_pec_sibc_identical_before_dzs(solvers):
    pec, sibc = solvers

    # Every electric DOF must match, not merely the number of DOFs.
    print("PEC active DOFs:", len(pec.free_E_idx))
    print("SIBC active DOFs:", len(sibc.free_E_idx))

    # Compare geometric conductor masks
    geom_pec = pec.active_conductor_E
    geom_sibc = sibc.active_conductor_E

    # Compare material masks
    mat_pec = pec.iDeps.diagonal() != 0.0
    mat_sibc = sibc.iDeps.diagonal() != 0.0

    # DOFs active only for SIBC
    extra_sibc = sibc.active_E & ~pec.active_E

    print("\n--- PEC / SIBC diagnostic ---")
    print("Geometry masks identical:", np.array_equal(geom_pec, geom_sibc))

    print("Material masks identical:", np.array_equal(mat_pec, mat_sibc))

    print("Extra SIBC DOFs:", np.count_nonzero(extra_sibc))

    print(
        "Extra SIBC DOFs inside geometric conductor mask:",
        np.count_nonzero(extra_sibc & ~geom_sibc),
    )

    print(
        "Extra SIBC DOFs outside geometric conductor mask:",
        np.count_nonzero(extra_sibc & geom_sibc),
    )

    print(
        "PEC-active but SIBC-inactive DOFs:",
        np.count_nonzero(pec.active_E & ~sibc.active_E),
    )

    extra_sibc = sibc.active_E & ~pec.active_E

    print("\n--- Classification of extra SIBC DOFs ---")

    for component, direction in enumerate(("x", "y", "z")):
        start_ids, end_ids = sibc.grid._get_edge_region_ids("primal", direction)

        conductor_ids = [
            region_id
            for region_id, material_type in sibc.region_material_types.items()
            if material_type in ("pec", "sibc")
        ]

        start_conductor = np.isin(start_ids, conductor_ids)
        end_conductor = np.isin(end_ids, conductor_ids)

        normal_normal = ~start_conductor & ~end_conductor
        normal_conductor = start_conductor ^ end_conductor
        conductor_conductor = start_conductor & end_conductor

        # Map full geometric edge arrays to stored electric DOFs.
        nn = sibc._edge_array_to_field_component(normal_normal, direction).ravel(
            order="C"
        )

        nc = sibc._edge_array_to_field_component(normal_conductor, direction).ravel(
            order="C"
        )

        cc = sibc._edge_array_to_field_component(conductor_conductor, direction).ravel(
            order="C"
        )

        offset = component * sibc.N
        extra = extra_sibc[offset : offset + sibc.N]

        print(f"\n{direction}-edges:")
        print("  Extra DOFs:", np.count_nonzero(extra))
        print("  Normal-normal:", np.count_nonzero(extra & nn))
        print("  Normal-conductor:", np.count_nonzero(extra & nc))
        print("  Conductor-conductor:", np.count_nonzero(extra & cc))

    extra = sibc.active_E & ~pec.active_E

    pec_ieps = pec.iDeps.diagonal()
    sibc_ieps = sibc.iDeps.diagonal()

    pec_sigma = pec.Dsigma.diagonal()
    sibc_sigma = sibc.Dsigma.diagonal()

    print("\n--- Material values at extra SIBC DOFs ---")

    print("PEC ieps == 0:", np.count_nonzero(pec_ieps[extra] == 0.0))

    print("SIBC ieps == 0:", np.count_nonzero(sibc_ieps[extra] == 0.0))

    if np.any(extra):
        print("PEC sigma min/max:", np.min(pec_sigma[extra]), np.max(pec_sigma[extra]))

        print(
            "SIBC sigma min/max:", np.min(sibc_sigma[extra]), np.max(sibc_sigma[extra])
        )
    else:
        print("No additional SIBC DOFs. Material diagnostic skipped.")

    assert np.array_equal(pec.active_conductor_E, sibc.active_conductor_E), (
        "Geometric conductor mask differs"
    )
    assert np.array_equal(pec.active_E, sibc.active_E), (
        "Active E mask differs: inspect iDeps / material assignment"
    )
    assert np.array_equal(pec.free_E_idx, sibc.free_E_idx)

    # The actual sparse operators must be equal as well.
    matrices = ("C_red", "Mnu", "Meps", "Msigma", "SL")
    differences = {}
    for name in matrices:
        a, b = getattr(pec, name), getattr(sibc, name)
        differences[name] = _max_sparse_diff(a, b)
        print(f"{name:8s}: max |PEC - SIBC| = {differences[name]:.6e}")

    # Relative comparison for matrices that carry very different SI scales.
    for name in matrices:
        a, b = getattr(pec, name), getattr(sibc, name)
        scale = max(
            float(np.max(np.abs(a.data))) if a.nnz else 0.0,
            float(np.max(np.abs(b.data))) if b.nnz else 0.0,
            1e-300,
        )
        assert differences[name] <= 1e-12 * scale, (
            f"{name} differs: abs={differences[name]:.6e}, scale={scale:.6e}"
        )

    for frequency in (1e8, 1e9):
        omega = 2 * np.pi * frequency
        a = pec.build_system_matrix(omega)
        b = sibc.build_system_matrix(omega)
        print(f"A({frequency:.3e} Hz): max diff = {_max_sparse_diff(a, b):.6e}")
        scale = max(
            float(np.max(np.abs(a.data))) if a.nnz else 0.0,
            float(np.max(np.abs(b.data))) if b.nnz else 0.0,
            1e-300,
        )
        assert _max_sparse_diff(a, b) <= 1e-12 * scale
