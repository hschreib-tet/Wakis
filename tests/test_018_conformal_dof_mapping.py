"""
Conformal FIT: grid-geometry to field-DOF mappings must follow the curl matrix.

The curl matrix C defines which primal edges bound the primal face of each
magnetic DOF, and C^T defines which magnetic DOFs bound the dual face of each
electric DOF. All conformal fractions and face materials of one DOF must be
taken from that same face/edge.
"""

import sys

import numpy as np
import pytest
import pyvista as pv

sys.path.append("../wakis")

from wakis import GridFIT3D, SolverFIT3D  # noqa: E402

AXES = ("x", "y", "z")


def _flat(solver, direction, index):
    i, j, k = index
    return AXES.index(direction) * solver.N + i + solver.Nx * (j + solver.Ny * k)


def _unflat(solver, flat):
    component, rest = divmod(int(flat), solver.N)
    i = rest % solver.Nx
    j = (rest // solver.Nx) % solver.Ny
    k = rest // (solver.Nx * solver.Ny)
    return AXES[component], (i, j, k)


def _mapped_dofs(solver, mapping, array_shape, direction, index, **kwargs):
    """Field DOFs that receive the value placed at one array entry."""
    values = np.zeros(array_shape, dtype=float)
    values[index] = 1.0
    field = mapping(values, direction, **kwargs)
    return {tuple(int(v) for v in idx) for idx in np.argwhere(field == 1.0)}


@pytest.fixture(scope="module")
def plain_solver():
    grid = GridFIT3D(0, 1, 0, 1, 0, 1, 4, 5, 6, verbose=0)
    return SolverFIT3D(
        grid,
        use_stl=False,
        verbose=0,
        bc_low=["periodic"] * 3,
        bc_high=["periodic"] * 3,
    )


def test_primal_face_mapping_matches_curl(plain_solver):
    """Hd(i,j,k) is the flux through the primal d-face (i,j,k)."""
    s = plain_solver
    C = s.C.tocsr()
    face_shapes = {
        "x": (s.Nx + 1, s.Ny, s.Nz),
        "y": (s.Nx, s.Ny + 1, s.Nz),
        "z": (s.Nx, s.Ny, s.Nz + 1),
    }

    for d in AXES:
        index = (2, 2, 3)
        dofs = _mapped_dofs(
            s,
            s._primal_face_array_to_dual_edge_component,
            face_shapes[d],
            d,
            index,
        )
        assert dofs == {index}

        # Primal edges in this row of C must all lie on the face plane.
        axis = AXES.index(d)
        row = C[_flat(s, d, index)]
        for col in row.nonzero()[1]:
            edge_dir, edge_index = _unflat(s, col)
            assert edge_dir != d
            assert edge_index[axis] == index[axis]


def test_dual_face_mapping_matches_curl_transpose(plain_solver):
    """The dual face of Ed(i,j,k) is bounded by the H DOFs in row C^T."""
    s = plain_solver
    CT = s.C.transpose().tocsr()
    dual_edge_shapes = {
        "x": (s.Nx, s.Ny + 1, s.Nz + 1),
        "y": (s.Nx + 1, s.Ny, s.Nz + 1),
        "z": (s.Nx + 1, s.Ny + 1, s.Nz),
    }
    dual_face_shapes = {
        "x": (s.Nx + 1, s.Ny, s.Nz),
        "y": (s.Nx, s.Ny + 1, s.Nz),
        "z": (s.Nx, s.Ny, s.Nz + 1),
    }
    # Dual edges bounding dual d-face (i,j,k), as (direction, index).
    boundary = {
        "x": lambda i, j, k: [
            ("y", (i, j, k)),
            ("y", (i, j, k + 1)),
            ("z", (i, j, k)),
            ("z", (i, j + 1, k)),
        ],
        "y": lambda i, j, k: [
            ("x", (i, j, k)),
            ("x", (i, j, k + 1)),
            ("z", (i, j, k)),
            ("z", (i + 1, j, k)),
        ],
        "z": lambda i, j, k: [
            ("x", (i, j, k)),
            ("x", (i, j + 1, k)),
            ("y", (i, j, k)),
            ("y", (i + 1, j, k)),
        ],
    }

    for d in AXES:
        face = (2, 2, 3)
        e_dofs = _mapped_dofs(
            s,
            s._dual_face_array_to_primal_edge_component,
            dual_face_shapes[d],
            d,
            face,
        )
        assert len(e_dofs) == 1
        (e_index,) = e_dofs

        h_from_geometry = set()
        for edge_dir, edge_index in boundary[d](*face):
            mapped = _mapped_dofs(
                s,
                s._dual_edge_array_to_field_component,
                dual_edge_shapes[edge_dir],
                edge_dir,
                edge_index,
            )
            assert len(mapped) == 1
            h_from_geometry.add((edge_dir, mapped.pop()))

        row = CT[_flat(s, d, e_index)]
        h_from_curl = {_unflat(s, col) for col in row.nonzero()[1]}

        assert h_from_geometry == h_from_curl


def test_cut_areas_and_face_materials_use_same_dofs(tmp_path):
    """
    Inclined PEC plane: every H DOF with a reduced primal face area must be
    the DOF that receives the material of that same cut face, and vice
    versa. Conductor walls leave the dual grid uncut.
    """
    stl_file = tmp_path / "inclined_pec.stl"
    surface = pv.Box(bounds=(-3, 3, -3, 3, -6, 0.0)).triangulate()
    surface = surface.rotate_y(17, point=(0.5, 0.5, 0.0))
    surface = surface.rotate_x(11, point=(0.5, 0.5, 0.0))
    surface = surface.translate((0, 0, 0.47))
    surface.save(stl_file)

    N = 8
    grid = GridFIT3D(
        0,
        1,
        0,
        1,
        0,
        1,
        N,
        N,
        N,
        stl_solids={"pec": str(stl_file)},
        stl_materials={"pec": "pec"},
        stl_material_types={"pec": "pec"},
        geometry_mode="conformal",
        bg_material_type="normal",
        stl_method="interior_points",
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        bg=[2.0, 1.0, 0.0],
        use_stl=True,
        verbose=0,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
    )

    for d in AXES:
        primal_cut = solver.conformal_face_data["primal"][d]["normal_pec"]
        h_material = solver._primal_face_array_to_dual_edge_component(
            primal_cut, d
        ).astype(bool)
        h_area = solver.A_fraction.to_matrix(d) < 1.0 - 1e-12

        assert np.any(h_area)
        np.testing.assert_array_equal(h_area, h_material)

        # No dual intersections or cut dual faces at a PEC wall.
        assert not np.any(grid.interface_intersections["dual"][d]["has_crossing"])
        assert not np.any(solver.conformal_face_data["dual"][d]["normal_pec"])

    np.testing.assert_array_equal(solver.tL.toarray(), grid.tL.toarray())
    np.testing.assert_array_equal(solver.itA.toarray(), grid.itA.toarray())
