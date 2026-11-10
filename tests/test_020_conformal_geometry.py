"""Conformal STL geometry: point masks, interface topology, edge and face
cuts, DOF mapping, and conformal material/metric assignment."""

from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0

from wakis import GridFIT3D, SolverFIT3D


@pytest.mark.slow
class TestConformalGeometry:
    def test_conformal_geometry_masks(self):
        """
        Test conformal STL mask generation.

        Verifies that:
        1. the primal point mask is defined on all primal grid points,
        2. the cell mask stored in ``grid.grid[key]`` follows the
           5-of-8 corner-point rule,
        3. the dual point mask is classified directly on the dual FIT grid.
        """

        # Geometry & Materials
        solid_1 = "tests/stl/007_vacuum_cavity.stl"
        solid_2 = "tests/stl/007_lossymetal_shell.stl"

        stl_solids = {
            "cavity": solid_1,
            "shell": solid_2,
        }

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30, 1.0, 30],
        }

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

        # Number of primal nodes (60 x 60 x 140 physical cells)
        Nx = 61
        Ny = 61
        Nz = 141

        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            Nx,
            Ny,
            Nz,
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_method="voxelize_rectilinear",
            geometry_mode="conformal",
            subpixel_smoothing=False,
            stl_scale=1.0,
            stl_rotate=[0, 0, 0],
            stl_translate=[0, 0, 0],
            verbose=1,
        )

        # ----------------------------------------------------------
        # 1. Primal point mask
        # ----------------------------------------------------------
        primal = grid.primal_point_masks["shell"]

        assert primal.shape == (Nx, Ny, Nz)

        assert primal.dtype == bool

        # ----------------------------------------------------------
        # 2. Independently reconstruct the expected 5-of-8
        #    cell-center classification
        # ----------------------------------------------------------
        corner_count = (
            primal[:-1, :-1, :-1].astype(np.uint8)
            + primal[1:, :-1, :-1].astype(np.uint8)
            + primal[:-1, 1:, :-1].astype(np.uint8)
            + primal[1:, 1:, :-1].astype(np.uint8)
            + primal[:-1, :-1, 1:].astype(np.uint8)
            + primal[1:, :-1, 1:].astype(np.uint8)
            + primal[:-1, 1:, 1:].astype(np.uint8)
            + primal[1:, 1:, 1:].astype(np.uint8)
        )

        expected_cell_mask = corner_count > 4

        assert expected_cell_mask.shape == (Nx - 1, Ny - 1, Nz - 1)

        # Cell mask stored in the PyVista grid, including the zero-extent
        # high-side ghost cells of the symmetric stencil.
        cell_mask = np.reshape(
            np.asarray(grid.grid["shell"], dtype=bool),
            (Nx, Ny, Nz),
            order="C",
        )

        np.testing.assert_array_equal(
            cell_mask[:-1, :-1, :-1],
            expected_cell_mask,
        )

        # ----------------------------------------------------------
        # 3. Dual point mask
        # ----------------------------------------------------------
        dual = grid.dual_point_masks["shell"]

        assert dual.shape == (
            len(grid.tx),
            len(grid.ty),
            len(grid.tz),
        )

        assert dual.dtype == bool

        assert np.any(dual)
        assert np.any(~dual)

        # Dual points: both domain boundaries and the primal cell centres.
        assert dual.shape == (Nx + 1, Ny + 1, Nz + 1)
        assert grid.tx[0] == grid.x[0] and grid.tx[-1] == grid.x[-1]

    def test_average_dual_points_to_primal_edges(self):
        """Test dual-point averaging onto the dual faces of primal edges."""

        # Nx, Ny, Nz count primal nodes; dual points are (Nx+1, Ny+1, Nz+1).
        Nx = 3
        Ny = 4
        Nz = 5

        # Unique value at every grid point, so indexing errors are visible.
        values = np.arange(
            (Nx + 1) * (Ny + 1) * (Nz + 1),
            dtype=float,
        ).reshape(Nx + 1, Ny + 1, Nz + 1)

        values_x, values_y, values_z = SolverFIT3D._average_dual_points_to_primal_edges(
            values
        )

        assert values_x.shape == (Nx, Ny, Nz)
        assert values_y.shape == (Nx, Ny, Nz)
        assert values_z.shape == (Nx, Ny, Nz)

        # Ex(i,j,k) -> dual x-face (i+1, j, k): dual points (i+1, j..j+1, k..k+1)
        i, j, k = 1, 2, 3
        expected = 0.25 * (
            values[i + 1, j, k]
            + values[i + 1, j + 1, k]
            + values[i + 1, j + 1, k + 1]
            + values[i + 1, j, k + 1]
        )
        assert values_x[i, j, k] == pytest.approx(expected)

        # Ey(i,j,k) -> dual y-face (i, j+1, k)
        i, j, k = 2, 1, 3
        expected = 0.25 * (
            values[i, j + 1, k]
            + values[i + 1, j + 1, k]
            + values[i + 1, j + 1, k + 1]
            + values[i, j + 1, k + 1]
        )
        assert values_y[i, j, k] == pytest.approx(expected)

        # Ez(i,j,k) -> dual z-face (i, j, k+1)
        i, j, k = 2, 2, 1
        expected = 0.25 * (
            values[i, j, k + 1]
            + values[i + 1, j, k + 1]
            + values[i + 1, j + 1, k + 1]
            + values[i, j + 1, k + 1]
        )
        assert values_z[i, j, k] == pytest.approx(expected)

        # Boundary edges have complete (half-size) dual faces: the dual
        # grid includes the domain boundary, so no extrapolation is used.
        i, j, k = 1, 0, 0
        expected = 0.25 * (
            values[i + 1, 0, 0]
            + values[i + 1, 1, 0]
            + values[i + 1, 1, 1]
            + values[i + 1, 0, 1]
        )
        assert values_x[i, j, k] == pytest.approx(expected)

        i, j, k = 1, Ny - 1, Nz - 1
        expected = 0.25 * (
            values[i + 1, Ny - 1, Nz - 1]
            + values[i + 1, Ny, Nz - 1]
            + values[i + 1, Ny, Nz]
            + values[i + 1, Ny - 1, Nz]
        )
        assert values_x[i, j, k] == pytest.approx(expected)

    def test_average_primal_points_to_dual_edges(self):
        """Test primal-point averaging onto the primal faces of dual edges."""

        # Nx, Ny, Nz count primal nodes.
        Nx = 3
        Ny = 4
        Nz = 5

        values = np.arange(Nx * Ny * Nz, dtype=float).reshape(Nx, Ny, Nz)

        values_x, values_y, values_z = SolverFIT3D._average_primal_points_to_dual_edges(
            values
        )

        assert values_x.shape == (Nx, Ny, Nz)
        assert values_y.shape == (Nx, Ny, Nz)
        assert values_z.shape == (Nx, Ny, Nz)

        # Hx(i,j,k) -> primal yz face at x[i] (curl matrix C)
        i, j, k = 1, 2, 3
        expected = 0.25 * (
            values[i, j, k]
            + values[i, j + 1, k]
            + values[i, j + 1, k + 1]
            + values[i, j, k + 1]
        )
        assert values_x[i, j, k] == pytest.approx(expected)

        # Hy(i,j,k) -> primal xz face at y[j]
        i, j, k = 1, 1, 3
        expected = 0.25 * (
            values[i, j, k]
            + values[i + 1, j, k]
            + values[i + 1, j, k + 1]
            + values[i, j, k + 1]
        )
        assert values_y[i, j, k] == pytest.approx(expected)

        # Hz(i,j,k) -> primal xy face at z[k]
        i, j, k = 1, 2, 1
        expected = 0.25 * (
            values[i, j, k]
            + values[i + 1, j, k]
            + values[i + 1, j + 1, k]
            + values[i, j + 1, k]
        )
        assert values_z[i, j, k] == pytest.approx(expected)

        # Hx(0,j,k) belongs to the boundary face at x[0].
        i, j, k = 0, Ny - 2, Nz - 2
        expected = 0.25 * (
            values[0, Ny - 2, Nz - 2]
            + values[0, Ny - 1, Nz - 2]
            + values[0, Ny - 1, Nz - 1]
            + values[0, Ny - 2, Nz - 1]
        )
        assert values_x[i, j, k] == pytest.approx(expected)

        # Ghost DOFs (zero-area high-side faces) copy the nearest face.
        assert values_x[0, Ny - 1, Nz - 1] == pytest.approx(expected)

    def test_conformal_material_assignment(self):
        """Test point-based material assignment onto FIT material tensors."""

        # Primal nodes (2 x 2 x 2 physical cells); fields have this shape.
        Nx = 3
        Ny = 3
        Nz = 3

        shape_cells = (Nx, Ny, Nz)
        shape_primal = (Nx, Ny, Nz)
        shape_dual = (Nx + 1, Ny + 1, Nz + 1)

        # ----------------------------------------------------------
        # Synthetic primal and dual point masks
        # ----------------------------------------------------------
        primal_mask = np.zeros(shape_primal, dtype=bool)
        dual_mask = np.zeros(shape_dual, dtype=bool)

        # For eps_x[1, 1, 1], the associated dual x-face (2, 1, 1) uses
        #
        # (2,1,1), (2,2,1), (2,2,2), (2,1,2).
        #
        # Mark two of these four points as material.
        dual_mask[2, 2, 2] = True
        dual_mask[2, 1, 2] = True

        # For mu_x[1, 0, 0], the associated primal yz face at x[1]
        # (curl matrix C) uses
        #
        # (1,0,0), (1,1,0), (1,1,1), (1,0,1).
        #
        # Mark three of these four points as material.
        primal_mask[1, 0, 0] = True
        primal_mask[1, 1, 0] = True
        primal_mask[1, 1, 1] = True

        # ----------------------------------------------------------
        # Minimal grid object required by the conformal material path
        # ----------------------------------------------------------
        grid = SimpleNamespace(
            stl_solids={"solid": None},
            stl_materials={"solid": [5.0, 3.0, 8.0]},
            stl_colors={"solid": None},
            primal_point_masks={"solid": primal_mask},
            dual_point_masks={"solid": dual_mask},
        )

        # ----------------------------------------------------------
        # Minimal solver object
        # ----------------------------------------------------------
        solver = SolverFIT3D.__new__(SolverFIT3D)

        solver.Nx = Nx
        solver.Ny = Ny
        solver.Nz = Nz
        solver.grid = grid

        solver.dtype = np.float64

        solver.eps_bg = eps_0
        solver.mu_bg = mu_0
        solver.sigma_bg = 0.0

        solver.use_sibc = False
        solver.use_conductivity = False
        solver.verbose = 0

        solver.ieps = SimpleNamespace(
            field_x=np.zeros(shape_cells),
            field_y=np.zeros(shape_cells),
            field_z=np.zeros(shape_cells),
        )

        solver.imu = SimpleNamespace(
            field_x=np.zeros(shape_cells),
            field_y=np.zeros(shape_cells),
            field_z=np.zeros(shape_cells),
        )

        solver.sigma = SimpleNamespace(
            field_x=np.zeros(shape_cells),
            field_y=np.zeros(shape_cells),
            field_z=np.zeros(shape_cells),
        )

        # ----------------------------------------------------------
        # Apply new conformal material discretization
        # ----------------------------------------------------------
        solver._apply_stl_materials_conformal()

        # ----------------------------------------------------------
        # Electric material relation
        # ----------------------------------------------------------
        # Two of four dual-face points have eps_r = 5,
        # two remain background eps_r = 1.
        #
        # eps_eff / eps_0 = (5 + 5 + 1 + 1) / 4 = 3
        expected_eps = 3.0 * eps_0

        assert solver.ieps.field_x[1, 1, 1] == pytest.approx(1.0 / expected_eps)

        # Same two points carry sigma = 8 S/m:
        #
        # sigma_eff = (8 + 8 + 0 + 0) / 4 = 4 S/m
        assert solver.sigma.field_x[1, 1, 1] == pytest.approx(4.0)

        # ----------------------------------------------------------
        # Magnetic material relation
        # ----------------------------------------------------------
        # Three of four primal-face points have mu_r = 3,
        # one remains background mu_r = 1.
        #
        # mu_eff / mu_0 = (3 + 3 + 3 + 1) / 4 = 2.5
        expected_mu = 2.5 * mu_0

        assert solver.imu.field_x[1, 0, 0] == pytest.approx(1.0 / expected_mu)

        # Conductive material must enable conductivity treatment.
        assert solver.use_conductivity

    def test_conformal_material_assignment_homogeneous(self):
        """Test conformal material assignment for a homogeneous solid."""

        # Primal nodes; fields have this shape.
        Nx = 3
        Ny = 4
        Nz = 5

        shape_cells = (Nx, Ny, Nz)

        # All primal and dual grid points belong to the material.
        primal_mask = np.ones((Nx, Ny, Nz), dtype=bool)
        dual_mask = np.ones((Nx + 1, Ny + 1, Nz + 1), dtype=bool)

        eps_r = 4.0
        mu_r = 2.5
        sigma = 7.0

        grid = SimpleNamespace(
            stl_solids={"solid": None},
            stl_materials={"solid": [eps_r, mu_r, sigma]},
            stl_colors={"solid": None},
            primal_point_masks={"solid": primal_mask},
            dual_point_masks={"solid": dual_mask},
        )

        solver = SolverFIT3D.__new__(SolverFIT3D)

        solver.Nx = Nx
        solver.Ny = Ny
        solver.Nz = Nz
        solver.grid = grid

        solver.dtype = np.float64

        solver.eps_bg = eps_0
        solver.mu_bg = mu_0
        solver.sigma_bg = 0.0

        solver.use_sibc = False
        solver.use_conductivity = False
        solver.verbose = 0

        solver.ieps = SimpleNamespace(
            field_x=np.zeros(shape_cells),
            field_y=np.zeros(shape_cells),
            field_z=np.zeros(shape_cells),
        )

        solver.imu = SimpleNamespace(
            field_x=np.zeros(shape_cells),
            field_y=np.zeros(shape_cells),
            field_z=np.zeros(shape_cells),
        )

        solver.sigma = SimpleNamespace(
            field_x=np.zeros(shape_cells),
            field_y=np.zeros(shape_cells),
            field_z=np.zeros(shape_cells),
        )

        solver._apply_stl_materials_conformal()

        expected_ieps = 1.0 / (eps_r * eps_0)
        expected_imu = 1.0 / (mu_r * mu_0)

        for field in (
            solver.ieps.field_x,
            solver.ieps.field_y,
            solver.ieps.field_z,
        ):
            np.testing.assert_allclose(field, expected_ieps)

        for field in (
            solver.imu.field_x,
            solver.imu.field_y,
            solver.imu.field_z,
        ):
            np.testing.assert_allclose(field, expected_imu)

        for field in (
            solver.sigma.field_x,
            solver.sigma.field_y,
            solver.sigma.field_z,
        ):
            np.testing.assert_allclose(field, sigma)

        assert solver.use_conductivity

    def test_conformal_material_assignment_pec(self):
        """
        A PEC region must deactivate the corresponding electric DOFs.

        PEC is represented by eps_r = infinity. After conformal
        dual-point averaging this must result in

            ieps = 1 / eps = 0.
        """
        # Primal nodes; fields have this shape.
        Nx = 3
        Ny = 3
        Nz = 3

        shape_cells = (Nx, Ny, Nz)

        primal_mask = np.ones((Nx, Ny, Nz), dtype=bool)
        dual_mask = np.ones((Nx + 1, Ny + 1, Nz + 1), dtype=bool)

        grid = SimpleNamespace(
            stl_solids={
                "pec": None,
            },
            stl_materials={
                "pec": [
                    np.inf,
                    1.0,
                    0.0,
                ],
            },
            stl_colors={
                "pec": None,
            },
            primal_point_masks={
                "pec": primal_mask,
            },
            dual_point_masks={
                "pec": dual_mask,
            },
        )

        solver = SolverFIT3D.__new__(SolverFIT3D)

        solver.Nx = Nx
        solver.Ny = Ny
        solver.Nz = Nz
        solver.grid = grid

        solver.dtype = np.float64

        solver.eps_bg = eps_0
        solver.mu_bg = mu_0
        solver.sigma_bg = 0.0

        solver.use_sibc = False
        solver.use_conductivity = False
        solver.verbose = 0

        solver.ieps = SimpleNamespace(
            field_x=np.zeros(shape_cells),
            field_y=np.zeros(shape_cells),
            field_z=np.zeros(shape_cells),
        )

        solver.imu = SimpleNamespace(
            field_x=np.zeros(shape_cells),
            field_y=np.zeros(shape_cells),
            field_z=np.zeros(shape_cells),
        )

        solver.sigma = SimpleNamespace(
            field_x=np.zeros(shape_cells),
            field_y=np.zeros(shape_cells),
            field_z=np.zeros(shape_cells),
        )

        solver._apply_stl_materials_conformal()

        for field in (
            solver.ieps.field_x,
            solver.ieps.field_y,
            solver.ieps.field_z,
        ):
            np.testing.assert_array_equal(
                field,
                np.zeros_like(field),
            )

        # PEC does not alter mu in the present material model.
        expected_imu = 1.0 / mu_0

        for field in (
            solver.imu.field_x,
            solver.imu.field_y,
            solver.imu.field_z,
        ):
            np.testing.assert_allclose(
                field,
                expected_imu,
            )

        for field in (
            solver.sigma.field_x,
            solver.sigma.field_y,
            solver.sigma.field_z,
        ):
            np.testing.assert_array_equal(
                field,
                np.zeros_like(field),
            )

    def test_conformal_material_assignment_rejects_sibc(self):
        """Test that SIBC is explicitly rejected for conformal geometry."""

        Nx = 2
        Ny = 2
        Nz = 2

        shape_points = (Nx + 1, Ny + 1, Nz + 1)

        mask = np.ones(shape_points, dtype=bool)

        grid = SimpleNamespace(
            stl_solids={"conductor": None},
            stl_materials={"conductor": [1.0, 1.0, 1.0e6]},
            stl_colors={"conductor": None},
            primal_point_masks={"conductor": mask},
            dual_point_masks={"conductor": mask},
        )

        solver = SolverFIT3D.__new__(SolverFIT3D)

        solver.Nx = Nx
        solver.Ny = Ny
        solver.Nz = Nz
        solver.grid = grid

        solver.dtype = np.float64

        solver.eps_bg = eps_0
        solver.mu_bg = mu_0
        solver.sigma_bg = 0.0

        solver.use_sibc = True
        solver.use_conductivity = False
        solver.verbose = 0

        with pytest.raises(
            NotImplementedError,
            match="SIBC.*not supported",
        ):
            solver._apply_stl_materials_conformal()

    def test_stl_material_types(self):
        solid_1 = "tests/stl/007_vacuum_cavity.stl"
        solid_2 = "tests/stl/007_lossymetal_shell.stl"

        stl_solids = {
            "cavity": solid_1,
            "shell": solid_2,
        }

        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

        # Explicit SIBC type with missing entries defaulting to normal.
        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            7,
            7,
            15,
            stl_solids=stl_solids,
            stl_materials={
                "cavity": "vacuum",
                "shell": [30.0, 1.0, 30.0],
            },
            stl_material_types={
                "shell": "sibc",
            },
            verbose=0,
        )

        assert grid.stl_material_types == {
            "cavity": "normal",
            "shell": "sibc",
        }

        # Existing PEC definitions must remain identifiable without
        # explicitly specifying stl_material_types.
        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            7,
            7,
            15,
            stl_solids=stl_solids,
            stl_materials={
                "cavity": "vacuum",
                "shell": "pec",
            },
            verbose=0,
        )

        assert grid.stl_material_types == {
            "cavity": "normal",
            "shell": "pec",
        }

    def test_background_material_type(self):
        # --------------------------------------------------
        # Backward compatibility:
        #
        # If the grid does not explicitly define the
        # background treatment, the solver infers the
        # unambiguous type from the supplied material.
        # --------------------------------------------------
        grid = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            3,
            3,
            3,
            verbose=0,
        )

        solver = SolverFIT3D(
            grid,
            bg=[1.0, 1.0, 0.0],
            verbose=0,
        )

        assert solver.bg_material_type == "normal"

        solver = SolverFIT3D(
            grid,
            bg="pec",
            verbose=0,
        )

        assert solver.bg_material_type == "pec"

        # --------------------------------------------------
        # Explicit treatment type belongs to GridFIT3D.
        #
        # SIBC cannot be inferred from finite constitutive
        # parameters and must therefore be declared by the grid.
        # --------------------------------------------------
        grid_sibc = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            3,
            3,
            3,
            bg_material_type="sibc",
            verbose=0,
        )

        solver = SolverFIT3D(
            grid_sibc,
            bg=[1.0, 1.0, 0.0],
            verbose=0,
        )

        assert solver.bg_material_type == "sibc"

        # --------------------------------------------------
        # Inconsistent PEC declarations must be rejected.
        # --------------------------------------------------
        grid_normal = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            3,
            3,
            3,
            bg_material_type="normal",
            verbose=0,
        )

        with pytest.raises(
            ValueError,
            match="background",
        ):
            SolverFIT3D(
                grid_normal,
                bg="pec",
                verbose=0,
            )

        grid_pec = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            3,
            3,
            3,
            bg_material_type="pec",
            verbose=0,
        )

        with pytest.raises(
            ValueError,
            match="background",
        ):
            SolverFIT3D(
                grid_pec,
                bg=[1.0, 1.0, 0.0],
                verbose=0,
            )

    def test_primal_and_dual_region_topology(self):
        grid = GridFIT3D(
            0.0,
            3.0,
            0.0,
            1.0,
            0.0,
            1.0,
            4,
            2,
            2,
            verbose=0,
        )

        grid.stl_solids = {
            "material_1": "unused",
            "material_2": "unused",
        }

        # Primal points (4, 2, 2); dual points (5, 3, 3) with
        # tx = [0, 0.5, 1.5, 2.5, 3].
        shape = (4, 2, 2)
        dual_shape = (5, 3, 3)

        # --------------------------------------------------
        # Primal point masks
        #
        # resulting x sequence:
        #
        # 0  1  2  2
        # --------------------------------------------------

        primal_mask_1 = np.zeros(
            shape,
            dtype=bool,
        )

        primal_mask_2 = np.zeros(
            shape,
            dtype=bool,
        )

        primal_mask_1[1:3, :, :] = True
        primal_mask_2[2:, :, :] = True

        grid.primal_point_masks = {
            "material_1": primal_mask_1,
            "material_2": primal_mask_2,
        }

        # --------------------------------------------------
        # Dual point masks
        #
        # resulting x sequence:
        #
        # 0  0  0  1  2
        # --------------------------------------------------

        dual_mask_1 = np.zeros(
            dual_shape,
            dtype=bool,
        )

        dual_mask_2 = np.zeros(
            dual_shape,
            dtype=bool,
        )

        dual_mask_1[3:, :, :] = True
        dual_mask_2[4:, :, :] = True

        grid.dual_point_masks = {
            "material_1": dual_mask_1,
            "material_2": dual_mask_2,
        }

        grid._build_interface_topology()

        assert grid.region_id_to_key == {
            0: None,
            1: "material_1",
            2: "material_2",
        }

        assert grid.region_key_to_id == {
            "material_1": 1,
            "material_2": 2,
        }

        assert np.array_equal(
            grid.region_ids["primal"][:, 0, 0],
            np.array([0, 1, 2, 2]),
        )

        assert np.array_equal(
            grid.region_ids["dual"][:, 0, 0],
            np.array([0, 0, 0, 1, 2]),
        )

        assert np.array_equal(
            grid.interface_edge_masks["primal"]["x"][:, 0, 0],
            np.array([True, True, False]),
        )

        assert np.array_equal(
            grid.interface_edge_masks["dual"]["x"][:, 0, 0],
            np.array([False, False, True, True]),
        )

        for grid_type in ("primal", "dual"):
            assert not np.any(grid.interface_edge_masks[grid_type]["y"])

            assert not np.any(grid.interface_edge_masks[grid_type]["z"])

    def test_primal_and_dual_interface_transition_types(
        self,
    ):
        grid = GridFIT3D(
            0.0,
            4.0,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            2,
            2,
            geometry_mode="conformal",
            bg_material_type="normal",
            verbose=0,
        )

        grid.region_id_to_key = {
            0: None,
            1: "normal_1",
            2: "pec",
            3: "normal_2",
            4: "sibc",
        }

        grid.region_key_to_id = {
            "normal_1": 1,
            "pec": 2,
            "normal_2": 3,
            "sibc": 4,
        }

        grid.stl_material_types = {
            "normal_1": "normal",
            "pec": "pec",
            "normal_2": "normal",
            "sibc": "sibc",
        }

        shape = (5, 2, 2)

        primal_regions = np.zeros(
            shape,
            dtype=np.int32,
        )

        dual_regions = np.zeros(
            shape,
            dtype=np.int32,
        )

        # Primal:
        #
        # normal -> normal -> PEC -> normal -> SIBC
        for i, region_id in enumerate([0, 1, 2, 3, 4]):
            primal_regions[i, :, :] = region_id

        # Dual: same interfaces in reverse order
        #
        # SIBC -> normal -> PEC -> normal -> normal
        for i, region_id in enumerate([4, 3, 2, 1, 0]):
            dual_regions[i, :, :] = region_id

        grid.region_ids = {
            "primal": primal_regions,
            "dual": dual_regions,
        }

        grid.interface_edge_masks = {
            "primal": grid._build_edge_transition_masks(primal_regions),
            "dual": grid._build_edge_transition_masks(dual_regions),
        }

        solver = SolverFIT3D(
            grid,
            bg=[1.0, 1.0, 0.0],
            use_stl=False,
            verbose=0,
        )

        assert solver.region_material_types == {
            0: "normal",
            1: "normal",
            2: "pec",
            3: "normal",
            4: "sibc",
        }

        primal = solver.interface_transition_masks["primal"]["x"]

        assert np.array_equal(
            primal["normal_normal"][:, 0, 0],
            np.array([True, False, False, False]),
        )

        assert np.array_equal(
            primal["normal_pec"][:, 0, 0],
            np.array([False, True, True, False]),
        )

        assert np.array_equal(
            primal["normal_sibc"][:, 0, 0],
            np.array([False, False, False, True]),
        )

        dual = solver.interface_transition_masks["dual"]["x"]

        assert np.array_equal(
            dual["normal_sibc"][:, 0, 0],
            np.array([True, False, False, False]),
        )

        assert np.array_equal(
            dual["normal_pec"][:, 0, 0],
            np.array([False, True, True, False]),
        )

        assert np.array_equal(
            dual["normal_normal"][:, 0, 0],
            np.array([False, False, False, True]),
        )

        for grid_type in ("primal", "dual"):
            masks = solver.interface_transition_masks[grid_type]["x"]

            classified = np.zeros_like(grid.interface_edge_masks[grid_type]["x"])

            for mask in masks.values():
                classified |= mask

            assert np.array_equal(
                classified,
                grid.interface_edge_masks[grid_type]["x"],
            )

    def test_split_intersection_groups(self):
        crossing_group = [
            {
                "alpha": 0.25,
                "surface_id": 1,
                "cell_id": 10,
                "normal": np.array([1.0, 0.0, 0.0]),
            }
        ]

        contact_group = [
            {
                "alpha": 0.50,
                "surface_id": 1,
                "cell_id": 11,
                "normal": np.array([0.0, 1.0, 0.0]),
            }
        ]

        mixed_group = [
            {
                "alpha": 0.75,
                "surface_id": 1,
                "cell_id": 12,
                "normal": np.array([0.0, 1.0, 0.0]),
            },
            {
                "alpha": 0.75,
                "surface_id": 2,
                "cell_id": 13,
                "normal": np.array([0.5, 0.5, 0.0]),
            },
        ]

        crossing_groups, contact_groups = GridFIT3D._split_intersection_groups(
            [
                crossing_group,
                contact_group,
                mixed_group,
            ],
            direction="x",
        )

        assert crossing_groups == [
            crossing_group,
            mixed_group,
        ]

        assert contact_groups == [
            contact_group,
        ]

    def test_select_crossing_group(self):
        group_left = [
            {
                "alpha": 0.20,
                "surface_id": 1,
                "cell_id": 10,
                "normal": np.array([1.0, 0.0, 0.0]),
            }
        ]

        group_center = [
            {
                "alpha": 0.55,
                "surface_id": 1,
                "cell_id": 11,
                "normal": np.array([1.0, 0.0, 0.0]),
            }
        ]

        selected = GridFIT3D._select_crossing_group(
            [
                group_left,
                group_center,
            ]
        )

        assert selected is group_center

        # Equal distance from the edge midpoint:
        # prefer the group with the higher surface ID.
        group_a = [
            {
                "alpha": 0.40,
                "surface_id": 1,
                "cell_id": 12,
                "normal": np.array([1.0, 0.0, 0.0]),
            }
        ]

        group_b = [
            {
                "alpha": 0.60,
                "surface_id": 2,
                "cell_id": 13,
                "normal": np.array([1.0, 0.0, 0.0]),
            }
        ]

        selected = GridFIT3D._select_crossing_group(
            [
                group_a,
                group_b,
            ]
        )
        assert selected is group_b

    def test_conformal_line_fractions(self):
        grid = GridFIT3D(
            0.0,
            4.0,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            2,
            2,
            geometry_mode="conformal",
            bg_material_type="normal",
            verbose=0,
        )

        grid.region_id_to_key = {
            0: None,
            1: "normal",
            2: "pec",
            3: "sibc",
        }

        grid.region_key_to_id = {
            "normal": 1,
            "pec": 2,
            "sibc": 3,
        }

        grid.stl_material_types = {
            "normal": "normal",
            "pec": "pec",
            "sibc": "sibc",
        }

        # x-directed sequence:
        #
        # background(normal)
        #     -> normal
        #     -> PEC
        #     -> normal
        #     -> SIBC
        regions = np.zeros(
            (5, 2, 2),
            dtype=np.int32,
        )

        for i, region_id in enumerate([0, 1, 2, 1, 3]):
            regions[i, :, :] = region_id

        grid.region_ids = {
            "primal": regions.copy(),
            "dual": regions.copy(),
        }

        grid.interface_edge_masks = {
            "primal": (grid._build_edge_transition_masks(regions)),
            "dual": (grid._build_edge_transition_masks(regions)),
        }

        grid.resolved_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        grid.interface_intersections = {
            "primal": {},
            "dual": {},
        }

        for grid_type in (
            "primal",
            "dual",
        ):
            for direction in (
                "x",
                "y",
                "z",
            ):
                mask = grid.interface_edge_masks[grid_type][direction]

                alpha = np.full(
                    mask.shape,
                    np.nan,
                    dtype=float,
                )

                grid.resolved_interface_edge_masks[grid_type][direction] = mask.copy()

                grid.interface_intersections[grid_type][direction] = {
                    "alpha": alpha,
                }

        # x-edge intersection positions:
        #
        # normal -> normal : alpha irrelevant for length
        # normal -> PEC    : 0.25
        # PEC -> normal    : 0.40
        # normal -> SIBC   : 0.75
        alpha_x = np.array(
            [0.20, 0.25, 0.40, 0.75],
            dtype=float,
        )

        alpha_x = np.broadcast_to(
            alpha_x[:, None, None],
            (4, 2, 2),
        ).copy()

        for grid_type in (
            "primal",
            "dual",
        ):
            grid.interface_intersections[grid_type]["x"]["alpha"] = alpha_x

        grid.stl_materials = {
            "normal": [
                4.0,
                5.0,
                8.0,
            ],
            "pec": [
                np.inf,
                1.0,
                0.0,
            ],
            "sibc": [
                1.0,
                1.0,
                1.0e6,
            ],
        }

        solver = SolverFIT3D(
            grid,
            bg=[
                2.0,
                3.0,
                4.0,
            ],
            use_stl=False,
            verbose=0,
        )

        solver._build_conformal_effective_lengths()

        fractions = solver.conformal_line_data["primal"]["x"][
            "effective_length_fraction"
        ][:, 0, 0]

        np.testing.assert_allclose(
            fractions,
            np.array(
                [
                    1.0,
                    0.25,
                    0.60,
                    0.75,
                ]
            ),
        )

        primal_materials = solver.conformal_line_materials["primal"]["x"]

        dual_materials = solver.conformal_line_materials["dual"]["x"]

        np.testing.assert_allclose(
            primal_materials["eps_r"][:, 0, 0],
            np.array(
                [
                    10.0 / 3.0,
                    4.0,
                    4.0,
                    4.0,
                ]
            ),
        )

        np.testing.assert_allclose(
            primal_materials["sigma"][:, 0, 0],
            np.array(
                [
                    20.0 / 3.0,
                    8.0,
                    8.0,
                    8.0,
                ]
            ),
        )

        np.testing.assert_allclose(
            dual_materials["mu_r"][:, 0, 0],
            np.array(
                [
                    75.0 / 17.0,
                    5.0,
                    5.0,
                    5.0,
                ]
            ),
        )

        # The last stored Ex DOF is the zero-length ghost edge (uncut).
        expected_fraction = np.array(
            [
                1.0,
                0.25,
                0.60,
                0.75,
                1.0,
            ]
        )

        np.testing.assert_allclose(
            solver.L_fraction.to_matrix("x")[
                :,
                0,
                0,
            ],
            expected_fraction,
        )

        np.testing.assert_allclose(
            solver.L_eff.to_matrix("x")[
                :,
                0,
                0,
            ],
            solver.L.to_matrix("x")[
                :,
                0,
                0,
            ]
            * expected_fraction,
        )

        # Conductor cuts never modify the dual edge lengths. The solver's
        # tL may differ from grid.tL only through the (default periodic)
        # boundary conditions, so compare with an uncut solver.
        assert not hasattr(solver, "tL_eff")
        reference = SolverFIT3D(
            GridFIT3D(0.0, 4.0, 0.0, 1.0, 0.0, 1.0, 5, 2, 2, verbose=0),
            use_stl=False,
            verbose=0,
        )
        np.testing.assert_array_equal(solver.tL.toarray(), reference.tL.toarray())

    def test_edge_array_to_field_component(self):
        grid = GridFIT3D(
            0.0,
            2.0,
            0.0,
            3.0,
            0.0,
            4.0,
            3,
            4,
            5,
            verbose=0,
        )

        solver = SolverFIT3D(
            grid,
            use_stl=False,
            verbose=0,
        )

        # Primal edges (Nx-1, Ny, Nz) etc.; the stored DOFs append one
        # zero-length ghost layer along the edge direction.
        x_edges = np.arange(2 * 4 * 5).reshape(2, 4, 5)
        y_edges = np.arange(3 * 3 * 5).reshape(3, 3, 5)
        z_edges = np.arange(3 * 4 * 4).reshape(3, 4, 4)

        for direction, edges, axis in (
            ("x", x_edges, 0),
            ("y", y_edges, 1),
            ("z", z_edges, 2),
        ):
            field = solver._edge_array_to_field_component(edges, direction, -1)
            assert field.shape == (3, 4, 5)

            ghost = [slice(None)] * 3
            ghost[axis] = -1
            physical = [slice(None)] * 3
            physical[axis] = slice(0, -1)

            np.testing.assert_array_equal(field[tuple(physical)], edges)
            assert np.all(field[tuple(ghost)] == -1)

            # Region IDs copy the nearest edge into the ghost layer.
            nearest = solver._edge_array_to_field_component(edges, direction, "nearest")
            np.testing.assert_array_equal(
                np.take(nearest, -1, axis=axis), np.take(edges, -1, axis=axis)
            )

        # Boolean arrays default to False, floats to NaN on ghosts.
        field = solver._edge_array_to_field_component(np.ones((2, 4, 5), bool), "x")
        assert field.dtype == bool and not np.any(field[-1])
        field = solver._edge_array_to_field_component(np.ones((2, 4, 5)), "x")
        assert np.all(np.isnan(field[-1])) and np.all(field[:-1] == 1.0)

    def test_face_candidate_masks(self):
        grid = GridFIT3D(
            0.0,
            2.0,
            0.0,
            2.0,
            0.0,
            2.0,
            3,
            3,
            3,
            geometry_mode="conformal",
            verbose=0,
        )

        edge_shapes = {
            "x": (2, 3, 3),
            "y": (3, 2, 3),
            "z": (3, 3, 2),
        }

        face_shapes = {
            "x": (3, 2, 2),
            "y": (2, 3, 2),
            "z": (2, 2, 3),
        }

        cases = [
            (
                "x",
                (0, 1, 1),
                {
                    "y": [
                        (0, 1, 0),
                        (0, 1, 1),
                    ],
                    "z": [
                        (0, 0, 1),
                        (0, 1, 1),
                    ],
                },
            ),
            (
                "y",
                (1, 0, 1),
                {
                    "x": [
                        (1, 0, 0),
                        (1, 0, 1),
                    ],
                    "z": [
                        (0, 0, 1),
                        (1, 0, 1),
                    ],
                },
            ),
            (
                "z",
                (1, 1, 0),
                {
                    "x": [
                        (1, 0, 0),
                        (1, 1, 0),
                    ],
                    "y": [
                        (0, 1, 0),
                        (1, 1, 0),
                    ],
                },
            ),
        ]

        for (
            edge_direction,
            edge_index,
            expected_faces,
        ) in cases:
            grid.interior_interface_edge_masks = {
                "primal": {
                    direction: np.zeros(
                        edge_shapes[direction],
                        dtype=bool,
                    )
                    for direction in (
                        "x",
                        "y",
                        "z",
                    )
                },
                "dual": {
                    direction: np.zeros(
                        edge_shapes[direction],
                        dtype=bool,
                    )
                    for direction in (
                        "x",
                        "y",
                        "z",
                    )
                },
            }

            for grid_type in (
                "primal",
                "dual",
            ):
                grid.interior_interface_edge_masks[grid_type][edge_direction][
                    edge_index
                ] = True

            grid._build_face_candidate_masks()

            for grid_type in (
                "primal",
                "dual",
            ):
                for face_direction in (
                    "x",
                    "y",
                    "z",
                ):
                    expected = np.zeros(
                        face_shapes[face_direction],
                        dtype=bool,
                    )

                    for index in expected_faces.get(
                        face_direction,
                        [],
                    ):
                        expected[index] = True

                    np.testing.assert_array_equal(
                        grid.face_candidate_masks[grid_type][face_direction],
                        expected,
                    )

    def test_face_boundary_intersections(self):
        grid = GridFIT3D(
            0.0,
            2.0,
            0.0,
            2.0,
            0.0,
            2.0,
            3,
            3,
            3,
            geometry_mode="conformal",
            verbose=0,
        )

        point_shape = (
            3,
            3,
            3,
        )

        grid.region_ids = {
            "primal": np.zeros(
                point_shape,
                dtype=np.int32,
            ),
            "dual": np.zeros(
                point_shape,
                dtype=np.int32,
            ),
        }

        edge_shapes = {
            "x": (
                2,
                3,
                3,
            ),
            "y": (
                3,
                2,
                3,
            ),
            "z": (
                3,
                3,
                2,
            ),
        }

        grid.resolved_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        grid.interface_intersections = {
            "primal": {},
            "dual": {},
        }

        for grid_type in (
            "primal",
            "dual",
        ):
            for direction in (
                "x",
                "y",
                "z",
            ):
                shape = edge_shapes[direction]

                grid.resolved_interface_edge_masks[grid_type][direction] = np.zeros(
                    shape,
                    dtype=bool,
                )

                grid.interface_intersections[grid_type][direction] = {
                    "alpha": np.full(
                        shape,
                        np.nan,
                    ),
                    "surface_id": np.full(
                        shape,
                        -1,
                        dtype=np.int32,
                    ),
                    "cell_id": np.full(
                        shape,
                        -1,
                        dtype=np.int64,
                    ),
                    "normal": np.full(
                        shape + (3,),
                        np.nan,
                    ),
                    "multiple_crossing_positions": (
                        np.zeros(
                            shape,
                            dtype=bool,
                        )
                    ),
                }

        # --------------------------------------------------
        # z-normal primal face:
        #
        #     y
        #     ^
        #     |
        #   P1---------+
        #    \         |
        #     \        |
        #      \       |
        #       +------P0 --> x
        #
        # P0 lies exactly on a corner and therefore appears
        # on two boundary edges.
        # --------------------------------------------------

        face_index = (
            0,
            0,
            1,
        )

        # Bottom x-edge:
        # alpha = 1 -> P0 = (1, 0, 1)
        edge = (
            0,
            0,
            1,
        )

        grid.resolved_interface_edge_masks["primal"]["x"][edge] = True

        grid.interface_intersections["primal"]["x"]["alpha"][edge] = 1.0

        grid.interface_intersections["primal"]["x"]["surface_id"][edge] = 1

        grid.interface_intersections["primal"]["x"]["cell_id"][edge] = 10

        grid.interface_intersections["primal"]["x"]["normal"][edge] = [
            1.0,
            1.0,
            0.0,
        ]

        # Right y-edge:
        # alpha = 0 -> same P0
        edge = (
            1,
            0,
            1,
        )

        grid.resolved_interface_edge_masks["primal"]["y"][edge] = True

        grid.interface_intersections["primal"]["y"]["alpha"][edge] = 0.0

        grid.interface_intersections["primal"]["y"]["surface_id"][edge] = 1

        grid.interface_intersections["primal"]["y"]["cell_id"][edge] = 11

        grid.interface_intersections["primal"]["y"]["normal"][edge] = [
            1.0,
            1.0,
            0.0,
        ]

        # Top x-edge:
        # alpha = 0.5 -> P1 = (0.5, 1, 1)
        edge = (
            0,
            1,
            1,
        )

        grid.resolved_interface_edge_masks["primal"]["x"][edge] = True

        grid.interface_intersections["primal"]["x"]["alpha"][edge] = 0.5

        grid.interface_intersections["primal"]["x"]["surface_id"][edge] = 1

        grid.interface_intersections["primal"]["x"]["cell_id"][edge] = 12

        grid.interface_intersections["primal"]["x"]["normal"][edge] = [
            1.0,
            1.0,
            0.0,
        ]

        intersections = grid._get_face_boundary_intersections(
            "primal",
            "z",
            face_index,
        )

        # Three raw edge hits, but only two unique
        # geometric intersection points.
        assert len(intersections) == 2

        np.testing.assert_allclose(
            intersections[0]["point"],
            [
                1.0,
                0.0,
                1.0,
            ],
        )

        np.testing.assert_allclose(
            intersections[1]["point"],
            [
                0.5,
                1.0,
                1.0,
            ],
        )

        assert len(intersections[0]["hits"]) == 2

        assert len(intersections[1]["hits"]) == 1

        boundary_edges = grid._get_face_boundary_edges(
            "primal",
            "z",
            face_index,
        )

        assert [edge["boundary_sign"] for edge in boundary_edges] == [
            1,
            1,
            -1,
            -1,
        ]

    def test_face_intersection_segment(self):
        grid = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            2,
            2,
            2,
            geometry_mode="conformal",
            verbose=0,
        )

        shape = (
            2,
            2,
            2,
        )

        grid.region_ids = {
            "primal": np.zeros(
                shape,
                dtype=np.int32,
            ),
            "dual": np.zeros(
                shape,
                dtype=np.int32,
            ),
        }

        edge_shapes = {
            "x": (1, 2, 2),
            "y": (2, 1, 2),
            "z": (2, 2, 1),
        }

        grid.resolved_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        grid.interface_intersections = {
            "primal": {},
            "dual": {},
        }

        for grid_type in (
            "primal",
            "dual",
        ):
            for direction in (
                "x",
                "y",
                "z",
            ):
                edge_shape = edge_shapes[direction]

                grid.resolved_interface_edge_masks[grid_type][direction] = np.zeros(
                    edge_shape,
                    dtype=bool,
                )

                grid.interface_intersections[grid_type][direction] = {
                    "alpha": np.full(
                        edge_shape,
                        np.nan,
                    ),
                    "surface_id": np.full(
                        edge_shape,
                        -1,
                        dtype=np.int32,
                    ),
                    "cell_id": np.full(
                        edge_shape,
                        -1,
                        dtype=np.int64,
                    ),
                    "normal": np.full(
                        edge_shape + (3,),
                        np.nan,
                    ),
                    "multiple_crossing_positions": np.zeros(
                        edge_shape,
                        dtype=bool,
                    ),
                }

        # z-normal xy face at z = 0.
        #
        # Interface:
        #
        # y
        # ^
        # |      P1
        # |------x
        # |     /
        # |    /
        # x---+------> x
        # P0
        #
        # P0 = (0, 0.25, 0)
        # P1 = (1, 0.75, 0)

        y_edge_left = (
            0,
            0,
            0,
        )

        y_edge_right = (
            1,
            0,
            0,
        )

        for edge, alpha in (
            (
                y_edge_left,
                0.25,
            ),
            (
                y_edge_right,
                0.75,
            ),
        ):
            grid.resolved_interface_edge_masks["primal"]["y"][edge] = True

            grid.interface_intersections["primal"]["y"]["alpha"][edge] = alpha

            grid.interface_intersections["primal"]["y"]["surface_id"][edge] = 1

            grid.interface_intersections["primal"]["y"]["cell_id"][edge] = 0

            grid.interface_intersections["primal"]["y"]["normal"][edge] = [
                -0.5,
                1.0,
                0.0,
            ]

        segment = grid._get_face_intersection_segment(
            "primal",
            "z",
            (
                0,
                0,
                0,
            ),
        )

        assert segment["resolved"]

        np.testing.assert_allclose(
            segment["p0"],
            [
                1.0,
                0.75,
                0.0,
            ],
        )

        np.testing.assert_allclose(
            segment["p1"],
            [
                0.0,
                0.25,
                0.0,
            ],
        )

        expected_vector = np.array(
            [
                -1.0,
                -0.5,
                0.0,
            ]
        )

        np.testing.assert_allclose(
            segment["vector"],
            expected_vector,
        )

        expected_length = np.sqrt(1.0 + 0.25)

        assert segment["length"] == pytest.approx(expected_length)

        np.testing.assert_allclose(
            segment["tangent"],
            expected_vector / expected_length,
        )

        assert segment["surface_id"] == 1

        np.testing.assert_allclose(
            np.linalg.norm(segment["normal"]),
            1.0,
        )

    def test_split_face_by_interface(self):
        grid = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            2,
            2,
            2,
            geometry_mode="conformal",
            verbose=0,
        )

        regions = np.zeros(
            (
                2,
                2,
                2,
            ),
            dtype=np.int32,
        )

        # Lower y-side -> region 0
        # Upper y-side -> region 1
        regions[:, 1, :] = 1

        grid.region_ids = {
            "primal": regions.copy(),
            "dual": regions.copy(),
        }

        edge_shapes = {
            "x": (1, 2, 2),
            "y": (2, 1, 2),
            "z": (2, 2, 1),
        }

        grid.resolved_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        grid.interior_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        grid.interface_intersections = {
            "primal": {},
            "dual": {},
        }

        for grid_type in (
            "primal",
            "dual",
        ):
            for direction in (
                "x",
                "y",
                "z",
            ):
                shape = edge_shapes[direction]

                grid.resolved_interface_edge_masks[grid_type][direction] = np.zeros(
                    shape,
                    dtype=bool,
                )

                grid.interior_interface_edge_masks[grid_type][direction] = np.zeros(
                    shape,
                    dtype=bool,
                )

                grid.interface_intersections[grid_type][direction] = {
                    "alpha": np.full(
                        shape,
                        np.nan,
                    ),
                    "surface_id": np.full(
                        shape,
                        -1,
                        dtype=np.int32,
                    ),
                    "cell_id": np.full(
                        shape,
                        -1,
                        dtype=np.int64,
                    ),
                    "normal": np.full(
                        shape + (3,),
                        np.nan,
                    ),
                    "multiple_crossing_positions": np.zeros(
                        shape,
                        dtype=bool,
                    ),
                }

        # Left y-edge:
        # (0, 0.2, 0)
        left_edge = (
            0,
            0,
            0,
        )

        # Right y-edge:
        # (1, 0.6, 0)
        right_edge = (
            1,
            0,
            0,
        )

        for edge, alpha in (
            (
                left_edge,
                0.2,
            ),
            (
                right_edge,
                0.6,
            ),
        ):
            grid.resolved_interface_edge_masks["primal"]["y"][edge] = True

            grid.interior_interface_edge_masks["primal"]["y"][edge] = True

            grid.interface_intersections["primal"]["y"]["alpha"][edge] = alpha

            grid.interface_intersections["primal"]["y"]["surface_id"][edge] = 1

            grid.interface_intersections["primal"]["y"]["cell_id"][edge] = 0

            grid.interface_intersections["primal"]["y"]["normal"][edge] = [
                -0.4,
                1.0,
                0.0,
            ]

        split = grid._split_face_by_interface(
            "primal",
            "z",
            (
                0,
                0,
                0,
            ),
        )

        assert split["resolved"]

        assert split["face_area"] == pytest.approx(1.0)

        polygons = {polygon["region_id"]: polygon for polygon in split["polygons"]}

        assert set(polygons.keys()) == {
            0,
            1,
        }

        # Lower region
        assert polygons[0]["area"] == pytest.approx(0.4)

        assert polygons[0]["area_fraction"] == pytest.approx(0.4)

        # Upper region
        assert polygons[1]["area"] == pytest.approx(0.6)

        assert polygons[1]["area_fraction"] == pytest.approx(0.6)

        assert sum(polygon["area"] for polygon in split["polygons"]) == pytest.approx(
            1.0
        )

        np.testing.assert_allclose(
            polygons[0]["cut_vector"],
            -polygons[1]["cut_vector"],
        )

        assert polygons[0]["cut_length"] == pytest.approx(polygons[1]["cut_length"])

        grid._build_face_candidate_masks()
        grid._build_interface_face_data()

        face_data = grid.interface_face_data["primal"]["z"]

        index = (
            0,
            0,
            0,
        )

        assert face_data["candidate"][index]

        assert face_data["segment_resolved"][index]

        assert face_data["resolved"][index]

        assert face_data["face_area"][index] == pytest.approx(1.0)

        assert face_data["region_areas"][index][0] == pytest.approx(0.4)

        assert face_data["region_areas"][index][1] == pytest.approx(0.6)

        assert face_data["region_area_fractions"][index][0] == pytest.approx(0.4)

        assert face_data["region_area_fractions"][index][1] == pytest.approx(0.6)

        segment = face_data["segments"][index]

        assert segment["length"] > 0.0

        assert len(face_data["polygons"][index]) == 2

        # No dual intersections were inserted into
        # this synthetic test.
        for direction in (
            "x",
            "y",
            "z",
        ):
            assert not np.any(grid.interface_face_data["dual"][direction]["resolved"])

    def test_face_array_to_field_component(
        self,
    ):
        grid = GridFIT3D(
            0.0,
            2.0,
            0.0,
            3.0,
            0.0,
            4.0,
            3,
            4,
            5,
            verbose=0,
        )

        solver = SolverFIT3D(
            grid,
            use_stl=False,
            verbose=0,
        )

        # Primal faces (Nx, Ny-1, Nz-1) etc.; Hd(i,j,k) <-> d-face (i,j,k).
        # The stored DOFs append zero-area ghost layers transverse to d.
        x_faces = np.arange(3 * 3 * 4).reshape(3, 3, 4)
        y_faces = np.arange(2 * 4 * 4).reshape(2, 4, 4)
        z_faces = np.arange(2 * 3 * 5).reshape(2, 3, 5)

        for direction, faces in (
            ("x", x_faces),
            ("y", y_faces),
            ("z", z_faces),
        ):
            field = solver._face_array_to_field_component(faces, direction, -1)
            assert field.shape == (3, 4, 5)

            n0, n1, n2 = faces.shape
            np.testing.assert_array_equal(field[:n0, :n1, :n2], faces)

            ghost = np.ones(field.shape, dtype=bool)
            ghost[:n0, :n1, :n2] = False
            assert np.all(field[ghost] == -1)

            # Ghost faces are exactly the faces with zero area (iA = 0).
            np.testing.assert_array_equal(ghost, solver.iA.to_matrix(direction) == 0)

    def test_conformal_face_fractions(self):
        grid = GridFIT3D(
            0.0,
            3.0,
            0.0,
            1.0,
            0.0,
            1.0,
            4,
            2,
            2,
            geometry_mode="conformal",
            bg_material_type="normal",
            verbose=0,
        )

        # --------------------------------------------------
        # Region definitions
        #
        # 0: background normal
        # 1: STL normal
        # 2: PEC
        # --------------------------------------------------
        grid.region_id_to_key = {
            0: None,
            1: "normal",
            2: "pec",
        }

        grid.region_key_to_id = {
            "normal": 1,
            "pec": 2,
        }

        grid.stl_material_types = {
            "normal": "normal",
            "pec": "pec",
        }

        grid.stl_materials = {
            "normal": [
                4.0,  # eps_r
                5.0,  # mu_r
                8.0,  # sigma
            ],
            "pec": [
                np.inf,
                1.0,
                0.0,
            ],
        }

        # Region IDs only need to exist for Solver initialization.
        point_shape = (
            4,
            2,
            2,
        )

        grid.region_ids = {
            "primal": np.zeros(
                point_shape,
                dtype=np.int32,
            ),
            "dual": np.zeros(
                point_shape,
                dtype=np.int32,
            ),
        }

        grid.interface_edge_masks = {
            "primal": grid._build_edge_transition_masks(grid.region_ids["primal"]),
            "dual": grid._build_edge_transition_masks(grid.region_ids["dual"]),
        }

        # --------------------------------------------------
        # Full face-array shapes
        # --------------------------------------------------
        face_shapes = {
            "x": (
                4,
                1,
                1,
            ),
            "y": (
                3,
                2,
                1,
            ),
            "z": (
                3,
                1,
                2,
            ),
        }

        grid.interface_face_data = {
            "primal": {},
            "dual": {},
        }

        for grid_type in (
            "primal",
            "dual",
        ):
            for direction in (
                "x",
                "y",
                "z",
            ):
                shape = face_shapes[direction]

                grid.interface_face_data[grid_type][direction] = {
                    "resolved": np.zeros(
                        shape,
                        dtype=bool,
                    ),
                    "region_area_fractions": {},
                }

        # --------------------------------------------------
        # Three z-normal faces on the low-z plane
        # --------------------------------------------------
        face_nn = (
            0,
            0,
            0,
        )

        face_normal_pec = (
            1,
            0,
            0,
        )

        face_pec_normal = (
            2,
            0,
            0,
        )

        for grid_type in (
            "primal",
            "dual",
        ):
            data = grid.interface_face_data[grid_type]["z"]

            # normal-normal
            data["resolved"][face_nn] = True

            data["region_area_fractions"][face_nn] = {
                0: 0.4,
                1: 0.6,
            }

            # normal-PEC
            data["resolved"][face_normal_pec] = True

            data["region_area_fractions"][face_normal_pec] = {
                1: 0.25,
                2: 0.75,
            }

            # PEC-normal:
            # same physical area fractions, but deliberately
            # reversed region ordering.
            data["resolved"][face_pec_normal] = True

            data["region_area_fractions"][face_pec_normal] = {
                2: 0.75,
                1: 0.25,
            }

        solver = SolverFIT3D(
            grid,
            bg=[
                2.0,
                3.0,
                4.0,
            ],
            use_stl=False,
            verbose=0,
        )

        # use_stl=False intentionally prevents automatic application.
        solver._build_conformal_face_data()
        solver._build_conformal_face_materials()
        solver._build_conformal_effective_areas()

        # --------------------------------------------------
        # Check classification
        # --------------------------------------------------
        for grid_type in (
            "primal",
            "dual",
        ):
            data = solver.conformal_face_data[grid_type]["z"]

            assert data["normal_normal"][face_nn]

            assert not data["normal_pec"][face_nn]

            assert data["normal_pec"][face_normal_pec]

            assert data["normal_pec"][face_pec_normal]

            assert data["supported"][face_nn]

            assert data["supported"][face_normal_pec]

            assert data["supported"][face_pec_normal]

            assert data["effective_area_fraction"][face_nn] == pytest.approx(1.0)

            assert data["effective_area_fraction"][face_normal_pec] == pytest.approx(
                0.25
            )

            assert data["effective_area_fraction"][face_pec_normal] == pytest.approx(
                0.25
            )

        # --------------------------------------------------
        # Check mapped Field fractions
        #
        # Hz(i,j,k) <-> primal z-face (i,j,k). The last x entry is the
        # zero-area ghost face, which is reported as uncut.
        # --------------------------------------------------
        np.testing.assert_allclose(
            solver.A_fraction.to_matrix("z")[:, 0, 0],
            [
                1.0,
                0.25,
                0.25,
                1.0,
            ],
        )

        # Conductor cuts never modify the dual face areas. The solver's
        # itA may differ from grid.itA only through the (default periodic)
        # boundary conditions, so compare with an uncut solver.
        assert not hasattr(solver, "tA_eff")
        reference = SolverFIT3D(
            GridFIT3D(0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 4, 2, 2, verbose=0),
            use_stl=False,
            verbose=0,
        )
        np.testing.assert_array_equal(solver.itA.toarray(), reference.itA.toarray())

        # --------------------------------------------------
        # Effective primal areas must be base area * fraction.
        # --------------------------------------------------
        # Ghost faces (iA = 0) have zero area.
        iA_z = solver.iA.to_matrix("z")[:, 0, 0]
        primal_base_area = np.divide(
            1.0, iA_z, out=np.zeros_like(iA_z), where=iA_z != 0
        )

        expected_fraction = np.array(
            [
                1.0,
                0.25,
                0.25,
                1.0,
            ]
        )

        np.testing.assert_allclose(
            solver.A_eff.to_matrix("z")[:, 0, 0],
            primal_base_area * expected_fraction,
        )

        primal_materials = solver.conformal_face_materials["primal"]["z"]

        dual_materials = solver.conformal_face_materials["dual"]["z"]

        # normal-normal
        assert primal_materials["mu_r"][face_nn] == pytest.approx(4.2)

        assert dual_materials["eps_r"][face_nn] == pytest.approx(3.2)

        assert dual_materials["sigma"][face_nn] == pytest.approx(6.4)

        # normal-PEC
        assert primal_materials["mu_r"][face_normal_pec] == pytest.approx(5.0)

        assert dual_materials["eps_r"][face_normal_pec] == pytest.approx(4.0)

        assert dual_materials["sigma"][face_normal_pec] == pytest.approx(8.0)

        # PEC-normal: must give exactly the same material values.
        assert primal_materials["mu_r"][face_pec_normal] == pytest.approx(5.0)

        assert dual_materials["eps_r"][face_pec_normal] == pytest.approx(4.0)

        assert dual_materials["sigma"][face_pec_normal] == pytest.approx(8.0)

    def test_primal_and_dual_interface_intersections(
        self,
        tmp_path,
    ):
        interface_x = 0.40625

        stl_file = tmp_path / "right_region.stl"

        surface = pv.Box(
            bounds=(
                interface_x,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            )
        ).triangulate()

        surface.save(stl_file)

        grid = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            3,
            3,
            geometry_mode="conformal",
            verbose=0,
        )

        grid.stl_solids = {
            "right": str(stl_file),
        }

        grid.stl_rotate = {
            "right": [0.0, 0.0, 0.0],
        }

        grid.stl_translate = {
            "right": [0.0, 0.0, 0.0],
        }

        grid.stl_scale = {
            "right": 1.0,
        }

        grid.stl_material_types = {
            "right": "normal",
        }

        grid._build_region_maps()

        # Build exact artificial region IDs from the known interface
        # location independently on both grids.
        primal_regions = np.zeros(
            (
                len(grid.x),
                len(grid.y),
                len(grid.z),
            ),
            dtype=np.int32,
        )

        dual_regions = np.zeros(
            (
                len(grid.tx),
                len(grid.ty),
                len(grid.tz),
            ),
            dtype=np.int32,
        )

        primal_regions[
            np.asarray(grid.x) >= interface_x,
            :,
            :,
        ] = 1

        dual_regions[
            np.asarray(grid.tx) >= interface_x,
            :,
            :,
        ] = 1

        grid.region_ids = {
            "primal": primal_regions,
            "dual": dual_regions,
        }

        grid.interface_edge_masks = {
            "primal": grid._build_edge_transition_masks(primal_regions),
            "dual": grid._build_edge_transition_masks(dual_regions),
        }

        grid._build_interface_intersections()

        for grid_type in (
            "primal",
            "dual",
        ):
            mask = grid.interface_edge_masks[grid_type]["x"]

            # All x-interface edges have the same x index.
            i = np.where(
                np.any(
                    mask,
                    axis=(1, 2),
                )
            )[0][0]

            # Use an interior transverse edge to avoid box corners.
            index = (
                int(i),
                1,
                1,
            )

            assert mask[index]

            data = grid.interface_intersections[grid_type]["x"]

            assert data["has_intersection"][index]

            assert data["has_crossing"][index]
            assert not data["has_contact"][index]

            assert not data["multiple_crossing_positions"][index]

            assert len(data["crossing_groups"][index]) == 1

            assert index not in data["contact_groups"]

            assert not data["multiple_intersections"][index]

            groups = data["hit_groups"][index]

            assert len(groups) == 1
            assert len(groups[0]) >= 1

            (
                start_point,
                end_point,
                _,
                _,
            ) = grid._get_edge_geometry(
                grid_type,
                "x",
                index,
            )

            expected_alpha = (interface_x - start_point[0]) / (
                end_point[0] - start_point[0]
            )

            assert data["alpha"][index] == pytest.approx(
                expected_alpha,
                abs=1e-8,
            )

            assert data["surface_id"][index] == 1

            assert data["cell_id"][index] >= 0

            normal = data["normal"][index]

            assert np.linalg.norm(normal) == pytest.approx(
                1.0,
                abs=1e-8,
            )

            assert abs(normal[0]) == pytest.approx(
                1.0,
                abs=1e-8,
            )

            assert normal[1] == pytest.approx(
                0.0,
                abs=1e-8,
            )

            assert normal[2] == pytest.approx(
                0.0,
                abs=1e-8,
            )

            group_alpha = np.mean([hit["alpha"] for hit in groups[0]])

            assert group_alpha == pytest.approx(
                expected_alpha,
                abs=1e-8,
            )

            assert grid.resolved_interface_edge_masks[grid_type]["x"][index]

    def test_thin_stl_layer_inside_single_edge(
        self,
        tmp_path,
    ):
        """
        A thin STL layer entirely inside one grid edge must produce two
        intersections even though both edge endpoints belong to the
        background.
        """

        stl_file = tmp_path / "thin_layer.stl"

        surface = pv.Box(
            bounds=(
                0.35,
                0.40,
                0.0,
                1.0,
                0.0,
                1.0,
            )
        ).triangulate()

        surface.save(stl_file)

        grid = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            3,
            3,
            3,
            stl_solids={
                "layer": str(stl_file),
            },
            stl_materials={
                "layer": [4.0, 1.0, 0.0],
            },
            stl_material_types={
                "layer": "normal",
            },
            geometry_mode="conformal",
            stl_method="voxelize_rectilinear",
            verbose=0,
        )

        # First x-edge runs from x=0 to x=0.5.
        # The complete thin layer x=[0.35, 0.40] lies inside it.
        index = (
            0,
            1,
            1,
        )

        data = grid.interface_intersections["primal"]["x"]

        assert data["has_intersection"][index]

        assert data["has_crossing"][index]

        assert data["multiple_crossing_positions"][index]

        assert len(data["crossing_groups"][index]) == 2

        assert data["multiple_intersections"][index]

        groups = data["hit_groups"][index]

        assert len(groups) == 2

        alpha_1 = np.mean([hit["alpha"] for hit in groups[0]])

        alpha_2 = np.mean([hit["alpha"] for hit in groups[1]])

        assert alpha_1 == pytest.approx(
            0.70,
            abs=1e-6,
        )

        assert alpha_2 == pytest.approx(
            0.80,
            abs=1e-6,
        )

        assert data["has_crossing"][index]

        assert not grid.interface_edge_masks["primal"]["x"][index]

        assert not grid.resolved_interface_edge_masks["primal"]["x"][index]

    def test_three_normal_material_regions_on_single_edge(
        self,
        tmp_path,
    ):
        """
        A single grid edge crosses two normal-material interfaces:

            background -> middle -> right

        The two geometric interface positions must be preserved separately.
        """

        middle_file = tmp_path / "middle_region.stl"

        right_file = tmp_path / "right_region.stl"

        # Use binary-exact coordinates to avoid STL float32 noise.
        x_1 = 0.375
        x_2 = 0.4375

        middle_surface = pv.Box(
            bounds=(
                x_1,
                x_2,
                0.0,
                1.0,
                0.0,
                1.0,
            )
        ).triangulate()

        right_surface = pv.Box(
            bounds=(
                x_2,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            )
        ).triangulate()

        middle_surface.save(middle_file)

        right_surface.save(right_file)

        grid = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            3,
            3,
            3,
            stl_solids={
                "middle": str(middle_file),
                "right": str(right_file),
            },
            stl_materials={
                "middle": [4.0, 1.0, 0.0],
                "right": [9.0, 1.0, 0.0],
            },
            stl_material_types={
                "middle": "normal",
                "right": "normal",
            },
            geometry_mode="conformal",
            stl_method="voxelize_rectilinear",
            verbose=0,
        )

        # First primal x-edge:
        #
        # x = 0 ------------------------------ 0.5
        #               |           |
        #             0.375       0.4375
        #       background -> middle -> right
        #
        index = (
            0,
            1,
            1,
        )

        data = grid.interface_intersections["primal"]["x"]

        assert data["has_intersection"][index]

        assert data["multiple_intersections"][index]

        groups = data["hit_groups"][index]

        assert len(groups) == 2

        alpha_1 = np.mean([hit["alpha"] for hit in groups[0]])

        alpha_2 = np.mean([hit["alpha"] for hit in groups[1]])

        assert alpha_1 == pytest.approx(
            0.75,
            abs=1e-8,
        )

        assert alpha_2 == pytest.approx(
            0.875,
            abs=1e-8,
        )

        middle_id = grid.region_key_to_id["middle"]

        right_id = grid.region_key_to_id["right"]

        first_surface_ids = {hit["surface_id"] for hit in groups[0]}

        second_surface_ids = {hit["surface_id"] for hit in groups[1]}

        # At x = 0.375 only the middle STL starts.
        assert first_surface_ids == {middle_id}

        # At x = 0.4375 the middle STL ends and the right STL starts.
        # Both surfaces therefore occur at the same geometric position.
        assert second_surface_ids == {
            middle_id,
            right_id,
        }

        assert data["multiple_crossing_positions"][index]

        assert grid.resolved_interface_edge_masks["primal"]["x"][index]

        assert data["alpha"][index] == pytest.approx(
            0.75,
            abs=1e-8,
        )

    def test_real_stl_interface_intersections(self):
        """
        Diagnostic integration test for geometric interface intersections
        on the existing cavity/shell STL geometry.

        The test checks all detected primal and dual interface edges in
        x-, y-, and z-direction and reports missing and ambiguous ray hits.
        """

        solid_1 = "tests/stl/007_vacuum_cavity.stl"
        solid_2 = "tests/stl/007_lossymetal_shell.stl"

        stl_solids = {
            "cavity": solid_1,
            "shell": solid_2,
        }

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30.0, 1.0, 30.0],
        }

        stl_material_types = {
            "cavity": "normal",
            "shell": "normal",
        }

        solids = pv.read(solid_1) + pv.read(solid_2)

        (
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        ) = solids.bounds

        # Deliberately smaller than the full 60 x 60 x 140 regression grid.
        # This is a geometry/intersection diagnostic, not a convergence test.
        Nx = 20
        Ny = 20
        Nz = 40

        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            Nx,
            Ny,
            Nz,
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_material_types=stl_material_types,
            stl_method="voxelize_rectilinear",
            geometry_mode="conformal",
            subpixel_smoothing=False,
            stl_scale=1.0,
            stl_rotate=[0.0, 0.0, 0.0],
            stl_translate=[0.0, 0.0, 0.0],
            verbose=0,
        )

        total_expected_interfaces = 0
        total_geometric_intersections = 0
        total_missing_expected = 0
        total_multiple = 0
        total_hidden_intersections = 0

        print()
        print("Real STL interface intersection diagnostic")
        print("------------------------------------------")

        for grid_type in (
            "primal",
            "dual",
        ):
            print(f"{grid_type}:")

            for direction in (
                "x",
                "y",
                "z",
            ):
                interface_mask = grid.interface_edge_masks[grid_type][direction]

                data = grid.interface_intersections[grid_type][direction]

                has_intersection = data["has_intersection"]

                multiple = data["multiple_intersections"]

                # Endpoint classification says that this edge crosses
                # a region boundary, but the exact STL tracing found none.
                missing_expected = interface_mask & ~has_intersection

                # Exact geometry found an STL intersection although the
                # endpoint regions are identical.
                hidden_intersections = has_intersection & ~interface_mask

                n_expected = int(np.count_nonzero(interface_mask))

                n_geometric = int(np.count_nonzero(has_intersection))

                n_missing_expected = int(np.count_nonzero(missing_expected))

                n_multiple = int(np.count_nonzero(multiple))

                n_hidden = int(np.count_nonzero(hidden_intersections))

                total_expected_interfaces += n_expected
                total_geometric_intersections += n_geometric
                total_missing_expected += n_missing_expected
                total_multiple += n_multiple
                total_hidden_intersections += n_hidden

                print(
                    f"  {direction}: "
                    f"{n_expected:6d} endpoint interfaces, "
                    f"{n_geometric:6d} geometric intersections, "
                    f"{n_missing_expected:6d} expected-but-missing, "
                    f"{n_hidden:6d} hidden intersections, "
                    f"{n_multiple:6d} multiple"
                )

                # Every representative intersection that is actually used by the
                # conformal discretization must contain valid geometric data.
                valid = grid.resolved_interface_edge_masks[grid_type][direction]

                if np.any(valid):
                    alpha = data["alpha"][valid]

                    if not np.all(np.isfinite(alpha)):
                        invalid_indices = np.argwhere(
                            has_intersection & ~np.isfinite(data["alpha"])
                        )

                        print(
                            f"\nInvalid representative intersections "
                            f"for {grid_type} {direction}:"
                        )

                        for index_array in invalid_indices:
                            index = tuple(int(value) for value in index_array)

                            print(f"  index = {index}")
                            print(f"    alpha = {data['alpha'][index]}")
                            print(f"    surface_id = {data['surface_id'][index]}")
                            print(f"    cell_id = {data['cell_id'][index]}")
                            print(f"    normal = {data['normal'][index]}")
                            print(
                                f"    multiple = {data['multiple_intersections'][index]}"
                            )
                            print(
                                "    multiple_crossing_positions = "
                                f"{data['multiple_crossing_positions'][index]}"
                            )

                    assert np.all(np.isfinite(alpha))

                    assert np.all(alpha >= 0.0)

                    assert np.all(alpha <= 1.0)

                    assert np.all(data["surface_id"][valid] > 0)

                    assert np.all(data["cell_id"][valid] >= 0)

                    normals = data["normal"][valid]

                    assert np.all(np.isfinite(normals))

                    np.testing.assert_allclose(
                        np.linalg.norm(
                            normals,
                            axis=1,
                        ),
                        1.0,
                        atol=1e-10,
                        rtol=1e-10,
                    )

        print("------------------------------------------")
        print(
            f"total: "
            f"{total_expected_interfaces} endpoint interfaces, "
            f"{total_geometric_intersections} geometric intersections, "
            f"{total_missing_expected} expected-but-missing, "
            f"{total_hidden_intersections} hidden intersections, "
            f"{total_multiple} multiple"
        )

        assert total_geometric_intersections > 0

    def test_conformal_geometry_h5_roundtrip(
        self,
        tmp_path,
    ):
        """
        Test HDF5 roundtrip of conformal geometry data.

        Verifies that the primal and directly classified dual point masks,
        STL ordering, material treatment types, and interface topology are
        preserved when a conformal grid is saved and loaded again.
        """
        solid_z = pv.Box(
            bounds=(
                0.20,
                0.45,
                0.20,
                0.80,
                0.20,
                0.80,
            )
        ).triangulate()

        solid_a = pv.Box(
            bounds=(
                0.55,
                0.80,
                0.20,
                0.80,
                0.20,
                0.80,
            )
        ).triangulate()

        path_z = tmp_path / "z_region.stl"
        path_a = tmp_path / "a_region.stl"

        solid_z.save(path_z)
        solid_a.save(path_a)

        # Deliberately use a non-alphabetical order. The order defines
        # region IDs and must therefore survive the HDF5 roundtrip.
        stl_solids = {
            "z_region": str(path_z),
            "a_region": str(path_a),
        }

        stl_materials = {
            "z_region": [4.0, 1.0, 0.0],
            "a_region": [1.0, 1.0, 0.0],
        }

        stl_material_types = {
            "z_region": "normal",
            "a_region": "pec",
        }

        grid = GridFIT3D(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            7,
            7,
            7,
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_material_types=stl_material_types,
            geometry_mode="conformal",
            stl_method="interior_points",
            verbose=0,
        )

        h5_path = tmp_path / "conformal_grid.h5"

        grid.save_to_h5(str(h5_path))

        loaded = GridFIT3D(
            load_from_h5=str(h5_path),
            verbose=0,
        )

        # Geometry-pipeline metadata
        assert loaded.geometry_mode == "conformal"
        assert loaded.stl_method == "interior_points"
        assert loaded.stl_tol == pytest.approx(grid.stl_tol)

        # STL ordering is significant for region priority.
        assert list(loaded.stl_solids.keys()) == [
            "z_region",
            "a_region",
        ]

        # Material treatment types must survive the roundtrip.
        assert loaded.stl_material_types == (grid.stl_material_types)

        for key in stl_solids:
            # Existing cell mask
            assert np.array_equal(
                np.asarray(grid.grid[key]),
                np.asarray(loaded.grid[key]),
            )

            # Direct primal-point classification
            assert np.array_equal(
                grid.primal_point_masks[key],
                loaded.primal_point_masks[key],
            )

            # Direct dual-point classification
            assert np.array_equal(
                grid.dual_point_masks[key],
                loaded.dual_point_masks[key],
            )

        # Region IDs reconstructed from the loaded point masks
        # must be identical.
        assert np.array_equal(
            grid.region_ids["primal"],
            loaded.region_ids["primal"],
        )

        assert np.array_equal(
            grid.region_ids["dual"],
            loaded.region_ids["dual"],
        )

        # The endpoint-based interface topology must also survive.
        for grid_type in (
            "primal",
            "dual",
        ):
            for direction in (
                "x",
                "y",
                "z",
            ):
                assert np.array_equal(
                    grid.interface_edge_masks[grid_type][direction],
                    loaded.interface_edge_masks[grid_type][direction],
                )

    def test_conformal_slanted_material_interface(self, tmp_path):
        """
        Validate the conformal treatment for a slanted material interface.

        The interface is

            y_s(x) = 1.375 + 0.5 * (x - 2)

        and separates the background from a second material occupying

            y < y_s(x).

        Two cases are tested:

        1. normal <-> normal
        - intersection position alpha
        - effective edge/face geometry
        - harmonic line-material averaging
        - arithmetic face-material averaging

        2. normal <-> PEC
        - effective edge length
        - effective face area
        - material properties of the remaining normal region
        """

        # ============================================================
        # Geometry
        # ============================================================
        #
        # Computational domain:
        #
        #     0 <= x,y,z <= 5
        #
        # with
        #
        #     dx = dy = dz = 1.
        #
        # The STL solid is deliberately extended beyond the
        # computational domain in x, z and toward negative y.
        #
        # Therefore, inside the computational domain only the slanted
        # upper interface is relevant:
        #
        #     y_s(x) = 1.375 + 0.5 * (x - 2)
        #
        # Material "insert" occupies y < y_s(x).
        # ============================================================

        x0 = -1.0
        x1 = 6.0

        y0 = -1.0

        yt0 = 1.375 + 0.5 * (x0 - 2.0)
        yt1 = 1.375 + 0.5 * (x1 - 2.0)

        z0 = -1.0
        z1 = 6.0

        points = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, yt1, z0],
                [x0, yt0, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, yt1, z1],
                [x0, yt0, z1],
            ],
            dtype=float,
        )

        faces = np.hstack(
            [
                [4, 0, 1, 2, 3],
                [4, 4, 7, 6, 5],
                [4, 0, 4, 5, 1],
                [4, 3, 2, 6, 7],
                [4, 0, 3, 7, 4],
                [4, 1, 5, 6, 2],
            ]
        )

        surface = pv.PolyData(
            points,
            faces,
        ).triangulate()

        surface = surface.compute_normals(
            consistent_normals=True,
            auto_orient_normals=True,
            point_normals=False,
            cell_normals=True,
        )

        stl_file = tmp_path / "slanted_region.stl"

        surface.save(stl_file)

        # ============================================================
        # 1) NORMAL <-> NORMAL
        # ============================================================
        #
        # Background:
        #
        #     eps_r = 2
        #     mu_r  = 1
        #     sigma = 0
        #
        # Insert:
        #
        #     eps_r = 8
        #     mu_r  = 4
        #     sigma = 0
        # ============================================================

        grid = GridFIT3D(
            0.0,
            5.0,
            0.0,
            5.0,
            0.0,
            5.0,
            6,
            6,
            6,
            stl_solids={
                "insert": str(stl_file),
            },
            stl_materials={
                "insert": [8.0, 4.0, 0.0],
            },
            stl_material_types={
                "insert": "normal",
            },
            geometry_mode="conformal",
            stl_method="interior_points",
            verbose=0,
        )

        # use_stl=False intentionally prevents automatic application
        # of the conformal material/metric data to the solver operators.
        solver = SolverFIT3D(
            grid,
            bg=[2.0, 1.0, 0.0],
            use_stl=False,
            verbose=0,
        )

        solver._build_conformal_effective_lengths()
        solver._build_conformal_face_data()
        solver._build_conformal_face_materials()
        solver._build_conformal_effective_areas()
        solver._build_conformal_diagonal_materials()

        insert_id = grid.region_key_to_id["insert"]

        # ============================================================
        # Primal y-edge
        # ============================================================
        #
        # Edge:
        #
        #     (2, 1, 2) -> (2, 2, 2)
        #
        # At x = 2:
        #
        #     y_s(2) = 1.375
        #
        # Thus:
        #
        #     insert fraction     = 0.375 = 3/8
        #     background fraction = 0.625 = 5/8
        #
        # For a normal-normal interface the full geometrical edge
        # remains active.
        #
        # The line material is harmonically averaged:
        #
        # eps_eff =
        #
        #       1
        # -----------------
        # (3/8)/8 + (5/8)/2
        #
        # = 64/23
        # ============================================================

        edge_index = (2, 1, 2)

        line = solver.conformal_line_data["primal"]["y"]

        assert line["normal_normal"][edge_index]

        assert line["alpha"][edge_index] == pytest.approx(3.0 / 8.0)

        assert solver.L_fraction.to_matrix("y")[edge_index] == pytest.approx(1.0)

        assert solver.L_eff.to_matrix("y")[edge_index] == pytest.approx(1.0)

        eps_line_expected = 1.0 / ((3.0 / 8.0) / 8.0 + (5.0 / 8.0) / 2.0)

        eps_line = solver.conformal_line_materials["primal"]["y"]["eps_r"][edge_index]

        assert eps_line == pytest.approx(eps_line_expected)

        assert eps_line == pytest.approx(64.0 / 23.0)

        # ============================================================
        # Dual y-face
        # ============================================================
        #
        # Associated dual face:
        #
        #     y = 1.5
        #
        # with
        #
        #     1.5 <= x <= 2.5
        #     1.5 <= z <= 2.5
        #
        # Solve interface position:
        #
        #     1.5 = 1.375 + 0.5 * (x - 2)
        #
        # ->  x = 2.25
        #
        # Hence:
        #
        #     insert area fraction     = 1/4
        #     background area fraction = 3/4
        #
        # Arithmetic face average:
        #
        #     eps_eff =
        #         1/4 * 8
        #       + 3/4 * 2
        #
        #     = 3.5
        #
        # Since both materials are normal, the complete face remains
        # geometrically active.
        # ============================================================

        # Dual coordinates are [0, 0.5, 1.5, ..., 4.5, 5]: y = 1.5 is dual
        # node 2 and the x/z range [1.5, 2.5] is dual cell 2.
        dual_face_index = (2, 2, 2)

        dual_face_geometry = grid.interface_face_data["dual"]["y"]

        dual_face_fractions = dual_face_geometry["region_area_fractions"][
            dual_face_index
        ]

        assert dual_face_fractions[insert_id] == pytest.approx(1.0 / 4.0)

        assert dual_face_fractions[0] == pytest.approx(3.0 / 4.0)

        eps_face_expected = (1.0 / 4.0) * 8.0 + (3.0 / 4.0) * 2.0

        eps_face = solver.conformal_face_materials["dual"]["y"]["eps_r"][
            dual_face_index
        ]

        assert eps_face == pytest.approx(eps_face_expected)

        assert eps_face == pytest.approx(3.5)

        # ------------------------------------------------------------
        # Map the dual-face material to its associated primal y-edge.
        #
        # primal y-edge:
        #     (2, 1, 2)
        #
        # associated dual y-face:
        #     (2, 2, 2)  (Ey(i,j,k) <-> dual y-face (i, j+1, k))
        # ------------------------------------------------------------

        eps_parallel = solver._dual_face_array_to_primal_edge_component(
            solver.conformal_face_materials["dual"]["y"]["eps_r"],
            "y",
        )

        assert eps_parallel[edge_index] == pytest.approx(3.5)

        # ------------------------------------------------------------
        # Diagonal material approximation
        #
        # Interface:
        #
        #     y - 0.5 x = const.
        #
        # therefore:
        #
        #     n = (-0.5, 1, 0) / sqrt(1.25)
        #
        # and:
        #
        #     n_y^2 = 4/5
        #
        # eps_yy =
        #     n_y^2 * eps_perp
        #     + (1 - n_y^2) * eps_parallel
        # ------------------------------------------------------------

        normal = grid.interface_intersections["primal"]["y"]["normal"][edge_index]

        normal = normal / np.linalg.norm(normal)

        ny2 = normal[1] ** 2

        assert ny2 == pytest.approx(4.0 / 5.0)

        eps_perp = eps_line

        eps_yy = ny2 * eps_perp + (1.0 - ny2) * eps_parallel[edge_index]

        assert eps_yy == pytest.approx(673.0 / 230.0)

        assert solver.conformal_face_data["dual"]["y"]["effective_area_fraction"][
            dual_face_index
        ] == pytest.approx(1.0)

        eps_yy_solver = solver.conformal_diagonal_materials["eps_r"]["y"][edge_index]

        assert eps_yy_solver == pytest.approx(673.0 / 230.0)

        sigma_yy_solver = solver.conformal_diagonal_materials["sigma"]["y"][edge_index]

        assert sigma_yy_solver == pytest.approx(0.0)
        # ============================================================
        # Primal z-face
        # ============================================================
        #
        # Face:
        #
        #     2 <= x <= 3
        #     1 <= y <= 2
        #
        # At x = 2:
        #
        #     y_s = 1.375
        #
        # At x = 3:
        #
        #     y_s = 1.875
        #
        # Material area is therefore the trapezoid
        #
        #       (0.375 + 0.875)
        # A = ------------------- * 1
        #               2
        #
        #   = 0.625 = 5/8
        #
        # Thus:
        #
        #     insert fraction     = 5/8
        #     background fraction = 3/8
        #
        # Arithmetic mu average:
        #
        #     mu_eff =
        #         5/8 * 4
        #       + 3/8 * 1
        #
        #     = 23/8
        # ============================================================

        primal_face_index = (2, 1, 2)

        primal_face_geometry = grid.interface_face_data["primal"]["z"]

        primal_face_fractions = primal_face_geometry["region_area_fractions"][
            primal_face_index
        ]

        assert primal_face_fractions[insert_id] == pytest.approx(5.0 / 8.0)

        assert primal_face_fractions[0] == pytest.approx(3.0 / 8.0)

        mu_face_expected = (5.0 / 8.0) * 4.0 + (3.0 / 8.0) * 1.0

        mu_face = solver.conformal_face_materials["primal"]["z"]["mu_r"][
            primal_face_index
        ]

        assert mu_face == pytest.approx(mu_face_expected)

        assert mu_face == pytest.approx(23.0 / 8.0)

        # ------------------------------------------------------------
        # Tangential magnetic component
        #
        # The interface normal is
        #
        #     n = (-0.5, 1, 0) / sqrt(1.25)
        #
        # Hence n_z = 0.
        #
        # The z-directed dual edge associated with the primal z-face
        # is therefore tangential to the interface. It does not need
        # a harmonic line contribution:
        #
        #     mu_zz = mu_parallel
        # ------------------------------------------------------------

        mu_parallel = solver._primal_face_array_to_dual_edge_component(
            solver.conformal_face_materials["primal"]["z"]["mu_r"],
            "z",
        )

        # Hz(i,j,k) is the flux through primal z-face (i,j,k) (curl matrix C).
        dual_edge_index = primal_face_index

        assert dual_edge_index == (
            2,
            1,
            2,
        )

        assert mu_parallel[dual_edge_index] == pytest.approx(23.0 / 8.0)

        face_segment = grid.interface_face_data["primal"]["z"]["segments"][
            primal_face_index
        ]

        normal = np.asarray(
            face_segment["normal"],
            dtype=float,
        )

        normal /= np.linalg.norm(normal)

        nz2 = normal[2] ** 2

        assert nz2 == pytest.approx(
            0.0,
            abs=1e-12,
        )

        mu_zz = mu_parallel[dual_edge_index]

        assert mu_zz == pytest.approx(23.0 / 8.0)

        assert solver.A_fraction.to_matrix("z")[primal_face_index] == pytest.approx(1.0)

        mu_zz_solver = solver.conformal_diagonal_materials["mu_r"]["z"][dual_edge_index]

        assert mu_zz_solver == pytest.approx(23.0 / 8.0)

        # ============================================================
        # Applied conformal material tensors
        # ============================================================
        #
        # Up to this point, the conformal preprocessing quantities
        #
        #     eps_perp, eps_parallel
        #     mu_perp,  mu_parallel
        #
        # and the resulting diagonal approximation have been tested
        # independently.
        #
        # Now initialize a real solver with use_stl=True and verify
        # that the final material Fields used by the FIT time stepping
        # contain the same analytically expected values.
        # ============================================================

        solver_applied = SolverFIT3D(
            grid,
            bg=[
                2.0,
                1.0,
                0.0,
            ],
            use_stl=True,
            verbose=0,
        )
        # ------------------------------------------------------------
        # Electric y-component
        #
        # Expected:
        #
        #     eps_yy = 673 / 230
        #
        # solver.ieps stores
        #
        #     1 / epsilon
        #
        # with
        #
        #     epsilon = eps_r * eps_0
        # ------------------------------------------------------------

        eps_yy_applied = 1.0 / (solver_applied.ieps.to_matrix("y")[edge_index] * eps_0)

        assert eps_yy_applied == pytest.approx(673.0 / 230.0)

        # The automatically constructed diagonal material data must
        # contain the same value.
        assert solver_applied.conformal_diagonal_materials["eps_r"]["y"][
            edge_index
        ] == pytest.approx(673.0 / 230.0)

        # ------------------------------------------------------------
        # Magnetic z-component
        #
        # n_z = 0, therefore the field component is purely tangential:
        #
        #     mu_zz = mu_parallel = 23 / 8
        #
        # solver.imu stores
        #
        #     1 / mu
        #
        # with
        #
        #     mu = mu_r * mu_0
        # ------------------------------------------------------------

        mu_zz_applied = 1.0 / (
            solver_applied.imu.to_matrix("z")[dual_edge_index] * mu_0
        )

        assert mu_zz_applied == pytest.approx(23.0 / 8.0)

        assert solver_applied.conformal_diagonal_materials["mu_r"]["z"][
            dual_edge_index
        ] == pytest.approx(23.0 / 8.0)

        # ------------------------------------------------------------
        # Conductivity
        #
        # Both normal materials are insulating in this test:
        #
        #     sigma = 0
        # ------------------------------------------------------------

        sigma_yy_applied = solver_applied.sigma.to_matrix("y")[edge_index]

        assert sigma_yy_applied == pytest.approx(0.0)

        assert solver_applied.conformal_diagonal_materials["sigma"]["y"][
            edge_index
        ] == pytest.approx(0.0)
        # ============================================================
        # 2) SAME GEOMETRY, INSERT = PEC
        # ============================================================

        grid_pec = GridFIT3D(
            0.0,
            5.0,
            0.0,
            5.0,
            0.0,
            5.0,
            6,
            6,
            6,
            stl_solids={
                "insert": str(stl_file),
            },
            stl_materials={
                "insert": [1.0, 1.0, 0.0],
            },
            stl_material_types={
                "insert": "pec",
            },
            geometry_mode="conformal",
            stl_method="interior_points",
            verbose=0,
        )

        solver_pec = SolverFIT3D(
            grid_pec,
            bg=[2.0, 1.0, 0.0],
            use_stl=False,
            verbose=0,
        )

        solver_pec._build_conformal_effective_lengths()
        solver_pec._build_conformal_face_data()
        solver_pec._build_conformal_face_materials()
        solver_pec._build_conformal_effective_areas()

        # ============================================================
        # PEC: primal y-edge
        # ============================================================
        #
        # First 3/8 of the edge lies inside PEC.
        # Remaining 5/8 lies inside the normal background.
        #
        # Hence:
        #
        #     L_fraction = 5/8
        #     L_eff      = 5/8
        #
        # Material of the active part:
        #
        #     eps_r = eps_r,bg = 2
        # ============================================================

        line_pec = solver_pec.conformal_line_data["primal"]["y"]

        assert line_pec["conductor_to_normal"][edge_index]

        assert line_pec["alpha"][edge_index] == pytest.approx(3.0 / 8.0)

        assert solver_pec.L_fraction.to_matrix("y")[edge_index] == pytest.approx(
            5.0 / 8.0
        )

        assert solver_pec.L_eff.to_matrix("y")[edge_index] == pytest.approx(5.0 / 8.0)

        eps_line_pec = solver_pec.conformal_line_materials["primal"]["y"]["eps_r"][
            edge_index
        ]

        assert eps_line_pec == pytest.approx(2.0)

        # ============================================================
        # PEC: dual y-face
        # ============================================================
        #
        # Conductor walls never cut the dual metrics, so the dual grid is
        # not traced against the normal-PEC interface at all: the dual
        # face stays unresolved and its area remains the grid value.
        # ============================================================

        dual_face_data_pec = solver_pec.conformal_face_data["dual"]["y"]

        assert not dual_face_data_pec["resolved"][dual_face_index]

        assert not dual_face_data_pec["normal_pec"][dual_face_index]

        assert not hasattr(solver_pec, "tA_eff")

        # The solver's itA differs from grid.itA only through the (default
        # periodic) boundary conditions, so compare with an uncut solver.
        reference = SolverFIT3D(
            GridFIT3D(0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 6, 6, 6, verbose=0),
            use_stl=False,
            verbose=0,
        )
        np.testing.assert_array_equal(
            solver_pec.itA.toarray(),
            reference.itA.toarray(),
        )

        # ============================================================
        # PEC: primal z-face
        # ============================================================
        #
        # Face fractions:
        #
        #     PEC        = 5/8
        #     background = 3/8
        #
        # Therefore:
        #
        #     A_fraction = 3/8
        #     A_eff      = 3/8
        #
        # Remaining material:
        #
        #     mu_r = 1
        # ============================================================

        primal_face_data_pec = solver_pec.conformal_face_data["primal"]["z"]

        assert primal_face_data_pec["normal_pec"][primal_face_index]

        assert primal_face_data_pec["effective_area_fraction"][
            primal_face_index
        ] == pytest.approx(3.0 / 8.0)

        assert solver_pec.A_fraction.to_matrix("z")[primal_face_index] == pytest.approx(
            3.0 / 8.0
        )

        assert solver_pec.A_eff.to_matrix("z")[primal_face_index] == pytest.approx(
            3.0 / 8.0
        )

        mu_face_pec = solver_pec.conformal_face_materials["primal"]["z"]["mu_r"][
            primal_face_index
        ]

        assert mu_face_pec == pytest.approx(1.0)

        # ============================================================
        # Applied conformal FIT metric operators
        # ============================================================

        solver_pec_applied = SolverFIT3D(
            grid_pec,
            bg=[
                2.0,
                1.0,
                0.0,
            ],
            use_stl=True,
            verbose=0,
        )

        # ------------------------------------------------------------
        # The sparse metric matrices must use the effective
        # conformal geometry.
        # ------------------------------------------------------------

        np.testing.assert_allclose(
            solver_pec_applied.Ds.diagonal(),
            solver_pec_applied.L.toarray(),
        )

        np.testing.assert_allclose(
            solver_pec_applied.tDs.diagonal(),
            solver_pec_applied.tL.toarray(),
        )

        np.testing.assert_allclose(
            solver_pec_applied.iDa.diagonal(),
            solver_pec_applied.iA.toarray(),
        )

        np.testing.assert_allclose(
            solver_pec_applied.itDa.diagonal(),
            solver_pec_applied.itA.toarray(),
        )

        # Primal y-edge:
        # active normal length = 5/8
        assert solver_pec_applied.L.to_matrix("y")[edge_index] == pytest.approx(
            5.0 / 8.0
        )

        # Dual y-face: conductor walls do not cut dual areas.
        assert solver_pec_applied.itA.to_matrix("y")[edge_index] == pytest.approx(
            grid_pec.itA.to_matrix("y")[edge_index]
        )

        # Primal z-face:
        # active normal area = 3/8
        assert solver_pec_applied.iA.to_matrix("z")[primal_face_index] == pytest.approx(
            8.0 / 3.0
        )

    def test_legacy_solver_uses_original_metric_operators(self):
        grid = GridFIT3D(
            0.0,
            2.0,
            0.0,
            3.0,
            0.0,
            4.0,
            3,
            4,
            5,
            geometry_mode="legacy",
            verbose=0,
        )

        solver = SolverFIT3D(
            grid,
            use_stl=False,
            verbose=0,
        )

        np.testing.assert_allclose(
            solver.Ds.diagonal(),
            solver.L.toarray(),
        )

        np.testing.assert_allclose(
            solver.iDa.diagonal(),
            solver.iA.toarray(),
        )

        np.testing.assert_allclose(
            solver.tDs.diagonal(),
            solver.tL.toarray(),
        )

        np.testing.assert_allclose(
            solver.itDa.diagonal(),
            solver.itA.toarray(),
        )
