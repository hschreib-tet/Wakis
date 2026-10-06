import sys

from types import SimpleNamespace

from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0

import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


@pytest.mark.slow
class TestGridFIT3DMeshing:
    # Reference data
    tol = dict(rtol=50e-5, atol=50e-4)
    dtype = np.float32

    # fmt: off
    WP = np.array([ 3.03999377e-18, -1.57436678e-14, -5.81744067e-11, -4.16962878e-08,
                    -8.37576052e-06, -5.10188367e-04, -9.65024317e-03, -5.62474460e-02,
                    -8.92658495e-02,  1.94104416e-02,  1.20731894e-01,  6.94716656e-02,
                    -2.25460627e-02, -4.91365346e-02, -2.81756714e-02, -1.07040993e-02,
                    6.24067287e-02,  7.76899724e-02, -6.41804617e-02, -1.09100607e-01,
                    3.17122367e-02,  9.95300771e-02,  1.59930539e-02, -6.11959223e-02,
                    -3.38419464e-02, -3.31974531e-03,  1.97809615e-02,  7.32837232e-02,
                    2.14655914e-03, -9.54641911e-02, -3.93029761e-02,  7.85793202e-02,
                    6.92603412e-02, -4.44469131e-02, -5.61724850e-02, -4.31207455e-03,
                    1.37319512e-02,  4.47538136e-02,  3.30728875e-02, -4.91526078e-02,
                    -7.17331871e-02,  2.30389026e-02,  9.07601155e-02,  4.60074950e-03,
                    -6.82039506e-02, -2.42327580e-02,  1.47484248e-02,  3.33755469e-02,
                    3.07827068e-02, -1.10044189e-02, -6.00957767e-02, -2.83398905e-02,
                    7.01179625e-02,  5.07391705e-02, -4.55863327e-02, -5.23725044e-02,
                    3.64085044e-03,  3.55489172e-02,  2.52530595e-02,  3.40678159e-03,
                    -3.39325723e-02, -4.70488102e-02,  2.66807614e-02,  6.77546427e-02,
                    -3.49550941e-03, -5.86094480e-02, -2.46501822e-02,  3.33310938e-02,
                    3.19647879e-02,  3.80802801e-03, -1.52754979e-02, -4.11351141e-02,
                    -7.28641157e-03,  5.23813004e-02,  3.22026092e-02, -3.76132883e-02,
                    -4.92387915e-02,  1.42774059e-02,  4.05807388e-02,  1.10590376e-02,
                    -1.09048635e-02, -2.85400383e-02, -2.08799885e-02,  2.56681667e-02,
                    4.33091263e-02, -4.11216871e-03, -5.22413075e-02, -1.57761296e-02,
                    3.85064574e-02,  2.44861946e-02, -5.94644029e-03, -2.37867277e-02,
                    -2.08974540e-02,  6.12977218e-03,  3.41808685e-02,  2.11225797e-02,
                    -3.48074853e-02, -3.68432258e-02,  1.84538236e-02,  3.62614155e-02,
                    5.71392888e-03, -2.22282929e-02, -2.07595900e-02, -2.14059628e-03])


    Z = np.array([ 4.87770399e+00   -0.j,         -9.37231333e-01   +9.76902j,
                    -4.24056580e-02   +5.75895605j,  7.64718800e+00  +15.05633851j,
                    2.25805798e-02  +22.94919652j,  1.81227395e+00  +18.44213153j,
                    9.25440294e+00  +29.07627434j,  2.80600688e-01  +36.56571882j,
                    3.21773007e+00  +31.43991496j,  1.09670695e+01  +44.23182763j,
                    4.43604660e-02  +51.73186851j,  4.57961315e+00  +45.61130327j,
                    1.31989216e+01  +61.78568579j, -7.53366266e-01  +69.74316051j,
                    6.17983511e+00  +62.00941222j,  1.65219994e+01  +83.7460061j,
                    -2.43993772e+00  +92.90805422j,  8.45866414e+00  +82.50488264j,
                    2.21727722e+01 +114.27862094j, -5.88329858e+00 +126.41846549j,
                    1.25234418e+01 +111.41469792j,  3.38300774e+01 +164.58570432j,
                    -1.36120025e+01 +186.02864687j,  2.28328211e+01 +163.1007426j,
                    6.98071354e+01 +281.32459768j, -3.41036324e+01 +359.14919895j,
                    9.25735014e+01 +348.74599877j,  8.21857579e+02+1199.51083954j,
                    1.12890901e+03-1003.57407371j, -1.01266995e+01 -186.88815086j,
                    -4.70789813e+01 -262.98926442j,  1.12077022e+02  -91.51376311j,
                    -1.54554302e+01  +54.34806626j, -2.88613554e+01  -36.43009739j,
                    8.69109385e+01  +55.12056128j, -1.08808342e+01 +164.39050487j,
                    -1.17645744e+01  +86.56414324j,  1.18563803e+02 +213.48816771j,
                    1.20556558e+01 +395.18631788j,  9.57219766e+01 +399.30569715j,
                    1.16332782e+03 +970.86145883j,  1.04839496e+03-1020.32813672j,
                    6.54871934e+01 -429.14499028j,  2.24857075e+01 -363.85175326j,
                    6.78880242e+01 -188.44076494j, -1.98911757e+01 -125.63772553j,
                    2.05538171e+01 -141.29209289j,  3.79217091e+01  -41.08975005j,
                    -2.73138490e+01  -38.01589641j,  3.61895934e+01  -59.80608412j])

    #Ez = np.array([])

    # fmt: on
    def test_voxelize_rectilinear(self, use_gpu):
        """
        Tests 'voxelize_rectilinear' and subpixel smoothing using the
        exact cavity and shell gridLogs configuration.
        """

        # Geometry & Materials
        solid_1 = "tests/stl/007_vacuum_cavity.stl"  # logs["stl_solids"]["cavity"]
        solid_2 = "tests/stl/007_lossymetal_shell.stl"  # logs["stl_solids"]["shell"]

        stl_solids = {"cavity": solid_1, "shell": solid_2}

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
        }

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

        # Number of mesh cells
        Nx = 60  # logs["Nx"]
        Ny = 60  # logs["Ny"]
        Nz = 140  # logs["Nz"]

        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,  # Global domain zmin
            zmax,  # Global domain zmax
            Nx,
            Ny,
            Nz,  # Global domain Nz
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_method="voxelize_rectilinear",
            subpixel_smoothing=False,
            stl_scale=1.0,
            stl_rotate=[0, 0, 0],
            stl_translate=[0, 0, 0],
            verbose=1,
        )

        # number of cells in the mask
        n_inside = grid.grid.threshold(scalars="shell", value=0.5).n_cells
        n_inside_expected = 61258
        assert n_inside == n_inside_expected, (
            f"Number of cells masked inside the shell is {n_inside}, expected {n_inside_expected}"
        )

        # volume
        vol = n_inside * np.min(grid.dx) * np.min(grid.dy) * np.min(grid.dz)
        vol_expected = 0.02629232100267493
        assert np.allclose(vol, vol_expected, rtol=1e-5), (
            f"Volume of the shell mask is {vol}, expected {vol_expected}"
        )

    def test_subpixel_smoothing(self):
        """
        Tests 'voxelize_rectilinear' and subpixel smoothing using the
        exact cavity and shell gridLogs configuration.
        """

        # Geometry & Materials
        solid_1 = "tests/stl/007_vacuum_cavity.stl"  # logs["stl_solids"]["cavity"]
        solid_2 = "tests/stl/007_lossymetal_shell.stl"  # logs["stl_solids"]["shell"]

        stl_solids = {"cavity": solid_1, "shell": solid_2}

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
        }

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

        # Number of mesh cells
        Nx = 60  # logs["Nx"]
        Ny = 60  # logs["Ny"]
        Nz = 140  # logs["Nz"]

        global grid
        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,  # Global domain zmin
            zmax,  # Global domain zmax
            Nx,
            Ny,
            Nz,  # Global domain Nz
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_method="voxelize_rectilinear",
            subpixel_smoothing=True,
            subpixel_smoothing_factor=4,
            subpixel_smoothing_bool=True,
            subpixel_smoothing_threshold=0.3,
            stl_scale=1.0,
            stl_rotate=[0, 0, 0],
            stl_translate=[0, 0, 0],
            verbose=1,
        )

        # number of cells in the mask
        n_inside = grid.grid.threshold(scalars="shell", value=0.5).n_cells
        n_inside_expected = 89256
        assert n_inside == n_inside_expected, (
            f"Number of cells masked inside the shell is {n_inside}, expected {n_inside_expected}"
        )

        # volume
        vol = n_inside * np.min(grid.dx) * np.min(grid.dy) * np.min(grid.dz)
        vol_expected = 0.038309239665264186
        assert np.allclose(vol, vol_expected, rtol=1e-5), (
            f"Volume of the shell mask is {vol}, expected {vol_expected}"
        )

    def test_conformal_geometry_masks(self):
        """
        Test conformal STL mask generation.

        Verifies that:
        1. the primal point mask is defined on all primal grid points,
        2. the cell mask stored in ``grid.grid[key]`` follows the
           5-of-8 corner-point rule,
        3. the dual point mask is derived consistently from the
           cell-center classification.
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

        # Number of mesh cells
        Nx = 60
        Ny = 60
        Nz = 140

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

        assert primal.shape == (
            Nx + 1,
            Ny + 1,
            Nz + 1,
        )

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

        assert expected_cell_mask.shape == (
            Nx,
            Ny,
            Nz,
        )

        # Cell mask actually stored in the PyVista grid
        cell_mask = np.reshape(
            np.asarray(grid.grid["shell"], dtype=bool),
            (Nx, Ny, Nz),
            order="C",
        )

        np.testing.assert_array_equal(
            cell_mask,
            expected_cell_mask,
        )

        # ----------------------------------------------------------
        # 3. Dual point mask
        # ----------------------------------------------------------
        dual = grid.dual_point_masks["shell"]

        assert dual.shape == (
            Nx + 1,
            Ny + 1,
            Nz + 1,
        )

        expected_dual_mask = np.pad(
            expected_cell_mask,
            (
                (0, 1),
                (0, 1),
                (0, 1),
            ),
            mode="edge",
        )

        np.testing.assert_array_equal(
            dual,
            expected_dual_mask,
        )


    def test_average_dual_points_to_primal_edges(self):
        """Test dual-point averaging onto the faces of primal edges."""

        Nx = 3
        Ny = 4
        Nz = 5

        # Unique value at every grid point, so indexing errors are visible.
        values = np.arange(
            (Nx + 1) * (Ny + 1) * (Nz + 1),
            dtype=float,
        ).reshape(Nx + 1, Ny + 1, Nz + 1)

        values_x, values_y, values_z = (
            SolverFIT3D._average_dual_points_to_primal_edges(values)
        )

        assert values_x.shape == (Nx, Ny, Nz)
        assert values_y.shape == (Nx, Ny, Nz)
        assert values_z.shape == (Nx, Ny, Nz)

        # ----------------------------------------------------------
        # Interior faces
        # ----------------------------------------------------------

        # x-directed primal edge -> dual yz face
        i, j, k = 1, 2, 3
        expected = 0.25 * (
            values[i, j, k]
            + values[i, j - 1, k]
            + values[i, j - 1, k - 1]
            + values[i, j, k - 1]
        )
        assert values_x[i, j, k] == pytest.approx(expected)

        # y-directed primal edge -> dual xz face
        i, j, k = 2, 1, 3
        expected = 0.25 * (
            values[i, j, k]
            + values[i - 1, j, k]
            + values[i - 1, j, k - 1]
            + values[i, j, k - 1]
        )
        assert values_y[i, j, k] == pytest.approx(expected)

        # z-directed primal edge -> dual xy face
        i, j, k = 2, 2, 1
        expected = 0.25 * (
            values[i, j, k]
            + values[i - 1, j, k]
            + values[i - 1, j - 1, k]
            + values[i, j - 1, k]
        )
        assert values_z[i, j, k] == pytest.approx(expected)

        # ----------------------------------------------------------
        # Low-side boundary treatment
        # ----------------------------------------------------------

        # x edge on ymin: only two distinct dual points contribute
        i, j, k = 1, 0, 3
        expected = 0.5 * (
            values[i, 0, k]
            + values[i, 0, k - 1]
        )
        assert values_x[i, j, k] == pytest.approx(expected)

        # x edge on ymin and zmin: only one dual point contributes
        i, j, k = 1, 0, 0
        expected = values[i, 0, 0]
        assert values_x[i, j, k] == pytest.approx(expected)


    def test_average_primal_points_to_dual_edges(self):
        """Test primal-point averaging onto the faces of dual edges."""

        Nx = 3
        Ny = 4
        Nz = 5

        values = np.arange(
            (Nx + 1) * (Ny + 1) * (Nz + 1),
            dtype=float,
        ).reshape(Nx + 1, Ny + 1, Nz + 1)

        values_x, values_y, values_z = (
            SolverFIT3D._average_primal_points_to_dual_edges(values)
        )

        assert values_x.shape == (Nx, Ny, Nz)
        assert values_y.shape == (Nx, Ny, Nz)
        assert values_z.shape == (Nx, Ny, Nz)

        # ----------------------------------------------------------
        # Interior faces
        # ----------------------------------------------------------

        # x-directed dual edge -> primal yz face at x[i+1]
        i, j, k = 1, 2, 3
        expected = 0.25 * (
            values[i + 1, j, k]
            + values[i + 1, j + 1, k]
            + values[i + 1, j + 1, k + 1]
            + values[i + 1, j, k + 1]
        )
        assert values_x[i, j, k] == pytest.approx(expected)

        # y-directed dual edge -> primal xz face at y[j+1]
        i, j, k = 1, 1, 3
        expected = 0.25 * (
            values[i, j + 1, k]
            + values[i + 1, j + 1, k]
            + values[i + 1, j + 1, k + 1]
            + values[i, j + 1, k + 1]
        )
        assert values_y[i, j, k] == pytest.approx(expected)

        # z-directed dual edge -> primal xy face at z[k+1]
        i, j, k = 1, 2, 1
        expected = 0.25 * (
            values[i, j, k + 1]
            + values[i + 1, j, k + 1]
            + values[i + 1, j + 1, k + 1]
            + values[i, j + 1, k + 1]
        )
        assert values_z[i, j, k] == pytest.approx(expected)

        # ----------------------------------------------------------
        # High-side boundary
        # ----------------------------------------------------------

        # Last stored x-directed dual edge terminates at x[Nx].
        # All four primal points of its associated face exist.
        i = Nx - 1
        j = Ny - 1
        k = Nz - 1

        expected = 0.25 * (
            values[Nx, Ny - 1, Nz - 1]
            + values[Nx, Ny, Nz - 1]
            + values[Nx, Ny, Nz]
            + values[Nx, Ny - 1, Nz]
        )
        assert values_x[i, j, k] == pytest.approx(expected)


    def test_conformal_material_assignment(self):
        """Test point-based material assignment onto FIT material tensors."""

        Nx = 2
        Ny = 2
        Nz = 2

        shape_points = (Nx + 1, Ny + 1, Nz + 1)
        shape_cells = (Nx, Ny, Nz)

        # ----------------------------------------------------------
        # Synthetic primal and dual point masks
        # ----------------------------------------------------------
        primal_mask = np.zeros(shape_points, dtype=bool)
        dual_mask = np.zeros(shape_points, dtype=bool)

        # For eps_x[1, 1, 1], the associated dual yz face uses
        #
        # (1,1,1), (1,0,1), (1,0,0), (1,1,0).
        #
        # Mark two of these four points as material.
        dual_mask[1, 1, 1] = True
        dual_mask[1, 0, 1] = True

        # For mu_x[0, 0, 0], the associated primal yz face uses
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

        assert solver.ieps.field_x[1, 1, 1] == pytest.approx(
            1.0 / expected_eps
        )

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

        assert solver.imu.field_x[0, 0, 0] == pytest.approx(
            1.0 / expected_mu
        )

        # Conductive material must enable conductivity treatment.
        assert solver.use_conductivity


    def test_conformal_material_assignment_homogeneous(self):
        """Test conformal material assignment for a homogeneous solid."""

        Nx = 2
        Ny = 3
        Nz = 4

        shape_points = (Nx + 1, Ny + 1, Nz + 1)
        shape_cells = (Nx, Ny, Nz)

        # All primal and dual grid points belong to the material.
        primal_mask = np.ones(shape_points, dtype=bool)
        dual_mask = np.ones(shape_points, dtype=bool)

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

    def test_long_wake_potential_and_impedance(self, use_gpu, plot_comparison):
        global grid
        # ------------ Beam source ----------------
        # Beam parameters
        sigmaz = 10e-2  # [m] -> 2 GHz
        q = 1e-9  # [C]
        beta = 1.0  # beam beta
        xs = 0.0  # x source position [m]
        ys = 0.0  # y source position [m]
        xt = 0.0  # x test position [m]
        yt = 0.0  # y test position [m]
        # [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s]

        # ----------- Wake Solver  setup  ----------
        # Wakefield post-processor
        wakelength = 10.0  # [m] -> Partially decayed
        skip_cells = 20  # no. cells to skip at zlo/zhi for wake integration
        results_folder = "tests/013_results/"

        wake = WakeSolver(
            q=q,
            sigmaz=sigmaz,
            beta=beta,
            xsource=xs,
            ysource=ys,
            xtest=xt,
            ytest=yt,
            skip_cells=skip_cells,
            results_folder=results_folder,
            Ez_file=results_folder + "Ez.h5",
        )

        # ----------- Solver & Simulation ----------
        # boundary conditions
        bc_low = ["pec", "pec", "pec"]
        bc_high = ["pec", "pec", "pec"]

        # Solver setup
        solver = SolverFIT3D(
            grid,
            wake,
            bc_low=bc_low,
            bc_high=bc_high,
            use_stl=True,
            bg="pec",  # Background material
            dtype=self.dtype,
            use_gpu=use_gpu,
        )

        # Run simulation
        solver.wakesolve(wakelength=wakelength)

        # print(wake.WP[::50])
        np.cumsum(np.abs(wake.WP))[-1]
        assert len(wake.WP) == 5195, "Wake potential mesh samples length mismatch"
        plot_comparison(wake.WP[::50], self.WP, "Wake potential")
        assert np.allclose(wake.WP[::50], self.WP, **self.tol), (
            "Wake potential mesh samples failed"
        )
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
            179.95393780891274, 0.1
        ), "Wake potential cumsum mesh failed"

        assert len(wake.Z) == 998, "Impedance samples length mismatch"
        plot_comparison(np.abs(wake.Z)[::20], np.abs(self.Z), "Impedance magnitude")
        assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **self.tol), (
            "Abs Impedance samples mesh failed"
        )
        plot_comparison(np.real(wake.Z)[::20], np.real(self.Z), "Impedance real part")
        assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **self.tol), (
            "Real Impedance samples mesh failed"
        )
        plot_comparison(
            np.imag(wake.Z)[::20], np.imag(self.Z), "Impedance imaginary part"
        )
        assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z), **self.tol), (
            "Imag Impedance samples mesh failed"
        )
        assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(
            249395.46953432143, 0.1
        ), "Abs Impedance cumsum mesh failed"
