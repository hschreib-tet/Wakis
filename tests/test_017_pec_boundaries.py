import numpy as np
import pytest

from wakis.gridFIT3D import GridFIT3D
from wakis.solverFIT3D import SolverFIT3D


class TestBoundaryConditions:
    @pytest.mark.parametrize(
        "axis, tangential_components",
        [
            (0, ("y", "z")),  # x boundary
            (1, ("x", "z")),  # y boundary
            (2, ("x", "y")),  # z boundary
        ],
    )
    def test_pec_low_and_high_boundary_mask(
        self,
        axis,
        tangential_components,
    ):
        """
        Regression test for PEC boundary masking.

        With the symmetric FIT stencil, Nx, Ny, Nz count primal nodes and
        both boundary node planes are stored. Tangential E-field DOFs on
        the low- and high-side PEC walls must therefore both be masked.
        """

        grid = GridFIT3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            Nx=3,
            Ny=4,
            Nz=5,
            verbose=0,
        )

        # Use periodic boundaries in the other two directions so that
        # only the PEC boundary in the tested direction affects BC.
        bc_low = ["periodic", "periodic", "periodic"]
        bc_high = ["periodic", "periodic", "periodic"]

        bc_low[axis] = "pec"
        bc_high[axis] = "pec"

        solver = SolverFIT3D(
            grid=grid,
            bc_low=bc_low,
            bc_high=bc_high,
            verbose=0,
        )

        for component in tangential_components:
            if axis == 0:
                # Exclude edges and corners shared by multiple boundary faces.
                # Their treatment depends on the combination of boundary conditions
                # and is outside the scope of this regression test.
                low_mask = solver.BC[0, 1:-1, 1:-1, component]
                high_mask = solver.BC[-1, 1:-1, 1:-1, component]

            elif axis == 1:
                low_mask = solver.BC[1:-1, 0, 1:-1, component]
                high_mask = solver.BC[1:-1, -1, 1:-1, component]

            else:
                low_mask = solver.BC[1:-1, 1:-1, 0, component]
                high_mask = solver.BC[1:-1, 1:-1, -1, component]

            # Tangential DOFs on both PEC walls are removed.
            assert np.all(low_mask == 0)
            assert np.all(high_mask == 0)
