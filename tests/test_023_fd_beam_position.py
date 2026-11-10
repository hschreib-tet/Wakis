"""
The FD impedance sweep must place the beam on the Ez grid line at
(xsource, ysource), as the time-domain beam source does.
"""

import numpy as np

from wakis import GridFIT3D, WakeSolver
from wakis.solverFIT3D_FD import SolverFIT3D_FD


def test_impedance_sweep_beam_on_requested_ez_line():
    # Nx, Ny, Nz count primal nodes: 8 x 8 x 6 cells.
    grid = GridFIT3D(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 9, 9, 7, verbose=0)
    solver = SolverFIT3D_FD(
        grid,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        use_stl=False,
        verbose=0,
    )

    sources = []

    def capture(omega, source):
        sources.append(np.asarray(source).copy())
        return np.zeros(3 * solver.N, dtype=np.complex128)

    solver.solve = capture

    for xs, ys in ((0.0, 0.0), (0.5, -0.25)):
        sources.clear()
        solver.impedance_sweep(
            np.array([1e8]),
            wake=WakeSolver(beta=1.0, xsource=xs, ysource=ys, xtest=xs, ytest=ys),
        )

        ez = sources[0][2 * solver.N :].reshape(
            (solver.Nx, solver.Ny, solver.Nz), order="F"
        )
        i, j = np.argwhere(np.abs(ez).sum(axis=2) > 0)[0]

        # Ez(i, j, :) lies on the primal grid line (grid.x[i], grid.y[j]).
        assert grid.x[i] == xs
        assert grid.y[j] == ys

        # The beam drives every physical Ez edge and not the zero-length
        # ghost edge at the high z boundary.
        assert np.all(np.abs(ez[i, j, :-1]) > 0)
        assert ez[i, j, -1] == 0
