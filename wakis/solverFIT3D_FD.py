# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .routines_FD import RoutinesFDMixin
from .solverFIT3D import SolverFIT3D


class SolverFIT3D_FD(
    RoutinesFDMixin,
    SolverFIT3D,
):
    def __init__(self, *args, **kwargs):
        bc_low = kwargs.get(
            "bc_low",
            ["Periodic", "Periodic", "Periodic"],
        )

        bc_high = kwargs.get(
            "bc_high",
            ["Periodic", "Periodic", "Periodic"],
        )

        supported_bcs = {
            "periodic",
            "pec",
            "electric",
            "pmc",
            "magnetic",
        }

        requested_bcs = {bc.lower() for bc in (*bc_low, *bc_high)}

        unsupported_bcs = requested_bcs - supported_bcs

        if unsupported_bcs:
            raise NotImplementedError(
                "SolverFIT3D_FD currently supports only "
                "Periodic, PEC/Electric and PMC/Magnetic "
                f"boundary conditions. Unsupported: "
                f"{sorted(unsupported_bcs)}"
            )

        super().__init__(*args, **kwargs)

        self._build_active_e_dofs()
        self._assemble_fd_operators()

    def _build_active_e_dofs(self):
        """
        Build the active electric degree-of-freedom set.

        Electric DOFs are inactive if they belong either to
        a PEC material or to an electric Dirichlet boundary.
        """

        active_material = self.iDeps.diagonal() != 0.0

        active_boundary = self.bc_mask_E.toarray().astype(bool)

        self.active_E = active_material & active_boundary

        self.free_E_idx = np.flatnonzero(self.active_E)

    def _assemble_fd_operators(self):
        """
        Assemble the frequency-independent FIT operators.

        Only the electric unknown space is reduced. Magnetic DOFs remain
        in the full space because magnetic/PMC boundary conditions are
        already incorporated as zero rows in the curl matrix C.

        Builds

            Mnu
            Meps
            Msigma
            SL = C_red.T @ Mnu @ C_red

        with

            C_red = C[:, free_E_idx].
        """

        # ----------------------------------------------------------
        # Reduced curl matrix
        #
        # Only electric DOFs are eliminated. The rows of C remain in
        # the full magnetic space.
        # ----------------------------------------------------------

        C = self.C.tocsr()

        self.C_red = C[
            :,
            self.free_E_idx,
        ].tocsr()

        # ----------------------------------------------------------
        # Magnetic reluctance matrix
        #
        # Mnu = tDs * iDmu * iDa
        #
        # Mnu remains in the full magnetic space.
        # ----------------------------------------------------------

        Mnu_diag = self.tDs.diagonal() * self.iDmu.diagonal() * self.iDa.diagonal()

        if np.any(~np.isfinite(Mnu_diag)):
            raise RuntimeError("Non-finite entries found in magnetic material matrix.")

        self.Mnu = diags(Mnu_diag).tocsr()

        # ----------------------------------------------------------
        # Electric permittivity matrix
        #
        # Meps = inv(Ds * iDeps * itDa)
        #
        # Only active electric DOFs are retained.
        # ----------------------------------------------------------

        inverse_Meps_diag = (
            self.Ds.diagonal() * self.iDeps.diagonal() * self.itDa.diagonal()
        )

        inverse_Meps_red = inverse_Meps_diag[self.free_E_idx]

        if np.any(~np.isfinite(inverse_Meps_red)):
            raise RuntimeError("Non-finite entries found in electric material matrix.")

        if np.any(inverse_Meps_red <= 0.0):
            raise RuntimeError(
                "Non-positive electric material/geometry "
                "entry found on an active electric DOF."
            )

        self.Meps = diags(1.0 / inverse_Meps_red).tocsr()

        # ----------------------------------------------------------
        # Electric conductivity matrix
        #
        # Msigma = tDa * Dsigma * inv(Ds)
        #
        # Since itDa = inv(tDa):
        #
        # Msigma_diag = sigma / (Ds * itDa)
        # ----------------------------------------------------------

        conductivity_geometry = self.Ds.diagonal() * self.itDa.diagonal()

        conductivity_geometry_red = conductivity_geometry[self.free_E_idx]

        if np.any(~np.isfinite(conductivity_geometry_red)):
            raise RuntimeError("Non-finite conductivity geometry factor.")

        if np.any(conductivity_geometry_red <= 0.0):
            raise RuntimeError(
                "Non-positive conductivity geometry factor on an active electric DOF."
            )

        Msigma_diag = (
            self.Dsigma.diagonal()[self.free_E_idx] / conductivity_geometry_red
        )

        self.Msigma = diags(Msigma_diag).tocsr()

        # ----------------------------------------------------------
        # Curl-curl / stiffness matrix
        #
        # C_red : (3N) x NE
        # Mnu   : (3N) x (3N)
        # SL    : NE x NE
        # ----------------------------------------------------------

        self.SL = (self.C_red.transpose() @ self.Mnu @ self.C_red).tocsr()

    def build_system_matrix(
        self,
        omega,
    ):
        """
        Build the reduced frequency-domain FIT system matrix.

        The phasor convention is exp(+j*omega*t):

            A(omega)
                = Msigma
                + j*omega*Meps
                + 1/(j*omega) * SL
        """

        omega = float(omega)

        if not np.isfinite(omega):
            raise ValueError("omega must be finite.")

        if omega <= 0.0:
            raise ValueError("omega must be greater than zero.")

        return (
            self.Msigma + 1j * omega * self.Meps + 1.0 / (1j * omega) * self.SL
        ).tocsr()

    def solve(self, omega, source):
        """
        Solve the frequency-domain FIT system

            A(omega) e = -j

        where e contains electric grid voltages and j is the
        integrated electric current source on the dual faces.

        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].

        source : array_like
            Full electric source vector with shape (3*N,).
            Entries have units of ampere.

        Returns
        -------
        e : ndarray
            Full electric grid-voltage vector with shape (3*N,).
            Inactive PEC DOFs are zero.
        """

        source = np.asarray(source)

        if source.shape != (3 * self.N,):
            raise ValueError(
                f"source must have shape {(3 * self.N,)}, got {source.shape}."
            )

        A = self.build_system_matrix(omega)

        source_red = source[self.free_E_idx]

        e_red = spsolve(
            A,
            -source_red,
        )

        e = np.zeros(
            3 * self.N,
            dtype=np.complex128,
        )

        e[self.free_E_idx] = e_red

        return e
