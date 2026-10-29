# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh, spsolve

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
        self._build_sibc_geometry()

    def _build_conductor_e_dof_mask(self):
        """Return a boolean mask of primal electric edges in the field domain.

        PEC and SIBC use identical *geometric* exclusion rules. Edges with
        two conductor endpoints are removed; resolved normal-to-conductor
        edges are retained because they have a nonzero active line segment.

        This function does not impose the SIBC itself; the FD surface term
        must be added separately. Unsupported/hidden conductor interfaces
        are rejected rather than silently treated as fully active edges.
        """
        active = np.ones(3 * self.N, dtype=bool)

        if not self.use_stl or self.grid.geometry_mode != "conformal":
            return active

        if not self.region_material_types:
            self._build_region_material_types()

        for component, direction in enumerate(("x", "y", "z")):
            start_ids, end_ids = self.grid._get_edge_region_ids("primal", direction)
            conductor_ids = [
                region_id
                for region_id, kind in self.region_material_types.items()
                if kind in ("pec", "sibc")
            ]
            start_conductor = np.isin(start_ids, conductor_ids)
            end_conductor = np.isin(end_ids, conductor_ids)
            fully_conductor = start_conductor & end_conductor
            crossing = start_conductor ^ end_conductor

            line_data = self.conformal_line_data.get("primal", {}).get(direction)
            if line_data is None:
                raise RuntimeError(f"Missing conformal primal {direction}-edge data.")

            resolved = np.asarray(line_data["supported"], dtype=bool)
            alpha_fraction = np.asarray(
                line_data["effective_length_fraction"], dtype=float
            )
            invalid = crossing & (
                ~resolved | ~np.isfinite(alpha_fraction) | (alpha_fraction <= 0.0)
            )

            # The full arrays include high-side edges absent from Field.
            # Only validate edges that actually belong to the solver DOFs.
            invalid_stored = self._edge_array_to_field_component(invalid, direction)
            if np.any(invalid_stored):
                raise RuntimeError(
                    f"Unsupported normal-to-conductor geometry on "
                    f"{np.count_nonzero(invalid_stored)} stored {direction}-edges."
                )

            active_component = self._edge_array_to_field_component(
                ~fully_conductor, direction
            )
            active[component * self.N : (component + 1) * self.N] = (
                active_component.ravel(order="C")
            )

        return active

    def _build_active_e_dofs(self):
        """Build active electric DOFs from material, boundary and geometry.

        The geometric PEC/SIBC exclusion is shared. The impedance of SIBC
        boundaries is *not* included in this mask and must enter the FD
        operator through D_Zs later.
        """
        active_material = self.iDeps.diagonal() != 0.0
        active_boundary = self.bc_mask_E.toarray().astype(bool)
        self.active_conductor_E = self._build_conductor_e_dof_mask()

        self.active_E = active_material & active_boundary & self.active_conductor_E
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

    def _build_sibc_geometry(self):
        """Prepare scalar SIBC coefficients on *primal faces* once.

        Stored magnetic unknowns correspond to primal faces shifted in the
        normal direction. Therefore use _primal_face_array_to_dual_edge_component,
        NOT _face_array_to_field_component (which uses low-side faces).

        First-order approximation: H is taken parallel to the FIT face normal.
        Geometry coefficient g = cut_length * sin(theta) / active_face_area.
        The normal-region permeability used in the conversion b -> H is saved
        separately; SIBC material permeability belongs in Zs, not here.
        """
        self.sibc_g = np.zeros(3 * self.N, dtype=float)
        self.sibc_sigma = np.zeros(3 * self.N, dtype=float)
        self.sibc_mu_wall = np.zeros(3 * self.N, dtype=float)
        self.sibc_mu_field = np.ones(3 * self.N, dtype=float)
        self.has_sibc = False

        if not self.use_stl or self.grid.geometry_mode != "conformal":
            return
        for direction in ("x", "y", "z"):
            geom = self.grid.interface_face_data["primal"][direction]
            classified = self.conformal_face_data["primal"][direction]
            shape = geom["resolved"].shape
            g = np.zeros(shape, dtype=float)
            sig = np.zeros(shape, dtype=float)
            mu_wall = np.zeros(shape, dtype=float)
            mu_field = np.ones(shape, dtype=float)
            axis = {"x": 0, "y": 1, "z": 2}[direction]

            # A recorded SIBC segment must not be quietly dropped when its
            # polygon classification is unresolved or not normal <-> SIBC.
            for index, segment in geom["segments"].items():
                region_id = int(segment["surface_id"])
                if self.region_material_types[region_id] != "sibc":
                    continue
                if not classified["normal_sibc"][index] or not geom["resolved"][index]:
                    raise RuntimeError(
                        f"Unresolved/unsupported SIBC face: primal {direction} {index}"
                    )
                field_ids = classified["normal_region_ids"][index]
                wall_ids = classified["conductor_region_ids"][index]
                if (
                    len(field_ids) != 1
                    or len(wall_ids) != 1
                    or wall_ids[0] != region_id
                ):
                    raise RuntimeError(f"Ambiguous SIBC interface: {direction} {index}")
                area = float(geom["region_areas"][index][field_ids[0]])
                normal = np.asarray(segment["normal"], dtype=float)
                length = float(segment["length"])
                if normal.shape != (3,) or not np.all(np.isfinite(normal)):
                    raise RuntimeError(f"Missing SIBC normal: {direction} {index}")
                n_norm = np.linalg.norm(normal)
                if area <= 0 or length <= 0 or n_norm <= 0:
                    raise RuntimeError(f"Invalid SIBC geometry: {direction} {index}")
                normal /= n_norm
                sin_theta = np.sqrt(max(0.0, 1.0 - normal[axis] ** 2))
                if sin_theta <= 1e-10:
                    raise RuntimeError(f"Degenerate SIBC angle: {direction} {index}")
                wall_material = self._get_region_material_properties(region_id)
                field_material = self._get_region_material_properties(field_ids[0])
                if (
                    wall_material[2] <= 0
                    or wall_material[1] <= 0
                    or field_material[1] <= 0
                ):
                    raise ValueError(f"Invalid SIBC material: {direction} {index}")
                g[index] = length * sin_theta / area
                sig[index] = wall_material[2]
                mu_wall[index] = wall_material[1]
                mu_field[index] = field_material[1]

            # Shift from primal face index to stored H/magnetic-flux DOF.
            offset = axis * self.N
            target = slice(offset, offset + self.N)
            self.sibc_g[target] = self._primal_face_array_to_dual_edge_component(
                g, direction
            ).ravel(order="C")
            self.sibc_sigma[target] = self._primal_face_array_to_dual_edge_component(
                sig, direction
            ).ravel(order="C")
            self.sibc_mu_wall[target] = self._primal_face_array_to_dual_edge_component(
                mu_wall, direction
            ).ravel(order="C")
            self.sibc_mu_field[target] = self._primal_face_array_to_dual_edge_component(
                mu_field, direction
            ).ravel(order="C")

            # The stored magnetic space omits low-side primal faces. Reject
            # SIBC contributions there rather than ignoring them silently.
            low = [slice(None)] * 3
            low[axis] = 0
            if np.any(g[tuple(low)] != 0):
                raise NotImplementedError(
                    f"SIBC on low-side {direction} faces is outside the stored magnetic DOF space"
                )

        self.has_sibc = bool(np.any(self.sibc_g > 0))

    def _build_sibc_diagonal(self, omega):
        """Diagonal D_Zs for exp(+j omega t), good-conductor Leontovich SIBC."""
        D = np.ones(3 * self.N, dtype=np.complex128)
        if not self.has_sibc:
            return D
        from scipy.constants import mu_0

        active = self.sibc_g > 0
        z_s = np.sqrt(
            1j * omega * mu_0 * self.sibc_mu_wall[active] / self.sibc_sigma[active]
        )
        D[active] += (
            self.sibc_g[active] * z_s / (1j * omega * mu_0 * self.sibc_mu_field[active])
        )
        if not np.all(np.isfinite(D)) or np.any(np.abs(D) == 0):
            raise RuntimeError("Non-finite or zero D_Zs diagonal")
        return D

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

        if self.has_sibc:
            D = self._build_sibc_diagonal(omega)
            mnu_eff = self.Mnu.diagonal() / D
            stiffness = (self.C_red.transpose() @ diags(mnu_eff) @ self.C_red).tocsr()
        else:
            stiffness = self.SL

        return (
            self.Msigma + 1j * omega * self.Meps + 1.0 / (1j * omega) * stiffness
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

    def solve_eigenmodes(
        self,
        n_modes=6,
        target_frequency=None,
    ):
        """
        Solve the lossless generalized electric-field eigenvalue problem.

        The source-free frequency-domain FIT system is

            SL @ e = omega**2 * Meps @ e,

        where ``e`` is the reduced electric grid-voltage vector.

        Parameters
        ----------
        n_modes : int, optional
            Number of eigenmodes to compute.

        target_frequency : float, optional
            Target frequency in Hz. If provided, shift-invert is used to
            find the eigenmodes closest to this frequency.

            If omitted, the smallest-magnitude eigenvalues are requested.
            This may return curl-curl nullspace modes unless the nullspace
            has been removed separately.

        Returns
        -------
        frequencies : ndarray
            Eigenfrequencies in Hz, sorted in ascending order.

        modes : ndarray
            Reduced electric grid-voltage eigenvectors. Shape is
            ``(n_active_E_dofs, n_modes)``.

        eigenvalues : ndarray
            Eigenvalues ``omega**2``.
        """

        if self.has_sibc:
            raise NotImplementedError(
                "Frequency-dependent SIBC is not supported by the lossless eigenmode solver."
            )

        # ----------------------------------------------------------
        # Lossless formulation only
        # ----------------------------------------------------------

        if self.Msigma.nnz != 0:
            if np.any(np.abs(self.Msigma.diagonal()) > 0.0):
                raise NotImplementedError(
                    "The current eigenmode solver supports only "
                    "lossless systems with Msigma = 0."
                )

        # ----------------------------------------------------------
        # Generalized eigenvalue problem
        #
        #     SL e = lambda Meps e
        #
        # with
        #
        #     lambda = omega^2
        # ----------------------------------------------------------

        if target_frequency is not None:
            target_frequency = float(target_frequency)

            if not np.isfinite(target_frequency) or target_frequency <= 0.0:
                raise ValueError(
                    "target_frequency must be finite and greater than zero."
                )

            omega_target = 2.0 * np.pi * target_frequency

            sigma = omega_target**2

            eigenvalues, modes = eigsh(
                self.SL,
                k=n_modes,
                M=self.Meps,
                sigma=sigma,
                which="LM",
            )

        else:
            eigenvalues, modes = eigsh(
                self.SL,
                k=n_modes,
                M=self.Meps,
                which="SM",
            )

        # ----------------------------------------------------------
        # Sort
        # ----------------------------------------------------------

        order = np.argsort(eigenvalues)

        eigenvalues = eigenvalues[order]
        modes = modes[:, order]

        # ----------------------------------------------------------
        # Convert omega^2 -> frequency
        # ----------------------------------------------------------

        positive = eigenvalues > 0.0

        frequencies = np.full(
            eigenvalues.shape,
            np.nan,
            dtype=float,
        )

        frequencies[positive] = np.sqrt(eigenvalues[positive]) / (2.0 * np.pi)

        return (
            frequencies,
            modes,
            eigenvalues,
        )
