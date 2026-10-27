import time

import numpy as np
from scipy.constants import c as c_light
from tqdm import tqdm

from .field import Field


class RoutinesFDMixin:
    def impedance_sweep(
        self,
        frequencies,
        wake=None,
        field_monitor=None,
        current=1.0,
    ):
        """
        Compute the longitudinal beam-coupling impedance directly
        in the frequency domain.

        Parameters
        ----------
        frequencies : array_like
            Frequencies [Hz] at which the longitudinal impedance
            is evaluated.

        wake : WakeSolver, optional
            WakeSolver object providing beta, source position and
            test position. If omitted, self.wake is used.

        field_monitor : FieldMonitor, optional
            Frequency-domain field monitor. Its configured
            frequencies are solved in addition to the impedance
            sweep frequencies and the corresponding complex
            electric field distributions are stored.

        current : float, optional
            Beam current amplitude [A]. Default is 1 A.

        Returns
        -------
        frequencies : ndarray
            Requested impedance frequencies [Hz].

        Z_long : ndarray
            Longitudinal impedance [Ohm].
        """

        # ----------------------------------------------------------
        # Input
        # ----------------------------------------------------------

        frequencies = np.asarray(
            frequencies,
            dtype=float,
        )

        if frequencies.ndim != 1:
            raise ValueError("frequencies must be a one-dimensional array.")

        if len(frequencies) == 0:
            raise ValueError("frequencies must not be empty.")

        if np.any(~np.isfinite(frequencies)):
            raise ValueError("frequencies contains non-finite values.")

        if np.any(frequencies <= 0.0):
            raise ValueError("All frequencies must be greater than zero.")

        if current == 0.0:
            raise ValueError("current must be non-zero.")

        # ----------------------------------------------------------
        # Wake / beam parameters
        # ----------------------------------------------------------

        if wake is not None:
            self.wake = wake

        if self.wake is None:
            raise AttributeError(
                "Wake solver information not passed to SolverFIT3D_FD."
            )

        beta = self.wake.beta

        if beta <= 0.0:
            raise ValueError("wake.beta must be greater than zero.")

        velocity = beta * c_light

        xs = self.wake.xsource
        ys = self.wake.ysource

        xt = self.wake.xtest
        yt = self.wake.ytest

        ixs = np.abs(self.x - xs).argmin()

        iys = np.abs(self.y - ys).argmin()

        ixt = np.abs(self.x - xt).argmin()

        iyt = np.abs(self.y - yt).argmin()

        # ----------------------------------------------------------
        # Frequencies to solve
        # ----------------------------------------------------------

        if field_monitor is not None:
            monitor_frequencies = np.asarray(
                field_monitor.frequencies,
                dtype=float,
            )

            solve_frequencies = np.unique(
                np.concatenate(
                    (
                        frequencies,
                        monitor_frequencies,
                    )
                )
            )

        else:
            monitor_frequencies = np.array(
                [],
                dtype=float,
            )

            solve_frequencies = np.unique(frequencies)

        # Map the requested impedance frequencies back to the
        # original user-defined ordering.
        sweep_indices = {float(f): i for i, f in enumerate(frequencies)}

        # ----------------------------------------------------------
        # Source and monitor fields
        # ----------------------------------------------------------

        source = Field(
            self.Nx,
            self.Ny,
            self.Nz,
            dtype=np.complex128,
        )

        E_monitor = None

        if field_monitor is not None:
            E_monitor = Field(
                self.Nx,
                self.Ny,
                self.Nz,
                dtype=np.complex128,
            )

        zero_source = np.zeros(
            3 * self.N,
            dtype=np.complex128,
        )

        z_path = self.z

        # ----------------------------------------------------------
        # Check beam trajectory once
        # ----------------------------------------------------------

        source.fromarray(zero_source.copy())

        source[
            ixs,
            iys,
            :,
            "z",
        ] = 1.0

        beam_source_idx = np.flatnonzero(np.abs(source.toarray()) > 0.0)

        active_lookup = np.zeros(
            3 * self.N,
            dtype=bool,
        )

        active_lookup[self.free_E_idx] = True

        inactive_source_idx = beam_source_idx[~active_lookup[beam_source_idx]]

        if inactive_source_idx.size:
            raise RuntimeError(
                f"{inactive_source_idx.size} beam-source "
                "DOFs are inactive. Check source trajectory "
                "and PEC geometry."
            )

        # ----------------------------------------------------------
        # Result arrays
        # ----------------------------------------------------------

        Z_long = np.zeros(
            len(frequencies),
            dtype=np.complex128,
        )

        Ds_diag = self.Ds.diagonal()

        # ----------------------------------------------------------
        # Frequency sweep
        # ----------------------------------------------------------

        t0 = time.time()

        print("Running electromagnetic frequency-domain sweep...")

        for f in tqdm(solve_frequencies):
            omega = 2.0 * np.pi * f

            k = omega / velocity

            source.fromarray(zero_source.copy())

            # Beam travelling in +z with exp(+j omega t)
            source[
                ixs,
                iys,
                :,
                "z",
            ] = current * np.exp(-1j * k * z_path)

            # ------------------------------------------------------
            # Solve for electric grid voltages e = Ds E
            # ------------------------------------------------------

            e = self.solve(
                omega,
                source.toarray(),
            )

            # ------------------------------------------------------
            # Longitudinal impedance
            # ------------------------------------------------------

            e_field = Field(
                self.Nx,
                self.Ny,
                self.Nz,
                dtype=np.complex128,
            )

            if float(f) in sweep_indices:
                test_phase = np.exp(1j * k * z_path)

                e_field.fromarray(e)

                Z_long[sweep_indices[float(f)]] = (
                    -e_field[
                        ixt,
                        iyt,
                        :,
                        "z",
                    ]
                    @ test_phase
                    / current
                )

            # ------------------------------------------------------
            # Field monitor
            # ------------------------------------------------------

            if field_monitor is not None and np.any(
                np.isclose(
                    monitor_frequencies,
                    f,
                    rtol=1e-12,
                    atol=0.0,
                )
            ):
                # Convert grid voltage e [V]
                # back to physical E field [V/m].
                E_array = np.divide(
                    e,
                    Ds_diag,
                    out=np.zeros_like(e),
                    where=Ds_diag != 0.0,
                )

                E_monitor.fromarray(E_array)

                field_monitor.store(
                    E_monitor,
                    f,
                )

        # ----------------------------------------------------------
        # Store results
        # ----------------------------------------------------------

        self.f = frequencies.copy()
        self.Z_long = Z_long.copy()

        self.wake.f = frequencies.copy()
        self.wake.Z = Z_long.copy()

        self.fd_simulation_time = time.time() - t0

        return (
            frequencies,
            Z_long,
        )

    def eigenmode_longitudinal_impedance(
        self,
        frequencies,
        eigenfrequencies,
        eigenvectors,
        wake=None,
        skip_cells=0,
    ):
        """
        Compute the longitudinal impedance from lossless electric eigenmodes.

        The modal expansion is based on

            SL @ e_n = omega_n**2 * Meps @ e_n

        and the phasor convention exp(+j*omega*t).

        Parameters
        ----------
        frequencies : array_like
            Frequencies [Hz] at which the impedance is evaluated.

        eigenfrequencies : array_like
            Eigenfrequencies [Hz].

        eigenvectors : ndarray
            Reduced electric grid-voltage eigenvectors with shape

                (n_active_E_dofs, n_modes).

        wake : WakeSolver, optional
            WakeSolver object providing beta, source position and
            test position. If omitted, ``self.wake`` is used.

        skip_cells : int, optional
            Number of cells excluded at each end of the longitudinal
            integration path. Default is zero.

        Returns
        -------
        Z_long : ndarray
            Total longitudinal impedance [Ohm].

        Z_modes : ndarray
            Individual modal impedance contributions with shape

                (n_modes, n_frequencies).
        """

        frequencies = np.asarray(
            frequencies,
            dtype=float,
        )

        eigenfrequencies = np.asarray(
            eigenfrequencies,
            dtype=float,
        )

        eigenvectors = np.asarray(
            eigenvectors,
        )

        if frequencies.ndim != 1:
            raise ValueError("frequencies must be one-dimensional.")

        if eigenfrequencies.ndim != 1:
            raise ValueError("eigenfrequencies must be one-dimensional.")

        if eigenvectors.ndim != 2:
            raise ValueError("eigenvectors must be a two-dimensional array.")

        if eigenvectors.shape[0] != len(self.free_E_idx):
            raise ValueError(
                "First eigenvector dimension must equal the "
                "number of active electric DOFs."
            )

        if eigenvectors.shape[1] != len(eigenfrequencies):
            raise ValueError("Number of eigenvectors and eigenfrequencies must agree.")

        if np.any(frequencies <= 0.0):
            raise ValueError("All evaluation frequencies must be positive.")

        if np.any(eigenfrequencies <= 0.0):
            raise ValueError("All supplied eigenfrequencies must be positive.")

        # ----------------------------------------------------------
        # Wake / beam parameters
        # ----------------------------------------------------------

        if wake is not None:
            self.wake = wake

        if self.wake is None:
            raise AttributeError("Wake solver information not provided.")

        beta = self.wake.beta

        if beta <= 0.0:
            raise ValueError("wake.beta must be greater than zero.")

        velocity = beta * c_light

        ixs = np.abs(self.x - self.wake.xsource).argmin()

        iys = np.abs(self.y - self.wake.ysource).argmin()

        ixt = np.abs(self.x - self.wake.xtest).argmin()

        iyt = np.abs(self.y - self.wake.ytest).argmin()

        # ----------------------------------------------------------
        # Longitudinal integration path
        # ----------------------------------------------------------

        if skip_cells < 0:
            raise ValueError("skip_cells must be non-negative.")

        if 2 * skip_cells >= self.Nz:
            raise ValueError("skip_cells removes the complete z path.")

        if skip_cells == 0:
            z_slice = slice(None)
        else:
            z_slice = slice(
                skip_cells,
                -skip_cells,
            )

        z_path = self.z[z_slice]

        # ----------------------------------------------------------
        # Reconstruct longitudinal grid voltages of each mode
        # ----------------------------------------------------------

        n_modes = len(eigenfrequencies)

        mode_source = np.zeros(
            (
                n_modes,
                len(z_path),
            ),
            dtype=np.complex128,
        )

        mode_test = np.zeros_like(mode_source)

        mode_field = Field(
            self.Nx,
            self.Ny,
            self.Nz,
            dtype=np.complex128,
        )

        full_mode = np.zeros(
            3 * self.N,
            dtype=np.complex128,
        )

        for n in range(n_modes):
            full_mode.fill(0.0)

            full_mode[self.free_E_idx] = eigenvectors[:, n]

            mode_field.fromarray(full_mode)

            mode_source[n] = mode_field[
                ixs,
                iys,
                z_slice,
                "z",
            ]

            mode_test[n] = mode_field[
                ixt,
                iyt,
                z_slice,
                "z",
            ]

        # ----------------------------------------------------------
        # Modal electric coefficients
        #
        # C_n = e_n^H Meps e_n
        #
        # This makes the result independent of eigenvector
        # normalization.
        # ----------------------------------------------------------

        Cn = np.zeros(
            n_modes,
            dtype=float,
        )

        for n in range(n_modes):
            mode = eigenvectors[
                :,
                n,
            ]

            Cn[n] = np.real(
                np.vdot(
                    mode,
                    self.Meps @ mode,
                )
            )

        if np.any(Cn <= 0.0):
            raise RuntimeError("Non-positive modal electric coefficient found.")

        # ----------------------------------------------------------
        # Beam coupling
        # ----------------------------------------------------------

        omega = 2.0 * np.pi * frequencies

        k = omega / velocity

        phase_plus = np.exp(
            1j
            * np.outer(
                k,
                z_path,
            )
        )

        phase_minus = np.exp(
            -1j
            * np.outer(
                k,
                z_path,
            )
        )

        # Test voltage:
        #
        #   T_n = sum e_n exp(+j k z)
        #
        test_coupling = mode_test @ phase_plus.T

        # Source overlap:
        #
        #   S_n = sum e_n^* exp(-j k z)
        #
        source_coupling = np.conjugate(mode_source) @ phase_minus.T

        # ----------------------------------------------------------
        # Modal impedance
        # ----------------------------------------------------------

        omega_n = 2.0 * np.pi * eigenfrequencies

        denominator = Cn[:, None] * (omega_n[:, None] ** 2 - omega[None, :] ** 2)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            Z_modes = (
                1j * omega[None, :] * test_coupling * source_coupling / denominator
            )

        Z_long = np.sum(
            Z_modes,
            axis=0,
        )

        return (
            Z_long,
            Z_modes,
        )
