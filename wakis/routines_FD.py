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
