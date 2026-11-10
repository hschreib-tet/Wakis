# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Conformal FIT geometry and material handling.

The routines in this module resolve how STL surfaces intersect the
staggered FIT grid and turn those intersections into conformal line and
face metrics and material averages. They are attached to the main classes
as mixins:

- ``ConformalGridMixin`` is mixed into ``GridFIT3D`` and builds the region
  maps, edge/face intersections, PEC snapping and partial face areas.
- ``ConformalSolverMixin`` is mixed into ``SolverFIT3D`` and turns the grid
  data into effective lengths/areas and conformal material tensors.

Both are only used when ``geometry_mode="conformal"``. The entry points
are ``GridFIT3D._mark_cells_in_stl_conformal`` and
``SolverFIT3D._apply_stl_materials_conformal``, called from the
``_mark_cells_in_stl`` / ``_apply_stl_materials`` dispatchers.
"""

import warnings

import numpy as np
import pyvista as pv
import vtk
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0

from .field import Field
from .materials import material_lib


class ConformalGridMixin:
    """Conformal STL intersection handling for ``GridFIT3D``."""

    def _get_stabilization_fraction(self):
        """Fraction of the active small-cut-cell treatment (None if off)."""
        if self.conformal_stabilization == "snapping":
            return self.conformal_pec_snap_fraction
        if self.conformal_stabilization == "area_floor":
            return self.conformal_area_floor
        return None

    def _set_conformal_stabilization(
        self,
        method,
        fraction,
        min_edge_fraction,
        legacy_snap_fraction,
    ):
        """
        Validate and store the small-cut-cell treatment.

        Sets ``conformal_stabilization`` ("none", "snapping" or
        "area_floor"), ``conformal_pec_snap_fraction`` (used by snapping),
        ``conformal_area_floor`` and ``conformal_min_edge_fraction`` (used
        by the solver for the area floor). Unused values are 0.
        """
        if legacy_snap_fraction is not None and float(legacy_snap_fraction) > 0.0:
            warnings.warn(
                "conformal_pec_snap_fraction is deprecated; use "
                "conformal_stabilization='snapping' and "
                "conformal_stabilization_fraction instead.",
                DeprecationWarning,
                stacklevel=3,
            )

            if method not in (None, "snapping"):
                raise ValueError(
                    "conformal_pec_snap_fraction cannot be combined with "
                    f"conformal_stabilization={method!r}."
                )

            if fraction is not None and float(fraction) != float(legacy_snap_fraction):
                raise ValueError(
                    "Conflicting conformal_pec_snap_fraction and "
                    "conformal_stabilization_fraction."
                )

            method = "snapping"
            fraction = float(legacy_snap_fraction)

        method = "none" if method is None else str(method).lower()

        if method not in ("none", "snapping", "area_floor"):
            raise ValueError(
                "conformal_stabilization must be None, 'snapping' or 'area_floor'."
            )

        fraction = 0.1 if fraction is None else float(fraction)
        min_edge_fraction = (
            min(fraction, 0.49)
            if min_edge_fraction is None
            else float(min_edge_fraction)
        )

        self.conformal_stabilization = method
        self.conformal_pec_snap_fraction = 0.0
        self.conformal_area_floor = 0.0
        self.conformal_min_edge_fraction = 0.0

        if method == "snapping":
            if not (0.0 < fraction < 0.5):
                raise ValueError(
                    "Snapping requires 0 < conformal_stabilization_fraction < 0.5."
                )

            if self.geometry_mode == "conformal" and self.bg_material_type is None:
                raise ValueError(
                    "Conformal PEC snapping requires "
                    "bg_material_type to be specified "
                    "in GridFIT3D."
                )

            self.conformal_pec_snap_fraction = fraction

        elif method == "area_floor":
            if not (0.0 < fraction <= 1.0):
                raise ValueError(
                    "Area floor requires 0 < conformal_stabilization_fraction <= 1."
                )

            if not (0.0 <= min_edge_fraction < 0.5):
                raise ValueError(
                    "conformal_min_edge_fraction must satisfy 0 <= value < 0.5."
                )

            self.conformal_area_floor = fraction
            self.conformal_min_edge_fraction = min_edge_fraction

    def _get_grid_coordinates(self, grid_type):
        """
        Return the coordinates of the primal or dual grid.

        Parameters
        ----------
        grid_type : str
            Either "primal" or "dual".
        """
        if grid_type == "primal":
            return self.x, self.y, self.z

        if grid_type == "dual":
            return self.tx, self.ty, self.tz

        raise ValueError(
            f"Unknown grid type '{grid_type}'. Expected 'primal' or 'dual'."
        )

    def _point_mask_to_cell_mask(self, point_mask, threshold=0.5):
        """
        Derive a cell mask from a boolean mask on primal grid points.

        The cell centers correspond to the interior dual grid points.
        The cell classification is obtained from the eight primal
        corner-point classifications using a majority rule.

        With threshold=0.5, at least five of the eight primal corner
        points must lie inside the solid.

        In conformal mode, this mask is stored in ``self.grid[key]``.
        It is also used as a backward-compatibility fallback to reconstruct
        dual point masks from older HDF5 files.

        Parameters
        ----------
        point_mask : ndarray of bool
            Boolean mask with shape (Nx+1, Ny+1, Nz+1).
        threshold : float, optional
            Occupancy threshold used for the cell classification.

        Returns
        -------
        cell_mask : ndarray of bool
            Boolean mask with shape (Nx, Ny, Nz).
        """
        point_mask = np.asarray(point_mask, dtype=bool)

        expected_shape = (self.Nx + 1, self.Ny + 1, self.Nz + 1)
        if point_mask.shape != expected_shape:
            raise ValueError(
                f"Expected primal point mask shape {expected_shape}, "
                f"got {point_mask.shape}."
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        # Count how many of the eight corner points lie inside the STL.
        # uint8 is sufficient (maximum value is 8) and avoids a large
        # temporary floating-point array.
        count = point_mask[:-1, :-1, :-1].astype(np.uint8)

        count += point_mask[1:, :-1, :-1]
        count += point_mask[:-1, 1:, :-1]
        count += point_mask[:-1, :-1, 1:]
        count += point_mask[1:, 1:, :-1]
        count += point_mask[1:, :-1, 1:]
        count += point_mask[:-1, 1:, 1:]
        count += point_mask[1:, 1:, 1:]

        return count > 8.0 * threshold

    def _cell_mask_to_dual_point_mask(self, cell_mask):
        """
        Reconstruct a dual-point mask from a primal cell mask.

        This is a backward-compatibility helper for HDF5 files written before
        direct dual-grid point classification was introduced. The current
        conformal geometry pipeline classifies dual grid points directly.

        Interior dual grid points coincide with primal cell centers. Values on
        the additional high-side boundary planes are extended from the adjacent
        cell-center values.

        Parameters
        ----------
        cell_mask : ndarray of bool
            Boolean cell mask with shape (Nx, Ny, Nz).

        Returns
        -------
        dual_point_mask : ndarray of bool
            Boolean mask with shape (Nx+1, Ny+1, Nz+1).
        """
        cell_mask = np.asarray(cell_mask, dtype=bool)

        expected_shape = (self.Nx, self.Ny, self.Nz)
        if cell_mask.shape != expected_shape:
            raise ValueError(
                f"Expected cell mask shape {expected_shape}, got {cell_mask.shape}."
            )

        return np.pad(
            cell_mask,
            ((0, 1), (0, 1), (0, 1)),
            mode="edge",
        )

    def _classify_coordinate_block_in_stl(
        self,
        surf,
        x,
        y,
        z,
        method,
        stl_tolerance,
        progress_bar=False,
    ):
        """
        Classify a Cartesian block of points with respect to an STL surface.

        Parameters
        ----------
        surf : pyvista.PolyData
            Closed STL surface.
        x, y, z : array_like
            Coordinates of the Cartesian point block.
        method : str
            STL classification method.
        stl_tolerance : float
            Tolerance used by the point-in-surface classifier.
        progress_bar : bool, optional
            Show PyVista progress information.

        Returns
        -------
        ndarray of bool
            Boolean mask with shape (len(x), len(y), len(z)).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        shape = (
            len(x),
            len(y),
            len(z),
        )

        method = method.lower()

        # ----------------------------------------------------------
        # interior points
        # ----------------------------------------------------------
        if method in (
            "legacy",
            "interior_points",
        ):
            xx, yy, zz = np.meshgrid(
                x,
                y,
                z,
                indexing="ij",
            )

            points = np.column_stack(
                (
                    xx.ravel(order="C"),
                    yy.ravel(order="C"),
                    zz.ravel(order="C"),
                )
            )

            point_cloud = pv.PolyData(points)

            try:
                selected = point_cloud.select_interior_points(
                    surf,
                    method="cell_locator",
                    locator_tolerance=stl_tolerance,
                )

            except Exception:
                selected = point_cloud.select_interior_points(
                    surf,
                    method="cell_locator",
                    locator_tolerance=stl_tolerance,
                    check_surface=False,
                )

            return np.reshape(
                np.asarray(
                    selected.point_data["selected_points"],
                    dtype=bool,
                ),
                shape,
                order="C",
            )

        # ----------------------------------------------------------
        # implicit distance
        # ----------------------------------------------------------
        elif method == "implicit_distance":
            xx, yy, zz = np.meshgrid(
                x,
                y,
                z,
                indexing="ij",
            )

            points = np.column_stack(
                (
                    xx.ravel(order="C"),
                    yy.ravel(order="C"),
                    zz.ravel(order="C"),
                )
            )

            point_cloud = pv.PolyData(points)

            selected = point_cloud.compute_implicit_distance(surf)

            return np.reshape(
                np.asarray(selected.point_data["implicit_distance"]) <= 0.0,
                shape,
                order="C",
            )

        # ----------------------------------------------------------
        # voxelize rectilinear
        # ----------------------------------------------------------
        elif method == "voxelize_rectilinear":
            coordinate_arrays = (
                x,
                y,
                z,
            )

            fallback_spacings = (
                np.min(self.dx),
                np.min(self.dy),
                np.min(self.dz),
            )

            spacings = []

            for coordinates, fallback in zip(
                coordinate_arrays,
                fallback_spacings,
            ):
                if len(coordinates) > 1:
                    differences = np.diff(coordinates)

                    if not np.allclose(
                        differences,
                        differences[0],
                    ):
                        raise ValueError(
                            "voxelize_rectilinear currently "
                            "requires uniformly spaced "
                            "classification coordinates."
                        )

                    spacings.append(float(differences[0]))

                else:
                    spacings.append(float(fallback))

            reference_volume = pv.ImageData(
                dimensions=shape,
                origin=(
                    x[0],
                    y[0],
                    z[0],
                ),
                spacing=tuple(spacings),
            )

            vox = surf.voxelize_rectilinear(
                reference_volume=reference_volume,
                progress_bar=progress_bar,
            )

            voxel_mask = np.asarray(
                vox.cell_data["mask"],
                dtype=bool,
            )

            expected_size = int(np.prod(shape))

            if voxel_mask.size != expected_size:
                raise RuntimeError(
                    "Unexpected voxelize_rectilinear "
                    f"mask size {voxel_mask.size}; "
                    f"expected {expected_size}."
                )

            return np.reshape(
                voxel_mask,
                shape,
                order="F",
            )

        else:
            raise ValueError(
                f"[!] Error: stl_method "
                f"{method} not recognized. "
                "Use 'legacy', "
                "'interior_points', "
                "'implicit_distance', or "
                "'voxelize_rectilinear'."
            )

    def _classify_grid_points_in_stl(
        self,
        surf,
        grid_type,
        method,
        stl_tolerance,
        progress_bar=False,
    ):
        """
        Classify primal or dual FIT grid points with respect to an STL.

        For the primal grid, all points are classified using the selected
        STL method.

        For the dual grid, the regular inner dual grid is classified using
        the selected STL method. Points on the high-side boundary planes are
        classified separately using interior_points.
        """
        x, y, z = self._get_grid_coordinates(grid_type)

        if grid_type == "primal":
            return self._classify_coordinate_block_in_stl(
                surf,
                x,
                y,
                z,
                method,
                stl_tolerance,
                progress_bar=progress_bar,
            )

        if grid_type != "dual":
            raise ValueError(f"Unknown grid type: {grid_type}")

        # ----------------------------------------------------------
        # Regular inner dual grid
        # ----------------------------------------------------------
        point_mask = np.zeros(
            (
                len(x),
                len(y),
                len(z),
            ),
            dtype=bool,
        )

        point_mask[
            :-1,
            :-1,
            :-1,
        ] = self._classify_coordinate_block_in_stl(
            surf,
            x[:-1],
            y[:-1],
            z[:-1],
            method,
            stl_tolerance,
            progress_bar=progress_bar,
        )

        # ----------------------------------------------------------
        # High-side x plane
        #
        # Includes the y/z high-side edges and the high-side corner.
        # ----------------------------------------------------------
        point_mask[
            -1,
            :,
            :,
        ] = self._classify_coordinate_block_in_stl(
            surf,
            x[-1:],
            y,
            z,
            "interior_points",
            stl_tolerance,
            progress_bar=False,
        )[0, :, :]

        # ----------------------------------------------------------
        # High-side y plane
        #
        # x-high points have already been classified above.
        # ----------------------------------------------------------
        point_mask[
            :-1,
            -1,
            :,
        ] = self._classify_coordinate_block_in_stl(
            surf,
            x[:-1],
            y[-1:],
            z,
            "interior_points",
            stl_tolerance,
            progress_bar=False,
        )[:, 0, :]

        # ----------------------------------------------------------
        # High-side z plane
        #
        # x-high and y-high points have already been classified.
        # ----------------------------------------------------------
        point_mask[
            :-1,
            :-1,
            -1,
        ] = self._classify_coordinate_block_in_stl(
            surf,
            x[:-1],
            y[:-1],
            z[-1:],
            "interior_points",
            stl_tolerance,
            progress_bar=False,
        )[:, :, 0]

        return point_mask

    def _store_stl_masks(
        self,
        key,
        primal_point_mask,
        dual_point_mask,
        cell_threshold=0.5,
    ):
        """
        Store primal-point, dual-point, and cell masks for one STL solid.

        The primal and dual point masks are obtained independently from
        geometric STL classification. The cell mask stored in
        ``self.grid[key]`` is derived from the eight primal corner-point
        classifications.

        Parameters
        ----------
        key : str
            STL solid key.
        primal_point_mask : ndarray of bool
            Boolean mask on primal grid points with shape
            (Nx+1, Ny+1, Nz+1).
        dual_point_mask : ndarray of bool
            Boolean mask on dual grid points with shape
            (Nx+1, Ny+1, Nz+1).
        cell_threshold : float, optional
            Threshold used to classify the cell center from the eight
            primal corner-point values. With the default value 0.5,
            at least five of eight corner points must lie inside the solid.
        """
        primal_point_mask = np.asarray(
            primal_point_mask,
            dtype=bool,
        )

        dual_point_mask = np.asarray(
            dual_point_mask,
            dtype=bool,
        )

        cell_center_mask = self._point_mask_to_cell_mask(
            primal_point_mask,
            threshold=cell_threshold,
        )

        expected_primal_shape = (
            len(self.x),
            len(self.y),
            len(self.z),
        )

        expected_dual_shape = (
            len(self.tx),
            len(self.ty),
            len(self.tz),
        )

        if primal_point_mask.shape != expected_primal_shape:
            raise ValueError(
                "Expected primal point mask shape "
                f"{expected_primal_shape}, "
                f"got {primal_point_mask.shape}."
            )

        if dual_point_mask.shape != expected_dual_shape:
            raise ValueError(
                "Expected dual point mask shape "
                f"{expected_dual_shape}, "
                f"got {dual_point_mask.shape}."
            )
        self.primal_point_masks[key] = primal_point_mask
        self.dual_point_masks[key] = dual_point_mask

        self.grid[key] = np.reshape(
            cell_center_mask,
            self.Nx * self.Ny * self.Nz,
            order="C",
        )

    def _build_region_maps(self):
        """
        Assign a unique region ID to the background and every STL solid.

        Region 0 is reserved for the background. STL solids are assigned
        IDs 1, 2, ... following the order of ``self.stl_solids``.
        """
        self.region_id_to_key = {0: None}
        self.region_key_to_id = {}

        for region_id, key in enumerate(
            self.stl_solids.keys(),
            start=1,
        ):
            self.region_id_to_key[region_id] = key
            self.region_key_to_id[key] = region_id

    def _build_region_ids(self, point_masks, grid_type):
        """
        Build unique region IDs from point masks on one grid.

        Later STL solids intentionally overwrite earlier solids in
        overlapping regions.

        Parameters
        ----------
        point_masks : dict
            Boolean point masks indexed by STL key.
        grid_type : str
            Either "primal" or "dual".

        Returns
        -------
        region_ids : ndarray of int
            Unique region ID at every grid point.
        """
        x, y, z = self._get_grid_coordinates(grid_type)

        expected_shape = (
            len(x),
            len(y),
            len(z),
        )

        region_ids = np.zeros(
            expected_shape,
            dtype=np.int32,
        )

        for key in self.stl_solids.keys():
            if key not in point_masks:
                raise RuntimeError(
                    f"No {grid_type} point mask available for STL solid '{key}'."
                )

            point_mask = np.asarray(
                point_masks[key],
                dtype=bool,
            )

            if point_mask.shape != expected_shape:
                raise ValueError(
                    f"Expected {grid_type} point mask shape "
                    f"{expected_shape} for STL solid '{key}', "
                    f"got {point_mask.shape}."
                )

            region_id = self.region_key_to_id[key]

            # Later STL solids overwrite earlier ones.
            region_ids[point_mask] = region_id

        return region_ids

    @staticmethod
    def _build_edge_transition_masks(region_ids):
        """
        Detect edges whose endpoints belong to different regions.

        Parameters
        ----------
        region_ids : ndarray
            Region IDs defined on grid points.

        Returns
        -------
        dict
            Boolean interface-edge masks for x, y, and z directions.
        """
        return {
            "x": (region_ids[:-1, :, :] != region_ids[1:, :, :]),
            "y": (region_ids[:, :-1, :] != region_ids[:, 1:, :]),
            "z": (region_ids[:, :, :-1] != region_ids[:, :, 1:]),
        }

    def _build_interface_topology(self):
        """
        Build region IDs and interface-edge masks on both grids.
        """
        self._build_region_maps()

        self.region_ids["primal"] = self._build_region_ids(
            self.primal_point_masks,
            "primal",
        )

        self.region_ids["dual"] = self._build_region_ids(
            self.dual_point_masks,
            "dual",
        )

        self.interface_edge_masks["primal"] = self._build_edge_transition_masks(
            self.region_ids["primal"]
        )

        self.interface_edge_masks["dual"] = self._build_edge_transition_masks(
            self.region_ids["dual"]
        )

    def _get_edge_region_ids(self, grid_type, direction):
        """
        Return start and end region IDs of all edges in one direction.

        Parameters
        ----------
        grid_type : str
            Either "primal" or "dual".
        direction : str
            Edge direction: "x", "y", or "z".
        """
        if grid_type not in self.region_ids:
            raise ValueError(
                f"Unknown grid type '{grid_type}'. Expected 'primal' or 'dual'."
            )

        region_ids = self.region_ids[grid_type]

        if region_ids is None:
            raise RuntimeError(
                f"{grid_type.capitalize()} region IDs have not been built."
            )

        if direction == "x":
            return (
                region_ids[:-1, :, :],
                region_ids[1:, :, :],
            )

        if direction == "y":
            return (
                region_ids[:, :-1, :],
                region_ids[:, 1:, :],
            )

        if direction == "z":
            return (
                region_ids[:, :, :-1],
                region_ids[:, :, 1:],
            )

        raise ValueError(
            f"Unknown edge direction '{direction}'. Expected 'x', 'y', or 'z'."
        )

    def _get_edge_geometry(self, grid_type, direction, index):
        """
        Return geometry and endpoint region IDs of one grid edge.

        Parameters
        ----------
        grid_type : str
            Either "primal" or "dual".
        direction : str
            Edge direction: "x", "y", or "z".
        index : tuple of int
            Edge index (i, j, k).

        Returns
        -------
        start_point, end_point : ndarray
            Cartesian edge endpoints.
        start_region, end_region : int
            Region IDs at the edge endpoints.
        """
        x, y, z = self._get_grid_coordinates(grid_type)
        region_ids = self.region_ids[grid_type]

        if region_ids is None:
            raise RuntimeError(
                f"{grid_type.capitalize()} region IDs have not been built."
            )

        i, j, k = (int(value) for value in index)

        start_point = np.array(
            [x[i], y[j], z[k]],
            dtype=float,
        )

        start_region = int(region_ids[i, j, k])

        if direction == "x":
            end_point = np.array(
                [x[i + 1], y[j], z[k]],
                dtype=float,
            )
            end_region = int(region_ids[i + 1, j, k])

        elif direction == "y":
            end_point = np.array(
                [x[i], y[j + 1], z[k]],
                dtype=float,
            )
            end_region = int(region_ids[i, j + 1, k])

        elif direction == "z":
            end_point = np.array(
                [x[i], y[j], z[k + 1]],
                dtype=float,
            )
            end_region = int(region_ids[i, j, k + 1])

        else:
            raise ValueError(
                f"Unknown edge direction '{direction}'. Expected 'x', 'y', or 'z'."
            )

        return (
            start_point,
            end_point,
            start_region,
            end_region,
        )

    def _locate_intersection_edge(
        self,
        grid_type,
        direction,
        point,
        alpha_tolerance=1e-8,
    ):
        """
        Map a geometric intersection point on a grid line to one grid edge
        and compute its local edge coordinate alpha in [0, 1].
        """
        x, y, z = self._get_grid_coordinates(grid_type)

        coordinates = {
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "z": np.asarray(z, dtype=float),
        }

        axis = {
            "x": 0,
            "y": 1,
            "z": 2,
        }[direction]

        coordinate = float(point[axis])
        grid_coordinates = coordinates[direction]

        edge_index = int(
            np.searchsorted(
                grid_coordinates,
                coordinate,
                side="right",
            )
            - 1
        )

        # A hit exactly on the last grid point belongs to the final edge.
        edge_index = min(
            edge_index,
            len(grid_coordinates) - 2,
        )

        if edge_index < 0:
            return None

        start_coordinate = grid_coordinates[edge_index]
        end_coordinate = grid_coordinates[edge_index + 1]

        edge_length = end_coordinate - start_coordinate

        if edge_length == 0.0:
            return None

        alpha = (coordinate - start_coordinate) / edge_length

        if alpha < -alpha_tolerance or alpha > 1.0 + alpha_tolerance:
            return None

        alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0,
            )
        )

        return edge_index, alpha

    @staticmethod
    def _get_triangle_normal(surface, cell_id):
        """
        Compute the unit normal of one triangular STL facet.
        """
        cell = surface.get_cell(int(cell_id))
        points = np.asarray(
            cell.points,
            dtype=float,
        )

        if points.shape != (3, 3):
            raise RuntimeError(
                f"Expected triangular STL cell, got "
                f"{points.shape[0]} points for cell {cell_id}."
            )

        normal = np.cross(
            points[1] - points[0],
            points[2] - points[0],
        )

        norm = np.linalg.norm(normal)

        if norm == 0.0:
            raise RuntimeError(
                f"Degenerate STL triangle encountered for cell {cell_id}."
            )

        return normal / norm

    def _trace_grid_line(
        self,
        grid_type,
        direction,
        transverse_index,
        surface_cache,
        alpha_tolerance=1e-8,
    ):
        """
        Trace one complete primal or dual grid line through all STL surfaces.

        Returns all geometric intersections, already assigned to their
        corresponding grid edges.
        """
        x, y, z = self._get_grid_coordinates(grid_type)

        axis_coordinates = {
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "z": np.asarray(z, dtype=float),
        }[direction]

        spacing = np.diff(axis_coordinates)

        if np.any(spacing <= 0.0):
            raise ValueError("Grid coordinates must be strictly increasing.")

        line_extension = np.min(spacing)

        if direction == "x":
            j, k = transverse_index

            start_point = np.array(
                [
                    x[0] - line_extension,
                    y[j],
                    z[k],
                ],
                dtype=float,
            )

            end_point = np.array(
                [
                    x[-1] + line_extension,
                    y[j],
                    z[k],
                ],
                dtype=float,
            )

        elif direction == "y":
            i, k = transverse_index

            start_point = np.array(
                [
                    x[i],
                    y[0] - line_extension,
                    z[k],
                ],
                dtype=float,
            )

            end_point = np.array(
                [
                    x[i],
                    y[-1] + line_extension,
                    z[k],
                ],
                dtype=float,
            )

        elif direction == "z":
            i, j = transverse_index

            start_point = np.array(
                [
                    x[i],
                    y[j],
                    z[0] - line_extension,
                ],
                dtype=float,
            )

            end_point = np.array(
                [
                    x[i],
                    y[j],
                    z[-1] + line_extension,
                ],
                dtype=float,
            )

        else:
            raise ValueError(f"Unknown edge direction '{direction}'.")

        edge_hits = {}

        for surface_id, key in self.region_id_to_key.items():
            if surface_id == 0:
                continue

            if key not in surface_cache:
                surface_cache[key] = self.read_stl(key)

            surface = surface_cache[key]

            # Cache one static locator per STL surface.
            # It can be reused for all grid lines.
            if not hasattr(self, "_conformal_stl_locators"):
                self._conformal_stl_locators = {}

            if key not in self._conformal_stl_locators:
                locator = vtk.vtkStaticCellLocator()
                locator.SetDataSet(surface)
                locator.BuildLocator()

                self._conformal_stl_locators[key] = locator

            locator = self._conformal_stl_locators[key]

            # Intersect the complete grid line with the STL.
            vtk_points = vtk.vtkPoints()
            vtk_cell_ids = vtk.vtkIdList()

            locator.IntersectWithLine(
                start_point,
                end_point,
                1e-8,
                vtk_points,
                vtk_cell_ids,
            )

            points = np.array(
                [vtk_points.GetPoint(n) for n in range(vtk_points.GetNumberOfPoints())],
                dtype=float,
            ).reshape(-1, 3)

            cell_ids = np.array(
                [vtk_cell_ids.GetId(n) for n in range(vtk_cell_ids.GetNumberOfIds())],
                dtype=int,
            )

            for point, cell_id in zip(
                np.asarray(points, dtype=float),
                np.asarray(cell_ids, dtype=int),
            ):
                result = self._locate_intersection_edge(
                    grid_type,
                    direction,
                    point,
                    alpha_tolerance=alpha_tolerance,
                )

                if result is None:
                    continue

                edge_axis_index, alpha = result

                if direction == "x":
                    edge_index = (
                        edge_axis_index,
                        int(transverse_index[0]),
                        int(transverse_index[1]),
                    )

                elif direction == "y":
                    edge_index = (
                        int(transverse_index[0]),
                        edge_axis_index,
                        int(transverse_index[1]),
                    )

                else:
                    edge_index = (
                        int(transverse_index[0]),
                        int(transverse_index[1]),
                        edge_axis_index,
                    )

                normal = self._get_triangle_normal(
                    surface,
                    cell_id,
                )

                hit = {
                    "alpha": alpha,
                    "surface_id": int(surface_id),
                    "cell_id": int(cell_id),
                    "normal": normal,
                }

                edge_hits.setdefault(
                    edge_index,
                    [],
                ).append(hit)

        return edge_hits

    def _get_edge_shape_and_transverse_indices(
        self,
        grid_type,
        direction,
    ):
        x, y, z = self._get_grid_coordinates(grid_type)

        nx = len(x)
        ny = len(y)
        nz = len(z)

        if direction == "x":
            shape = (
                nx - 1,
                ny,
                nz,
            )

            transverse_indices = ((j, k) for j in range(ny) for k in range(nz))

        elif direction == "y":
            shape = (
                nx,
                ny - 1,
                nz,
            )

            transverse_indices = ((i, k) for i in range(nx) for k in range(nz))

        elif direction == "z":
            shape = (
                nx,
                ny,
                nz - 1,
            )

            transverse_indices = ((i, j) for i in range(nx) for j in range(ny))

        else:
            raise ValueError(f"Unknown edge direction '{direction}'.")

        return shape, transverse_indices

    @staticmethod
    def _group_intersection_hits(
        hits,
        alpha_tolerance=1e-8,
    ):
        """
        Group ray-tracing hits at the same geometric position.
        """
        if not hits:
            return []

        hits = sorted(
            hits,
            key=lambda hit: hit["alpha"],
        )

        groups = [[hits[0]]]

        for hit in hits[1:]:
            reference_alpha = np.mean(
                [grouped_hit["alpha"] for grouped_hit in groups[-1]]
            )

            if abs(hit["alpha"] - reference_alpha) <= alpha_tolerance:
                groups[-1].append(hit)

            else:
                groups.append([hit])

        return groups

    @staticmethod
    def _split_intersection_groups(
        hit_groups,
        direction,
        normal_projection_tolerance=1e-8,
    ):
        """
        Split grouped STL hits into crossings and tangential contacts.

        A hit group is classified as a crossing if at least one facet normal
        has a non-zero component along the grid-edge direction. If all facet
        normals are perpendicular to the edge direction, the group is treated
        as a tangential or coplanar contact.

        Parameters
        ----------
        hit_groups : list
            Groups of ray-tracing hits at identical geometric positions.
        direction : str
            Grid-edge direction: "x", "y", or "z".
        normal_projection_tolerance : float, optional
            Tolerance for the absolute projection of the unit facet normal
            onto the grid-edge direction.

        Returns
        -------
        crossing_groups : list
            Hit groups representing actual crossings of an STL surface.
        contact_groups : list
            Hit groups representing tangential or coplanar contacts.
        """
        axis = {
            "x": 0,
            "y": 1,
            "z": 2,
        }

        if direction not in axis:
            raise ValueError(
                f"Unknown edge direction '{direction}'. Expected 'x', 'y', or 'z'."
            )

        if not 0.0 <= normal_projection_tolerance <= 1.0:
            raise ValueError("normal_projection_tolerance must be between 0 and 1.")

        axis_index = axis[direction]

        crossing_groups = []
        contact_groups = []

        for group in hit_groups:
            is_crossing = any(
                abs(
                    float(
                        np.asarray(
                            hit["normal"],
                            dtype=float,
                        )[axis_index]
                    )
                )
                > normal_projection_tolerance
                for hit in group
            )

            if is_crossing:
                crossing_groups.append(group)
            else:
                contact_groups.append(group)

        return crossing_groups, contact_groups

    @staticmethod
    def _select_crossing_group(
        crossing_groups,
    ):
        """
        Select one representative crossing position on an edge.

        If several distinct crossing positions exist, the position closest
        to the edge midpoint is selected. If two positions are equally close
        to the midpoint, the group containing the highest surface ID is
        preferred.

        The complete set of crossing groups is preserved separately for
        diagnostics and future multi-interface treatment.
        """
        if not crossing_groups:
            raise ValueError("Cannot select a crossing group from an empty list.")

        if len(crossing_groups) == 1:
            return crossing_groups[0]

        def group_key(group):
            alpha = float(np.mean([hit["alpha"] for hit in group]))

            highest_surface_id = max(hit["surface_id"] for hit in group)

            return (
                abs(alpha - 0.5),
                -highest_surface_id,
            )

        return min(
            crossing_groups,
            key=group_key,
        )

    def _select_effective_crossing_group(
        self,
        crossing_groups,
        start_region,
        end_region,
    ):
        """
        Select the effective physical crossing for one endpoint-region
        transition.

        For normal-conductor interfaces, the geometry of the conductor
        surface is authoritative:

            normal <-> PEC
            normal <-> SIBC

        This is important when two separate STL solids describe the same
        material interface but their triangulated surfaces are not exactly
        coincident.

        The raw crossing groups are not modified.

        Returns
        -------
        selected_group : list
            Crossing group used for the effective conformal interface.

        multiple_effective_crossings : bool
            True if the physical crossing remains ambiguous after applying
            the material-aware selection.
        """
        if not crossing_groups:
            raise ValueError("Cannot select an effective crossing from an empty list.")

        if len(crossing_groups) == 1:
            return (
                crossing_groups[0],
                False,
            )

        start_region = int(start_region)

        end_region = int(end_region)

        if start_region == 0:
            start_type = self.bg_material_type
        else:
            start_type = self._get_region_material_type(start_region)

        if end_region == 0:
            end_type = self.bg_material_type
        else:
            end_type = self._get_region_material_type(end_region)

        conductor_types = {
            "pec",
            "sibc",
        }

        conductor_region = None

        if start_type == "normal" and end_type in conductor_types:
            conductor_region = end_region

        elif end_type == "normal" and start_type in conductor_types:
            conductor_region = start_region

        # ----------------------------------------------------------
        # Normal <-> conductor:
        #
        # Prefer the crossing generated by the conductor STL itself.
        # surface_id and STL-region ID use the same region mapping.
        # ----------------------------------------------------------

        if conductor_region is not None:
            conductor_groups = [
                group
                for group in crossing_groups
                if any(int(hit["surface_id"]) == conductor_region for hit in group)
            ]

            if len(conductor_groups) == 1:
                return (
                    conductor_groups[0],
                    False,
                )

        # ----------------------------------------------------------
        # No unambiguous material-aware selection is possible.
        #
        # Keep the previous geometric fallback, but retain the fact
        # that several effective crossing positions remain.
        # ----------------------------------------------------------

        return (
            self._select_crossing_group(crossing_groups),
            True,
        )

    def _select_representative_hit(
        self,
        hits,
    ):
        """
        Select one representative STL facet from coincident hits.

        SIBC and PEC surfaces are preferred because their local facet
        normals are particularly relevant for boundary-condition treatment.
        """

        material_priority = {
            "sibc": 0,
            "pec": 1,
            "normal": 2,
        }

        def priority(hit):
            surface_id = hit["surface_id"]

            key = self.region_id_to_key[surface_id]

            material_type = self.stl_material_types[key]

            return (
                material_priority[material_type],
                surface_id,
                hit["cell_id"],
            )

        return min(
            hits,
            key=priority,
        )

    def _get_region_normal_flags(self):
        """
        Return a boolean array: True where a region ID is a normal material.

        If ``bg_material_type`` is not given, the background is treated as
        normal. This is conservative: it can only add dual-grid work, never
        drop a normal-normal interface.
        """
        flags = np.zeros(len(self.region_id_to_key), dtype=bool)

        flags[0] = self.bg_material_type in (None, "normal")

        for region_id, key in self.region_id_to_key.items():
            if region_id == 0:
                continue
            flags[region_id] = self.stl_material_types[key] == "normal"

        return flags

    def _get_normal_normal_edge_mask(self, grid_type, direction):
        """
        Edges whose two endpoints lie in different normal-material regions.
        """
        start, end = self._get_edge_region_ids(grid_type, direction)
        is_normal = self._get_region_normal_flags()

        return (start != end) & is_normal[start] & is_normal[end]

    def _build_interface_intersections(
        self,
        alpha_tolerance=1e-8,
    ):
        """
        Compute STL intersections with primal and dual grid edges.

        Intersections are obtained by tracing complete grid lines through all
        STL surfaces. Therefore, intersections are also detected when an STL
        region is thinner than one grid edge and both edge endpoints belong
        to the same effective region.

        Primal grid: all interfaces are traced. They define the conformal
        cuts at PEC/SIBC walls and the material interfaces.

        Dual grid: only edges between two different normal materials are
        kept, and only grid lines containing such edges are traced. Dual
        intersections are needed solely for the effective materials at
        dielectric/magnetic interfaces; PEC/SIBC walls never cut the dual
        metrics.
        """
        surface_cache = {}

        self.interface_intersections = {
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
                (
                    shape,
                    transverse_indices,
                ) = self._get_edge_shape_and_transverse_indices(
                    grid_type,
                    direction,
                )

                (
                    start_regions,
                    end_regions,
                ) = self._get_edge_region_ids(
                    grid_type,
                    direction,
                )

                intersections = {
                    "alpha": np.full(
                        shape,
                        np.nan,
                        dtype=float,
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
                        dtype=float,
                    ),
                    # Raw geometric ray-tracing information.
                    "has_intersection": np.zeros(
                        shape,
                        dtype=bool,
                    ),
                    "multiple_intersections": np.zeros(
                        shape,
                        dtype=bool,
                    ),
                    "hit_groups": {},
                    # Classified interface information.
                    "has_crossing": np.zeros(
                        shape,
                        dtype=bool,
                    ),
                    "has_contact": np.zeros(
                        shape,
                        dtype=bool,
                    ),
                    "multiple_crossing_positions": np.zeros(
                        shape,
                        dtype=bool,
                    ),
                    "raw_multiple_crossing_positions": np.zeros(
                        shape,
                        dtype=bool,
                    ),
                    "crossing_groups": {},
                    "contact_groups": {},
                }

                raw_hits = {}

                # Dual grid: restrict to normal-normal interface edges and
                # trace only the grid lines that contain such edges.
                if grid_type == "dual":
                    scope = self._get_normal_normal_edge_mask(grid_type, direction)
                    traced_lines = np.any(scope, axis="xyz".index(direction))
                    transverse_indices = (
                        index for index in transverse_indices if traced_lines[index]
                    )
                else:
                    scope = None

                for transverse_index in transverse_indices:
                    line_hits = self._trace_grid_line(
                        grid_type,
                        direction,
                        transverse_index,
                        surface_cache,
                        alpha_tolerance=alpha_tolerance,
                    )

                    for edge_index, hits in line_hits.items():
                        if scope is not None and not scope[edge_index]:
                            continue

                        raw_hits.setdefault(
                            edge_index,
                            [],
                        ).extend(hits)

                for edge_index, hits in raw_hits.items():
                    hit_groups = self._group_intersection_hits(
                        hits,
                        alpha_tolerance=alpha_tolerance,
                    )

                    if not hit_groups:
                        continue

                    # ----------------------------------------------------------
                    # Raw geometric ray-tracing information
                    # ----------------------------------------------------------
                    intersections["has_intersection"][edge_index] = True

                    intersections["multiple_intersections"][edge_index] = (
                        len(hit_groups) > 1
                    )

                    intersections["hit_groups"][edge_index] = hit_groups

                    # ----------------------------------------------------------
                    # Crossing/contact classification
                    # ----------------------------------------------------------
                    (
                        crossing_groups,
                        contact_groups,
                    ) = self._split_intersection_groups(
                        hit_groups,
                        direction,
                    )

                    if crossing_groups:
                        intersections["has_crossing"][edge_index] = True

                        # Preserve the complete raw geometric information.
                        intersections["crossing_groups"][edge_index] = crossing_groups

                        intersections["raw_multiple_crossing_positions"][edge_index] = (
                            len(crossing_groups) > 1
                        )

                    if contact_groups:
                        intersections["has_contact"][edge_index] = True

                        intersections["contact_groups"][edge_index] = contact_groups

                    # ----------------------------------------------------------
                    # Representative physical crossing
                    #
                    # Dense alpha/surface/normal arrays describe an actual
                    # crossing only. Pure tangential contacts therefore keep
                    # their default NaN / -1 values.
                    # ----------------------------------------------------------
                    if crossing_groups:
                        (
                            selected_group,
                            multiple_effective_crossings,
                        ) = self._select_effective_crossing_group(
                            crossing_groups,
                            start_region=(start_regions[edge_index]),
                            end_region=(end_regions[edge_index]),
                        )

                        intersections["multiple_crossing_positions"][edge_index] = (
                            multiple_effective_crossings
                        )

                        representative = self._select_representative_hit(selected_group)

                        intersections["alpha"][edge_index] = representative["alpha"]

                        intersections["surface_id"][edge_index] = representative[
                            "surface_id"
                        ]

                        intersections["cell_id"][edge_index] = representative["cell_id"]

                        intersections["normal"][edge_index] = representative["normal"]

                self.interface_intersections[grid_type][direction] = intersections

                if self.verbose:
                    n_intersections = int(
                        np.count_nonzero(intersections["has_intersection"])
                    )

                    n_crossings = int(np.count_nonzero(intersections["has_crossing"]))

                    n_contacts = int(np.count_nonzero(intersections["has_contact"]))

                    n_multiple_crossings = int(
                        np.count_nonzero(intersections["multiple_crossing_positions"])
                    )

                    print(
                        f"    * {grid_type} {direction}-edges: "
                        f"{n_intersections} geometric hits, "
                        f"{n_crossings} crossings, "
                        f"{n_contacts} contacts, "
                        f"{n_multiple_crossings} with multiple "
                        "crossing positions"
                    )
        self._build_resolved_interface_edge_masks()

    def _build_resolved_interface_edge_masks(self):
        """
        Build the interface masks used by the conformal FIT discretization.

        Before PEC snapping, every resolved interface is an interior
        geometrical crossing:

            endpoint material transition
            AND
            geometric STL crossing

        PEC snapping may later replace some interior crossings by
        endpoint contacts.
        """
        self.interior_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        self.endpoint_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        self.resolved_interface_edge_masks = {
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
                endpoint_mask = self.interface_edge_masks[grid_type][direction]

                has_crossing = self.interface_intersections[grid_type][direction][
                    "has_crossing"
                ]

                interior = endpoint_mask & has_crossing

                endpoint_contact = np.zeros_like(
                    interior,
                    dtype=bool,
                )

                self.interior_interface_edge_masks[grid_type][direction] = interior

                self.endpoint_interface_edge_masks[grid_type][direction] = (
                    endpoint_contact
                )

                self.resolved_interface_edge_masks[grid_type][direction] = (
                    interior.copy()
                )

    @staticmethod
    def _get_edge_endpoint_indices(
        direction,
        index,
    ):
        """
        Return the start and end grid-point indices of one edge.
        """
        i, j, k = (int(value) for value in index)

        start_index = (
            i,
            j,
            k,
        )

        if direction == "x":
            end_index = (
                i + 1,
                j,
                k,
            )

        elif direction == "y":
            end_index = (
                i,
                j + 1,
                k,
            )

        elif direction == "z":
            end_index = (
                i,
                j,
                k + 1,
            )

        else:
            raise ValueError(f"Unknown edge direction {direction!r}.")

        return (
            start_index,
            end_index,
        )

    def _get_region_material_type(
        self,
        region_id,
    ):
        region_id = int(region_id)

        if region_id == 0:
            if self.bg_material_type is None:
                raise RuntimeError(
                    "Background material type is "
                    "required for this conformal "
                    "geometry operation."
                )

            return self.bg_material_type

        key = self.region_id_to_key[region_id]

        return self.stl_material_types[key]

    def _rebuild_point_masks_from_region_ids(
        self,
    ):
        """
        Rebuild effective STL point masks from the current region IDs.

        After PEC snapping the region IDs describe the effective
        discretized geometry. The STL point masks are therefore
        reconstructed from these IDs so that material assignment and
        topology use the same geometry.
        """
        for (
            grid_type,
            point_masks,
        ) in (
            (
                "primal",
                self.primal_point_masks,
            ),
            (
                "dual",
                self.dual_point_masks,
            ),
        ):
            regions = self.region_ids[grid_type]

            for (
                key,
                region_id,
            ) in self.region_key_to_id.items():
                point_masks[key] = regions == region_id

        # Keep the existing cell masks consistent with the
        # effective primal-point geometry.
        for key in self.stl_solids:
            cell_mask = self._point_mask_to_cell_mask(
                self.primal_point_masks[key],
                threshold=0.5,
            )

            self.grid[key] = np.reshape(
                cell_mask,
                self.Nx * self.Ny * self.Nz,
                order="C",
            )

    def _apply_conformal_pec_snapping(
        self,
        max_iterations=20,
    ):
        """
        Remove small active normal edge fractions at normal-PEC
        interfaces by snapping the interface onto a grid point.

        Only single-crossing resolved interfaces are considered.

        normal -> PEC:
            alpha < eta
            => start point becomes PEC

        PEC -> normal:
            1 - alpha < eta
            => end point becomes PEC

        The operation is monotonic: normal points may become PEC,
        but PEC points are never converted back to normal.

        Only the primal grid is snapped. Conductor walls do not cut the dual
        metrics, so dual points keep their geometric classification.
        """
        eta = self.conformal_pec_snap_fraction

        if eta <= 0.0:
            return

        # ----------------------------------------------------------
        # Preserve the original geometric representative data.
        #
        # hit_groups / crossing_groups remain untouched as well.
        # The dense alpha/surface/normal arrays will later describe
        # the effective discretized interface.
        # ----------------------------------------------------------
        for grid_type in (
            "primal",
            "dual",
        ):
            for direction in (
                "x",
                "y",
                "z",
            ):
                data = self.interface_intersections[grid_type][direction]

                if "raw_alpha" not in data:
                    data["raw_alpha"] = data["alpha"].copy()

                    data["raw_surface_id"] = data["surface_id"].copy()

                    data["raw_cell_id"] = data["cell_id"].copy()

                    data["raw_normal"] = data["normal"].copy()

        self.snapped_point_data = {
            "primal": {},
            "dual": {},
        }

        conflict_count = 0

        for iteration in range(max_iterations):
            proposals = {
                "primal": {},
                "dual": {},
            }

            # Rebuild endpoint topology from the current
            # effective point regions.
            for grid_type in ("primal",):
                self.interface_edge_masks[grid_type] = (
                    self._build_edge_transition_masks(self.region_ids[grid_type])
                )

            # ------------------------------------------------------
            # Find small active normal portions.
            # ------------------------------------------------------
            for grid_type in ("primal",):
                for direction in (
                    "x",
                    "y",
                    "z",
                ):
                    (
                        start_regions,
                        end_regions,
                    ) = self._get_edge_region_ids(
                        grid_type,
                        direction,
                    )

                    data = self.interface_intersections[grid_type][direction]

                    raw_alpha = data["raw_alpha"]

                    candidate = (
                        self.interface_edge_masks[grid_type][direction]
                        & data["has_crossing"]
                    )

                    for index_array in np.argwhere(candidate):
                        index = tuple(int(value) for value in index_array)

                        alpha = float(raw_alpha[index])

                        if not np.isfinite(alpha):
                            continue

                        start_region = int(start_regions[index])

                        end_region = int(end_regions[index])

                        start_type = self._get_region_material_type(start_region)

                        end_type = self._get_region_material_type(end_region)

                        (
                            start_index,
                            end_index,
                        ) = self._get_edge_endpoint_indices(
                            direction,
                            index,
                        )

                        node_index = None
                        target_region = None

                        # Normal -> conductor (PEC or SIBC)
                        if (
                            start_type == "normal"
                            and end_type in ("pec", "sibc")
                            and alpha < eta
                        ):
                            node_index = start_index
                            target_region = end_region

                        # Conductor -> Normal (PEC or SIBC)
                        elif (
                            start_type in ("pec", "sibc")
                            and end_type == "normal"
                            and (1.0 - alpha) < eta
                        ):
                            node_index = end_index
                            target_region = start_region

                        if node_index is None:
                            continue

                        proposal = {
                            "target_region": (int(target_region)),
                            "surface_id": int(data["raw_surface_id"][index]),
                            "cell_id": int(data["raw_cell_id"][index]),
                            "normal": np.asarray(
                                data["raw_normal"][index],
                                dtype=float,
                            ).copy(),
                        }

                        proposals[grid_type].setdefault(
                            node_index,
                            [],
                        ).append(proposal)

            # ------------------------------------------------------
            # Apply all non-conflicting point snaps simultaneously.
            # ------------------------------------------------------
            n_changes = 0

            for grid_type in ("primal",):
                regions = self.region_ids[grid_type]

                for (
                    node_index,
                    node_proposals,
                ) in proposals[grid_type].items():
                    target_regions = {
                        proposal["target_region"] for proposal in node_proposals
                    }

                    # Different requested PEC regions are a genuine
                    # topological conflict.
                    if len(target_regions) != 1:
                        conflict_count += 1
                        continue

                    target_region = next(iter(target_regions))

                    # Prefer geometry belonging directly to the
                    # requested target STL region.
                    target_surface_proposals = [
                        proposal
                        for proposal in node_proposals
                        if proposal["surface_id"] == target_region
                    ]

                    if target_surface_proposals:
                        metadata_proposals = target_surface_proposals

                    else:
                        # For example, a PEC background has no STL
                        # surface of its own. In that case accept the
                        # metadata only when the contributing
                        # interface surface is unambiguous.
                        surface_ids = {
                            proposal["surface_id"] for proposal in node_proposals
                        }

                        if len(surface_ids) != 1:
                            conflict_count += 1
                            continue

                        metadata_proposals = node_proposals

                    selected_surface_id = int(metadata_proposals[0]["surface_id"])

                    current_region = int(regions[node_index])

                    if self._get_region_material_type(current_region) != "normal":
                        continue

                    regions[node_index] = target_region

                    normals = [
                        proposal["normal"]
                        for proposal in metadata_proposals
                        if np.all(np.isfinite(proposal["normal"]))
                    ]

                    if normals:
                        normal = np.mean(
                            normals,
                            axis=0,
                        )

                        norm = np.linalg.norm(normal)

                        if norm > 0.0:
                            normal = normal / norm

                        else:
                            normal = np.full(
                                3,
                                np.nan,
                            )

                    else:
                        normal = np.full(
                            3,
                            np.nan,
                        )

                    cell_ids = [
                        proposal["cell_id"]
                        for proposal in metadata_proposals
                        if proposal["cell_id"] >= 0
                    ]

                    self.snapped_point_data[grid_type][node_index] = {
                        "region_id": (int(target_region)),
                        "surface_id": (selected_surface_id),
                        "cell_id": (min(cell_ids) if cell_ids else -1),
                        "normal": (normal.copy()),
                    }

                    n_changes += 1

            if n_changes == 0:
                break

        else:
            raise RuntimeError(
                f"PEC snapping did not converge within {max_iterations} iterations."
            )

        if conflict_count:
            warnings.warn(
                f"{conflict_count} conformal PEC "
                "snapping proposals were ignored "
                "because multiple incompatible "
                "regions or surfaces met at the "
                "same grid point.",
                UserWarning,
                stacklevel=2,
            )

        # Point masks used by material assignment must
        # represent the same effective geometry.
        self._rebuild_point_masks_from_region_ids()

        self._rebuild_effective_interface_edges_after_snapping()

    def _rebuild_effective_interface_edges_after_snapping(
        self,
    ):
        """
        Rebuild the effective interface topology after PEC snapping.

        Interior cuts:
            genuine STL crossings inside endpoint-transition edges.

        Endpoint contacts:
            new normal-PEC transitions caused by a snapped PEC node,
            with the interface located exactly at alpha=0 or alpha=1.
        """
        self.interior_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        self.endpoint_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        self.resolved_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        for grid_type in (
            "primal",
            "dual",
        ):
            self.interface_edge_masks[grid_type] = self._build_edge_transition_masks(
                self.region_ids[grid_type]
            )

            snapped_points = self.snapped_point_data[grid_type]

            for direction in (
                "x",
                "y",
                "z",
            ):
                data = self.interface_intersections[grid_type][direction]

                endpoint_transition = self.interface_edge_masks[grid_type][direction]

                raw_alpha = data["raw_alpha"]

                raw_surface_id = data["raw_surface_id"]

                raw_cell_id = data["raw_cell_id"]

                raw_normal = data["raw_normal"]

                # Existing geometrical cuts that still connect
                # two different effective endpoint regions.
                interior = endpoint_transition & data["has_crossing"]

                endpoint_contact = np.zeros_like(
                    endpoint_transition,
                    dtype=bool,
                )

                # Dense arrays now describe the EFFECTIVE
                # conformal interface.
                alpha = np.full(
                    raw_alpha.shape,
                    np.nan,
                    dtype=float,
                )

                surface_id = np.full(
                    raw_surface_id.shape,
                    -1,
                    dtype=np.int32,
                )

                cell_id = np.full(
                    raw_cell_id.shape,
                    -1,
                    dtype=np.int64,
                )

                normal = np.full(
                    raw_normal.shape,
                    np.nan,
                    dtype=float,
                )

                alpha[interior] = raw_alpha[interior]

                surface_id[interior] = raw_surface_id[interior]

                cell_id[interior] = raw_cell_id[interior]

                normal[interior] = raw_normal[interior]

                (
                    start_regions,
                    end_regions,
                ) = self._get_edge_region_ids(
                    grid_type,
                    direction,
                )

                # Only transitions without an interior STL crossing
                # can be synthetic endpoint contacts.
                endpoint_candidates = endpoint_transition & ~interior

                for index_array in np.argwhere(endpoint_candidates):
                    index = tuple(int(value) for value in index_array)

                    (
                        start_index,
                        end_index,
                    ) = self._get_edge_endpoint_indices(
                        direction,
                        index,
                    )

                    start_region = int(start_regions[index])

                    end_region = int(end_regions[index])

                    start_type = self._get_region_material_type(start_region)

                    end_type = self._get_region_material_type(end_region)

                    metadata = None
                    endpoint_alpha = None

                    # normal -> PEC:
                    # interface at the PEC end point.
                    if start_type == "normal" and end_type in ("pec", "sibc"):
                        metadata = snapped_points.get(end_index)

                        if metadata is not None and metadata["region_id"] == end_region:
                            endpoint_alpha = 1.0

                    # PEC -> normal:
                    # interface at the PEC start point.
                    elif start_type in ("pec", "sibc") and end_type == "normal":
                        metadata = snapped_points.get(start_index)

                        if (
                            metadata is not None
                            and metadata["region_id"] == start_region
                        ):
                            endpoint_alpha = 0.0

                    if endpoint_alpha is None:
                        continue

                    endpoint_contact[index] = True

                    alpha[index] = endpoint_alpha

                    surface_id[index] = metadata["surface_id"]

                    cell_id[index] = metadata["cell_id"]

                    normal[index] = metadata["normal"]

                resolved = interior | endpoint_contact

                self.interior_interface_edge_masks[grid_type][direction] = interior

                self.endpoint_interface_edge_masks[grid_type][direction] = (
                    endpoint_contact
                )

                self.resolved_interface_edge_masks[grid_type][direction] = resolved

                data["alpha"] = alpha
                data["surface_id"] = surface_id
                data["cell_id"] = cell_id
                data["normal"] = normal

                data["effective_endpoint_contact"] = endpoint_contact.copy()

    def _get_interface_diagnostics(self):
        """
        Summarize unresolved and approximated conformal-interface cases.
        """
        diagnostics = {
            "endpoint_interfaces": 0,
            "resolved_interfaces": 0,
            "missing_crossings": 0,
            "hidden_crossings": 0,
            "multiple_crossing_edges": 0,
            "multiple_resolved_edges": 0,
            "repeated_stl_crossing_edges": [],
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
                endpoint_mask = self.interface_edge_masks[grid_type][direction]

                # Dual intersections only exist for normal-normal edges.
                if grid_type == "dual":
                    endpoint_mask = endpoint_mask & self._get_normal_normal_edge_mask(
                        grid_type, direction
                    )

                resolved_mask = self.resolved_interface_edge_masks[grid_type][direction]

                data = self.interface_intersections[grid_type][direction]

                has_crossing = data["has_crossing"]

                multiple = data["multiple_crossing_positions"]

                endpoint_contact = self.endpoint_interface_edge_masks[grid_type][
                    direction
                ]

                missing = endpoint_mask & ~has_crossing & ~endpoint_contact

                hidden = ~endpoint_mask & has_crossing

                diagnostics["endpoint_interfaces"] += int(
                    np.count_nonzero(endpoint_mask)
                )

                diagnostics["resolved_interfaces"] += int(
                    np.count_nonzero(resolved_mask)
                )

                diagnostics["missing_crossings"] += int(np.count_nonzero(missing))

                diagnostics["hidden_crossings"] += int(np.count_nonzero(hidden))

                diagnostics["multiple_crossing_edges"] += int(
                    np.count_nonzero(multiple)
                )

                diagnostics["multiple_resolved_edges"] += int(
                    np.count_nonzero(multiple & resolved_mask)
                )

                for index, groups in data["crossing_groups"].items():
                    if len(groups) <= 1:
                        continue

                    surface_ids_per_group = [
                        {hit["surface_id"] for hit in group} for group in groups
                    ]

                    all_surface_ids = set().union(*surface_ids_per_group)

                    repeated_surface_ids = {
                        surface_id
                        for surface_id in all_surface_ids
                        if sum(
                            surface_id in group_ids
                            for group_ids in surface_ids_per_group
                        )
                        > 1
                    }

                    if repeated_surface_ids:
                        diagnostics["repeated_stl_crossing_edges"].append(
                            {
                                "grid_type": grid_type,
                                "direction": direction,
                                "index": tuple(int(v) for v in index),
                                "surface_ids": tuple(sorted(repeated_surface_ids)),
                            }
                        )
        return diagnostics

    def _warn_interface_diagnostics(self):
        """
        Warn about conformal-interface cases that are currently ignored or
        approximated.
        """
        diagnostics = self._get_interface_diagnostics()

        messages = []

        if diagnostics["missing_crossings"]:
            messages.append(
                f"{diagnostics['missing_crossings']} endpoint "
                "interface edges have no geometric crossing and "
                "will not receive a conformal correction."
            )

        if diagnostics["hidden_crossings"]:
            messages.append(
                f"{diagnostics['hidden_crossings']} edges contain "
                "geometric crossings that are hidden from endpoint "
                "topology and are ignored by the current conformal "
                "discretization."
            )

        if diagnostics["multiple_crossing_edges"]:
            messages.append(
                f"{diagnostics['multiple_crossing_edges']} edges "
                "contain multiple crossing positions. "
                f"{diagnostics['multiple_resolved_edges']} of these "
                "are currently used by the conformal discretization; "
                "one representative crossing is selected on each edge."
            )

        repeated = diagnostics["repeated_stl_crossing_edges"]

        if repeated:
            messages.append(
                f"{len(repeated)} edges cross the same STL at more "
                "than one distinct position. The geometry on these "
                "edges is not fully resolved by the current "
                "single-crossing treatment."
            )

            for item in repeated[:5]:
                surface_names = [
                    self.region_id_to_key[surface_id]
                    for surface_id in item["surface_ids"]
                ]

                messages.append(
                    "  "
                    f"{item['grid_type']} "
                    f"{item['direction']}-edge "
                    f"{item['index']}: "
                    f"{', '.join(surface_names)}"
                )

            if len(repeated) > 5:
                messages.append(f"  ... and {len(repeated) - 5} more.")

        if messages:
            warnings.warn(
                "Conformal geometry diagnostics:\n" + "\n".join(messages),
                UserWarning,
                stacklevel=2,
            )

    def _build_face_candidate_masks(self):
        """
        Build masks of grid faces adjacent to resolved interface edges.

        A face is marked as a candidate when at least one of its four
        boundary edges contains a resolved geometric interface crossing.

        Face directions denote the face normal:
            x -> yz face
            y -> xz face
            z -> xy face
        """
        self.face_candidate_masks = {
            "primal": {},
            "dual": {},
        }

        for grid_type in (
            "primal",
            "dual",
        ):
            edge_masks = self.interior_interface_edge_masks.get(
                grid_type,
                {},
            )

            if not all(
                direction in edge_masks
                for direction in (
                    "x",
                    "y",
                    "z",
                )
            ):
                raise RuntimeError(
                    f"Resolved interface-edge masks are incomplete "
                    f"for the {grid_type} grid."
                )

            x_edges = np.asarray(
                edge_masks["x"],
                dtype=bool,
            )
            y_edges = np.asarray(
                edge_masks["y"],
                dtype=bool,
            )
            z_edges = np.asarray(
                edge_masks["z"],
                dtype=bool,
            )

            # ------------------------------------------------------
            # x-normal faces (yz)
            #
            # Boundary:
            #   two y-directed edges at z_k and z_{k+1}
            #   two z-directed edges at y_j and y_{j+1}
            #
            # Shape:
            #   (Nx + 1, Ny, Nz)
            # ------------------------------------------------------
            x_faces = (
                y_edges[:, :, :-1]
                | y_edges[:, :, 1:]
                | z_edges[:, :-1, :]
                | z_edges[:, 1:, :]
            )

            # ------------------------------------------------------
            # y-normal faces (xz)
            #
            # Boundary:
            #   two x-directed edges at z_k and z_{k+1}
            #   two z-directed edges at x_i and x_{i+1}
            #
            # Shape:
            #   (Nx, Ny + 1, Nz)
            # ------------------------------------------------------
            y_faces = (
                x_edges[:, :, :-1]
                | x_edges[:, :, 1:]
                | z_edges[:-1, :, :]
                | z_edges[1:, :, :]
            )

            # ------------------------------------------------------
            # z-normal faces (xy)
            #
            # Boundary:
            #   two x-directed edges at y_j and y_{j+1}
            #   two y-directed edges at x_i and x_{i+1}
            #
            # Shape:
            #   (Nx, Ny, Nz + 1)
            # ------------------------------------------------------
            z_faces = (
                x_edges[:, :-1, :]
                | x_edges[:, 1:, :]
                | y_edges[:-1, :, :]
                | y_edges[1:, :, :]
            )

            self.face_candidate_masks[grid_type] = {
                "x": x_faces,
                "y": y_faces,
                "z": z_faces,
            }

    def _get_face_boundary_edges(
        self,
        grid_type,
        face_direction,
        index,
    ):
        """
        Return the four boundary edges of one grid face.

        The edges are ordered counter-clockwise with respect to the
        positive face normal. ``boundary_sign`` gives the orientation
        of the stored grid edge relative to this boundary traversal.

        Face directions denote the face normal:
            x -> yz face
            y -> xz face
            z -> xy face
        """
        x, y, z = self._get_grid_coordinates(grid_type)

        nx = len(x)
        ny = len(y)
        nz = len(z)

        face_shapes = {
            "x": (
                nx,
                ny - 1,
                nz - 1,
            ),
            "y": (
                nx - 1,
                ny,
                nz - 1,
            ),
            "z": (
                nx - 1,
                ny - 1,
                nz,
            ),
        }

        if face_direction not in face_shapes:
            raise ValueError(
                f"Unknown face direction '{face_direction}'. Expected 'x', 'y', or 'z'."
            )

        i, j, k = (int(value) for value in index)

        face_shape = face_shapes[face_direction]

        if (
            i < 0
            or j < 0
            or k < 0
            or i >= face_shape[0]
            or j >= face_shape[1]
            or k >= face_shape[2]
        ):
            raise IndexError(
                f"Face index {index} is outside "
                f"{grid_type} {face_direction}-face "
                f"shape {face_shape}."
            )

        if face_direction == "x":
            # Local coordinates:
            # u = y, v = z
            #
            # e_y x e_z = e_x
            return [
                {
                    "direction": "y",
                    "index": (i, j, k),
                    "boundary_sign": 1,
                },
                {
                    "direction": "z",
                    "index": (i, j + 1, k),
                    "boundary_sign": 1,
                },
                {
                    "direction": "y",
                    "index": (i, j, k + 1),
                    "boundary_sign": -1,
                },
                {
                    "direction": "z",
                    "index": (i, j, k),
                    "boundary_sign": -1,
                },
            ]

        if face_direction == "y":
            # Local coordinates:
            # u = z, v = x
            #
            # e_z x e_x = e_y
            return [
                {
                    "direction": "z",
                    "index": (i, j, k),
                    "boundary_sign": 1,
                },
                {
                    "direction": "x",
                    "index": (i, j, k + 1),
                    "boundary_sign": 1,
                },
                {
                    "direction": "z",
                    "index": (i + 1, j, k),
                    "boundary_sign": -1,
                },
                {
                    "direction": "x",
                    "index": (i, j, k),
                    "boundary_sign": -1,
                },
            ]

        # face_direction == "z"
        #
        # Local coordinates:
        # u = x, v = y
        #
        # e_x x e_y = e_z
        return [
            {
                "direction": "x",
                "index": (i, j, k),
                "boundary_sign": 1,
            },
            {
                "direction": "y",
                "index": (i + 1, j, k),
                "boundary_sign": 1,
            },
            {
                "direction": "x",
                "index": (i, j + 1, k),
                "boundary_sign": -1,
            },
            {
                "direction": "y",
                "index": (i, j, k),
                "boundary_sign": -1,
            },
        ]

    def _get_face_boundary_intersections(
        self,
        grid_type,
        face_direction,
        index,
        point_tolerance=None,
    ):
        """
        Return unique resolved interface intersections on a face boundary.

        Coincident intersections at a face corner are grouped into one
        geometric point while the individual edge-hit information is
        preserved.
        """
        boundary_edges = self._get_face_boundary_edges(
            grid_type,
            face_direction,
            index,
        )

        resolved_masks = self.resolved_interface_edge_masks[grid_type]

        intersections = self.interface_intersections[grid_type]

        raw_hits = []
        edge_lengths = []

        for boundary_order, edge in enumerate(boundary_edges):
            direction = edge["direction"]
            edge_index = edge["index"]
            boundary_sign = edge["boundary_sign"]

            (
                start_point,
                end_point,
                start_region,
                end_region,
            ) = self._get_edge_geometry(
                grid_type,
                direction,
                edge_index,
            )

            edge_length = np.linalg.norm(end_point - start_point)

            edge_lengths.append(edge_length)

            if not resolved_masks[direction][edge_index]:
                continue

            intersection_data = intersections[direction]

            alpha = float(intersection_data["alpha"][edge_index])

            if not np.isfinite(alpha):
                raise RuntimeError(
                    f"Resolved {grid_type} "
                    f"{direction}-edge {edge_index} "
                    "has no finite intersection alpha."
                )

            point = start_point + alpha * (end_point - start_point)

            if boundary_sign > 0:
                local_boundary_coordinate = alpha
            else:
                local_boundary_coordinate = 1.0 - alpha

            boundary_parameter = boundary_order + local_boundary_coordinate

            multiple_crossing_positions = False

            if "multiple_crossing_positions" in intersection_data:
                multiple_crossing_positions = bool(
                    intersection_data["multiple_crossing_positions"][edge_index]
                )

            raw_hits.append(
                {
                    "point": point,
                    "edge_direction": direction,
                    "edge_index": edge_index,
                    "alpha": alpha,
                    "boundary_order": boundary_order,
                    "boundary_sign": boundary_sign,
                    "boundary_parameter": (boundary_parameter),
                    "start_region": start_region,
                    "end_region": end_region,
                    "surface_id": int(intersection_data["surface_id"][edge_index]),
                    "cell_id": int(intersection_data["cell_id"][edge_index]),
                    "normal": np.asarray(
                        intersection_data["normal"][edge_index],
                        dtype=float,
                    ).copy(),
                    "multiple_crossing_positions": (multiple_crossing_positions),
                }
            )

        if not raw_hits:
            return []

        if point_tolerance is None:
            characteristic_length = max(edge_lengths)

            point_tolerance = 1e-8 * characteristic_length

        raw_hits.sort(key=lambda hit: hit["boundary_parameter"])

        point_groups = []

        for hit in raw_hits:
            matching_group = None

            for group in point_groups:
                if np.linalg.norm(hit["point"] - group["point"]) <= point_tolerance:
                    matching_group = group
                    break

            if matching_group is None:
                point_groups.append(
                    {
                        "point": (hit["point"].copy()),
                        "boundary_parameter": (hit["boundary_parameter"]),
                        "hits": [hit],
                    }
                )

            else:
                matching_group["hits"].append(hit)

                matching_group["boundary_parameter"] = min(
                    matching_group["boundary_parameter"],
                    hit["boundary_parameter"],
                )

        point_groups.sort(key=lambda group: group["boundary_parameter"])

        return point_groups

    def _get_face_intersection_segment(
        self,
        grid_type,
        face_direction,
        index,
        point_tolerance=None,
    ):
        """
        Reconstruct one simple interface segment crossing a grid face.

        The current conformal V1 resolves only faces with exactly two
        unique boundary-intersection points belonging to one common STL
        surface. More complicated cases are returned as unresolved.

        The returned segment orientation follows the order of the
        boundary-intersection points. It is purely geometric and must not
        yet be interpreted as the oriented boundary direction required
        by Stokes' theorem.
        """
        point_groups = self._get_face_boundary_intersections(
            grid_type,
            face_direction,
            index,
            point_tolerance=point_tolerance,
        )

        result = {
            "resolved": False,
            "point_groups": point_groups,
            "p0": None,
            "p1": None,
            "vector": None,
            "length": np.nan,
            "tangent": None,
            "surface_id": -1,
            "normal": None,
        }

        # Simple V1 case:
        # exactly two distinct boundary-intersection positions.
        if len(point_groups) != 2:
            return result

        # Reject an edge for which more than one crossing position
        # existed inside the same grid edge.
        for group in point_groups:
            if any(hit["multiple_crossing_positions"] for hit in group["hits"]):
                return result

        surface_ids_0 = {
            hit["surface_id"]
            for hit in point_groups[0]["hits"]
            if hit["surface_id"] >= 0
        }

        surface_ids_1 = {
            hit["surface_id"]
            for hit in point_groups[1]["hits"]
            if hit["surface_id"] >= 0
        }

        common_surface_ids = surface_ids_0 & surface_ids_1

        # For now only resolve an unambiguous interface surface.
        if len(common_surface_ids) != 1:
            return result

        surface_id = next(iter(common_surface_ids))

        p0 = np.asarray(
            point_groups[0]["point"],
            dtype=float,
        )

        p1 = np.asarray(
            point_groups[1]["point"],
            dtype=float,
        )

        vector = p1 - p0

        length = float(np.linalg.norm(vector))

        if length <= 0.0:
            return result

        tangent = vector / length

        # Collect facet normals belonging to the selected STL surface.
        normals = []

        for group in point_groups:
            for hit in group["hits"]:
                if hit["surface_id"] == surface_id and np.all(
                    np.isfinite(hit["normal"])
                ):
                    normals.append(
                        np.asarray(
                            hit["normal"],
                            dtype=float,
                        )
                    )

        normal = None

        if normals:
            mean_normal = np.mean(
                normals,
                axis=0,
            )

            norm = np.linalg.norm(mean_normal)

            if norm > 0.0:
                normal = mean_normal / norm

        result.update(
            {
                "resolved": True,
                "p0": p0.copy(),
                "p1": p1.copy(),
                "vector": vector.copy(),
                "length": length,
                "tangent": tangent.copy(),
                "surface_id": int(surface_id),
                "normal": (None if normal is None else normal.copy()),
            }
        )

        return result

    def _get_face_corners(
        self,
        grid_type,
        face_direction,
        index,
    ):
        """
        Return the four corners of one grid face in positive boundary order.

        The ordering is counter-clockwise with respect to the positive
        face normal.
        """
        # Also validates the face index.
        self._get_face_boundary_edges(
            grid_type,
            face_direction,
            index,
        )

        x, y, z = self._get_grid_coordinates(grid_type)

        region_ids = self.region_ids[grid_type]

        i, j, k = (int(value) for value in index)

        if face_direction == "x":
            corner_indices = [
                (i, j, k),
                (i, j + 1, k),
                (i, j + 1, k + 1),
                (i, j, k + 1),
            ]

        elif face_direction == "y":
            corner_indices = [
                (i, j, k),
                (i, j, k + 1),
                (i + 1, j, k + 1),
                (i + 1, j, k),
            ]

        elif face_direction == "z":
            corner_indices = [
                (i, j, k),
                (i + 1, j, k),
                (i + 1, j + 1, k),
                (i, j + 1, k),
            ]

        else:
            raise ValueError(f"Unknown face direction '{face_direction}'.")

        corners = []

        for boundary_parameter, corner_index in enumerate(corner_indices):
            ci, cj, ck = corner_index

            corners.append(
                {
                    "point": np.array(
                        [
                            x[ci],
                            y[cj],
                            z[ck],
                        ],
                        dtype=float,
                    ),
                    "grid_index": corner_index,
                    "region_id": int(region_ids[corner_index]),
                    "boundary_parameter": float(boundary_parameter),
                }
            )

        return corners

    @staticmethod
    def _compute_face_polygon_area(
        vertices,
        face_direction,
    ):
        """
        Compute the area of a polygon lying in one coordinate-aligned grid face.
        """
        vertices = np.asarray(
            vertices,
            dtype=float,
        )

        if len(vertices) < 3:
            return 0.0

        if face_direction == "x":
            points_2d = vertices[:, [1, 2]]

        elif face_direction == "y":
            points_2d = vertices[:, [2, 0]]

        elif face_direction == "z":
            points_2d = vertices[:, [0, 1]]

        else:
            raise ValueError(f"Unknown face direction '{face_direction}'.")

        u = points_2d[:, 0]
        v = points_2d[:, 1]

        signed_area = 0.5 * (
            np.dot(
                u,
                np.roll(v, -1),
            )
            - np.dot(
                v,
                np.roll(u, -1),
            )
        )

        return float(abs(signed_area))

    @staticmethod
    def _get_face_boundary_path(
        corners,
        start_group,
        end_group,
    ):
        """
        Return the positive face-boundary path from one intersection
        point to another, including all face corners encountered.

        The face boundary is parameterized periodically over [0, 4).
        If the path crosses the parameter wrap-around at 4 -> 0, corner
        parameters are shifted by one period and explicitly sorted before
        the polygon path is assembled.
        """
        start_parameter = float(start_group["boundary_parameter"])

        end_parameter = float(end_group["boundary_parameter"])

        if end_parameter <= start_parameter:
            end_parameter += 4.0

        vertices = [
            np.asarray(
                start_group["point"],
                dtype=float,
            )
        ]

        # ----------------------------------------------------------
        # Map all face corners onto the same unwrapped boundary
        # interval as start_parameter -> end_parameter.
        #
        # The original corner list always has the geometric order
        #
        #     0, 1, 2, 3
        #
        # but after a periodic wrap the correct path may instead be
        #
        #     3, 0
        #
        # or
        #
        #     2, 3, 0, 1.
        #
        # Therefore the shifted boundary parameters must be sorted
        # explicitly before constructing the polygon boundary.
        # ----------------------------------------------------------

        path_corner_candidates = []

        for corner in corners:
            parameter = float(corner["boundary_parameter"])

            if parameter <= start_parameter:
                parameter += 4.0

            if start_parameter < parameter < end_parameter:
                path_corner_candidates.append(
                    (
                        parameter,
                        corner,
                    )
                )

        path_corner_candidates.sort(key=lambda item: item[0])

        path_corners = [corner for _, corner in path_corner_candidates]

        for corner in path_corners:
            vertices.append(
                np.asarray(
                    corner["point"],
                    dtype=float,
                )
            )

        vertices.append(
            np.asarray(
                end_group["point"],
                dtype=float,
            )
        )

        return (
            np.asarray(
                vertices,
                dtype=float,
            ),
            path_corners,
        )

    def _split_face_by_interface(
        self,
        grid_type,
        face_direction,
        index,
        point_tolerance=None,
    ):
        """
        Split one rectangular grid face into two region polygons.

        The current V1 handles a single resolved interface segment with
        exactly two distinct boundary-intersection points. Material
        treatment is intentionally not performed here.

        Returns both geometric region polygons together with their areas
        and the orientation of the cut segment on each polygon boundary.
        """
        segment = self._get_face_intersection_segment(
            grid_type,
            face_direction,
            index,
            point_tolerance=point_tolerance,
        )

        result = {
            "resolved": False,
            "segment": segment,
            "face_area": np.nan,
            "polygons": [],
        }

        if not segment["resolved"]:
            return result

        point_groups = segment["point_groups"]

        if len(point_groups) != 2:
            return result

        corners = self._get_face_corners(
            grid_type,
            face_direction,
            index,
        )

        # Full rectangular face area.
        corner_points = np.asarray(
            [corner["point"] for corner in corners],
            dtype=float,
        )

        face_area = self._compute_face_polygon_area(
            corner_points,
            face_direction,
        )

        if face_area <= 0.0:
            return result

        p0_group = point_groups[0]
        p1_group = point_groups[1]

        # --------------------------------------------------
        # Polygon 0:
        #
        # p0 -> positive outer boundary -> p1
        # then interface p1 -> p0
        # --------------------------------------------------
        (
            path_0,
            corners_0,
        ) = self._get_face_boundary_path(
            corners,
            p0_group,
            p1_group,
        )

        # --------------------------------------------------
        # Polygon 1:
        #
        # p1 -> positive outer boundary -> p0
        # then interface p0 -> p1
        # --------------------------------------------------
        (
            path_1,
            corners_1,
        ) = self._get_face_boundary_path(
            corners,
            p1_group,
            p0_group,
        )

        # Each region must contain at least one actual
        # face corner away from the interface.
        if not corners_0 or not corners_1:
            return result

        region_ids_0 = {corner["region_id"] for corner in corners_0}

        region_ids_1 = {corner["region_id"] for corner in corners_1}

        # V1 requires one unique effective region on
        # each side of the interface.
        if len(region_ids_0) != 1 or len(region_ids_1) != 1:
            return result

        region_id_0 = next(iter(region_ids_0))

        region_id_1 = next(iter(region_ids_1))

        if region_id_0 == region_id_1:
            return result

        area_0 = self._compute_face_polygon_area(
            path_0,
            face_direction,
        )

        area_1 = self._compute_face_polygon_area(
            path_1,
            face_direction,
        )

        area_tolerance = 1e-12 * face_area

        if area_0 <= area_tolerance or area_1 <= area_tolerance:
            return result

        if not np.isclose(
            area_0 + area_1,
            face_area,
            rtol=1e-10,
            atol=area_tolerance,
        ):
            return result

        segment_vector = np.asarray(
            segment["vector"],
            dtype=float,
        )

        segment_length = float(segment["length"])

        # Polygon 0 closes from p1 back to p0.
        cut_vector_0 = -segment_vector

        # Polygon 1 closes from p0 to p1.
        cut_vector_1 = segment_vector.copy()

        polygons = [
            {
                "region_id": int(region_id_0),
                "vertices": path_0.copy(),
                "area": float(area_0),
                "area_fraction": float(area_0 / face_area),
                "cut_vector": (cut_vector_0.copy()),
                "cut_length": (segment_length),
                "cut_tangent": (cut_vector_0 / segment_length),
            },
            {
                "region_id": int(region_id_1),
                "vertices": path_1.copy(),
                "area": float(area_1),
                "area_fraction": float(area_1 / face_area),
                "cut_vector": (cut_vector_1.copy()),
                "cut_length": (segment_length),
                "cut_tangent": (cut_vector_1 / segment_length),
            },
        ]

        result.update(
            {
                "resolved": True,
                "face_area": float(face_area),
                "polygons": polygons,
            }
        )

        return result

    def _build_interface_face_data(self):
        """
        Build geometric conformal-interface data for all candidate faces.

        The grid layer stores geometry and region IDs only. Material
        interpretation is intentionally deferred to the solver.
        """
        self.interface_face_data = {
            "primal": {},
            "dual": {},
        }

        for grid_type in (
            "primal",
            "dual",
        ):
            if not self.face_candidate_masks.get(grid_type):
                raise RuntimeError(
                    f"No face-candidate masks available for the {grid_type} grid."
                )

            for face_direction in (
                "x",
                "y",
                "z",
            ):
                candidate = np.asarray(
                    self.face_candidate_masks[grid_type][face_direction],
                    dtype=bool,
                )

                segment_resolved = np.zeros_like(
                    candidate,
                    dtype=bool,
                )

                resolved = np.zeros_like(
                    candidate,
                    dtype=bool,
                )

                face_area = np.full(
                    candidate.shape,
                    np.nan,
                    dtype=float,
                )

                segments = {}
                polygons = {}
                region_areas = {}
                region_area_fractions = {}

                for index_array in np.argwhere(candidate):
                    index = tuple(int(value) for value in index_array)

                    split = self._split_face_by_interface(
                        grid_type,
                        face_direction,
                        index,
                    )

                    segment = split["segment"]

                    # --------------------------------------------------
                    # Preserve a successfully reconstructed segment even
                    # if the later polygon classification is unresolved.
                    # --------------------------------------------------
                    if segment["resolved"]:
                        segment_resolved[index] = True

                        segments[index] = {
                            "surface_id": int(segment["surface_id"]),
                            "p0": (segment["p0"].copy()),
                            "p1": (segment["p1"].copy()),
                            "vector": (segment["vector"].copy()),
                            "length": float(segment["length"]),
                            "tangent": (segment["tangent"].copy()),
                            "normal": (
                                None
                                if segment["normal"] is None
                                else segment["normal"].copy()
                            ),
                        }

                    if not split["resolved"]:
                        continue

                    resolved[index] = True

                    face_area[index] = float(split["face_area"])

                    polygon_records = []

                    areas = {}
                    area_fractions = {}

                    for polygon in split["polygons"]:
                        region_id = int(polygon["region_id"])

                        area = float(polygon["area"])

                        area_fraction = float(polygon["area_fraction"])

                        areas[region_id] = area

                        area_fractions[region_id] = area_fraction

                        polygon_records.append(
                            {
                                "region_id": region_id,
                                "vertices": polygon["vertices"].copy(),
                                "area": area,
                                "area_fraction": area_fraction,
                                "cut_vector": polygon["cut_vector"].copy(),
                                "cut_length": float(polygon["cut_length"]),
                                "cut_tangent": polygon["cut_tangent"].copy(),
                            }
                        )

                    polygons[index] = polygon_records

                    region_areas[index] = areas

                    region_area_fractions[index] = area_fractions

                self.interface_face_data[grid_type][face_direction] = {
                    "candidate": (candidate.copy()),
                    "segment_resolved": (segment_resolved),
                    "resolved": resolved,
                    "face_area": face_area,
                    "segments": segments,
                    "polygons": polygons,
                    "region_areas": (region_areas),
                    "region_area_fractions": (region_area_fractions),
                }

                if self.verbose:
                    n_candidate = int(np.count_nonzero(candidate))

                    n_segment = int(np.count_nonzero(segment_resolved))

                    n_resolved = int(np.count_nonzero(resolved))

                    print(
                        f"    * {grid_type} "
                        f"{face_direction}-faces: "
                        f"{n_candidate} candidates, "
                        f"{n_segment} segments, "
                        f"{n_resolved} resolved splits"
                    )

    def _mark_cells_in_stl_conformal(self, method):
        """
        Generate conformal STL masks on the primal and dual FIT grids.

        Both primal and dual grid-point masks are generated using the same
        selected STL classification method. For the dual grid, the selected
        method is applied to the regular inner dual points, while high-side
        boundary points are classified separately using ``interior_points``.

        The cell mask stored in ``self.grid[key]`` is derived from the primal
        point mask using the existing corner-point threshold rule.

        Parameters
        ----------
        method : str
            STL classification method. Supported methods are
            "legacy", "interior_points", "implicit_distance",
            and "voxelize_rectilinear".
        """
        stl_tolerance = (
            np.min(
                [
                    np.min(self.dx),
                    np.min(self.dy),
                    np.min(self.dz),
                ]
            )
            * self.stl_tol
        )

        progress_bar = False
        if self.verbose > 1:
            progress_bar = True

        method = method.lower()

        valid_methods = (
            "legacy",
            "interior_points",
            "implicit_distance",
            "voxelize_rectilinear",
        )

        if method not in valid_methods:
            raise ValueError(
                f"[!] Error: stl_method {method} not recognized. "
                "Use 'legacy', 'interior_points', "
                "'implicit_distance', or "
                "'voxelize_rectilinear'."
            )

        for key in self.stl_solids.keys():
            surf = self.read_stl(key)

            if self.verbose:
                print(f" * Marking grid points inside STL solid '{key}'...")

            # ----------------------------------------------------------
            # Primal grid-point classification
            # ----------------------------------------------------------
            primal_point_mask = self._classify_grid_points_in_stl(
                surf=surf,
                grid_type="primal",
                method=method,
                stl_tolerance=stl_tolerance,
                progress_bar=progress_bar,
            )

            # ----------------------------------------------------------
            # Dual grid-point classification
            # ----------------------------------------------------------
            dual_point_mask = self._classify_grid_points_in_stl(
                surf=surf,
                grid_type="dual",
                method=method,
                stl_tolerance=stl_tolerance,
                progress_bar=progress_bar,
            )

            # ----------------------------------------------------------
            # Store primal/dual point masks and derive the cell mask
            # ----------------------------------------------------------
            self._store_stl_masks(
                key,
                primal_point_mask,
                dual_point_mask,
                cell_threshold=0.5,
            )

            # ----------------------------------------------------------
            # Diagnostics
            # ----------------------------------------------------------
            if self.verbose and np.sum(self.grid[key]) == 0:
                print(
                    f"[!] Warning: no cells were marked "
                    f"inside STL solid '{key}'. "
                    "Consider checking the STL geometry, "
                    "classification method, or grid resolution."
                )

            if self.verbose:
                print(
                    f"    * STL solid {key}: "
                    f"{np.sum(self.primal_point_masks[key])} "
                    "primal points, "
                    f"{np.sum(self.grid[key])} "
                    "cell centers, and "
                    f"{np.sum(self.dual_point_masks[key])} "
                    "dual points marked inside the solid."
                )

        # Build material-region topology
        #
        # Must be done only after all STL point masks are available.
        # --------------------------------------------------------------
        self._build_interface_topology()

        # --------------------------------------------------------------
        # Determine exact STL intersections with primal and dual edges
        #
        # This uses the geometric full-grid-line tracing developed above
        # and is independent of the endpoint transition masks.
        # --------------------------------------------------------------
        self._build_interface_intersections()

        if self.conformal_pec_snap_fraction > 0.0:
            self._apply_conformal_pec_snapping()

        # Produce Warnings for ignored interface cases (hidden crossings, missing crossings)
        self._warn_interface_diagnostics()

        # Build masks of grid faces adjacent to resolved interface edges
        self._build_face_candidate_masks()

        self._build_interface_face_data()


class ConformalSolverMixin:
    """Conformal metrics and material assignment for ``SolverFIT3D``."""

    def _build_region_material_types(self):
        """
        Associate every geometry region ID with its material treatment type.

        Region 0 represents the background. STL region IDs are mapped through
        ``grid.region_id_to_key`` and ``grid.stl_material_types``.
        """
        region_material_types = {
            0: self.bg_material_type,
        }

        for region_id, key in self.grid.region_id_to_key.items():
            if region_id == 0:
                continue

            if key not in self.grid.stl_material_types:
                raise RuntimeError(f"No material type available for STL solid '{key}'.")

            region_material_types[region_id] = self.grid.stl_material_types[key]

        self.region_material_types = region_material_types

    def _build_interface_transition_masks(self):
        """
        Classify primal and dual interface edges by the material types
        on their two endpoint regions.

        Transition orientation is ignored for the classification itself.
        For example, normal-to-PEC and PEC-to-normal edges both belong
        to "normal_pec".
        """
        self._build_region_material_types()

        type_codes = {
            "normal": 0,
            "pec": 1,
            "sibc": 2,
        }

        max_region_id = max(self.region_material_types)

        region_type_codes = np.empty(
            max_region_id + 1,
            dtype=np.int8,
        )

        for (
            region_id,
            material_type,
        ) in self.region_material_types.items():
            region_type_codes[region_id] = type_codes[material_type]

        normal = type_codes["normal"]
        pec = type_codes["pec"]
        sibc = type_codes["sibc"]

        self.interface_transition_masks = {
            "primal": {},
            "dual": {},
        }

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                (
                    start_regions,
                    end_regions,
                ) = self.grid._get_edge_region_ids(
                    grid_type,
                    direction,
                )

                interface_mask = self.grid.interface_edge_masks[grid_type][direction]

                start_types = region_type_codes[start_regions]

                end_types = region_type_codes[end_regions]

                normal_normal = (
                    interface_mask & (start_types == normal) & (end_types == normal)
                )

                normal_pec = interface_mask & (
                    ((start_types == normal) & (end_types == pec))
                    | ((start_types == pec) & (end_types == normal))
                )

                normal_sibc = interface_mask & (
                    ((start_types == normal) & (end_types == sibc))
                    | ((start_types == sibc) & (end_types == normal))
                )

                pec_pec = interface_mask & (start_types == pec) & (end_types == pec)

                pec_sibc = interface_mask & (
                    ((start_types == pec) & (end_types == sibc))
                    | ((start_types == sibc) & (end_types == pec))
                )

                sibc_sibc = interface_mask & (start_types == sibc) & (end_types == sibc)

                self.interface_transition_masks[grid_type][direction] = {
                    "normal_normal": normal_normal,
                    "normal_pec": normal_pec,
                    "normal_sibc": normal_sibc,
                    "pec_pec": pec_pec,
                    "pec_sibc": pec_sibc,
                    "sibc_sibc": sibc_sibc,
                }

    def _build_conformal_line_data(self):
        """
        Build solver-side line data for resolved conformal interfaces.

        For normal-normal interfaces the complete edge remains active.

        For normal-conductor interfaces, only the fraction of the edge inside
        the normal material is retained:

            normal -> conductor : fraction = alpha
            conductor -> normal : fraction = 1 - alpha

        Here, conductor means either PEC or SIBC.

        The method does not modify FIT metric or material operators. It only
        prepares the geometric line fractions required by later conformal
        discretization steps.
        """
        if not self.region_material_types:
            self._build_region_material_types()

        type_codes = {
            "normal": 0,
            "pec": 1,
            "sibc": 2,
        }

        max_region_id = max(self.region_material_types)

        region_type_codes = np.empty(
            max_region_id + 1,
            dtype=np.int8,
        )

        for (
            region_id,
            material_type,
        ) in self.region_material_types.items():
            region_type_codes[region_id] = type_codes[material_type]

        normal = type_codes["normal"]
        pec = type_codes["pec"]
        sibc = type_codes["sibc"]

        self.conformal_line_data = {
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
                (
                    start_regions,
                    end_regions,
                ) = self.grid._get_edge_region_ids(
                    grid_type,
                    direction,
                )

                data = self.grid.interface_intersections[grid_type][direction]

                alpha = np.asarray(
                    data["alpha"],
                    dtype=float,
                )

                resolved = np.asarray(
                    self.grid.resolved_interface_edge_masks[grid_type][direction],
                    dtype=bool,
                )

                if alpha.shape != resolved.shape:
                    raise RuntimeError(
                        "Conformal intersection and resolved-interface "
                        f"shapes differ for {grid_type} "
                        f"{direction}-edges."
                    )

                start_types = region_type_codes[start_regions]

                end_types = region_type_codes[end_regions]

                start_normal = start_types == normal

                end_normal = end_types == normal

                start_conductor = (start_types == pec) | (start_types == sibc)

                end_conductor = (end_types == pec) | (end_types == sibc)

                normal_normal = resolved & start_normal & end_normal

                normal_to_conductor = resolved & start_normal & end_conductor

                conductor_to_normal = resolved & start_conductor & end_normal

                supported = normal_normal | normal_to_conductor | conductor_to_normal

                # NaN deliberately marks edges for which no current
                # conformal line treatment is defined.
                effective_length_fraction = np.full(
                    alpha.shape,
                    np.nan,
                    dtype=float,
                )

                effective_length_fraction[normal_normal] = 1.0

                effective_length_fraction[normal_to_conductor] = alpha[
                    normal_to_conductor
                ]

                effective_length_fraction[conductor_to_normal] = (
                    1.0 - alpha[conductor_to_normal]
                )

                if np.any(~np.isfinite(effective_length_fraction[supported])):
                    raise RuntimeError(
                        "Non-finite conformal line fraction "
                        f"for {grid_type} "
                        f"{direction}-edges."
                    )

                if np.any(
                    (effective_length_fraction[supported] < 0.0)
                    | (effective_length_fraction[supported] > 1.0)
                ):
                    raise RuntimeError("Conformal line fraction outside [0, 1].")

                self.conformal_line_data[grid_type][direction] = {
                    "resolved": resolved,
                    "supported": supported,
                    "alpha": alpha,
                    "start_region": start_regions,
                    "end_region": end_regions,
                    "normal_normal": normal_normal,
                    "normal_to_conductor": (normal_to_conductor),
                    "conductor_to_normal": (conductor_to_normal),
                    "effective_length_fraction": (effective_length_fraction),
                }

    def _build_conformal_face_data(self):
        """
        Interpret geometric interface-face data in terms of material types.

        The grid provides geometric region-area fractions. This method
        determines which portions belong to volumetric normal materials.

        No FIT operators are modified here.
        """
        if not self.region_material_types:
            self._build_region_material_types()

        self.conformal_face_data = {
            "primal": {},
            "dual": {},
        }

        conductor_types = {
            "pec",
            "sibc",
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
                geometry = self.grid.interface_face_data[grid_type][direction]

                resolved = np.asarray(
                    geometry["resolved"],
                    dtype=bool,
                )

                supported = np.zeros_like(
                    resolved,
                    dtype=bool,
                )

                normal_normal = np.zeros_like(
                    resolved,
                    dtype=bool,
                )

                normal_pec = np.zeros_like(
                    resolved,
                    dtype=bool,
                )

                normal_sibc = np.zeros_like(
                    resolved,
                    dtype=bool,
                )

                effective_area_fraction = np.full(
                    resolved.shape,
                    np.nan,
                    dtype=float,
                )

                normal_region_ids = {}
                conductor_region_ids = {}

                for index, fractions in geometry["region_area_fractions"].items():
                    if not resolved[index]:
                        continue

                    region_ids = list(fractions.keys())

                    if len(region_ids) != 2:
                        continue

                    normal_regions = []
                    conductor_regions = []

                    for region_id in region_ids:
                        material_type = self.region_material_types[int(region_id)]

                        if material_type == "normal":
                            normal_regions.append(int(region_id))

                        elif material_type in conductor_types:
                            conductor_regions.append(int(region_id))

                    # ----------------------------------------------
                    # normal <-> normal
                    #
                    # Entire geometrical face remains active.
                    # ----------------------------------------------
                    if len(normal_regions) == 2:
                        supported[index] = True
                        normal_normal[index] = True

                        effective_area_fraction[index] = 1.0

                        normal_region_ids[index] = tuple(normal_regions)

                        continue

                    # ----------------------------------------------
                    # normal <-> PEC/SIBC
                    #
                    # Only the normal portion remains volumetrically
                    # active.
                    # ----------------------------------------------
                    if len(normal_regions) == 1 and len(conductor_regions) == 1:
                        normal_region = normal_regions[0]

                        conductor_region = conductor_regions[0]

                        supported[index] = True

                        effective_area_fraction[index] = float(fractions[normal_region])

                        normal_region_ids[index] = (normal_region,)

                        conductor_region_ids[index] = (conductor_region,)

                        conductor_type = self.region_material_types[conductor_region]

                        if conductor_type == "pec":
                            normal_pec[index] = True

                        elif conductor_type == "sibc":
                            normal_sibc[index] = True

                self.conformal_face_data[grid_type][direction] = {
                    "resolved": resolved.copy(),
                    "supported": supported,
                    "normal_normal": (normal_normal),
                    "normal_pec": normal_pec,
                    "normal_sibc": normal_sibc,
                    "effective_area_fraction": (effective_area_fraction),
                    "normal_region_ids": (normal_region_ids),
                    "conductor_region_ids": (conductor_region_ids),
                }

    def _build_conformal_face_materials(self):
        """
        Build effective material properties on resolved conformal faces.

        Material properties are averaged only over the active normal
        portion of each face. Geometric area reduction is handled
        separately through A_eff.

        Primal faces:
            mu_r

        Dual faces:
            eps_r, sigma
        """
        self.conformal_face_materials = {
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
                face_data = self.conformal_face_data[grid_type][direction]

                geometry = self.grid.interface_face_data[grid_type][direction]

                supported = face_data["supported"]

                shape = supported.shape

                if grid_type == "primal":
                    mu_r_eff = np.full(
                        shape,
                        np.nan,
                        dtype=float,
                    )

                else:
                    eps_r_eff = np.full(
                        shape,
                        np.nan,
                        dtype=float,
                    )

                    sigma_eff = np.full(
                        shape,
                        np.nan,
                        dtype=float,
                    )

                for index_array in np.argwhere(supported):
                    index = tuple(int(value) for value in index_array)

                    fractions = geometry["region_area_fractions"][index]

                    normal_regions = face_data["normal_region_ids"][index]

                    normal_fraction = sum(
                        float(fractions[region_id]) for region_id in normal_regions
                    )

                    if normal_fraction <= 0.0:
                        raise RuntimeError(
                            f"Supported {grid_type} "
                            f"{direction}-face {index} "
                            "has no positive normal-material area."
                        )

                    expected_fraction = float(
                        face_data["effective_area_fraction"][index]
                    )

                    if not np.isclose(
                        normal_fraction,
                        expected_fraction,
                        rtol=1e-10,
                        atol=1e-12,
                    ):
                        raise RuntimeError(
                            f"Inconsistent active area fraction on "
                            f"{grid_type} {direction}-face "
                            f"{index}: geometry gives "
                            f"{normal_fraction}, face data gives "
                            f"{expected_fraction}."
                        )

                    if grid_type == "primal":
                        value = 0.0

                        for region_id in normal_regions:
                            weight = float(fractions[region_id]) / normal_fraction

                            material = self._get_region_material_properties(region_id)

                            value += weight * material[1]

                        mu_r_eff[index] = value

                    else:
                        eps_value = 0.0
                        sigma_value = 0.0

                        for region_id in normal_regions:
                            weight = float(fractions[region_id]) / normal_fraction

                            material = self._get_region_material_properties(region_id)

                            eps_value += weight * material[0]

                            sigma_value += weight * material[2]

                        eps_r_eff[index] = eps_value

                        sigma_eff[index] = sigma_value

                if grid_type == "primal":
                    self.conformal_face_materials[grid_type][direction] = {
                        "mu_r": mu_r_eff,
                    }

                else:
                    self.conformal_face_materials[grid_type][direction] = {
                        "eps_r": eps_r_eff,
                        "sigma": sigma_eff,
                    }

    def _get_region_material_properties(self, region_id):
        """
        Return [eps_r, mu_r, sigma] for one geometry region.

        Region 0 is the background. All other regions correspond to STL solids.
        """
        region_id = int(region_id)

        if region_id == 0:
            material = self.background

            if isinstance(material, str):
                mat = material_lib[material.lower()]

                material = [
                    mat[0],
                    mat[1],
                    mat[2] if len(mat) == 3 else 0.0,
                ]

            else:
                material = list(material)

                if len(material) == 2:
                    material.append(0.0)

        else:
            key = self.grid.region_id_to_key[region_id]

            material = self.grid.stl_materials[key]

        return np.asarray(
            material,
            dtype=float,
        )

    def _build_region_material_properties(self):
        """
        Build numerical material-property tables indexed by region ID.
        """
        max_region_id = max(self.region_material_types)

        eps_r = np.full(
            max_region_id + 1,
            np.nan,
            dtype=float,
        )

        mu_r = np.full(
            max_region_id + 1,
            np.nan,
            dtype=float,
        )

        sigma = np.full(
            max_region_id + 1,
            np.nan,
            dtype=float,
        )

        for region_id in self.region_material_types:
            properties = self._get_region_material_properties(region_id)

            eps_r[region_id] = properties[0]
            mu_r[region_id] = properties[1]
            sigma[region_id] = properties[2]

        self.region_material_properties = {
            "eps_r": eps_r,
            "mu_r": mu_r,
            "sigma": sigma,
        }

    def _build_conformal_line_materials(self):
        """
        Build effective material values along resolved conformal edges.

        Primal edges:
            - normal-normal: harmonic eps_r and sigma
            - normal-conductor: material properties of the normal segment

        Dual edges:
            - normal-normal: harmonic mu_r
            - normal-conductor: mu_r of the normal segment

        No FIT operators are modified here.
        """
        self._build_region_material_properties()

        eps_r_regions = self.region_material_properties["eps_r"]

        mu_r_regions = self.region_material_properties["mu_r"]

        sigma_regions = self.region_material_properties["sigma"]

        self.conformal_line_materials = {
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
                line_data = self.conformal_line_data[grid_type][direction]

                alpha = line_data["alpha"]

                start_regions = line_data["start_region"]

                end_regions = line_data["end_region"]

                normal_normal = line_data["normal_normal"]

                normal_to_conductor = line_data["normal_to_conductor"]

                conductor_to_normal = line_data["conductor_to_normal"]

                shape = alpha.shape

                start_eps_r = eps_r_regions[start_regions]

                end_eps_r = eps_r_regions[end_regions]

                start_mu_r = mu_r_regions[start_regions]

                end_mu_r = mu_r_regions[end_regions]

                start_sigma = sigma_regions[start_regions]

                end_sigma = sigma_regions[end_regions]

                if grid_type == "primal":
                    eps_r_eff = np.full(
                        shape,
                        np.nan,
                        dtype=float,
                    )

                    sigma_eff = np.full(
                        shape,
                        np.nan,
                        dtype=float,
                    )

                    # ------------------------------------------
                    # Normal -> normal:
                    # harmonic average along the edge
                    # ------------------------------------------
                    eps_r_eff[normal_normal] = 1.0 / (
                        alpha[normal_normal] / start_eps_r[normal_normal]
                        + (1.0 - alpha[normal_normal]) / end_eps_r[normal_normal]
                    )

                    # Conductivity:
                    #
                    # For the present non-dispersive approximation,
                    # use the corresponding series (harmonic)
                    # conductivity. If either region is insulating,
                    # the effective DC conductivity is zero.
                    both_conductive = (
                        normal_normal & (start_sigma > 0.0) & (end_sigma > 0.0)
                    )

                    sigma_eff[normal_normal] = 0.0

                    sigma_eff[both_conductive] = 1.0 / (
                        alpha[both_conductive] / start_sigma[both_conductive]
                        + (1.0 - alpha[both_conductive]) / end_sigma[both_conductive]
                    )

                    # ------------------------------------------
                    # Normal -> PEC/SIBC
                    # ------------------------------------------
                    eps_r_eff[normal_to_conductor] = start_eps_r[normal_to_conductor]

                    sigma_eff[normal_to_conductor] = start_sigma[normal_to_conductor]

                    # ------------------------------------------
                    # PEC/SIBC -> normal
                    # ------------------------------------------
                    eps_r_eff[conductor_to_normal] = end_eps_r[conductor_to_normal]

                    sigma_eff[conductor_to_normal] = end_sigma[conductor_to_normal]

                    self.conformal_line_materials[grid_type][direction] = {
                        "eps_r": eps_r_eff,
                        "sigma": sigma_eff,
                    }

                else:
                    mu_r_eff = np.full(
                        shape,
                        np.nan,
                        dtype=float,
                    )

                    # ------------------------------------------
                    # Normal -> normal:
                    # harmonic average along dual edge
                    # ------------------------------------------
                    mu_r_eff[normal_normal] = 1.0 / (
                        alpha[normal_normal] / start_mu_r[normal_normal]
                        + (1.0 - alpha[normal_normal]) / end_mu_r[normal_normal]
                    )

                    # ------------------------------------------
                    # Normal -> PEC/SIBC
                    # ------------------------------------------
                    mu_r_eff[normal_to_conductor] = start_mu_r[normal_to_conductor]

                    # ------------------------------------------
                    # PEC/SIBC -> normal
                    # ------------------------------------------
                    mu_r_eff[conductor_to_normal] = end_mu_r[conductor_to_normal]

                    self.conformal_line_materials[grid_type][direction] = {
                        "mu_r": mu_r_eff,
                    }

    def _edge_array_to_field_component(
        self,
        edge_values,
        direction,
    ):
        """
        Map a full primal/dual edge array to the edge DOFs stored by Field.

        The full edge arrays include high-side boundary edges, whereas the
        solver stores Nx * Ny * Nz DOFs per component. Low-side boundary
        DOFs are included; high-side boundary DOFs are not.
        """
        edge_values = np.asarray(edge_values)

        expected_shapes = {
            "x": (
                self.Nx,
                self.Ny + 1,
                self.Nz + 1,
            ),
            "y": (
                self.Nx + 1,
                self.Ny,
                self.Nz + 1,
            ),
            "z": (
                self.Nx + 1,
                self.Ny + 1,
                self.Nz,
            ),
        }

        if direction not in expected_shapes:
            raise ValueError(f"Unknown edge direction '{direction}'.")

        if edge_values.shape != expected_shapes[direction]:
            raise ValueError(
                f"Expected {direction}-edge array shape "
                f"{expected_shapes[direction]}, "
                f"got {edge_values.shape}."
            )

        if direction == "x":
            return edge_values[
                :,
                : self.Ny,
                : self.Nz,
            ]

        if direction == "y":
            return edge_values[
                : self.Nx,
                :,
                : self.Nz,
            ]

        return edge_values[
            : self.Nx,
            : self.Ny,
            :,
        ]

    def _face_array_to_field_component(
        self,
        face_values,
        direction,
    ):
        """
        Map a full grid-face array to the face DOFs stored in Field.

        Full face-array shapes are:
            x: (Nx + 1, Ny, Nz)
            y: (Nx, Ny + 1, Nz)
            z: (Nx, Ny, Nz + 1)

        Field stores Nx * Ny * Nz values for every component and
        therefore contains the low-side faces.
        """
        face_values = np.asarray(face_values)

        expected_shapes = {
            "x": (
                self.Nx + 1,
                self.Ny,
                self.Nz,
            ),
            "y": (
                self.Nx,
                self.Ny + 1,
                self.Nz,
            ),
            "z": (
                self.Nx,
                self.Ny,
                self.Nz + 1,
            ),
        }

        if direction not in expected_shapes:
            raise ValueError(f"Unknown face direction '{direction}'.")

        expected_shape = expected_shapes[direction]

        if face_values.shape != expected_shape:
            raise ValueError(
                f"Expected {direction}-face array "
                f"shape {expected_shape}, "
                f"got {face_values.shape}."
            )

        if direction == "x":
            return face_values[
                : self.Nx,
                :,
                :,
            ]

        if direction == "y":
            return face_values[
                :,
                : self.Ny,
                :,
            ]

        return face_values[
            :,
            :,
            : self.Nz,
        ]

    def _dual_face_array_to_primal_edge_component(
        self,
        face_values,
        direction,
        fill_value=np.nan,
    ):
        """
        Map dual-face data to the associated stored primal-edge DOFs.

        For interior primal edges the associated dual face is shifted by one
        index in the two directions transverse to the edge:

            Ex(i,j,k) <-> dual x-face (i,   j-1, k-1)
            Ey(i,j,k) <-> dual y-face (i-1, j,   k-1)
            Ez(i,j,k) <-> dual z-face (i-1, j-1, k)

        Low-side boundary edges do not have a complete dual face inside the
        physical domain. They are therefore left at ``fill_value`` here.
        """
        face_values = np.asarray(face_values)

        expected_shapes = {
            "x": (
                self.Nx + 1,
                self.Ny,
                self.Nz,
            ),
            "y": (
                self.Nx,
                self.Ny + 1,
                self.Nz,
            ),
            "z": (
                self.Nx,
                self.Ny,
                self.Nz + 1,
            ),
        }

        if direction not in expected_shapes:
            raise ValueError(f"Unknown face direction '{direction}'.")

        if face_values.shape != expected_shapes[direction]:
            raise ValueError(
                f"Expected dual {direction}-face array shape "
                f"{expected_shapes[direction]}, "
                f"got {face_values.shape}."
            )

        dtype = np.result_type(
            face_values.dtype,
            np.asarray(fill_value).dtype,
        )

        result = np.full(
            (
                self.Nx,
                self.Ny,
                self.Nz,
            ),
            fill_value,
            dtype=dtype,
        )

        if direction == "x":
            result[:, 1:, 1:] = face_values[
                : self.Nx,
                : self.Ny - 1,
                : self.Nz - 1,
            ]

        elif direction == "y":
            result[1:, :, 1:] = face_values[
                : self.Nx - 1,
                : self.Ny,
                : self.Nz - 1,
            ]

        else:
            result[1:, 1:, :] = face_values[
                : self.Nx - 1,
                : self.Ny - 1,
                : self.Nz,
            ]

        return result

    def _primal_face_array_to_dual_edge_component(
        self,
        face_values,
        direction,
    ):
        """
        Map primal-face data to the associated stored dual-edge DOFs.

        The curl matrix C defines the magnetic DOF Hx(i,j,k) as the flux
        through the primal x-face at x[i]:

            Hx(i,j,k) <-> primal x-face (i, j, k)
            Hy(i,j,k) <-> primal y-face (i, j, k)
            Hz(i,j,k) <-> primal z-face (i, j, k)

        The high-side boundary faces (x[Nx], y[Ny], z[Nz]) have no stored
        magnetic DOF.
        """
        face_values = np.asarray(face_values)

        expected_shapes = {
            "x": (
                self.Nx + 1,
                self.Ny,
                self.Nz,
            ),
            "y": (
                self.Nx,
                self.Ny + 1,
                self.Nz,
            ),
            "z": (
                self.Nx,
                self.Ny,
                self.Nz + 1,
            ),
        }

        if direction not in expected_shapes:
            raise ValueError(f"Unknown face direction '{direction}'.")

        if face_values.shape != expected_shapes[direction]:
            raise ValueError(
                f"Expected primal {direction}-face array shape "
                f"{expected_shapes[direction]}, "
                f"got {face_values.shape}."
            )

        return self._face_array_to_field_component(
            face_values,
            direction,
        )

    def _dual_edge_array_to_field_component(
        self,
        edge_values,
        direction,
        fill_value=np.nan,
    ):
        """
        Map a full dual-edge array to the stored magnetic DOFs.

        Dual x-edge (i, j, k) connects the dual points tx[i] and tx[i+1]
        and therefore pierces the primal x-face at x[i+1]. Since Hx(i,j,k)
        belongs to the primal x-face at x[i] (see
        ``_primal_face_array_to_dual_edge_component``):

            Hx(i,j,k) <-> dual x-edge (i-1, j,   k)
            Hy(i,j,k) <-> dual y-edge (i,   j-1, k)
            Hz(i,j,k) <-> dual z-edge (i,   j,   k-1)

        The low-side DOFs (i=0, j=0 or k=0 along the edge direction) belong
        to half dual edges between the domain boundary and the first dual
        point. These are not part of the dual-edge arrays and are set to
        ``fill_value``.
        """
        edge_values = np.asarray(edge_values)

        expected_shapes = {
            "x": (self.Nx, self.Ny + 1, self.Nz + 1),
            "y": (self.Nx + 1, self.Ny, self.Nz + 1),
            "z": (self.Nx + 1, self.Ny + 1, self.Nz),
        }

        if direction not in expected_shapes:
            raise ValueError(f"Unknown edge direction '{direction}'.")

        if edge_values.shape != expected_shapes[direction]:
            raise ValueError(
                f"Expected dual {direction}-edge array shape "
                f"{expected_shapes[direction]}, "
                f"got {edge_values.shape}."
            )

        dtype = np.result_type(
            edge_values.dtype,
            np.asarray(fill_value).dtype,
        )

        result = np.full(
            (self.Nx, self.Ny, self.Nz),
            fill_value,
            dtype=dtype,
        )

        if direction == "x":
            result[1:, :, :] = edge_values[
                : self.Nx - 1,
                : self.Ny,
                : self.Nz,
            ]

        elif direction == "y":
            result[:, 1:, :] = edge_values[
                : self.Nx,
                : self.Ny - 1,
                : self.Nz,
            ]

        else:
            result[:, :, 1:] = edge_values[
                : self.Nx,
                : self.Ny,
                : self.Nz - 1,
            ]

        return result

    @staticmethod
    def _combine_conformal_diagonal_component(
        perpendicular,
        parallel,
        line_mask,
        face_mask,
        line_n2,
        face_n2,
        normal_tol=1e-12,
    ):
        """
        Combine normal and tangential effective material values into one
        diagonal Cartesian material component.

        The diagonal approximation is

            m_ii = n_i^2 * m_perp
                + (1 - n_i^2) * m_parallel

        where

            m_perp     : harmonic line value,
            m_parallel : arithmetic face value.

        Purely normal and purely tangential cases are treated explicitly so
        that missing values in the unused contribution do not produce NaNs.
        """
        perpendicular = np.asarray(
            perpendicular,
            dtype=float,
        )

        parallel = np.asarray(
            parallel,
            dtype=float,
        )

        line_mask = np.asarray(
            line_mask,
            dtype=bool,
        )

        face_mask = np.asarray(
            face_mask,
            dtype=bool,
        )

        line_n2 = np.asarray(
            line_n2,
            dtype=float,
        )

        face_n2 = np.asarray(
            face_n2,
            dtype=float,
        )

        if not (
            perpendicular.shape
            == parallel.shape
            == line_mask.shape
            == face_mask.shape
            == line_n2.shape
            == face_n2.shape
        ):
            raise ValueError(
                "Conformal diagonal material arrays must have identical shapes."
            )

        result = np.full(
            perpendicular.shape,
            np.nan,
            dtype=float,
        )

        # Prefer the normal obtained directly from the edge intersection.
        # For tangential edges without an edge crossing, use the normal
        # reconstructed from the associated cut face.
        n2 = np.where(
            np.isfinite(line_n2),
            line_n2,
            face_n2,
        )

        finite_n2 = np.isfinite(n2)

        # Numerical noise may produce values slightly outside [0, 1].
        n2 = np.clip(
            n2,
            0.0,
            1.0,
        )

        # ----------------------------------------------------------
        # General case:
        # both normal and tangential material values are available.
        # ----------------------------------------------------------
        combined = (
            line_mask
            & face_mask
            & finite_n2
            & np.isfinite(perpendicular)
            & np.isfinite(parallel)
        )

        result[combined] = (
            n2[combined] * perpendicular[combined]
            + (1.0 - n2[combined]) * parallel[combined]
        )

        # ----------------------------------------------------------
        # Pure normal component:
        #
        # n_i^2 = 1 -> m_ii = m_perp
        #
        # No face contribution is required.
        # ----------------------------------------------------------
        pure_normal = (
            line_mask
            & finite_n2
            & np.isfinite(perpendicular)
            & ((1.0 - n2) <= normal_tol)
        )

        result[pure_normal] = perpendicular[pure_normal]

        # ----------------------------------------------------------
        # Pure tangential component:
        #
        # n_i^2 = 0 -> m_ii = m_parallel
        #
        # No line crossing is required.
        # ----------------------------------------------------------
        pure_tangential = (
            face_mask & finite_n2 & np.isfinite(parallel) & (n2 <= normal_tol)
        )

        result[pure_tangential] = parallel[pure_tangential]

        return result

    def _build_conformal_diagonal_materials(
        self,
        normal_tol=1e-12,
    ):
        """
        Build diagonal effective material components for resolved
        normal-normal interfaces.

        Electric material components use

            primal edge  -> normal/harmonic contribution
            dual face    -> tangential/arithmetic contribution

        Magnetic material components use

            dual edge    -> normal/harmonic contribution
            primal face  -> tangential/arithmetic contribution

        The resulting arrays have shape (Nx, Ny, Nz) and contain NaN
        where no complete conformal normal-normal treatment is available.
        """
        self.conformal_diagonal_materials = {
            "eps_r": {},
            "mu_r": {},
            "sigma": {},
        }

        axis_index = {
            "x": 0,
            "y": 1,
            "z": 2,
        }

        for direction in (
            "x",
            "y",
            "z",
        ):
            axis = axis_index[direction]

            # ======================================================
            # Electric material:
            #
            # primal edge <-> dual face
            # ======================================================

            primal_line_data = self.conformal_line_data["primal"][direction]

            primal_line_materials = self.conformal_line_materials["primal"][direction]

            dual_face_data = self.conformal_face_data["dual"][direction]

            dual_face_materials = self.conformal_face_materials["dual"][direction]

            # ------------------------------------------------------
            # Normal / harmonic contribution
            # ------------------------------------------------------

            eps_perp = self._edge_array_to_field_component(
                primal_line_materials["eps_r"],
                direction,
            )

            sigma_perp = self._edge_array_to_field_component(
                primal_line_materials["sigma"],
                direction,
            )

            electric_line_mask = self._edge_array_to_field_component(
                primal_line_data["normal_normal"],
                direction,
            ).astype(bool)

            line_normals = np.asarray(
                self.grid.interface_intersections["primal"][direction]["normal"],
                dtype=float,
            )

            line_norm = np.linalg.norm(
                line_normals,
                axis=-1,
            )

            electric_line_n2_full = np.full(
                line_norm.shape,
                np.nan,
                dtype=float,
            )

            valid_line_normal = np.isfinite(line_norm) & (line_norm > 0.0)

            electric_line_n2_full[valid_line_normal] = (
                line_normals[..., axis][valid_line_normal]
                / line_norm[valid_line_normal]
            ) ** 2

            electric_line_n2 = self._edge_array_to_field_component(
                electric_line_n2_full,
                direction,
            )

            # ------------------------------------------------------
            # Tangential / arithmetic contribution
            # ------------------------------------------------------

            eps_parallel = self._dual_face_array_to_primal_edge_component(
                dual_face_materials["eps_r"],
                direction,
            )

            sigma_parallel = self._dual_face_array_to_primal_edge_component(
                dual_face_materials["sigma"],
                direction,
            )

            electric_face_mask = self._dual_face_array_to_primal_edge_component(
                dual_face_data["normal_normal"],
                direction,
                fill_value=False,
            ).astype(bool)

            dual_face_geometry = self.grid.interface_face_data["dual"][direction]

            face_n2_full = np.full(
                dual_face_data["normal_normal"].shape,
                np.nan,
                dtype=float,
            )

            for index, segment in dual_face_geometry["segments"].items():
                if not dual_face_data["normal_normal"][index]:
                    continue

                normal = np.asarray(
                    segment["normal"],
                    dtype=float,
                )

                norm = np.linalg.norm(normal)

                if not np.isfinite(norm) or norm <= 0.0:
                    continue

                face_n2_full[index] = (normal[axis] / norm) ** 2

            electric_face_n2 = self._dual_face_array_to_primal_edge_component(
                face_n2_full,
                direction,
            )

            # ------------------------------------------------------
            # Final diagonal electric material components
            # ------------------------------------------------------

            self.conformal_diagonal_materials["eps_r"][direction] = (
                self._combine_conformal_diagonal_component(
                    eps_perp,
                    eps_parallel,
                    electric_line_mask,
                    electric_face_mask,
                    electric_line_n2,
                    electric_face_n2,
                    normal_tol=normal_tol,
                )
            )

            self.conformal_diagonal_materials["sigma"][direction] = (
                self._combine_conformal_diagonal_component(
                    sigma_perp,
                    sigma_parallel,
                    electric_line_mask,
                    electric_face_mask,
                    electric_line_n2,
                    electric_face_n2,
                    normal_tol=normal_tol,
                )
            )

            # ======================================================
            # Magnetic material:
            #
            # dual edge <-> primal face
            # ======================================================

            dual_line_data = self.conformal_line_data["dual"][direction]

            dual_line_materials = self.conformal_line_materials["dual"][direction]

            primal_face_data = self.conformal_face_data["primal"][direction]

            primal_face_materials = self.conformal_face_materials["primal"][direction]

            # ------------------------------------------------------
            # Normal / harmonic contribution
            # ------------------------------------------------------

            mu_perp = self._dual_edge_array_to_field_component(
                dual_line_materials["mu_r"],
                direction,
            )

            magnetic_line_mask = self._dual_edge_array_to_field_component(
                dual_line_data["normal_normal"],
                direction,
                fill_value=False,
            ).astype(bool)

            line_normals = np.asarray(
                self.grid.interface_intersections["dual"][direction]["normal"],
                dtype=float,
            )

            line_norm = np.linalg.norm(
                line_normals,
                axis=-1,
            )

            magnetic_line_n2_full = np.full(
                line_norm.shape,
                np.nan,
                dtype=float,
            )

            valid_line_normal = np.isfinite(line_norm) & (line_norm > 0.0)

            magnetic_line_n2_full[valid_line_normal] = (
                line_normals[..., axis][valid_line_normal]
                / line_norm[valid_line_normal]
            ) ** 2

            magnetic_line_n2 = self._dual_edge_array_to_field_component(
                magnetic_line_n2_full,
                direction,
            )

            # ------------------------------------------------------
            # Tangential / arithmetic contribution
            # ------------------------------------------------------

            mu_parallel = self._primal_face_array_to_dual_edge_component(
                primal_face_materials["mu_r"],
                direction,
            )

            magnetic_face_mask = self._primal_face_array_to_dual_edge_component(
                primal_face_data["normal_normal"],
                direction,
            ).astype(bool)

            primal_face_geometry = self.grid.interface_face_data["primal"][direction]

            face_n2_full = np.full(
                primal_face_data["normal_normal"].shape,
                np.nan,
                dtype=float,
            )

            for index, segment in primal_face_geometry["segments"].items():
                if not primal_face_data["normal_normal"][index]:
                    continue

                normal = np.asarray(
                    segment["normal"],
                    dtype=float,
                )

                norm = np.linalg.norm(normal)

                if not np.isfinite(norm) or norm <= 0.0:
                    continue

                face_n2_full[index] = (normal[axis] / norm) ** 2

            magnetic_face_n2 = self._primal_face_array_to_dual_edge_component(
                face_n2_full,
                direction,
            )

            # ------------------------------------------------------
            # Final diagonal magnetic material component
            # ------------------------------------------------------

            self.conformal_diagonal_materials["mu_r"][direction] = (
                self._combine_conformal_diagonal_component(
                    mu_perp,
                    mu_parallel,
                    magnetic_line_mask,
                    magnetic_face_mask,
                    magnetic_line_n2,
                    magnetic_face_n2,
                    normal_tol=normal_tol,
                )
            )

    def _build_conformal_effective_lengths(self):
        """
        Build effective primal edge lengths at conformal conductor cuts.

        Normal-normal edges retain their full length. Normal-PEC/SIBC edges
        retain only the fraction of the edge in the normal region. Hidden,
        missing and unsupported interface cases keep their grid length.

        Dual edge lengths are never cut (Dey-Mittra-type treatment).
        The original grid metric self.L is not modified.
        """
        self.L_fraction = Field(self.Nx, self.Ny, self.Nz)
        self.L_eff = Field(self.Nx, self.Ny, self.Nz)

        for direction in ("x", "y", "z"):
            line_data = self.conformal_line_data["primal"][direction]

            full_fraction = np.ones(
                line_data["effective_length_fraction"].shape,
                dtype=float,
            )

            supported = line_data["supported"]

            full_fraction[supported] = line_data["effective_length_fraction"][supported]

            field_fraction = self._edge_array_to_field_component(
                full_fraction,
                direction,
            )

            setattr(
                self.L_fraction,
                f"field_{direction}",
                field_fraction.copy(),
            )

            setattr(
                self.L_eff,
                f"field_{direction}",
                self.L.to_matrix(direction) * field_fraction,
            )

    def _build_conformal_effective_areas(self):
        """
        Build effective primal face areas from conformal interface-area
        fractions. Hx(i,j,k) belongs to primal x-face (i,j,k).

        With ``GridFIT3D(conformal_stabilization="area_floor")``, faces cut
        by a PEC/SIBC wall use at least ``conformal_area_floor``
        as area fraction (relaxed Dey-Mittra), which bounds the reduction of
        the stable time step. ``A_fraction_geometric`` keeps the exact
        geometric fractions; ``n_area_floored`` counts the raised faces.

        Dual face areas are never cut (Dey-Mittra-type treatment).
        The original grid metric self.iA is not modified.
        """
        self.A_fraction = Field(self.Nx, self.Ny, self.Nz)
        self.A_fraction_geometric = Field(self.Nx, self.Ny, self.Nz)
        self.A_eff = Field(self.Nx, self.Ny, self.Nz)
        self.n_area_floored = 0

        area_floor = getattr(self, "conformal_area_floor", 0.0)

        for direction in ("x", "y", "z"):
            data = self.conformal_face_data["primal"][direction]

            # Unsupported/unresolved faces retain their original area.
            full_fraction = np.ones(
                data["supported"].shape,
                dtype=float,
            )

            supported = data["supported"]

            full_fraction[supported] = data["effective_area_fraction"][supported]

            setattr(
                self.A_fraction_geometric,
                f"field_{direction}",
                self._face_array_to_field_component(full_fraction, direction).copy(),
            )

            if area_floor > 0.0:
                conductor_cut = data["normal_pec"] | data["normal_sibc"]
                floored = conductor_cut & (full_fraction < area_floor)
                full_fraction[floored] = area_floor
                self.n_area_floored += int(
                    np.count_nonzero(
                        self._face_array_to_field_component(floored, direction)
                    )
                )

            field_fraction = self._face_array_to_field_component(
                full_fraction,
                direction,
            )

            # Grid stores inverse face areas.
            base_area = np.divide(1.0, self.iA.to_matrix(direction))

            setattr(
                self.A_fraction,
                f"field_{direction}",
                field_fraction.copy(),
            )

            setattr(
                self.A_eff,
                f"field_{direction}",
                base_area * field_fraction,
            )

    def _eliminate_small_conformal_edges(self):
        """
        Area-floor treatment: remove nearly covered conductor-cut edges.

        Primal edges cut by a PEC/SIBC wall whose active fraction is below
        ``conformal_min_edge_fraction`` are treated as inside the conductor
        (ieps = 0). With raised face areas, such edges would otherwise
        couple only weakly to the field and form spurious low-frequency
        modes. Region IDs and the stored geometry are not changed.
        """
        self.n_edges_eliminated = 0

        threshold = getattr(self, "conformal_min_edge_fraction", 0.0)

        if (
            threshold <= 0.0
            or getattr(self, "conformal_area_floor", 0.0) <= 0.0
            or self.L_fraction is None
        ):
            return

        for direction in ("x", "y", "z"):
            ieps = np.array(self.ieps.to_matrix(direction))
            tiny = (self.L_fraction.to_matrix(direction) < threshold) & (ieps != 0.0)
            ieps[tiny] = 0.0
            setattr(self.ieps, f"field_{direction}", ieps)
            self.n_edges_eliminated += int(np.count_nonzero(tiny))

    @staticmethod
    def _average_dual_points_to_primal_edges(point_values):
        """
        Average material values at dual grid points over the dual faces
        associated with the stored primal edges.

        Parameters
        ----------
        point_values : ndarray
            Material values on the dual grid points with shape
            (Nx+1, Ny+1, Nz+1).

        Returns
        -------
        values_x, values_y, values_z : ndarray
            Face-averaged material values associated with the x-, y-, and
            z-directed primal edges. Each array has shape (Nx, Ny, Nz).

        Notes
        -----
        At low-side boundaries, the dual face is truncated by the physical
        domain boundary. Missing dual-face vertices are represented by
        constant extrapolation of the nearest available dual-point value.
        """
        point_values = np.asarray(point_values)

        Nx = point_values.shape[0] - 1
        Ny = point_values.shape[1] - 1
        Nz = point_values.shape[2] - 1

        # ----------------------------------------------------------
        # x-directed primal edges -> dual yz faces
        # ----------------------------------------------------------
        values = point_values[:Nx, :, :]

        padded = np.pad(
            values,
            ((0, 0), (1, 0), (1, 0)),
            mode="edge",
        )

        values_x = 0.25 * (
            padded[:, 1 : Ny + 1, 1 : Nz + 1]
            + padded[:, :Ny, 1 : Nz + 1]
            + padded[:, :Ny, :Nz]
            + padded[:, 1 : Ny + 1, :Nz]
        )

        # ----------------------------------------------------------
        # y-directed primal edges -> dual xz faces
        # ----------------------------------------------------------
        values = point_values[:, :Ny, :]

        padded = np.pad(
            values,
            ((1, 0), (0, 0), (1, 0)),
            mode="edge",
        )

        values_y = 0.25 * (
            padded[1 : Nx + 1, :, 1 : Nz + 1]
            + padded[:Nx, :, 1 : Nz + 1]
            + padded[:Nx, :, :Nz]
            + padded[1 : Nx + 1, :, :Nz]
        )

        # ----------------------------------------------------------
        # z-directed primal edges -> dual xy faces
        # ----------------------------------------------------------
        values = point_values[:, :, :Nz]

        padded = np.pad(
            values,
            ((1, 0), (1, 0), (0, 0)),
            mode="edge",
        )

        values_z = 0.25 * (
            padded[1 : Nx + 1, 1 : Ny + 1, :]
            + padded[:Nx, 1 : Ny + 1, :]
            + padded[:Nx, :Ny, :]
            + padded[1 : Nx + 1, :Ny, :]
        )

        return values_x, values_y, values_z

    @staticmethod
    def _average_primal_points_to_dual_edges(point_values):
        """
        Average material values at primal grid points over the primal faces
        associated with the stored dual edges.

        Parameters
        ----------
        point_values : ndarray
            Material values on the primal grid points with shape
            (Nx+1, Ny+1, Nz+1).

        Returns
        -------
        values_x, values_y, values_z : ndarray
            Face-averaged material values associated with the x-, y-, and
            z-directed dual edges. Each array has shape (Nx, Ny, Nz).
        """
        point_values = np.asarray(point_values)

        # Hx(i,j,k) -> primal yz face at x[i] (see curl matrix C)
        values_x = 0.25 * (
            point_values[:-1, :-1, :-1]
            + point_values[:-1, 1:, :-1]
            + point_values[:-1, 1:, 1:]
            + point_values[:-1, :-1, 1:]
        )

        # Hy(i,j,k) -> primal xz face at y[j]
        values_y = 0.25 * (
            point_values[:-1, :-1, :-1]
            + point_values[1:, :-1, :-1]
            + point_values[1:, :-1, 1:]
            + point_values[:-1, :-1, 1:]
        )

        # Hz(i,j,k) -> primal xy face at z[k]
        values_z = 0.25 * (
            point_values[:-1, :-1, :-1]
            + point_values[1:, :-1, :-1]
            + point_values[1:, 1:, :-1]
            + point_values[:-1, 1:, :-1]
        )

        return values_x, values_y, values_z

    def _apply_stl_materials_conformal(self):
        """
        Assign STL materials using primal and dual point masks.

        Electric permittivity and conductivity are averaged over the
        dual faces associated with the stored primal electric edges.
        Magnetic permeability is averaged over the primal faces associated
        with the stored dual magnetic edges.
        """
        self.stl_solids = self.grid.stl_solids
        self.stl_materials = self.grid.stl_materials
        self.stl_colors = self.grid.stl_colors

        # SIBC currently relies on cell-based surface masks and is therefore
        # not yet compatible with the conformal material discretization.
        if self.use_sibc and any(
            self.stl_materials[key][2] > 0.0 for key in self.stl_solids.keys()
        ):
            raise NotImplementedError(
                "SIBC for conductive STL materials is currently not supported "
                "with geometry_mode='conformal'."
            )

        shape = (
            self.Nx + 1,
            self.Ny + 1,
            self.Nz + 1,
        )

        # Initialize point-wise material values with the background material.
        eps_dual = np.full(
            shape,
            self.eps_bg,
            dtype=self.dtype,
        )

        sigma_dual = np.full(
            shape,
            self.sigma_bg,
            dtype=self.dtype,
        )

        mu_primal = np.full(
            shape,
            self.mu_bg,
            dtype=self.dtype,
        )

        # Assign STL material values to primal and dual grid points.
        for key in self.stl_solids.keys():
            primal_mask = self.grid.primal_point_masks[key]
            dual_mask = self.grid.dual_point_masks[key]

            eps = self.stl_materials[key][0] * eps_0
            mu = self.stl_materials[key][1] * mu_0
            sigma = self.stl_materials[key][2]

            # TODO: Temporary workaround for FD SIBC implementation.
            # Treat SIBC regions as PEC in the volumetric material assignment.
            # The physical conductivity remains stored in stl_materials
            # and will later be used to calculate Z_s(omega).
            #
            # This does not fix the underlying conformal material averaging.
            if self.region_material_types[self.grid.region_key_to_id[key]] == "sibc":
                eps = np.inf
                sigma = 0.0

            eps_dual[dual_mask] = eps
            sigma_dual[dual_mask] = sigma
            mu_primal[primal_mask] = mu

            if sigma > 0.0:
                self.use_conductivity = True

                if sigma > 10 * eps / eps_0:
                    print(
                        f"[!] Warning: High conductivity sigma={sigma} S/m "
                        f"for solid '{key}' with low permittivity "
                        f"epsilon_r={eps / eps_0} will considerably reduce "
                        f"the maximal stable timestep."
                    )

        # Average material values over the corresponding FIT faces.
        eps_x, eps_y, eps_z = self._average_dual_points_to_primal_edges(eps_dual)

        sigma_x, sigma_y, sigma_z = self._average_dual_points_to_primal_edges(
            sigma_dual
        )

        mu_x, mu_y, mu_z = self._average_primal_points_to_dual_edges(mu_primal)

        # ----------------------------------------------------------
        # Replace the point-averaged material values by the
        # conformal diagonal approximation wherever a resolved
        # normal-normal interface treatment is available.
        #
        # All other DOFs retain the point-averaged values above.
        # ----------------------------------------------------------

        eps_components = {
            "x": eps_x,
            "y": eps_y,
            "z": eps_z,
        }

        mu_components = {
            "x": mu_x,
            "y": mu_y,
            "z": mu_z,
        }

        sigma_components = {
            "x": sigma_x,
            "y": sigma_y,
            "z": sigma_z,
        }

        # ----------------------------------------------------------
        # Geometry-based constitutive assignment (conformal FIT).
        #
        # Point averaging is retained only as a fallback for geometry
        # configurations not yet represented by resolved interface data.
        # For uncut edges with one normal region, use that region's
        # material; for conductor-conductor primal edges, set ieps=0.
        # For resolved conductor interfaces, use the *normal* part.
        # Normal-normal material interfaces are handled afterwards by
        # the existing diagonal-tensor approximation.
        # ----------------------------------------------------------
        if not self.region_material_types:
            self._build_region_material_types()
        if not hasattr(self, "region_material_properties"):
            self._build_region_material_properties()

        eps_regions = self.region_material_properties["eps_r"]
        mu_regions = self.region_material_properties["mu_r"]
        sigma_regions = self.region_material_properties["sigma"]
        region_is_normal = np.zeros(len(eps_regions), dtype=bool)
        region_is_conductor = np.zeros(len(eps_regions), dtype=bool)
        for region_id, kind in self.region_material_types.items():
            region_is_normal[region_id] = kind == "normal"
            region_is_conductor[region_id] = kind in ("pec", "sibc")

        for direction in ("x", "y", "z"):
            # Electric DOFs correspond to primal edges.
            start, end = self.grid._get_edge_region_ids("primal", direction)
            start = self._edge_array_to_field_component(start, direction)
            end = self._edge_array_to_field_component(end, direction)

            same_normal = (start == end) & region_is_normal[start]
            fully_conductor = region_is_conductor[start] & region_is_conductor[end]

            # Assign the actual region material, not a four-point
            # average that can be contaminated by a nearby PEC point.
            eps_components[direction][same_normal] = (
                eps_regions[start[same_normal]] * eps_0
            )
            sigma_components[direction][same_normal] = sigma_regions[start[same_normal]]

            # Interior PEC/SIBC electric DOFs are eliminated through
            # ieps=0, consistent with the existing material mask.
            eps_components[direction][fully_conductor] = np.inf
            sigma_components[direction][fully_conductor] = 0.0

            primal_data = self.conformal_line_data.get("primal", {}).get(direction)
            primal_mat = self.conformal_line_materials.get("primal", {}).get(direction)
            if primal_data is not None and primal_mat is not None:
                crossing_full = (
                    primal_data["normal_to_conductor"]
                    | primal_data["conductor_to_normal"]
                )
                crossing = self._edge_array_to_field_component(
                    crossing_full, direction
                ).astype(bool)
                line_eps = self._edge_array_to_field_component(
                    primal_mat["eps_r"], direction
                )
                line_sigma = self._edge_array_to_field_component(
                    primal_mat["sigma"], direction
                )
                valid = (
                    crossing
                    & np.isfinite(line_eps)
                    & (line_eps > 0.0)
                    & np.isfinite(line_sigma)
                    & (line_sigma >= 0.0)
                )
                eps_components[direction][valid] = line_eps[valid] * eps_0
                sigma_components[direction][valid] = line_sigma[valid]

            # Magnetic DOFs correspond to dual edges and primal faces.
            dual_start, dual_end = self.grid._get_edge_region_ids("dual", direction)
            # Low-side half dual edges (fill -1) keep the averaged value.
            dual_start = self._dual_edge_array_to_field_component(
                dual_start, direction, fill_value=-1
            )
            dual_end = self._dual_edge_array_to_field_component(
                dual_end, direction, fill_value=-1
            )
            has_dual_edge = dual_start >= 0
            dual_same_normal = has_dual_edge & (dual_start == dual_end)
            dual_same_normal[has_dual_edge] &= region_is_normal[
                dual_start[has_dual_edge]
            ]
            mu_components[direction][dual_same_normal] = (
                mu_regions[dual_start[dual_same_normal]] * mu_0
            )

            primal_face = self.conformal_face_data.get("primal", {}).get(direction)
            primal_face_mat = self.conformal_face_materials.get("primal", {}).get(
                direction
            )
            if primal_face is not None and primal_face_mat is not None:
                cut_full = primal_face["normal_pec"] | primal_face["normal_sibc"]
                cut = self._primal_face_array_to_dual_edge_component(
                    cut_full, direction
                ).astype(bool)
                face_mu = self._primal_face_array_to_dual_edge_component(
                    primal_face_mat["mu_r"], direction
                )
                valid = cut & np.isfinite(face_mu) & (face_mu > 0.0)
                mu_components[direction][valid] = face_mu[valid] * mu_0

        # The normal-normal diagonal approximation below deliberately
        # takes precedence over the assignments above. Its purpose is
        # to approximate anisotropic dielectric/magnetic interfaces.

        diagonal_materials = getattr(
            self,
            "conformal_diagonal_materials",
            None,
        )

        if diagonal_materials:
            expected_shape = (
                self.Nx,
                self.Ny,
                self.Nz,
            )

            for direction in (
                "x",
                "y",
                "z",
            ):
                # --------------------------------------------------
                # Permittivity
                # --------------------------------------------------
                eps_r_diag = diagonal_materials["eps_r"].get(
                    direction,
                    None,
                )

                if eps_r_diag is not None:
                    eps_r_diag = np.asarray(
                        eps_r_diag,
                        dtype=float,
                    )

                    if eps_r_diag.shape != expected_shape:
                        raise RuntimeError(
                            f"Conformal eps_r {direction}-component "
                            f"has shape {eps_r_diag.shape}, "
                            f"expected {expected_shape}."
                        )

                    valid_eps = np.isfinite(eps_r_diag) & (eps_r_diag > 0.0)

                    eps_components[direction][valid_eps] = eps_r_diag[valid_eps] * eps_0

                # --------------------------------------------------
                # Permeability
                # --------------------------------------------------
                mu_r_diag = diagonal_materials["mu_r"].get(
                    direction,
                    None,
                )

                if mu_r_diag is not None:
                    mu_r_diag = np.asarray(
                        mu_r_diag,
                        dtype=float,
                    )

                    if mu_r_diag.shape != expected_shape:
                        raise RuntimeError(
                            f"Conformal mu_r {direction}-component "
                            f"has shape {mu_r_diag.shape}, "
                            f"expected {expected_shape}."
                        )

                    valid_mu = np.isfinite(mu_r_diag) & (mu_r_diag > 0.0)

                    mu_components[direction][valid_mu] = mu_r_diag[valid_mu] * mu_0

                # --------------------------------------------------
                # Conductivity
                # --------------------------------------------------
                sigma_diag = diagonal_materials["sigma"].get(
                    direction,
                    None,
                )

                if sigma_diag is not None:
                    sigma_diag = np.asarray(
                        sigma_diag,
                        dtype=float,
                    )

                    if sigma_diag.shape != expected_shape:
                        raise RuntimeError(
                            f"Conformal sigma {direction}-component "
                            f"has shape {sigma_diag.shape}, "
                            f"expected {expected_shape}."
                        )

                    valid_sigma = np.isfinite(sigma_diag) & (sigma_diag >= 0.0)

                    sigma_components[direction][valid_sigma] = sigma_diag[valid_sigma]

        # Convert physical material values to the quantities used
        # in the explicit FIT update equations.
        self.ieps.field_x = 1.0 / eps_components["x"]
        self.ieps.field_y = 1.0 / eps_components["y"]
        self.ieps.field_z = 1.0 / eps_components["z"]

        self.imu.field_x = 1.0 / mu_components["x"]
        self.imu.field_y = 1.0 / mu_components["y"]
        self.imu.field_z = 1.0 / mu_components["z"]

        self.sigma.field_x = sigma_components["x"]
        self.sigma.field_y = sigma_components["y"]
        self.sigma.field_z = sigma_components["z"]
