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
  maps, edge/face intersections and partial face areas.
- ``ConformalSolverMixin`` is mixed into ``SolverFIT3D`` and turns the grid
  data into effective lengths/areas and conformal material tensors.

Both are only used when ``geometry_mode="conformal"``. The grid points
are classified together with the staircase masks in
``GridFIT3D._mark_cells_in_stl``, which then calls
``_build_conformal_interfaces``; the solver entry point is
``SolverFIT3D._apply_stl_materials_conformal``.
"""

import warnings

import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0

# Import only the needed VTK modules: the full ``import vtk`` loads modules
# whose factory overrides can change the results of PyVista filters used
# elsewhere in Wakis (see ImpedanCEI/Wakis#123).
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

from .field import Field
from .materials import material_lib


class ConformalGridMixin:
    """Conformal STL intersection handling for ``GridFIT3D``."""

    def _get_stabilization_fraction(self):
        """Fraction of the active small-cut-cell treatment (None if off)."""
        if self.conformal_stabilization == "area_floor":
            return self.conformal_area_floor
        return None

    def _set_conformal_stabilization(self, method, fraction, min_edge_fraction):
        """
        Validate and store the small-cut-cell treatment.

        Sets ``conformal_stabilization`` ("none" or "area_floor"),
        ``conformal_area_floor`` and ``conformal_min_edge_fraction`` (used
        by the solver for the area floor). Unused values are 0.
        """
        method = "none" if method is None else str(method).lower()

        if method not in ("none", "area_floor"):
            raise ValueError("conformal_stabilization must be None or 'area_floor'.")

        fraction = 0.1 if fraction is None else float(fraction)
        min_edge_fraction = (
            min(fraction, 0.49)
            if min_edge_fraction is None
            else float(min_edge_fraction)
        )

        self.conformal_stabilization = method
        self.conformal_area_floor = 0.0
        self.conformal_min_edge_fraction = 0.0

        if method == "area_floor":
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
        return self.tx, self.ty, self.tz

    def _build_region_maps(self):
        """
        Assign a unique region ID to the background and every STL solid.

        Region 0 is reserved for the background. STL solids are assigned
        IDs 1, 2, ... following the order of ``self.stl_solids``.
        """
        self.region_id_to_key = {0: None}
        self.region_key_to_id = {}

        for region_id, key in enumerate(self.stl_solids.keys(), start=1):
            self.region_id_to_key[region_id] = key
            self.region_key_to_id[key] = region_id

    def _build_region_ids(self, node_masks, grid_type):
        """
        Build unique region IDs from node masks on one grid.

        Later STL solids intentionally overwrite earlier solids in
        overlapping regions.

        Parameters
        ----------
        node_masks : dict
            Boolean node masks indexed by STL key.
        grid_type : str
            Either "primal" or "dual".

        Returns
        -------
        region_ids : ndarray of int
            Unique region ID at every grid point.
        """
        x, y, z = self._get_grid_coordinates(grid_type)
        expected_shape = (len(x), len(y), len(z))
        region_ids = np.zeros(expected_shape, dtype=np.int32)

        for key in self.stl_solids.keys():
            node_mask = np.asarray(node_masks[key], dtype=bool)
            region_id = self.region_key_to_id[key]

            # Later STL solids overwrite earlier ones.
            region_ids[node_mask] = region_id

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
            self.primal_node_masks, "primal"
        )

        self.region_ids["dual"] = self._build_region_ids(self.dual_node_masks, "dual")

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
        region_ids = self.region_ids[grid_type]

        if direction == "x":
            return (region_ids[:-1, :, :], region_ids[1:, :, :])
        if direction == "y":
            return (region_ids[:, :-1, :], region_ids[:, 1:, :])
        return (region_ids[:, :, :-1], region_ids[:, :, 1:])

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
        i, j, k = (int(value) for value in index)
        start_point = np.array([x[i], y[j], z[k]], dtype=float)
        start_region = int(region_ids[i, j, k])

        if direction == "x":
            end_point = np.array([x[i + 1], y[j], z[k]], dtype=float)
            end_region = int(region_ids[i + 1, j, k])
        elif direction == "y":
            end_point = np.array([x[i], y[j + 1], z[k]], dtype=float)
            end_region = int(region_ids[i, j + 1, k])
        else:
            end_point = np.array([x[i], y[j], z[k + 1]], dtype=float)
            end_region = int(region_ids[i, j, k + 1])

        return (start_point, end_point, start_region, end_region)

    def _locate_intersection_edge(
        self, grid_type, direction, point, alpha_tolerance=1e-8
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

        axis = {"x": 0, "y": 1, "z": 2}[direction]
        coordinate = float(point[axis])
        grid_coordinates = coordinates[direction]

        edge_index = int(
            np.searchsorted(grid_coordinates, coordinate, side="right") - 1
        )

        # A hit exactly on the last grid point belongs to the final edge.
        edge_index = min(edge_index, len(grid_coordinates) - 2)

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

        alpha = float(np.clip(alpha, 0.0, 1.0))

        return edge_index, alpha

    @staticmethod
    def _get_triangle_normal(surface, cell_id):
        """
        Compute the unit normal of one triangular STL facet.
        """
        cell = surface.get_cell(int(cell_id))
        points = np.asarray(cell.points, dtype=float)

        if points.shape != (3, 3):
            raise RuntimeError(
                f"Expected triangular STL cell, got "
                f"{points.shape[0]} points for cell {cell_id}."
            )

        normal = np.cross(points[1] - points[0], points[2] - points[0])
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

        line_extension = np.min(np.diff(axis_coordinates))

        if direction == "x":
            j, k = transverse_index
            start_point = np.array([x[0] - line_extension, y[j], z[k]], dtype=float)
            end_point = np.array([x[-1] + line_extension, y[j], z[k]], dtype=float)
        elif direction == "y":
            i, k = transverse_index
            start_point = np.array([x[i], y[0] - line_extension, z[k]], dtype=float)
            end_point = np.array([x[i], y[-1] + line_extension, z[k]], dtype=float)
        else:
            i, j = transverse_index
            start_point = np.array([x[i], y[j], z[0] - line_extension], dtype=float)
            end_point = np.array([x[i], y[j], z[-1] + line_extension], dtype=float)

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
                locator = vtkStaticCellLocator()
                locator.SetDataSet(surface)
                locator.BuildLocator()
                self._conformal_stl_locators[key] = locator

            locator = self._conformal_stl_locators[key]

            # Intersect the complete grid line with the STL.
            vtk_points = vtkPoints()
            vtk_cell_ids = vtkIdList()

            locator.IntersectWithLine(
                start_point, end_point, 1e-8, vtk_points, vtk_cell_ids
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
                np.asarray(points, dtype=float), np.asarray(cell_ids, dtype=int)
            ):
                result = self._locate_intersection_edge(
                    grid_type, direction, point, alpha_tolerance=alpha_tolerance
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

                normal = self._get_triangle_normal(surface, cell_id)

                hit = {
                    "alpha": alpha,
                    "surface_id": int(surface_id),
                    "cell_id": int(cell_id),
                    "normal": normal,
                }

                edge_hits.setdefault(edge_index, []).append(hit)

        return edge_hits

    def _get_edge_shape_and_transverse_indices(self, grid_type, direction):
        x, y, z = self._get_grid_coordinates(grid_type)
        nx = len(x)
        ny = len(y)
        nz = len(z)

        if direction == "x":
            shape = (nx - 1, ny, nz)
            transverse_indices = ((j, k) for j in range(ny) for k in range(nz))
        elif direction == "y":
            shape = (nx, ny - 1, nz)
            transverse_indices = ((i, k) for i in range(nx) for k in range(nz))
        else:
            shape = (nx, ny, nz - 1)
            transverse_indices = ((i, j) for i in range(nx) for j in range(ny))

        return shape, transverse_indices

    @staticmethod
    def _group_intersection_hits(hits, alpha_tolerance=1e-8):
        """
        Group ray-tracing hits at the same geometric position.
        """
        if not hits:
            return []

        hits = sorted(hits, key=lambda hit: hit["alpha"])
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
        hit_groups, direction, normal_projection_tolerance=1e-8
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
        axis_index = "xyz".index(direction)
        crossing_groups = []
        contact_groups = []

        for group in hit_groups:
            is_crossing = any(
                abs(float(np.asarray(hit["normal"], dtype=float)[axis_index]))
                > normal_projection_tolerance
                for hit in group
            )

            if is_crossing:
                crossing_groups.append(group)
            else:
                contact_groups.append(group)

        return crossing_groups, contact_groups

    # Interior pieces of an edge shorter than this fraction of the edge are
    # not treated as material of their own. Coincident surfaces of different
    # STL solids (the same wall triangulated twice) leave such slivers, as
    # do gaps and overlaps far below the cell size; neither can be resolved
    # on the grid. Measured spreads of coincident surfaces are a few percent
    # of an edge, real thin layers are much wider.
    _SLIVER_FRACTION = 0.1

    def _resolve_edge_interfaces(self, crossing_groups, start_inside):
        """
        Reconstruct the material along one edge and return its interfaces.

        Every crossing of a solid's surface toggles whether the edge is
        inside that solid; the material at a point is the last-listed solid
        containing it (later solids overwrite earlier ones), or the
        background. Interior pieces shorter than ``_SLIVER_FRACTION`` of the
        edge are removed, so that their two bounding crossings form a single
        interface. Crossings of surfaces covered by a later solid do not
        change the material and drop out.

        Parameters
        ----------
        crossing_groups : list
            Crossing groups of the edge, sorted by position.
        start_inside : dict
            Region ID -> whether the edge start lies inside that STL solid.

        Returns
        -------
        regions : list of int
            Material regions along the edge (first = start, last = end).
        interfaces : list of list
            For each change between consecutive regions, the crossing groups
            that form it.
        thin_layer : bool
            True if a removed sliver is bounded on both sides by the same
            surface: a real layer thinner than the sliver limit (a coating,
            a clipped corner), not the coincident walls of two solids.
        """
        inside = dict(start_inside)

        def region():
            return max((rid for rid, flag in inside.items() if flag), default=0)

        regions = [region()]
        positions = [0.0]
        interfaces = []

        for group in crossing_groups:
            for surface_id in {hit["surface_id"] for hit in group}:
                inside[surface_id] = not inside[surface_id]
            positions.append(float(np.mean([hit["alpha"] for hit in group])))
            regions.append(region())
            interfaces.append([group])

        positions.append(1.0)
        widths = list(np.diff(positions))

        def merge_equal_neighbours():
            t = 0
            while t < len(regions) - 1:
                if regions[t] == regions[t + 1]:
                    widths[t] += widths.pop(t + 1)
                    regions.pop(t + 1)
                    interfaces.pop(t)
                else:
                    t += 1

        def surfaces(groups):
            return {hit["surface_id"] for group in groups for hit in group}

        merge_equal_neighbours()
        thin_layer = False

        while len(regions) > 2:
            t = min(range(1, len(regions) - 1), key=lambda t: widths[t])
            if widths[t] >= self._SLIVER_FRACTION:
                break

            thin_layer |= bool(surfaces(interfaces[t - 1]) & surfaces(interfaces[t]))
            # The sliver's two bounding crossings form one interface.
            interfaces[t - 1] = interfaces[t - 1] + interfaces.pop(t)
            widths[t - 1] += 0.5 * widths[t]
            widths[t + 1] += 0.5 * widths[t]
            widths.pop(t)
            regions.pop(t)
            merge_equal_neighbours()

        return regions, interfaces, thin_layer

    def _select_interface_group(self, groups, left_region, right_region):
        """
        Choose the crossing that represents one interface.

        A conductor surface defines the cut wall of a normal-conductor
        interface; otherwise the surface of the later solid wins, as it does
        in the point classification. Remaining ties go to the first crossing.
        """

        def priority(group):
            sides = {
                surface_id
                for surface_id in {hit["surface_id"] for hit in group}
                if surface_id in (left_region, right_region)
            }
            conductor = any(
                self._is_conductor_region(surface_id) for surface_id in sides
            )
            return (not conductor, -max(sides, default=-1))

        return min(groups, key=priority)

    def _select_representative_hit(self, hits):
        """
        Select one representative STL facet from coincident hits.

        SIBC and PEC surfaces are preferred because their local facet
        normals are particularly relevant for boundary-condition treatment.
        """
        material_priority = {"sibc": 0, "pec": 1, "normal": 2}

        def priority(hit):
            surface_id = hit["surface_id"]
            key = self.region_id_to_key[surface_id]
            material_type = self.stl_material_types[key]

            return (material_priority[material_type], surface_id, hit["cell_id"])

        return min(hits, key=priority)

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

    def _build_interface_intersections(self, alpha_tolerance=1e-8):
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
        self.interface_intersections = {"primal": {}, "dual": {}}

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                (shape, transverse_indices) = (
                    self._get_edge_shape_and_transverse_indices(grid_type, direction)
                )

                (start_regions, end_regions) = self._get_edge_region_ids(
                    grid_type, direction
                )

                # One representative crossing per edge; pure tangential
                # contacts keep the NaN / -1 defaults.
                intersections = {
                    "alpha": np.full(shape, np.nan, dtype=float),
                    "surface_id": np.full(shape, -1, dtype=np.int32),
                    "normal": np.full(shape + (3,), np.nan, dtype=float),
                    "has_crossing": np.zeros(shape, dtype=bool),
                    "multiple_crossing_positions": np.zeros(shape, dtype=bool),
                    # Diagnostics: material structure thinner than the mesh,
                    # and crossings contradicting the point classification.
                    "thin_feature": np.zeros(shape, dtype=bool),
                    "inconsistent": np.zeros(shape, dtype=bool),
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

                        raw_hits.setdefault(edge_index, []).extend(hits)

                node_masks = (
                    self.primal_node_masks
                    if grid_type == "primal"
                    else self.dual_node_masks
                )

                for edge_index, hits in raw_hits.items():
                    hit_groups = self._group_intersection_hits(
                        hits, alpha_tolerance=alpha_tolerance
                    )
                    crossing_groups, _ = self._split_intersection_groups(
                        hit_groups, direction
                    )
                    if not crossing_groups:
                        continue

                    start = int(start_regions[edge_index])
                    end = int(end_regions[edge_index])
                    start_inside = {
                        self.region_key_to_id[key]: bool(node_masks[key][edge_index])
                        for key in self.stl_solids
                    }
                    regions, interfaces, thin_layer = self._resolve_edge_interfaces(
                        crossing_groups, start_inside
                    )

                    # Crossings that contradict the point classification (an
                    # open surface, a crossing on a grid point) cannot be
                    # resolved: keep all of them as one ambiguous interface.
                    consistent = regions[0] == start and regions[-1] == end
                    if not consistent:
                        regions, interfaces = [start, end], [crossing_groups]
                        intersections["inconsistent"][edge_index] = True

                    # Only covered surfaces or slivers: no material change.
                    if not interfaces:
                        intersections["thin_feature"][edge_index] = thin_layer
                        continue

                    # A layer thinner than the mesh: a thin coating, a feature
                    # inside one material (same material at both ends), or
                    # several interfaces of which only one is represented.
                    intersections["thin_feature"][edge_index] = consistent and (
                        thin_layer or start == end or len(interfaces) > 1
                    )

                    # Same material at both ends: no interface to represent.
                    if start == end:
                        continue

                    candidates = [
                        self._select_interface_group(groups, left, right)
                        for groups, left, right in zip(
                            interfaces, regions[:-1], regions[1:]
                        )
                    ]
                    # Several interfaces in one edge: only one can be
                    # represented. A conductor endpoint keeps its own wall
                    # (a coating on it is dropped, not the wall moved);
                    # otherwise use the interface nearest the edge midpoint.
                    if self._is_conductor_region(end):
                        selected_group = candidates[-1]
                    elif self._is_conductor_region(start):
                        selected_group = candidates[0]
                    else:
                        selected_group = min(
                            candidates,
                            key=lambda group: abs(
                                np.mean([hit["alpha"] for hit in group]) - 0.5
                            ),
                        )
                    multiple = len(interfaces) > 1 or not consistent
                    representative = self._select_representative_hit(selected_group)
                    intersections["has_crossing"][edge_index] = True
                    intersections["multiple_crossing_positions"][edge_index] = multiple
                    intersections["alpha"][edge_index] = representative["alpha"]
                    intersections["surface_id"][edge_index] = representative[
                        "surface_id"
                    ]
                    intersections["normal"][edge_index] = representative["normal"]

                self.interface_intersections[grid_type][direction] = intersections

                if self.verbose:
                    print(
                        f"    * {grid_type} {direction}-edges: "
                        f"{np.count_nonzero(intersections['has_crossing'])} "
                        "crossings, "
                        f"{np.count_nonzero(intersections['multiple_crossing_positions'])}"
                        " with multiple crossing positions"
                    )
        self._build_resolved_interface_edge_masks()

    def _build_resolved_interface_edge_masks(self):
        """
        Build the interface masks used by the conformal FIT discretization.

        Every resolved interface is an interior geometrical crossing:

            endpoint material transition
            AND
            geometric STL crossing
        """
        self.resolved_interface_edge_masks = {"primal": {}, "dual": {}}

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                endpoint_mask = self.interface_edge_masks[grid_type][direction]

                has_crossing = self.interface_intersections[grid_type][direction][
                    "has_crossing"
                ]

                self.resolved_interface_edge_masks[grid_type][direction] = (
                    endpoint_mask & has_crossing
                )

    def _get_region_material_type(self, region_id):
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

    def _is_conductor_region(self, region_id):
        # A background without a material type is treated as normal.
        if region_id == 0 and self.bg_material_type is None:
            return False
        return self._get_region_material_type(region_id) in ("pec", "sibc")

    def _get_edge_diagnostic_masks(self, grid_type, direction):
        """
        Edges on which the conformal geometry is approximated or unreliable.

        Returns
        -------
        thin_feature : ndarray of bool
            Edges containing material structure thinner than the mesh: a
            layer inside one material, a thin coating, or several interfaces
            of which only one is represented.
        inconsistent : ndarray of bool
            Edges where STL crossings and point classification disagree: a
            material transition without a crossing, or crossings that do not
            reproduce the endpoint materials (e.g. open surfaces, surfaces
            through grid points). These edges keep the staircase material
            or an ambiguous cut.
        """
        data = self.interface_intersections[grid_type][direction]
        endpoint_mask = self.interface_edge_masks[grid_type][direction]

        # Dual intersections only exist for normal-normal edges.
        if grid_type == "dual":
            endpoint_mask = endpoint_mask & self._get_normal_normal_edge_mask(
                grid_type, direction
            )

        missing = endpoint_mask & ~data["has_crossing"]
        return data["thin_feature"], missing | data["inconsistent"]

    def _warn_interface_diagnostics(self):
        """
        Warn about edges with geometry thinner than the mesh resolution and
        edges where the STL crossings and point classification disagree.
        """
        thin = inconsistent = 0

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                thin_mask, inconsistent_mask = self._get_edge_diagnostic_masks(
                    grid_type, direction
                )
                thin += np.count_nonzero(thin_mask)
                inconsistent += np.count_nonzero(inconsistent_mask)

        messages = []
        if thin:
            messages.append(
                f"{thin} edges contain material layers thinner than the mesh "
                "resolution; at most one interface per edge is represented. "
                "Refine the mesh there if these layers matter."
            )
        if inconsistent:
            messages.append(
                f"{inconsistent} edges have STL crossings that disagree with the "
                "grid point classification and receive no or an ambiguous "
                "conformal correction. Check that the STL solids are closed "
                "and not tangent to grid points."
            )
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
        self.face_candidate_masks = {"primal": {}, "dual": {}}

        for grid_type in ("primal", "dual"):
            edge_masks = self.resolved_interface_edge_masks[grid_type]
            x_edges = np.asarray(edge_masks["x"], dtype=bool)
            y_edges = np.asarray(edge_masks["y"], dtype=bool)
            z_edges = np.asarray(edge_masks["z"], dtype=bool)

            # x-normal faces (yz)
            #
            # Boundary:
            #   two y-directed edges at z_k and z_{k+1}
            #   two z-directed edges at y_j and y_{j+1}
            #
            # Shape:
            #   (nx, ny - 1, nz - 1) for nx, ny, nz grid points
            x_faces = (
                y_edges[:, :, :-1]
                | y_edges[:, :, 1:]
                | z_edges[:, :-1, :]
                | z_edges[:, 1:, :]
            )

            # y-normal faces (xz)
            #
            # Boundary:
            #   two x-directed edges at z_k and z_{k+1}
            #   two z-directed edges at x_i and x_{i+1}
            #
            # Shape:
            #   (nx - 1, ny, nz - 1)
            y_faces = (
                x_edges[:, :, :-1]
                | x_edges[:, :, 1:]
                | z_edges[:-1, :, :]
                | z_edges[1:, :, :]
            )

            # z-normal faces (xy)
            #
            # Boundary:
            #   two x-directed edges at y_j and y_{j+1}
            #   two y-directed edges at x_i and x_{i+1}
            #
            # Shape:
            #   (nx - 1, ny - 1, nz)
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

    def _get_face_boundary_edges(self, grid_type, face_direction, index):
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
        i, j, k = (int(value) for value in index)

        if face_direction == "x":
            # Local coordinates:
            # u = y, v = z
            #
            # e_y x e_z = e_x
            return [
                {"direction": "y", "index": (i, j, k), "boundary_sign": 1},
                {"direction": "z", "index": (i, j + 1, k), "boundary_sign": 1},
                {"direction": "y", "index": (i, j, k + 1), "boundary_sign": -1},
                {"direction": "z", "index": (i, j, k), "boundary_sign": -1},
            ]

        if face_direction == "y":
            # Local coordinates:
            # u = z, v = x
            #
            # e_z x e_x = e_y
            return [
                {"direction": "z", "index": (i, j, k), "boundary_sign": 1},
                {"direction": "x", "index": (i, j, k + 1), "boundary_sign": 1},
                {"direction": "z", "index": (i + 1, j, k), "boundary_sign": -1},
                {"direction": "x", "index": (i, j, k), "boundary_sign": -1},
            ]

        # face_direction == "z"
        #
        # Local coordinates:
        # u = x, v = y
        #
        # e_x x e_y = e_z
        return [
            {"direction": "x", "index": (i, j, k), "boundary_sign": 1},
            {"direction": "y", "index": (i + 1, j, k), "boundary_sign": 1},
            {"direction": "x", "index": (i, j + 1, k), "boundary_sign": -1},
            {"direction": "y", "index": (i, j, k), "boundary_sign": -1},
        ]

    def _get_face_boundary_intersections(
        self, grid_type, face_direction, index, point_tolerance=None
    ):
        """
        Return unique resolved interface intersections on a face boundary.

        Coincident intersections at a face corner are grouped into one
        geometric point while the individual edge-hit information is
        preserved.
        """
        boundary_edges = self._get_face_boundary_edges(grid_type, face_direction, index)
        resolved_masks = self.resolved_interface_edge_masks[grid_type]
        intersections = self.interface_intersections[grid_type]
        raw_hits = []
        edge_lengths = []

        for boundary_order, edge in enumerate(boundary_edges):
            direction = edge["direction"]
            edge_index = edge["index"]
            boundary_sign = edge["boundary_sign"]

            start_point, end_point, _, _ = self._get_edge_geometry(
                grid_type, direction, edge_index
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

            raw_hits.append(
                {
                    "point": point,
                    "boundary_parameter": boundary_order + local_boundary_coordinate,
                    "surface_id": int(intersection_data["surface_id"][edge_index]),
                    "normal": np.asarray(
                        intersection_data["normal"][edge_index], dtype=float
                    ),
                    "multiple_crossing_positions": bool(
                        intersection_data["multiple_crossing_positions"][edge_index]
                    ),
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
                    matching_group["boundary_parameter"], hit["boundary_parameter"]
                )

        point_groups.sort(key=lambda group: group["boundary_parameter"])

        return point_groups

    def _get_face_intersection_segment(
        self, grid_type, face_direction, index, point_tolerance=None
    ):
        """
        Reconstruct one simple interface segment crossing a grid face.

        Only faces with exactly two unique boundary-intersection points
        belonging to one common STL surface are resolved; other cases are
        returned as unresolved.

        The returned segment orientation follows the order of the
        boundary-intersection points. It is purely geometric, not the
        oriented boundary direction of Stokes' theorem.
        """
        point_groups = self._get_face_boundary_intersections(
            grid_type, face_direction, index, point_tolerance=point_tolerance
        )

        result = {
            "resolved": False,
            "point_groups": point_groups,
            "length": np.nan,
            "surface_id": -1,
            "normal": None,
        }

        # Resolved case: exactly two distinct boundary-intersection positions.
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
        length = float(
            np.linalg.norm(point_groups[1]["point"] - point_groups[0]["point"])
        )

        if length <= 0.0:
            return result

        # Collect facet normals belonging to the selected STL surface.
        normals = []

        for group in point_groups:
            for hit in group["hits"]:
                if hit["surface_id"] == surface_id and np.all(
                    np.isfinite(hit["normal"])
                ):
                    normals.append(np.asarray(hit["normal"], dtype=float))

        normal = None

        if normals:
            mean_normal = np.mean(normals, axis=0)
            norm = np.linalg.norm(mean_normal)

            if norm > 0.0:
                normal = mean_normal / norm

        result.update(
            {
                "resolved": True,
                "length": length,
                "surface_id": int(surface_id),
                "normal": normal,
            }
        )

        return result

    def _get_face_corners(self, grid_type, face_direction, index):
        """
        Return the four corners of one grid face in positive boundary order.

        The ordering is counter-clockwise with respect to the positive
        face normal.
        """
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
        else:
            corner_indices = [
                (i, j, k),
                (i + 1, j, k),
                (i + 1, j + 1, k),
                (i, j + 1, k),
            ]

        corners = []

        for boundary_parameter, corner_index in enumerate(corner_indices):
            ci, cj, ck = corner_index

            corners.append(
                {
                    "point": np.array([x[ci], y[cj], z[ck]], dtype=float),
                    "region_id": int(region_ids[corner_index]),
                    "boundary_parameter": float(boundary_parameter),
                }
            )

        return corners

    @staticmethod
    def _compute_face_polygon_area(vertices, face_direction):
        """
        Compute the area of a polygon lying in one coordinate-aligned grid face.
        """
        vertices = np.asarray(vertices, dtype=float)

        if len(vertices) < 3:
            return 0.0

        if face_direction == "x":
            points_2d = vertices[:, [1, 2]]
        elif face_direction == "y":
            points_2d = vertices[:, [2, 0]]
        else:
            points_2d = vertices[:, [0, 1]]

        u = points_2d[:, 0]
        v = points_2d[:, 1]
        signed_area = 0.5 * (np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))

        return float(abs(signed_area))

    @staticmethod
    def _get_face_boundary_path(corners, start_group, end_group):
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

        vertices = [np.asarray(start_group["point"], dtype=float)]

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

        path_corner_candidates = []

        for corner in corners:
            parameter = float(corner["boundary_parameter"])

            if parameter <= start_parameter:
                parameter += 4.0

            if start_parameter < parameter < end_parameter:
                path_corner_candidates.append((parameter, corner))

        path_corner_candidates.sort(key=lambda item: item[0])
        path_corners = [corner for _, corner in path_corner_candidates]

        for corner in path_corners:
            vertices.append(np.asarray(corner["point"], dtype=float))

        vertices.append(np.asarray(end_group["point"], dtype=float))

        return (np.asarray(vertices, dtype=float), path_corners)

    def _split_face_by_interface(
        self, grid_type, face_direction, index, point_tolerance=None
    ):
        """
        Split one rectangular grid face into two region polygons.

        Handles a single resolved interface segment with exactly two
        distinct boundary-intersection points. Material treatment is not
        performed here.

        Returns both geometric region polygons together with their areas
        and the orientation of the cut segment on each polygon boundary.
        """
        segment = self._get_face_intersection_segment(
            grid_type, face_direction, index, point_tolerance=point_tolerance
        )

        result = {"resolved": False, "segment": segment, "polygons": []}

        if not segment["resolved"]:
            return result

        point_groups = segment["point_groups"]

        if len(point_groups) != 2:
            return result

        corners = self._get_face_corners(grid_type, face_direction, index)

        # Full rectangular face area.
        corner_points = np.asarray([corner["point"] for corner in corners], dtype=float)
        face_area = self._compute_face_polygon_area(corner_points, face_direction)

        if face_area <= 0.0:
            return result

        p0_group = point_groups[0]
        p1_group = point_groups[1]

        # Polygon 0:
        #
        # p0 -> positive outer boundary -> p1
        # then interface p1 -> p0
        (path_0, corners_0) = self._get_face_boundary_path(corners, p0_group, p1_group)

        # Polygon 1:
        #
        # p1 -> positive outer boundary -> p0
        # then interface p0 -> p1
        (path_1, corners_1) = self._get_face_boundary_path(corners, p1_group, p0_group)

        # Each region must contain at least one actual
        # face corner away from the interface.
        if not corners_0 or not corners_1:
            return result

        region_ids_0 = {corner["region_id"] for corner in corners_0}
        region_ids_1 = {corner["region_id"] for corner in corners_1}

        # Require one unique effective region on each side of the interface.
        if len(region_ids_0) != 1 or len(region_ids_1) != 1:
            return result

        region_id_0 = next(iter(region_ids_0))
        region_id_1 = next(iter(region_ids_1))

        if region_id_0 == region_id_1:
            return result

        area_0 = self._compute_face_polygon_area(path_0, face_direction)
        area_1 = self._compute_face_polygon_area(path_1, face_direction)
        area_tolerance = 1e-12 * face_area

        if area_0 <= area_tolerance or area_1 <= area_tolerance:
            return result

        if not np.isclose(area_0 + area_1, face_area, rtol=1e-10, atol=area_tolerance):
            return result

        polygons = [
            {
                "region_id": int(region_id),
                "vertices": path,
                "area_fraction": float(area / face_area),
            }
            for region_id, path, area in (
                (region_id_0, path_0, area_0),
                (region_id_1, path_1, area_1),
            )
        ]

        result.update({"resolved": True, "polygons": polygons})

        return result

    def _build_interface_face_data(self):
        """
        Build geometric conformal-interface data for all candidate faces.

        The grid layer stores geometry and region IDs only. Material
        interpretation is intentionally deferred to the solver.
        """
        self.interface_face_data = {"primal": {}, "dual": {}}

        for grid_type in ("primal", "dual"):
            for face_direction in ("x", "y", "z"):
                candidate = np.asarray(
                    self.face_candidate_masks[grid_type][face_direction], dtype=bool
                )

                resolved = np.zeros_like(candidate, dtype=bool)
                segments = {}
                polygons = {}
                region_area_fractions = {}

                for index_array in np.argwhere(candidate):
                    index = tuple(int(value) for value in index_array)
                    split = self._split_face_by_interface(
                        grid_type, face_direction, index
                    )
                    segment = split["segment"]

                    # Keep a reconstructed segment even if the polygon split
                    # fails, so that the SIBC can detect unresolved faces.
                    if segment["resolved"]:
                        segments[index] = {
                            key: segment[key]
                            for key in ("surface_id", "length", "normal")
                        }

                    if not split["resolved"]:
                        continue

                    resolved[index] = True
                    polygons[index] = split["polygons"]
                    region_area_fractions[index] = {
                        polygon["region_id"]: polygon["area_fraction"]
                        for polygon in split["polygons"]
                    }

                self.interface_face_data[grid_type][face_direction] = {
                    "resolved": resolved,
                    "segments": segments,
                    "polygons": polygons,
                    "region_area_fractions": region_area_fractions,
                }

                if self.verbose:
                    print(
                        f"    * {grid_type} {face_direction}-faces: "
                        f"{np.count_nonzero(candidate)} candidates, "
                        f"{len(segments)} segments, "
                        f"{np.count_nonzero(resolved)} resolved splits"
                    )

    def _build_conformal_interfaces(self):
        """
        Build the conformal interface geometry from the STL node masks.

        Used both after classifying the grid points and after loading the
        node masks from HDF5, so that both paths give identical cut-cell
        geometry. The STL surfaces are traced again and must be readable.
        """
        self._build_interface_topology()

        # Determine exact STL intersections with primal and dual edges
        #
        # This uses the geometric full-grid-line tracing developed above
        # and is independent of the endpoint transition masks.
        self._build_interface_intersections()

        # Warn about thin features and inconsistent STL crossings.
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
        region_material_types = {0: self.bg_material_type}

        for region_id, key in self.grid.region_id_to_key.items():
            if region_id == 0:
                continue

            if key not in self.grid.stl_material_types:
                raise RuntimeError(f"No material type available for STL solid '{key}'.")

            region_material_types[region_id] = self.grid.stl_material_types[key]

        self.region_material_types = region_material_types

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

        type_codes = {"normal": 0, "pec": 1, "sibc": 2}
        max_region_id = max(self.region_material_types)
        region_type_codes = np.empty(max_region_id + 1, dtype=np.int8)

        for region_id, material_type in self.region_material_types.items():
            region_type_codes[region_id] = type_codes[material_type]

        normal = type_codes["normal"]
        pec = type_codes["pec"]
        sibc = type_codes["sibc"]
        self.conformal_line_data = {"primal": {}, "dual": {}}

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                (start_regions, end_regions) = self.grid._get_edge_region_ids(
                    grid_type, direction
                )

                data = self.grid.interface_intersections[grid_type][direction]
                alpha = np.asarray(data["alpha"], dtype=float)

                resolved = np.asarray(
                    self.grid.resolved_interface_edge_masks[grid_type][direction],
                    dtype=bool,
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
                effective_length_fraction = np.full(alpha.shape, np.nan, dtype=float)
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

        self.conformal_face_data = {"primal": {}, "dual": {}}
        conductor_types = {"pec", "sibc"}

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                geometry = self.grid.interface_face_data[grid_type][direction]
                resolved = np.asarray(geometry["resolved"], dtype=bool)
                supported = np.zeros_like(resolved, dtype=bool)
                normal_normal = np.zeros_like(resolved, dtype=bool)
                normal_pec = np.zeros_like(resolved, dtype=bool)
                normal_sibc = np.zeros_like(resolved, dtype=bool)
                effective_area_fraction = np.full(resolved.shape, np.nan, dtype=float)
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

                    # normal <-> normal
                    #
                    # Entire geometrical face remains active.
                    if len(normal_regions) == 2:
                        supported[index] = True
                        normal_normal[index] = True
                        effective_area_fraction[index] = 1.0
                        normal_region_ids[index] = tuple(normal_regions)
                        continue

                    # normal <-> PEC/SIBC
                    #
                    # Only the normal portion remains volumetrically
                    # active.
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
        Effective materials on resolved interface faces: the area-weighted
        (arithmetic) average over the normal-material part of the face.
        The area reduction itself enters through A_eff.

        Primal faces carry mu_r, dual faces eps_r and sigma.
        """
        self.conformal_face_materials = {"primal": {}, "dual": {}}
        names = {"primal": ("mu_r",), "dual": ("eps_r", "sigma")}

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                face_data = self.conformal_face_data[grid_type][direction]
                geometry = self.grid.interface_face_data[grid_type][direction]
                supported = face_data["supported"]
                values = {
                    name: np.full(supported.shape, np.nan, dtype=float)
                    for name in names[grid_type]
                }

                for index_array in np.argwhere(supported):
                    index = tuple(int(value) for value in index_array)
                    fractions = geometry["region_area_fractions"][index]
                    regions = face_data["normal_region_ids"][index]
                    normal_fraction = sum(float(fractions[r]) for r in regions)

                    for name in names[grid_type]:
                        table = self.region_material_properties[name]
                        values[name][index] = sum(
                            float(fractions[r]) / normal_fraction * table[r]
                            for r in regions
                        )

                self.conformal_face_materials[grid_type][direction] = values

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
                material = [mat[0], mat[1], mat[2] if len(mat) == 3 else 0.0]
            else:
                material = list(material)

                if len(material) == 2:
                    material.append(0.0)
        else:
            key = self.grid.region_id_to_key[region_id]
            material = self.grid.stl_materials[key]

        return np.asarray(material, dtype=float)

    def _build_region_material_properties(self):
        """
        Build numerical material-property tables indexed by region ID.
        """
        max_region_id = max(self.region_material_types)
        eps_r = np.full(max_region_id + 1, np.nan, dtype=float)
        mu_r = np.full(max_region_id + 1, np.nan, dtype=float)
        sigma = np.full(max_region_id + 1, np.nan, dtype=float)

        for region_id in self.region_material_types:
            properties = self._get_region_material_properties(region_id)
            eps_r[region_id] = properties[0]
            mu_r[region_id] = properties[1]
            sigma[region_id] = properties[2]

        self.region_material_properties = {"eps_r": eps_r, "mu_r": mu_r, "sigma": sigma}

    def _build_conformal_line_materials(self):
        """
        Build effective material values along resolved conformal edges.

        Primal edges:
            - normal-normal: harmonic eps_r and sigma
            - normal-conductor: material properties of the normal segment

        Dual edges (normal-normal only): harmonic mu_r

        No FIT operators are modified here.
        """
        self._build_region_material_properties()
        eps_r_regions = self.region_material_properties["eps_r"]
        mu_r_regions = self.region_material_properties["mu_r"]
        sigma_regions = self.region_material_properties["sigma"]
        self.conformal_line_materials = {"primal": {}, "dual": {}}

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
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
                    eps_r_eff = np.full(shape, np.nan, dtype=float)
                    sigma_eff = np.full(shape, np.nan, dtype=float)

                    # Normal -> normal:
                    # harmonic average along the edge
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

                    # Normal -> PEC/SIBC
                    eps_r_eff[normal_to_conductor] = start_eps_r[normal_to_conductor]
                    sigma_eff[normal_to_conductor] = start_sigma[normal_to_conductor]

                    # PEC/SIBC -> normal
                    eps_r_eff[conductor_to_normal] = end_eps_r[conductor_to_normal]
                    sigma_eff[conductor_to_normal] = end_sigma[conductor_to_normal]

                    self.conformal_line_materials[grid_type][direction] = {
                        "eps_r": eps_r_eff,
                        "sigma": sigma_eff,
                    }
                else:
                    mu_r_eff = np.full(shape, np.nan, dtype=float)

                    # Normal -> normal:
                    # harmonic average along dual edge
                    mu_r_eff[normal_normal] = 1.0 / (
                        alpha[normal_normal] / start_mu_r[normal_normal]
                        + (1.0 - alpha[normal_normal]) / end_mu_r[normal_normal]
                    )

                    # Dual edges are only traced at normal-normal interfaces.
                    self.conformal_line_materials[grid_type][direction] = {
                        "mu_r": mu_r_eff
                    }

    # Geometry array -> stored DOF mapping (symmetric FIT stencil)
    #
    # Nx, Ny, Nz count primal nodes. Every field component stores
    # Nx * Ny * Nz values; entries without a geometric counterpart are
    # zero-size ghost DOFs on the high side (L = 0 or iA = 0).
    #
    # Full geometry array shapes and the stored DOF they belong to:
    #
    #   primal x-edge (Nx-1, Ny,   Nz  )  (i,j,k)     -> Ex(i,j,k)
    #   primal x-face (Nx,   Ny-1, Nz-1)  (i,j,k)     -> Hx(i,j,k)
    #   dual   x-edge (Nx,   Ny+1, Nz+1)  (i,j+1,k+1) -> Hx(i,j,k)
    #   dual   x-face (Nx+1, Ny,   Nz  )  (i+1,j,k)   -> Ex(i,j,k)
    #
    # and cyclically for y and z. The dual offsets follow from the dual
    # coordinates tx = [x0, primal midpoints, x_end]: the dual line through
    # the primal face centre y_{j+1/2} is ty[j+1], and the dual face
    # around the primal edge midpoint x_{i+1/2} lies at tx[i+1].

    _AXIS = {"x": 0, "y": 1, "z": 2}

    def _geometry_array_shape(self, kind, direction):
        """Full geometry-array shape of one primal/dual edge/face family."""
        if direction not in self._AXIS:
            raise ValueError(f"Unknown direction '{direction}'.")

        axis = self._AXIS[direction]
        shape = [self.Nx, self.Ny, self.Nz]

        if kind == "primal_edge":
            shape[axis] -= 1
        elif kind == "primal_face":
            shape = [n - 1 for n in shape]
            shape[axis] += 1
        elif kind == "dual_edge":
            shape = [n + 1 for n in shape]
            shape[axis] -= 1
        elif kind == "dual_face":
            shape[axis] += 1
        else:
            raise ValueError(f"Unknown geometry array kind '{kind}'.")

        return tuple(shape)

    def _geometry_array_offset(self, kind, direction):
        """Index offset from stored DOF (i,j,k) to the geometry array."""
        axis = self._AXIS[direction]

        if kind in ("primal_edge", "primal_face"):
            return (0, 0, 0)
        if kind == "dual_edge":
            offset = [1, 1, 1]
            offset[axis] = 0
            return tuple(offset)
        if kind == "dual_face":
            offset = [0, 0, 0]
            offset[axis] = 1
            return tuple(offset)

        raise ValueError(f"Unknown geometry array kind '{kind}'.")

    def _geometry_array_to_field_component(
        self, values, kind, direction, fill_value=None
    ):
        """
        Map a full primal/dual edge or face array to stored DOFs.

        Stored DOF (i, j, k) takes ``values[(i, j, k) + offset]`` (see the
        table above). Ghost DOFs without a geometric counterpart get
        ``fill_value``:

        - None (default): False for boolean arrays, NaN for floating-point
          arrays (NaN means "no conformal value" to the material code).
        - "nearest": copy the nearest stored value (used for region IDs).
        - any scalar: that constant.
        """
        values = np.asarray(values)
        expected_shape = self._geometry_array_shape(kind, direction)

        if values.shape != expected_shape:
            raise ValueError(
                f"Expected {kind.replace('_', ' ')} {direction}-array shape "
                f"{expected_shape}, got {values.shape}."
            )

        shape = (self.Nx, self.Ny, self.Nz)
        oi, oj, ok = self._geometry_array_offset(kind, direction)
        source = values[oi : oi + shape[0], oj : oj + shape[1], ok : ok + shape[2]]

        if source.shape == shape:
            return source.copy()

        if isinstance(fill_value, str) and fill_value == "nearest":
            padding = [(0, n - m) for n, m in zip(shape, source.shape)]
            return np.pad(source, padding, mode="edge")

        if fill_value is None:
            if values.dtype == bool:
                fill_value = False
            elif np.issubdtype(values.dtype, np.floating):
                fill_value = np.nan
            else:
                raise ValueError(
                    f"Ghost DOFs of a {values.dtype} array need an explicit "
                    "fill_value (a constant or 'nearest')."
                )

        dtype = np.result_type(values.dtype, np.asarray(fill_value).dtype)
        result = np.full(shape, fill_value, dtype=dtype)
        result[: source.shape[0], : source.shape[1], : source.shape[2]] = source

        return result

    def _edge_array_to_field_component(self, edge_values, direction, fill_value=None):
        """Map a full primal-edge array to the stored electric DOFs."""
        return self._geometry_array_to_field_component(
            edge_values, "primal_edge", direction, fill_value
        )

    def _face_array_to_field_component(self, face_values, direction, fill_value=None):
        """
        Map a full primal-face array to the stored magnetic DOFs.

        The curl matrix C defines Hx(i,j,k) as the flux through the primal
        x-face (i, j, k) at x[i]; likewise for Hy and Hz.
        """
        return self._geometry_array_to_field_component(
            face_values, "primal_face", direction, fill_value
        )

    def _primal_face_array_to_dual_edge_component(
        self, face_values, direction, fill_value=None
    ):
        """Map primal-face data to the stored dual-edge (magnetic) DOFs."""
        return self._face_array_to_field_component(face_values, direction, fill_value)

    def _dual_face_array_to_primal_edge_component(
        self, face_values, direction, fill_value=None
    ):
        """
        Map dual-face data to the stored primal-edge (electric) DOFs.

            Ex(i,j,k) <-> dual x-face (i+1, j, k)
            Ey(i,j,k) <-> dual y-face (i, j+1, k)
            Ez(i,j,k) <-> dual z-face (i, j, k+1)

        Every electric DOF, including the boundary edges, has a dual face.
        """
        return self._geometry_array_to_field_component(
            face_values, "dual_face", direction, fill_value
        )

    def _dual_edge_array_to_field_component(
        self, edge_values, direction, fill_value=None
    ):
        """
        Map a full dual-edge array to the stored magnetic DOFs.

            Hx(i,j,k) <-> dual x-edge (i, j+1, k+1)
            Hy(i,j,k) <-> dual y-edge (i+1, j, k+1)
            Hz(i,j,k) <-> dual z-edge (i+1, j+1, k)

        Dual x-edge (i, ., .) spans tx[i]..tx[i+1] around x[i], including
        the half edges at both domain boundaries, so every magnetic DOF has
        a dual edge.
        """
        return self._geometry_array_to_field_component(
            edge_values, "dual_edge", direction, fill_value
        )

    def _build_conformal_diagonal_materials(self, normal_tol=1e-12):
        """
        Diagonal effective materials at normal-normal interfaces.

        For the DOF component along x_i, with n the interface normal,

            m_ii = n_i^2 * m_perp + (1 - n_i^2) * m_parallel,

        where m_perp is the harmonic (series) value along the DOF edge and
        m_parallel the arithmetic (parallel) value over its face:

            eps, sigma: primal edge (perp) and dual face (parallel)
            mu:         dual edge (perp) and primal face (parallel)

        The arrays have the stored DOF shape (Nx, Ny, Nz) and are NaN where
        neither the edge nor its face is cut by a normal-normal interface.
        """
        self.conformal_diagonal_materials = {"eps_r": {}, "mu_r": {}, "sigma": {}}
        electric = (
            "primal",
            "dual",
            self._edge_array_to_field_component,
            self._dual_face_array_to_primal_edge_component,
        )
        magnetic = (
            "dual",
            "primal",
            self._dual_edge_array_to_field_component,
            self._primal_face_array_to_dual_edge_component,
        )

        for direction in ("x", "y", "z"):
            for name, (line_grid, face_grid, line_to_dof, face_to_dof) in (
                ("eps_r", electric),
                ("sigma", electric),
                ("mu_r", magnetic),
            ):
                self.conformal_diagonal_materials[name][direction] = (
                    self._diagonal_material_component(
                        name,
                        direction,
                        line_grid,
                        face_grid,
                        lambda values, fill=None: line_to_dof(values, direction, fill),
                        lambda values, fill=None: face_to_dof(values, direction, fill),
                        normal_tol,
                    )
                )

    def _diagonal_material_component(
        self, name, direction, line_grid, face_grid, line_to_dof, face_to_dof, tol
    ):
        """One diagonal material component (see _build_conformal_diagonal_materials)."""
        axis = "xyz".index(direction)
        line_data = self.conformal_line_data[line_grid][direction]
        face_data = self.conformal_face_data[face_grid][direction]
        region_values = self.region_material_properties[name]

        line_cut = line_to_dof(line_data["normal_normal"], False).astype(bool)
        face_cut = face_to_dof(face_data["normal_normal"], False).astype(bool)

        # Squared normal component: from the edge crossing, or for an uncut
        # edge from the segment that cuts its face.
        normals = self.grid.interface_intersections[line_grid][direction]["normal"]
        norm = np.linalg.norm(normals, axis=-1)
        line_n2 = np.full(norm.shape, np.nan, dtype=float)
        valid = np.isfinite(norm) & (norm > 0.0)
        line_n2[valid] = (normals[..., axis][valid] / norm[valid]) ** 2

        face_n2 = np.full(face_data["normal_normal"].shape, np.nan, dtype=float)
        segments = self.grid.interface_face_data[face_grid][direction]["segments"]
        for index, segment in segments.items():
            if not face_data["normal_normal"][index]:
                continue
            normal = np.asarray(segment["normal"], dtype=float)
            norm = np.linalg.norm(normal)
            if np.isfinite(norm) and norm > 0.0:
                face_n2[index] = (normal[axis] / norm) ** 2

        n2 = line_to_dof(line_n2)
        n2 = np.where(np.isfinite(n2), n2, face_to_dof(face_n2))

        # Where only one of edge and face is cut, the uncut one lies in a
        # single region: an uncut edge in its endpoint region, an uncut face
        # in the region of the edge midpoint, which it passes through.
        start, end = line_data["start_region"], line_data["end_region"]
        uncut_edge = np.where(start == end, region_values[start], np.nan)
        # Crossing before the midpoint (alpha < 0.5): midpoint in the end region.
        # NaN alpha (uncut edge) compares False and selects the start region.
        midpoint_region = np.where(line_data["alpha"] < 0.5, end, start)

        perp = np.where(
            line_cut,
            line_to_dof(self.conformal_line_materials[line_grid][direction][name]),
            line_to_dof(uncut_edge),
        )
        parallel = np.where(
            face_cut,
            face_to_dof(self.conformal_face_materials[face_grid][direction][name]),
            line_to_dof(region_values[midpoint_region]),
        )

        cut = (line_cut | face_cut) & np.isfinite(n2)
        n2 = np.clip(n2, 0.0, 1.0)
        result = np.full(perp.shape, np.nan, dtype=float)

        both = cut & np.isfinite(perp) & np.isfinite(parallel)
        result[both] = n2[both] * perp[both] + (1.0 - n2[both]) * parallel[both]
        # Purely normal or tangential interfaces need only one of the values.
        pure_normal = cut & np.isfinite(perp) & (1.0 - n2 <= tol)
        result[pure_normal] = perp[pure_normal]
        pure_tangential = cut & np.isfinite(parallel) & (n2 <= tol)
        result[pure_tangential] = parallel[pure_tangential]

        return result

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
                line_data["effective_length_fraction"].shape, dtype=float
            )

            supported = line_data["supported"]
            full_fraction[supported] = line_data["effective_length_fraction"][supported]

            field_fraction = self._edge_array_to_field_component(
                full_fraction, direction, fill_value=1.0
            )

            setattr(self.L_fraction, f"field_{direction}", field_fraction.copy())

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
        as area fraction (a constant-floor form of the area limitation of
        Benkler et al. 2006), which bounds the reduction of
        the stable time step. ``A_fraction_geometric`` keeps the exact
        geometric fractions; ``n_area_floored`` counts the raised faces.

        Dual face areas are never cut (Dey-Mittra-type treatment).
        The original grid metric self.iA is not modified.
        """
        self.A_fraction = Field(self.Nx, self.Ny, self.Nz)
        self.A_fraction_geometric = Field(self.Nx, self.Ny, self.Nz)
        self.A_eff = Field(self.Nx, self.Ny, self.Nz)
        self.n_area_floored = 0
        area_floor = self.conformal_area_floor

        for direction in ("x", "y", "z"):
            data = self.conformal_face_data["primal"][direction]

            # Unsupported/unresolved faces retain their original area.
            full_fraction = np.ones(data["supported"].shape, dtype=float)
            supported = data["supported"]
            full_fraction[supported] = data["effective_area_fraction"][supported]

            setattr(
                self.A_fraction_geometric,
                f"field_{direction}",
                self._face_array_to_field_component(
                    full_fraction, direction, fill_value=1.0
                ),
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
                full_fraction, direction, fill_value=1.0
            )

            # Grid stores inverse face areas; ghost faces (iA = 0) have
            # zero area.
            iA = self.iA.to_matrix(direction)
            base_area = np.divide(1.0, iA, out=np.zeros_like(iA), where=iA != 0)
            setattr(self.A_fraction, f"field_{direction}", field_fraction.copy())
            setattr(self.A_eff, f"field_{direction}", base_area * field_fraction)

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
        threshold = self.conformal_min_edge_fraction
        if (
            threshold <= 0.0
            or self.conformal_area_floor <= 0.0
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
        Ex(i,j,k) belongs to the dual x-face (i+1, j, k), whose corners are
        the dual points (i+1, j..j+1, k..k+1). The dual grid includes the
        domain boundaries, so boundary dual faces are complete half faces.
        """
        p = np.asarray(point_values)

        # Ex(i,j,k) -> dual yz face at tx[i+1]
        values_x = 0.25 * (
            p[1:, :-1, :-1] + p[1:, 1:, :-1] + p[1:, 1:, 1:] + p[1:, :-1, 1:]
        )

        # Ey(i,j,k) -> dual xz face at ty[j+1]
        values_y = 0.25 * (
            p[:-1, 1:, :-1] + p[1:, 1:, :-1] + p[1:, 1:, 1:] + p[:-1, 1:, 1:]
        )

        # Ez(i,j,k) -> dual xy face at tz[k+1]
        values_z = 0.25 * (
            p[:-1, :-1, 1:] + p[1:, :-1, 1:] + p[1:, 1:, 1:] + p[:-1, 1:, 1:]
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
            (Nx, Ny, Nz).

        Returns
        -------
        values_x, values_y, values_z : ndarray
            Face-averaged material values associated with the x-, y-, and
            z-directed dual edges. Each array has shape (Nx, Ny, Nz).
            Ghost DOFs (zero-area high-side faces) copy the nearest face.
        """
        p = np.asarray(point_values)

        # Hx(i,j,k) -> primal yz face at x[i] (see curl matrix C)
        values_x = 0.25 * (
            p[:, :-1, :-1] + p[:, 1:, :-1] + p[:, 1:, 1:] + p[:, :-1, 1:]
        )

        # Hy(i,j,k) -> primal xz face at y[j]
        values_y = 0.25 * (
            p[:-1, :, :-1] + p[1:, :, :-1] + p[1:, :, 1:] + p[:-1, :, 1:]
        )

        # Hz(i,j,k) -> primal xy face at z[k]
        values_z = 0.25 * (
            p[:-1, :-1, :] + p[1:, :-1, :] + p[1:, 1:, :] + p[:-1, 1:, :]
        )

        def pad(values):
            padding = [(0, n - m) for n, m in zip(p.shape, values.shape)]
            return np.pad(values, padding, mode="edge")

        return pad(values_x), pad(values_y), pad(values_z)

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

        # Point shapes: primal (Nx, Ny, Nz), dual (Nx+1, Ny+1, Nz+1).
        primal_shape = (self.Nx, self.Ny, self.Nz)
        dual_shape = (self.Nx + 1, self.Ny + 1, self.Nz + 1)

        # Initialize point-wise material values with the background material.
        eps_dual = np.full(dual_shape, self.eps_bg, dtype=self.dtype)
        sigma_dual = np.full(dual_shape, self.sigma_bg, dtype=self.dtype)
        mu_primal = np.full(primal_shape, self.mu_bg, dtype=self.dtype)

        # Assign STL material values to primal and dual grid points.
        for key in self.stl_solids.keys():
            primal_mask = self.grid.primal_node_masks[key]
            dual_mask = self.grid.dual_node_masks[key]
            eps = self.stl_materials[key][0] * eps_0
            mu = self.stl_materials[key][1] * mu_0
            sigma = self.stl_materials[key][2]
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

        # Replace the point-averaged material values by the
        # conformal diagonal approximation wherever a resolved
        # normal-normal interface treatment is available.
        #
        # All other DOFs retain the point-averaged values above.

        eps_components = {"x": eps_x, "y": eps_y, "z": eps_z}
        mu_components = {"x": mu_x, "y": mu_y, "z": mu_z}
        sigma_components = {"x": sigma_x, "y": sigma_y, "z": sigma_z}

        # Geometry-based constitutive assignment (conformal FIT).
        #
        # Uncut edges with one normal region get that region's
        # material; conductor-conductor primal edges get ieps=0.
        # Resolved conductor interfaces use the *normal* part.
        # Normal-normal interfaces are handled afterwards by the
        # diagonal-tensor approximation. The point averages above
        # remain only on edges without resolved crossings, which the
        # conformal geometry diagnostics already warn about, and on
        # the zero-length high-side ghost edges.
        if not self.region_material_types:
            self._build_region_material_types()
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
            start = self._edge_array_to_field_component(start, direction, "nearest")
            end = self._edge_array_to_field_component(end, direction, "nearest")
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
            dual_start = self._dual_edge_array_to_field_component(dual_start, direction)
            dual_end = self._dual_edge_array_to_field_component(dual_end, direction)
            dual_same_normal = (dual_start == dual_end) & region_is_normal[dual_start]
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

        for name, components, unit in (
            ("eps_r", eps_components, eps_0),
            ("mu_r", mu_components, mu_0),
            ("sigma", sigma_components, 1.0),
        ):
            for direction, values in self.conformal_diagonal_materials[name].items():
                values = np.asarray(values, dtype=float)
                # NaN marks DOFs without a normal-normal interface value.
                valid = np.isfinite(values) & (
                    values >= 0.0 if name == "sigma" else values > 0.0
                )
                components[direction][valid] = values[valid] * unit

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
