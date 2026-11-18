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


class ConformalGridMixin:
    """Conformal STL intersection handling for ``GridFIT3D``."""

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

    def _build_region_ids(self):
        """
        Assign region IDs and build them on the primal and dual grid points.

        Region 0 is the background; STL solids get IDs 1, 2, ... in the
        order of ``self.stl_solids``, and later solids overwrite earlier ones
        where they overlap. ``region_material_types`` maps every region ID to
        its material type; a background without ``bg_material_type`` is
        treated as normal (the solver rejects a conductor background then).
        """
        self.region_id_to_key = {0: None}
        self.region_key_to_id = {}
        self.region_material_types = {0: self.bg_material_type or "normal"}

        for region_id, key in enumerate(self.stl_solids.keys(), start=1):
            self.region_id_to_key[region_id] = key
            self.region_key_to_id[key] = region_id
            self.region_material_types[region_id] = self.stl_material_types[key]

        for grid_type, node_masks in (
            ("primal", self.primal_node_masks),
            ("dual", self.dual_node_masks),
        ):
            shape = tuple(len(c) for c in self._get_grid_coordinates(grid_type))
            region_ids = np.zeros(shape, dtype=np.int32)
            for key, region_id in self.region_key_to_id.items():
                region_ids[np.asarray(node_masks[key], dtype=bool)] = region_id
            self.region_ids[grid_type] = region_ids

    def _region_type_flags(self):
        """
        Boolean arrays indexed by region ID: normal material, and
        conductor (PEC or SIBC).
        """
        types = np.array(
            [self.region_material_types[r] for r in range(len(self.region_id_to_key))]
        )
        return types == "normal", (types == "pec") | (types == "sibc")

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

    def _trace_grid_line(
        self,
        grid_type,
        direction,
        transverse_index,
        surfaces,
        alpha_tolerance=1e-8,
    ):
        """
        Trace one complete primal or dual grid line through all STL surfaces.

        ``surfaces`` maps region ID -> (STL surface, VTK cell locator).
        Returns all geometric intersections, already assigned to their
        corresponding grid edges.
        """
        coordinates = self._get_grid_coordinates(grid_type)
        axis = "xyz".index(direction)
        axis_coordinates = np.asarray(coordinates[axis], dtype=float)

        # Extend the line beyond the grid so that hits on the outer grid
        # points are found.
        line_extension = np.min(np.diff(axis_coordinates))
        transverse_axes = [a for a in range(3) if a != axis]
        start_point = np.empty(3)
        for a, t in zip(transverse_axes, transverse_index):
            start_point[a] = coordinates[a][t]
        end_point = start_point.copy()
        start_point[axis] = axis_coordinates[0] - line_extension
        end_point[axis] = axis_coordinates[-1] + line_extension

        edge_hits = {}
        n_edges = len(axis_coordinates) - 1

        for surface_id, (locator, normals) in surfaces.items():
            vtk_points = vtkPoints()
            vtk_cell_ids = vtkIdList()
            locator.IntersectWithLine(
                start_point, end_point, 1e-8, vtk_points, vtk_cell_ids
            )
            n_hits = vtk_points.GetNumberOfPoints()
            if n_hits == 0:
                continue

            coordinate = np.array([vtk_points.GetPoint(n)[axis] for n in range(n_hits)])
            cell_ids = [vtk_cell_ids.GetId(n) for n in range(n_hits)]

            # Edge containing each hit; a hit exactly on the last grid point
            # belongs to the final edge. Hits on the line extension beyond
            # the grid are dropped.
            edge = np.searchsorted(axis_coordinates, coordinate, side="right") - 1
            edge = np.minimum(edge, n_edges - 1)
            start = axis_coordinates[np.maximum(edge, 0)]
            alpha = (coordinate - start) / (
                axis_coordinates[np.maximum(edge, 0) + 1] - start
            )
            inside = (
                (edge >= 0)
                & (alpha >= -alpha_tolerance)
                & (alpha <= 1.0 + alpha_tolerance)
            )

            for n in np.nonzero(inside)[0]:
                edge_index = list(transverse_index)
                edge_index.insert(axis, int(edge[n]))
                hit = {
                    "alpha": float(np.clip(alpha[n], 0.0, 1.0)),
                    "surface_id": int(surface_id),
                    "cell_id": int(cell_ids[n]),
                    "normal": normals[cell_ids[n]],
                }
                edge_hits.setdefault(tuple(edge_index), []).append(hit)

        return edge_hits

    @staticmethod
    def _group_crossings(hits, axis, alpha_tolerance=1e-8):
        """
        Group the STL hits of one edge by position and keep the crossings.

        Hits within ``alpha_tolerance`` belong to one geometric point (a
        line through a shared triangle edge or vertex, or coincident
        surfaces of different solids). A group is a crossing if at least one
        facet normal has a component along the edge; otherwise it is a
        tangential contact and dropped.
        """
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

        # VTK can report a hit for a facet coplanar with the grid line (e.g.
        # an axis-aligned wall on a grid plane). Counted as a crossing, it
        # would toggle inside/outside wrongly. The normals are unit vectors,
        # so 1e-8 separates "line in the facet plane up to rounding" from any
        # real tilt.
        return [
            group
            for group in groups
            if any(abs(hit["normal"][axis]) > 1e-8 for hit in group)
        ]

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

    def _select_interface_hit(self, groups, left_region, right_region):
        """
        Choose the STL hit that represents one interface: its position,
        surface and facet normal.

        Hits on the surfaces of the two adjacent regions come first, and
        among them a conductor surface, which defines the cut wall of a
        normal-conductor interface. Then the surface of the later solid wins,
        as it does in the point classification. Remaining ties go to the
        first crossing along the edge and the lowest facet ID.
        """

        def priority(hit):
            surface_id = hit["surface_id"]
            side = surface_id in (left_region, right_region)
            side_conductor = side and self._is_conductor_region(surface_id)
            return (
                not side_conductor,
                not side,
                -surface_id,
                hit["alpha"],
                hit["cell_id"],
            )

        return min((hit for group in groups for hit in group), key=priority)

    def _get_normal_normal_edge_mask(self, grid_type, direction):
        """
        Edges whose two endpoints lie in different normal-material regions.
        """
        start, end = self._get_edge_region_ids(grid_type, direction)
        is_normal, _ = self._region_type_flags()

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
        # One static cell locator and the unit facet normals per solid,
        # reused for all grid lines. STL facets are triangles.
        surfaces = {}
        for key, region_id in self.region_key_to_id.items():
            surface = self.read_stl(key)
            locator = vtkStaticCellLocator()
            locator.SetDataSet(surface)
            locator.BuildLocator()
            points = np.asarray(surface.points, dtype=float)  # STL stores float32
            triangles = points[surface.faces.reshape(-1, 4)[:, 1:]]
            normals = np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            )
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            surfaces[region_id] = (locator, normals)

        self.interface_intersections = {"primal": {}, "dual": {}}

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                axis = "xyz".index(direction)
                start_regions, end_regions = self._get_edge_region_ids(
                    grid_type, direction
                )
                shape = start_regions.shape
                transverse_shape = np.delete(self.region_ids[grid_type].shape, axis)
                transverse_indices = np.ndindex(*transverse_shape)

                # One representative crossing per edge with different
                # endpoint regions (has_crossing); all other edges keep the
                # NaN / -1 defaults.
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
                    traced_lines = np.any(scope, axis=axis)
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
                        surfaces,
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
                    crossing_groups = self._group_crossings(
                        hits, axis, alpha_tolerance=alpha_tolerance
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
                        self._select_interface_hit(groups, left, right)
                        for groups, left, right in zip(
                            interfaces, regions[:-1], regions[1:]
                        )
                    ]
                    # Several interfaces in one edge: only one can be
                    # represented. A conductor endpoint keeps its own wall
                    # (a coating on it is dropped, not the wall moved);
                    # otherwise use the interface nearest the edge midpoint.
                    if self._is_conductor_region(end):
                        representative = candidates[-1]
                    elif self._is_conductor_region(start):
                        representative = candidates[0]
                    else:
                        representative = min(
                            candidates, key=lambda hit: abs(hit["alpha"] - 0.5)
                        )
                    multiple = len(interfaces) > 1 or not consistent
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

    def _is_conductor_region(self, region_id):
        return self.region_material_types[region_id] in ("pec", "sibc")

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
        start, end = self._get_edge_region_ids(grid_type, direction)
        endpoint_mask = start != end

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

    @staticmethod
    def _polygon_area(points_2d):
        """Area of a simple polygon from its vertices in order (shoelace)."""
        u, v = points_2d[:, 0], points_2d[:, 1]
        return float(abs(0.5 * (np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))))

    def _split_face(self, grid_type, face_direction, index):
        """
        Split one grid face along the interface segment that crosses it.

        The cut points are the represented crossings of the four boundary
        edges. A face with exactly two distinct cut points of one common STL
        surface, none from an edge with several crossings, has an interface
        segment. It is split into two polygons when each side contains
        corners of a single region and the two regions differ.

        Returns
        -------
        segment : dict or None
            ``surface_id``, ``length`` and unit mean facet ``normal`` (None
            if the facet normals cancel) of the interface segment.
        polygons : list of dict or None
            For both sides: ``region_id``, ``vertices`` (Cartesian, in
            boundary order) and ``area_fraction`` of the face.
        """
        normal_axis = "xyz".index(face_direction)
        a, b = (normal_axis + 1) % 3, (normal_axis + 2) % 3
        coordinates = self._get_grid_coordinates(grid_type)
        intersections = self.interface_intersections[grid_type]

        # Corners counter-clockwise about the face normal (e_a x e_b = e_n);
        # boundary edge m runs from corner m to corner m + 1, so a point at
        # boundary parameter m + t lies on it at fraction t.
        corner_indices = []
        for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
            corner = list(index)
            corner[a] += da
            corner[b] += db
            corner_indices.append(tuple(corner))
        corners = np.array(
            [[coordinates[d][c[d]] for d in range(3)] for c in corner_indices],
            dtype=float,
        )

        cut_points = []
        for m in range(4):
            c0, c1 = corner_indices[m], corner_indices[(m + 1) % 4]
            axis = next(d for d in range(3) if c0[d] != c1[d])
            forward = c1[axis] > c0[axis]
            edge = c0 if forward else c1
            data = intersections["xyz"[axis]]
            if not data["has_crossing"][edge]:
                continue
            alpha = float(data["alpha"][edge])
            start, end = (m, (m + 1) % 4) if forward else ((m + 1) % 4, m)
            cut_points.append(
                {
                    "point": corners[start] + alpha * (corners[end] - corners[start]),
                    "parameter": m + (alpha if forward else 1.0 - alpha),
                    "surface_id": int(data["surface_id"][edge]),
                    "normal": data["normal"][edge],
                    "multiple": bool(data["multiple_crossing_positions"][edge]),
                }
            )

        # Cut points of two edges at their shared corner are one point.
        tolerance = 1e-8 * np.max(
            np.linalg.norm(np.roll(corners, -1, 0) - corners, axis=1)
        )
        groups = []
        for cut in sorted(cut_points, key=lambda cut: cut["parameter"]):
            for group in groups:
                if np.linalg.norm(cut["point"] - group["point"]) <= tolerance:
                    group["cuts"].append(cut)
                    group["parameter"] = min(group["parameter"], cut["parameter"])
                    break
            else:
                groups.append(
                    {
                        "point": cut["point"],
                        "parameter": cut["parameter"],
                        "cuts": [cut],
                    }
                )
        groups.sort(key=lambda group: group["parameter"])

        if len(groups) != 2 or any(
            cut["multiple"] for group in groups for cut in group["cuts"]
        ):
            return None, None
        common = {cut["surface_id"] for cut in groups[0]["cuts"]} & {
            cut["surface_id"] for cut in groups[1]["cuts"]
        }
        if len(common) != 1:
            return None, None

        surface_id = common.pop()
        mean_normal = np.mean(
            [
                cut["normal"]
                for group in groups
                for cut in group["cuts"]
                if cut["surface_id"] == surface_id
            ],
            axis=0,
        )
        norm = np.linalg.norm(mean_normal)
        segment = {
            "surface_id": surface_id,
            "length": float(np.linalg.norm(groups[1]["point"] - groups[0]["point"])),
            "normal": mean_normal / norm if norm > 0.0 else None,
        }

        # Both sides: from one cut point along the boundary, through the
        # corners passed, to the other cut point.
        region_ids = self.region_ids[grid_type]
        face_area = self._polygon_area(corners[:, [a, b]])
        polygons = []
        for p, q in ((groups[0], groups[1]), (groups[1], groups[0])):
            start, end = p["parameter"], q["parameter"]
            if end <= start:
                end += 4.0
            # Corner m lies at boundary parameter m, unwrapped past start.
            unwrapped = sorted((m + 4.0 if m <= start else m, m) for m in range(4))
            passed = [m for parameter, m in unwrapped if start < parameter < end]
            regions = {int(region_ids[corner_indices[m]]) for m in passed}
            if len(regions) != 1:
                return segment, None
            vertices = np.array([p["point"], *corners[passed], q["point"]])
            area = self._polygon_area(vertices[:, [a, b]])
            if area <= 1e-12 * face_area:
                return segment, None
            polygons.append(
                {
                    "region_id": regions.pop(),
                    "vertices": vertices,
                    "area_fraction": area / face_area,
                }
            )

        if polygons[0]["region_id"] == polygons[1]["region_id"]:
            return segment, None
        return segment, polygons

    def _build_interface_face_data(self):
        """
        Build the interface segments and split polygons of all grid faces
        with a cut boundary edge.

        The grid layer stores geometry and region IDs only; the solver
        interprets them in terms of materials.
        """
        self.interface_face_data = {"primal": {}, "dual": {}}

        def edge_pair(cut, axis):
            # The two parallel boundary edges of each face along ``axis``.
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            return cut[tuple(low)] | cut[tuple(high)]

        for grid_type in ("primal", "dual"):
            intersections = self.interface_intersections[grid_type]
            for normal_axis, face_direction in enumerate("xyz"):
                a, b = (normal_axis + 1) % 3, (normal_axis + 2) % 3
                candidate = edge_pair(
                    intersections["xyz"[a]]["has_crossing"], b
                ) | edge_pair(intersections["xyz"[b]]["has_crossing"], a)

                resolved = np.zeros_like(candidate)
                segments = {}
                polygons = {}
                region_area_fractions = {}

                for index in np.argwhere(candidate):
                    index = tuple(int(value) for value in index)
                    segment, split = self._split_face(grid_type, face_direction, index)

                    # Keep a segment even if the split fails, so that the
                    # SIBC can detect unresolved faces.
                    if segment is not None:
                        segments[index] = segment
                    if split is None:
                        continue

                    resolved[index] = True
                    polygons[index] = split
                    region_area_fractions[index] = {
                        polygon["region_id"]: polygon["area_fraction"]
                        for polygon in split
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
        self._build_region_ids()

        # Determine exact STL intersections with primal and dual edges
        #
        # This uses the geometric full-grid-line tracing developed above
        # and is independent of the endpoint transition masks.
        self._build_interface_intersections()

        # Warn about thin features and inconsistent STL crossings.
        self._warn_interface_diagnostics()

        # Split the grid faces crossed by an interface.
        self._build_interface_face_data()


class ConformalSolverMixin:
    """Conformal metrics and material assignment for ``SolverFIT3D``."""

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
        is_normal, is_conductor = self.grid._region_type_flags()
        self.conformal_line_data = {"primal": {}, "dual": {}}

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                (start_regions, end_regions) = self.grid._get_edge_region_ids(
                    grid_type, direction
                )

                data = self.grid.interface_intersections[grid_type][direction]
                alpha = data["alpha"]
                resolved = data["has_crossing"]
                start_normal = is_normal[start_regions]
                end_normal = is_normal[end_regions]
                start_conductor = is_conductor[start_regions]
                end_conductor = is_conductor[end_regions]
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

                self.conformal_line_data[grid_type][direction] = {
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
        self.conformal_face_data = {"primal": {}, "dual": {}}
        conductor_types = {"pec", "sibc"}

        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                geometry = self.grid.interface_face_data[grid_type][direction]
                resolved = geometry["resolved"]
                supported = np.zeros_like(resolved, dtype=bool)
                normal_normal = np.zeros_like(resolved, dtype=bool)
                normal_conductor = np.zeros_like(resolved, dtype=bool)
                normal_sibc = np.zeros_like(resolved, dtype=bool)
                effective_area_fraction = np.full(resolved.shape, np.nan, dtype=float)
                normal_region_ids = {}
                conductor_region_ids = {}

                # Only resolved faces, each split into two regions.
                for index, fractions in geometry["region_area_fractions"].items():
                    region_ids = list(fractions.keys())
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

                        normal_conductor[index] = True
                        normal_sibc[index] = conductor_type == "sibc"

                self.conformal_face_data[grid_type][direction] = {
                    "supported": supported,
                    "normal_normal": (normal_normal),
                    "normal_conductor": normal_conductor,
                    "normal_sibc": normal_sibc,
                    "effective_area_fraction": (effective_area_fraction),
                    "normal_region_ids": (normal_region_ids),
                    "conductor_region_ids": (conductor_region_ids),
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

    def _geometry_array_to_field_component(
        self, values, kind, direction, fill_value=None
    ):
        """
        Map a full primal/dual edge or face array to stored DOFs.

        ``kind`` is "primal_edge", "primal_face", "dual_edge" or
        "dual_face". Stored DOF (i, j, k) takes ``values[(i, j, k) +
        offset]`` (see the table above). Ghost DOFs without a geometric
        counterpart get ``fill_value``:

        - None (default): False for boolean arrays, NaN for floating-point
          arrays (NaN means "no conformal value" to the material code).
        - "nearest": copy the nearest stored value (used for region IDs).
        - any scalar: that constant.
        """
        values = np.asarray(values)
        offset = [0, 0, 0]
        if kind == "dual_edge":
            offset = [1, 1, 1]
            offset["xyz".index(direction)] = 0
        elif kind == "dual_face":
            offset["xyz".index(direction)] = 1

        shape = (self.Nx, self.Ny, self.Nz)
        oi, oj, ok = offset
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

    @staticmethod
    def _harmonic_mean(alpha, start_values, end_values):
        """
        Series average along an edge, ``alpha`` of it in the start region.
        A zero value (an insulator for sigma) makes the average zero.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = 1.0 / (alpha / start_values + (1.0 - alpha) / end_values)
        return np.where((start_values == 0.0) | (end_values == 0.0), 0.0, mean)

    def _diagonal_material(self, name, direction, line_grid, face_grid, tol=1e-12):
        """
        Diagonal effective material of one DOF component at normal-normal
        interfaces.

        For the DOF component along x_i, with n the interface normal,

            m_ii = n_i^2 * m_perp + (1 - n_i^2) * m_parallel,

        where m_perp is the harmonic (series) value along the DOF edge and
        m_parallel the area-weighted arithmetic (parallel) value over its
        face:

            eps, sigma: primal edge (perp) and dual face (parallel)
            mu:         dual edge (perp) and primal face (parallel)

        Returns an array of the stored DOF shape (Nx, Ny, Nz), NaN where
        neither the edge nor its face is cut by a normal-normal interface.
        """

        def line_to_dof(values, fill=None):
            return self._geometry_array_to_field_component(
                values, f"{line_grid}_edge", direction, fill
            )

        def face_to_dof(values, fill=None):
            return self._geometry_array_to_field_component(
                values, f"{face_grid}_face", direction, fill
            )

        axis = "xyz".index(direction)
        line_data = self.conformal_line_data[line_grid][direction]
        face_data = self.conformal_face_data[face_grid][direction]
        face_geometry = self.grid.interface_face_data[face_grid][direction]
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

        # Area-weighted value over the face and its normal component.
        face_n2 = np.full(face_data["normal_normal"].shape, np.nan, dtype=float)
        face_mean = np.full(face_data["normal_normal"].shape, np.nan, dtype=float)
        for index in np.argwhere(face_data["normal_normal"]):
            index = tuple(index)
            fractions = face_geometry["region_area_fractions"][index]
            regions = face_data["normal_region_ids"][index]
            total = sum(float(fractions[r]) for r in regions)
            face_mean[index] = sum(
                float(fractions[r]) / total * region_values[r] for r in regions
            )
            normal = face_geometry["segments"][index]["normal"]
            if normal is not None:
                face_n2[index] = normal[axis] ** 2

        n2 = line_to_dof(line_n2)
        n2 = np.where(np.isfinite(n2), n2, face_to_dof(face_n2))

        # Where only one of edge and face is cut, the uncut one lies in a
        # single region: an uncut edge in its endpoint region, an uncut face
        # in the region of the edge midpoint, which it passes through.
        start, end = line_data["start_region"], line_data["end_region"]
        alpha = line_data["alpha"]
        edge_value = np.where(
            line_data["normal_normal"],
            self._harmonic_mean(alpha, region_values[start], region_values[end]),
            np.where(start == end, region_values[start], np.nan),
        )
        # Crossing before the midpoint (alpha < 0.5): midpoint in the end region.
        # NaN alpha (uncut edge) compares False and selects the start region.
        midpoint_region = np.where(alpha < 0.5, end, start)

        perp = line_to_dof(edge_value)
        parallel = np.where(
            face_cut,
            face_to_dof(face_mean),
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

    def _build_conformal_metrics(self):
        """
        Cut the primal metrics L and iA at conformal interfaces.

        Edges cut by a PEC/SIBC wall keep only the fraction ``L_fraction``
        in the normal material; faces cut by it keep the area fraction
        ``A_fraction`` of their normal part. Normal-normal interfaces and
        edges/faces without a resolved cut keep their grid metric. The dual
        metrics are never cut (Dey and Mittra). The grid metrics are not
        modified; ``self.L`` and ``self.iA`` become new fields.

        With ``GridFIT3D(conformal_stabilization="area_floor")``, faces cut
        by a PEC/SIBC wall use at least ``conformal_area_floor`` as area
        fraction (a constant-floor form of the area limitation of Benkler et
        al. 2006), which bounds the reduction of the stable time step.
        ``A_fraction_geometric`` keeps the exact geometric fractions;
        ``n_area_floored`` counts the raised faces.
        """
        self.L_fraction = Field(self.Nx, self.Ny, self.Nz)
        self.A_fraction = Field(self.Nx, self.Ny, self.Nz)
        self.A_fraction_geometric = Field(self.Nx, self.Ny, self.Nz)
        L = Field(self.Nx, self.Ny, self.Nz)
        iA = Field(self.Nx, self.Ny, self.Nz)
        self.n_area_floored = 0
        area_floor = self.conformal_area_floor

        for direction in ("x", "y", "z"):
            line_data = self.conformal_line_data["primal"][direction]
            length_fraction = np.where(
                line_data["supported"], line_data["effective_length_fraction"], 1.0
            )
            length_fraction = self._geometry_array_to_field_component(
                length_fraction, "primal_edge", direction, fill_value=1.0
            )
            setattr(self.L_fraction, f"field_{direction}", length_fraction)
            setattr(
                L, f"field_{direction}", self.L.to_matrix(direction) * length_fraction
            )

            face_data = self.conformal_face_data["primal"][direction]
            area_fraction = np.where(
                face_data["supported"], face_data["effective_area_fraction"], 1.0
            )
            setattr(
                self.A_fraction_geometric,
                f"field_{direction}",
                self._geometry_array_to_field_component(
                    area_fraction, "primal_face", direction, fill_value=1.0
                ),
            )
            if area_floor > 0.0:
                floored = face_data["normal_conductor"] & (area_fraction < area_floor)
                area_fraction[floored] = area_floor
                self.n_area_floored += int(np.count_nonzero(floored))
            area_fraction = self._geometry_array_to_field_component(
                area_fraction, "primal_face", direction, fill_value=1.0
            )
            setattr(self.A_fraction, f"field_{direction}", area_fraction)

            # Ghost faces of the symmetric stencil have zero area (iA = 0).
            grid_iA = self.iA.to_matrix(direction)
            physical = grid_iA != 0
            area = np.divide(1.0, grid_iA, out=np.zeros_like(grid_iA), where=physical)
            area = area * area_fraction
            setattr(
                iA,
                f"field_{direction}",
                np.divide(1.0, area, out=np.zeros_like(area), where=physical),
            )

        self.L = L
        self.iA = iA

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
        # Zero unless the area floor is active.
        threshold = self.conformal_min_edge_fraction
        if threshold <= 0.0:
            return

        for direction in ("x", "y", "z"):
            ieps = np.array(self.ieps.to_matrix(direction))
            tiny = (self.L_fraction.to_matrix(direction) < threshold) & (ieps != 0.0)
            ieps[tiny] = 0.0
            setattr(self.ieps, f"field_{direction}", ieps)
            self.n_edges_eliminated += int(np.count_nonzero(tiny))

    def _apply_stl_materials_conformal(self):
        """
        Assign STL materials from the conformal region geometry.

        Every DOF first takes the material of its edge endpoints (primal
        edges for E, dual edges for H): the region material if both lie in
        one region, the mean over the normal-material endpoints otherwise,
        and a conductor (ieps = 0) if both lie in PEC/SIBC; at a cut E edge
        this is already the material of its normal part. H DOFs at faces cut
        by a conductor take the material of the normal part of the face, and
        resolved normal-normal interfaces the diagonal effective material
        (``_diagonal_material``), which takes precedence. Only edges without
        a resolved crossing, which the conformal diagnostics warn about,
        keep the endpoint value at a material transition.
        """
        self.stl_solids = self.grid.stl_solids
        self.stl_materials = self.grid.stl_materials
        self.stl_colors = self.grid.stl_colors

        for key in self.stl_solids.keys():
            eps = self.stl_materials[key][0] * eps_0
            sigma = self.stl_materials[key][2]

            if sigma > 0.0:
                self.use_conductivity = True

                if sigma > 10 * eps / eps_0:
                    print(
                        f"[!] Warning: High conductivity sigma={sigma} S/m "
                        f"for solid '{key}' with low permittivity "
                        f"epsilon_r={eps / eps_0} will considerably reduce "
                        f"the maximal stable timestep."
                    )

        # Material table indexed by region ID; region 0 is the background.
        table = np.array(
            [[self.eps_bg / eps_0, self.mu_bg / mu_0, self.sigma_bg]]
            + [self.stl_materials[key] for key in self.grid.region_key_to_id],
            dtype=float,
        )
        self.region_material_properties = region_values = {
            "eps_r": table[:, 0],
            "mu_r": table[:, 1],
            "sigma": table[:, 2],
        }
        is_normal, is_conductor = self.grid._region_type_flags()

        def endpoint_material(name, start, end):
            # A conductor endpoint carries no volumetric material of the edge.
            values = region_values[name]
            only_start = is_normal[start] & ~is_normal[end]
            only_end = is_normal[end] & ~is_normal[start]
            return np.where(
                only_start,
                values[start],
                np.where(only_end, values[end], 0.5 * (values[start] + values[end])),
            )

        def to_dof(values, kind, direction, fill_value=None):
            return self._geometry_array_to_field_component(
                values, kind, direction, fill_value
            )

        materials = {"eps_r": {}, "mu_r": {}, "sigma": {}}

        for direction in ("x", "y", "z"):
            # Electric DOFs: primal edges. Ghost edges copy the nearest
            # region IDs; their length is zero.
            start, end = self.grid._get_edge_region_ids("primal", direction)
            start = to_dof(start, "primal_edge", direction, "nearest")
            end = to_dof(end, "primal_edge", direction, "nearest")
            fully_conductor = is_conductor[start] & is_conductor[end]
            for name in ("eps_r", "sigma"):
                materials[name][direction] = endpoint_material(name, start, end)

            # Interior PEC/SIBC electric DOFs are eliminated through ieps = 0.
            materials["eps_r"][direction][fully_conductor] = np.inf
            materials["sigma"][direction][fully_conductor] = 0.0

            # Magnetic DOFs: dual edges, conductor cuts through primal faces.
            start, end = self.grid._get_edge_region_ids("dual", direction)
            start = to_dof(start, "dual_edge", direction)
            end = to_dof(end, "dual_edge", direction)
            face_data = self.conformal_face_data["primal"][direction]
            face_mu = np.full(face_data["normal_conductor"].shape, np.nan)
            for index, regions in face_data["normal_region_ids"].items():
                if face_data["normal_conductor"][index]:
                    face_mu[index] = region_values["mu_r"][regions[0]]
            cut = to_dof(face_data["normal_conductor"], "primal_face", direction)
            values = endpoint_material("mu_r", start, end)
            values[cut] = to_dof(face_mu, "primal_face", direction)[cut]
            materials["mu_r"][direction] = values

            # The diagonal approximation at normal-normal interfaces takes
            # precedence; NaN marks DOFs without such an interface.
            for name, line_grid, face_grid in (
                ("eps_r", "primal", "dual"),
                ("sigma", "primal", "dual"),
                ("mu_r", "dual", "primal"),
            ):
                values = self._diagonal_material(name, direction, line_grid, face_grid)
                valid = np.isfinite(values)
                materials[name][direction][valid] = values[valid]

        for direction in ("x", "y", "z"):
            setattr(
                self.ieps,
                f"field_{direction}",
                1.0 / (materials["eps_r"][direction] * eps_0),
            )
            setattr(
                self.imu,
                f"field_{direction}",
                1.0 / (materials["mu_r"][direction] * mu_0),
            )
            setattr(self.sigma, f"field_{direction}", materials["sigma"][direction])

        self._eliminate_small_conformal_edges()
