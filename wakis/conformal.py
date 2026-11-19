# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Ray-tracing backend of the conformal geometry.

Every primal grid line, and every dual grid line holding an edge between two
normal materials, is intersected exactly with the STL triangles (VTK
``IntersectWithLine``). Per edge, the crossings are grouped, checked against
the node classification and reduced to one represented crossing; the grid
faces are split along their cut points. The result is the sparse
intersection contract of ``intersection.py`` (``edge_intersections`` and
``face_intersections``) plus the geometry diagnostics.

``ConformalGridMixin`` is mixed into ``GridFIT3D``; its entry point is
``_build_conformal_interfaces``, called by ``GridFIT3D._build_intersections``
for ``conformal_method="raytracing"``.
"""

import numpy as np

# Import only the needed VTK modules: the full ``import vtk`` loads modules
# whose factory overrides can change the results of PyVista filters used
# elsewhere in Wakis (see ImpedanCEI/Wakis#123).
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

from .intersection import EdgeIntersections, FaceIntersections


class ConformalGridMixin:
    """Ray-tracing intersection backend for ``GridFIT3D``."""

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
                edge_alphas = [(int(edge[n]), float(np.clip(alpha[n], 0.0, 1.0)))]
                # A hit on a grid node (within the STL tolerance) belongs to
                # both edges at that node.
                if edge_alphas[0][1] <= self.stl_tol and edge[n] > 0:
                    edge_alphas.append((int(edge[n]) - 1, 1.0))
                elif edge_alphas[0][1] >= 1.0 - self.stl_tol and edge[n] < n_edges - 1:
                    edge_alphas.append((int(edge[n]) + 1, 0.0))
                for edge_axis_index, edge_alpha in edge_alphas:
                    edge_index = list(transverse_index)
                    edge_index.insert(axis, edge_axis_index)
                    hit = {
                        "alpha": edge_alpha,
                        "surface_id": int(surface_id),
                        "cell_id": int(cell_ids[n]),
                        "normal": normals[cell_ids[n]],
                    }
                    edge_hits.setdefault(tuple(edge_index), []).append(hit)

        return edge_hits

    @staticmethod
    def _group_crossings(hits, axis, node_tolerance, alpha_tolerance=1e-8):
        """
        Group the STL hits of one edge by position and keep the crossings.

        Hits within ``alpha_tolerance`` belong to one geometric point (a
        line through a shared triangle edge or vertex, or coincident
        surfaces of different solids). A surface crosses the edge there if
        its facet normals have a component along the edge, all with the same
        sign. Hits of a surface lying in the line (no component) or touching
        it (both signs: the line enters and leaves at the same point, e.g.
        tangent to a cylinder) are dropped, and so are groups left empty. A
        touch on a grid node (within ``node_tolerance`` of an edge end) is
        kept; the caller drops it if it contradicts the node classification.
        """

        def position(hit):
            # Hits within the node tolerance of an edge end are on that node.
            if hit["alpha"] <= node_tolerance:
                return 0.0
            if hit["alpha"] >= 1.0 - node_tolerance:
                return 1.0
            return hit["alpha"]

        hits = sorted(hits, key=position)
        groups = [[hits[0]]]

        for hit in hits[1:]:
            reference = np.mean([position(grouped) for grouped in groups[-1]])
            if abs(position(hit) - reference) <= alpha_tolerance:
                groups[-1].append(hit)
            else:
                groups.append([hit])

        # VTK can report a hit for a facet coplanar with the grid line (e.g.
        # an axis-aligned wall on a grid plane), and a line grazing a curved
        # surface hits the facets on both sides of the contact. Counted as
        # crossings, both would toggle inside/outside wrongly. The normals
        # are unit vectors, so 1e-8 separates "line in the facet plane up to
        # rounding" from any real tilt.
        crossings = []
        for group in groups:
            signs = {}
            for hit in group:
                if abs(hit["normal"][axis]) > 1e-8:
                    signs.setdefault(hit["surface_id"], set()).add(
                        hit["normal"][axis] > 0
                    )
            # A touch on a grid node is kept: the node classification may
            # put the node inside, and the consistency check drops it if not.
            at_node = not (node_tolerance < group[0]["alpha"] < 1.0 - node_tolerance)
            crossing = [
                hit
                for hit in group
                if len(signs.get(hit["surface_id"], ())) == 1
                or (at_node and hit["surface_id"] in signs)
            ]
            if crossing:
                crossings.append(crossing)
        return crossings

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

    def _load_stl_surfaces(self):
        """
        One static cell locator, the unit facet normals and the axis-aligned
        facet planes per solid, reused for all grid lines. STL facets are
        triangles.
        """
        surfaces = {}
        facet_planes = {}
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
            # Coordinates of the axis-aligned planes holding facets, e.g. a
            # flat end cap: a grid line in such a plane runs along the solid.
            aligned = np.abs(normals) > 1.0 - 1e-8
            facet_planes[region_id] = [
                np.unique(triangles[aligned[:, k], 0, k]) for k in range(3)
            ]
        return surfaces, facet_planes

    def _resolve_edge(
        self, hits, axis, start, end, start_inside, planar, alpha_tolerance
    ):
        """
        Reduce the STL hits of one edge to its represented crossing.

        Returns
        -------
        crossing : dict or None
            The representative hit (``alpha``, ``surface_id``, ``normal``)
            and ``multiple`` (several interfaces, or an ambiguous one), or
            None if the edge has no crossing to represent.
        thin_feature, inconsistent : bool
            Diagnostics of the edge.
        """
        crossing_groups = self._group_crossings(
            hits, axis, self.stl_tol, alpha_tolerance=alpha_tolerance
        )
        if not crossing_groups:
            return None, False, False

        regions, interfaces, thin_layer = self._resolve_edge_interfaces(
            crossing_groups, start_inside
        )

        # Where the surface passes through a grid node or the line runs
        # along a surface, the line touches the surface rather than
        # crossing it, and the node classification already decided the
        # side. If such crossings contradict it, they must not toggle the
        # material again.
        consistent = regions[0] == start and regions[-1] == end
        if not consistent:
            clean = [
                group
                for group in crossing_groups
                if self.stl_tol
                < np.mean([hit["alpha"] for hit in group])
                < 1.0 - self.stl_tol
                and not {hit["surface_id"] for hit in group} <= planar
            ]
            retry = self._resolve_edge_interfaces(clean, start_inside)
            if retry[0][0] == start and retry[0][-1] == end:
                crossing_groups = clean
                regions, interfaces, thin_layer = retry
                consistent = True

        # Crossings that contradict the point classification (an open
        # surface) cannot be resolved: keep all of them as one ambiguous
        # interface.
        if not consistent:
            regions, interfaces = [start, end], [crossing_groups]

        # Only covered surfaces or slivers: no material change.
        if not interfaces:
            return None, thin_layer, not consistent

        # A layer thinner than the mesh: a thin coating, a feature inside one
        # material (same material at both ends), or several interfaces of
        # which only one is represented.
        thin_feature = consistent and (
            thin_layer or start == end or len(interfaces) > 1
        )

        # Same material at both ends: no interface to represent.
        if start == end:
            return None, thin_feature, not consistent

        candidates = [
            self._select_interface_hit(groups, left, right)
            for groups, left, right in zip(interfaces, regions[:-1], regions[1:])
        ]
        # Several interfaces in one edge: only one can be represented. A
        # conductor endpoint keeps its own wall (a coating on it is dropped,
        # not the wall moved); otherwise use the interface nearest the edge
        # midpoint.
        if self._is_conductor_region(end):
            representative = candidates[-1]
        elif self._is_conductor_region(start):
            representative = candidates[0]
        else:
            representative = min(candidates, key=lambda hit: abs(hit["alpha"] - 0.5))
        crossing = {
            "alpha": representative["alpha"],
            "surface_id": representative["surface_id"],
            "normal": representative["normal"],
            "multiple": len(interfaces) > 1 or not consistent,
        }
        return crossing, thin_feature, not consistent

    def _trace_edges(
        self, grid_type, direction, surfaces, facet_planes, alpha_tolerance
    ):
        """
        Trace the grid lines of one direction and resolve their edges.

        Primal grid: all lines and edges. Dual grid: only edges between two
        different normal materials, and only the lines holding them; dual
        intersections serve the effective materials only, PEC/SIBC walls
        never cut the dual metric. Each line is resolved right after it is
        traced, so only the represented crossings are kept.

        Returns
        -------
        cuts : dict of ndarray
            ``index`` (n, 3) geometry indices of the cut edges, ``alpha``,
            ``normal`` (n, 3), and the backend-internal ``surface`` and
            ``multiple``.
        diagnostics : dict of ndarray
            ``thin_feature`` and ``inconsistent`` edges as geometry indices
            (n, 3); inconsistent includes material transitions without a
            crossing.
        """
        axis = "xyz".index(direction)
        coordinates = self._get_grid_coordinates(grid_type)
        transverse_axes = [k for k in range(3) if k != axis]
        start_regions, end_regions = self._get_edge_region_ids(grid_type, direction)
        transition = start_regions != end_regions
        transverse_shape = np.delete(self.region_ids[grid_type].shape, axis)
        lines = np.ndindex(*transverse_shape)
        if grid_type == "dual":
            transition &= self._get_normal_normal_edge_mask(grid_type, direction)
            traced = np.any(transition, axis=axis)
            lines = (line for line in lines if traced[line])
        node_masks = (
            self.primal_node_masks if grid_type == "primal" else self.dual_node_masks
        )
        # A plane closer to a grid line than the STL tolerance contains it.
        plane_tolerance = self.stl_tol * min(
            np.min(np.diff(c)) for c in self._get_grid_coordinates("primal")
        )

        records = {
            "index": [],
            "alpha": [],
            "normal": [],
            "surface": [],
            "multiple": [],
        }
        diagnostics = {"thin_feature": [], "inconsistent": []}
        for line in lines:
            line_hits = self._trace_grid_line(
                grid_type, direction, line, surfaces, alpha_tolerance=alpha_tolerance
            )
            if not line_hits:
                continue

            # Surfaces with a facet in the plane of this line: the line runs
            # along them, and its crossings of them are at the rim of that
            # planar part. Axis-aligned facet planes are matched by
            # coordinate; VTK also reports coplanar hits for other planes.
            line_coordinates = [
                (k, coordinates[k][index]) for k, index in zip(transverse_axes, line)
            ]
            planar = {
                surface_id
                for surface_id, planes in facet_planes.items()
                if any(
                    np.any(np.abs(planes[k] - value) <= plane_tolerance)
                    for k, value in line_coordinates
                )
            } | {
                hit["surface_id"]
                for hits in line_hits.values()
                for hit in hits
                if abs(hit["normal"][axis]) <= 1e-8
            }

            for edge_index, hits in line_hits.items():
                if grid_type == "dual" and not transition[edge_index]:
                    continue
                start_inside = {
                    self.region_key_to_id[key]: bool(node_masks[key][edge_index])
                    for key in self.stl_solids
                }
                crossing, thin, inconsistent = self._resolve_edge(
                    hits,
                    axis,
                    int(start_regions[edge_index]),
                    int(end_regions[edge_index]),
                    start_inside,
                    planar,
                    alpha_tolerance,
                )
                if thin:
                    diagnostics["thin_feature"].append(edge_index)
                if inconsistent:
                    diagnostics["inconsistent"].append(edge_index)
                if crossing is None:
                    continue
                records["index"].append(edge_index)
                records["alpha"].append(crossing["alpha"])
                records["normal"].append(crossing["normal"])
                records["surface"].append(crossing["surface_id"])
                records["multiple"].append(crossing["multiple"])

        cuts = {
            "index": np.array(records["index"], dtype=np.int64).reshape(-1, 3),
            "alpha": np.array(records["alpha"], dtype=float),
            "normal": np.array(records["normal"], dtype=float).reshape(-1, 3),
            "surface": np.array(records["surface"], dtype=np.int32),
            "multiple": np.array(records["multiple"], dtype=bool),
        }
        # Material transitions without a represented crossing.
        missing = transition.copy()
        missing[tuple(cuts["index"].T)] = False
        inconsistent = np.zeros(transition.shape, dtype=bool)
        if diagnostics["inconsistent"]:
            inconsistent[tuple(np.array(diagnostics["inconsistent"]).T)] = True
        diagnostics = {
            "thin_feature": np.array(
                diagnostics["thin_feature"], dtype=np.int64
            ).reshape(-1, 3),
            "inconsistent": np.argwhere(inconsistent | missing),
        }

        if self.verbose:
            print(
                f"    * {grid_type} {direction}-edges: {len(cuts['alpha'])} "
                f"crossings, {np.count_nonzero(cuts['multiple'])} with multiple "
                "crossing positions"
            )
        return cuts, diagnostics

    def _split_faces(self, grid_type, cuts):
        """
        Split the grid faces of one grid along their cut points.

        A face with exactly two distinct cut points of one common STL
        surface, none from an edge with several crossings, is split when
        each side holds corners of a single region and the two regions
        differ (``_face_polygons``). Its interface normal is the mean facet
        normal at the two cut points.

        Returns
        -------
        FaceIntersections
        """
        shape = self.region_ids[grid_type].shape
        rows = {}
        for axis, direction in enumerate("xyz"):
            edge_shape = list(shape)
            edge_shape[axis] -= 1
            keys = np.ravel_multi_index(cuts[direction]["index"].T, edge_shape)
            rows[axis] = (edge_shape, dict(zip(keys.tolist(), range(len(keys)))))

        def lookup(axis, edge_index):
            edge_shape, row_of = rows[axis]
            row = row_of.get(int(np.ravel_multi_index(edge_index, edge_shape)))
            if row is None:
                return None
            data = cuts["xyz"[axis]]
            return float(data["alpha"][row]), (
                int(data["surface"][row]),
                data["normal"][row],
                bool(data["multiple"][row]),
            )

        records = {
            "face": [],
            "direction": [],
            "a": [],
            "b": [],
            "fraction_a": [],
            "normal": [],
        }
        for normal_axis, face_direction in enumerate("xyz"):
            a, b = (normal_axis + 1) % 3, (normal_axis + 2) % 3
            face_shape = np.array(shape) - 1
            face_shape[normal_axis] += 1
            # Faces next to a cut edge: an edge along a borders the faces at
            # its own index and one below along b, and vice versa.
            candidates = []
            for edge_axis, other in ((a, b), (b, a)):
                index = cuts["xyz"[edge_axis]]["index"]
                for shift in (0, -1):
                    face = index.copy()
                    face[:, other] += shift
                    inside = np.all((face >= 0) & (face < face_shape), axis=1)
                    candidates.append(face[inside])
            candidates = np.unique(np.concatenate(candidates), axis=0)

            for index in candidates:
                index = tuple(int(value) for value in index)
                corner_indices, corners, plane_axes, groups = self._face_cut_groups(
                    grid_type, face_direction, index, lookup
                )
                if len(groups) != 2 or any(
                    cut["data"][2] for group in groups for cut in group["cuts"]
                ):
                    continue
                common = {cut["data"][0] for cut in groups[0]["cuts"]} & {
                    cut["data"][0] for cut in groups[1]["cuts"]
                }
                if len(common) != 1:
                    continue
                polygons = self._face_polygons(
                    grid_type, corner_indices, corners, plane_axes, groups
                )
                if polygons is None:
                    continue

                surface_id = common.pop()
                mean_normal = np.mean(
                    [
                        cut["data"][1]
                        for group in groups
                        for cut in group["cuts"]
                        if cut["data"][0] == surface_id
                    ],
                    axis=0,
                )
                norm = np.linalg.norm(mean_normal)
                (region_a, _, fraction_a), (region_b, _, _) = sorted(
                    polygons, key=lambda polygon: polygon[0]
                )
                records["face"].append(index)
                records["direction"].append(normal_axis)
                records["a"].append(region_a)
                records["b"].append(region_b)
                records["fraction_a"].append(fraction_a)
                records["normal"].append(
                    mean_normal / norm if norm > 0.0 else np.full(3, np.nan)
                )

            if self.verbose:
                print(
                    f"    * {grid_type} {face_direction}-faces: {len(candidates)} "
                    "candidates"
                )

        index = np.array(records["face"], dtype=np.int64).reshape(-1, 3)
        normal_axis = np.array(records["direction"], dtype=np.int64)
        face = np.empty(len(index), dtype=np.int64)
        for axis, face_direction in enumerate("xyz"):
            selected = normal_axis == axis
            face[selected] = self._flat_field_index(
                grid_type, "face", face_direction, index[selected]
            )
        # Faces without a field DOF (dual faces on the low domain boundary)
        # are not part of the contract.
        keep = face >= 0
        return FaceIntersections(
            face=face[keep],
            region_a=np.array(records["a"], dtype=self.region_ids[grid_type].dtype)[
                keep
            ],
            region_b=np.array(records["b"], dtype=self.region_ids[grid_type].dtype)[
                keep
            ],
            fraction_a=np.array(records["fraction_a"], dtype=float)[keep],
            normal=np.array(records["normal"], dtype=float).reshape(-1, 3)[keep],
        )

    def _build_conformal_interfaces(self):
        """
        Build the intersection contract by ray tracing.

        Used both after classifying the grid points and after loading the
        node masks from HDF5, so that both paths give identical cut-cell
        geometry. The STL surfaces are traced again and must be readable.
        """
        alpha_tolerance = 1e-8
        surfaces, facet_planes = self._load_stl_surfaces()
        self.edge_intersections = {}
        self.face_intersections = {}
        self.conformal_diagnostics = {}
        for grid_type in ("primal", "dual"):
            cuts = {}
            self.conformal_diagnostics[grid_type] = {}
            for direction in "xyz":
                cuts[direction], self.conformal_diagnostics[grid_type][direction] = (
                    self._trace_edges(
                        grid_type, direction, surfaces, facet_planes, alpha_tolerance
                    )
                )
            self.edge_intersections[grid_type] = EdgeIntersections(
                edge=np.concatenate(
                    [
                        self._flat_field_index(
                            grid_type, "edge", direction, cuts[direction]["index"]
                        )
                        for direction in "xyz"
                    ]
                ),
                alpha=np.concatenate([cuts[d]["alpha"] for d in "xyz"]),
                normal=np.concatenate([cuts[d]["normal"] for d in "xyz"]),
            )
            self.face_intersections[grid_type] = self._split_faces(grid_type, cuts)

        # Edges without a field DOF (dual edges on the low domain boundary)
        # serve only the face split.
        for grid_type, cuts in self.edge_intersections.items():
            keep = cuts.edge >= 0
            self.edge_intersections[grid_type] = EdgeIntersections(
                cuts.edge[keep], cuts.alpha[keep], cuts.normal[keep]
            )

        self._warn_interface_diagnostics()
