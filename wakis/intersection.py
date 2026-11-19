# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Conformal geometry from STL intersection data.

A conformal backend (``conformal.py``: ray tracing) intersects the STL
solids with the primal and dual grid and describes the result with two
sparse lists per grid, one record per cut edge (``EdgeIntersections``) and
per cut face (``FaceIntersections``). This module holds that contract and
everything computed from it alone, so the backends are interchangeable:

- ``IntersectionGridMixin`` (mixed into ``GridFIT3D``): region helpers, the
  geometric split of a face along its cut points, and the geometry
  diagnostics shared by the backends.
- ``IntersectionSolverMixin`` (mixed into ``SolverFIT3D``): the cut primal
  metrics and the effective materials.

Edges and faces are given by their flat field index (see
``GridFIT3D._flat_field_index``); the solid IDs on both sides of an edge
are those of its end nodes (``GridFIT3D.region_ids``). Edges and faces that
are not listed are uncut.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0

from .field import Field


@dataclass
class EdgeIntersections:
    """
    Cut edges of one grid (primal or dual).

    On the primal grid every edge whose end nodes have different solid IDs
    is cut; on the dual grid only edges between two different normal
    materials, since conductor walls never cut the dual metric. Each edge
    carries one crossing.

    Attributes
    ----------
    edge : ndarray of int64, shape (n,)
        Flat field index of the edge.
    alpha : ndarray of float, shape (n,)
        Crossing position along the edge, 0 to 1, from its start node.
    normal : ndarray of float, shape (n, 3)
        Unit normal of the interface at the crossing; its sign is not used.
    """

    edge: np.ndarray
    alpha: np.ndarray
    normal: np.ndarray

    def __post_init__(self):
        n = len(self.edge)
        if len(self.alpha) != n or np.shape(self.normal) != (n, 3):
            raise ValueError(
                "EdgeIntersections needs edge (n,), alpha (n,) and normal (n, 3)."
            )


@dataclass
class FaceIntersections:
    """
    Cut faces of one grid (primal or dual).

    Each face is split between two solids. On the primal grid every such
    face is listed; on the dual grid only faces between two different
    normal materials.

    Attributes
    ----------
    face : ndarray of int64, shape (m,)
        Flat field index of the face.
    region_a, region_b : ndarray of int, shape (m,)
        Solid IDs of the two parts, ``region_a < region_b``.
    fraction_a : ndarray of float, shape (m,)
        Area fraction of ``region_a``; ``region_b`` has ``1 - fraction_a``.
    normal : ndarray of float, shape (m, 3)
        Unit normal of the interface in the face (NaN if unknown); its sign
        is not used.
    """

    face: np.ndarray
    region_a: np.ndarray
    region_b: np.ndarray
    fraction_a: np.ndarray
    normal: np.ndarray

    def __post_init__(self):
        m = len(self.face)
        if any(len(v) != m for v in (self.region_a, self.region_b, self.fraction_a)):
            raise ValueError(
                "FaceIntersections needs face, region_a, region_b and fraction_a "
                "of equal length."
            )
        if np.shape(self.normal) != (m, 3):
            raise ValueError("FaceIntersections needs normal of shape (m, 3).")


def _sparse_lookup(keys, query):
    """Positions of ``query`` in the (unsorted, unique) ``keys``, -1 where
    absent."""
    if len(keys) == 0:
        return np.full(np.shape(query), -1, dtype=np.int64)
    order = np.argsort(keys)
    position = np.minimum(np.searchsorted(keys[order], query), len(keys) - 1)
    return np.where(keys[order][position] == query, order[position], -1)


def _gather(values, position, fill):
    """``values[position]``, and ``fill`` where ``position`` is -1."""
    result = np.full((len(position),) + np.shape(values)[1:], fill, dtype=float)
    found = position >= 0
    result[found] = values[position[found]]
    return result


class IntersectionGridMixin:
    """Region helpers, face splits and diagnostics for ``GridFIT3D``."""

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
        """Node coordinates of the primal or dual grid."""
        if grid_type == "primal":
            return self.x, self.y, self.z
        return self.tx, self.ty, self.tz

    def _region_type_flags(self):
        """
        Boolean arrays indexed by region ID: normal material, and
        conductor (PEC or SIBC).
        """
        types = np.array(
            [self.region_material_types[r] for r in range(len(self.region_id_to_key))]
        )
        return types == "normal", (types == "pec") | (types == "sibc")

    def _is_conductor_region(self, region_id):
        return self.region_material_types[region_id] in ("pec", "sibc")

    def _get_edge_region_ids(self, grid_type, direction):
        """Start and end region IDs of all edges in one direction (views)."""
        region_ids = self.region_ids[grid_type]
        if direction == "x":
            return (region_ids[:-1, :, :], region_ids[1:, :, :])
        if direction == "y":
            return (region_ids[:, :-1, :], region_ids[:, 1:, :])
        return (region_ids[:, :, :-1], region_ids[:, :, 1:])

    def _get_normal_normal_edge_mask(self, grid_type, direction):
        """Edges whose two endpoints lie in different normal-material regions."""
        start, end = self._get_edge_region_ids(grid_type, direction)
        is_normal, _ = self._region_type_flags()
        return (start != end) & is_normal[start] & is_normal[end]

    def _flat_index_to_geometry(self, grid_type, kind, flat):
        """
        Inverse of ``_flat_field_index``: direction axis and geometry
        indices (n, 3) of the edges or faces with the given flat indices.
        """
        flat = np.asarray(flat, dtype=np.int64)
        N = self.Nx * self.Ny * self.Nz
        axis = flat // N
        index = np.column_stack(
            np.unravel_index(flat % N, (self.Nx, self.Ny, self.Nz), order="F")
        )
        if grid_type == "dual":
            offset = np.zeros_like(index)
            for k in range(3):
                if kind == "edge":
                    offset[:, k] = axis != k
                else:
                    offset[:, k] = axis == k
            index = index + offset
        return axis, index

    # ------------------------------------------------------------------
    # Face split along the cut points of its boundary edges
    # ------------------------------------------------------------------

    @staticmethod
    def _polygon_area(points_2d):
        """Area of a simple polygon from its vertices in order (shoelace)."""
        u, v = points_2d[:, 0], points_2d[:, 1]
        return float(abs(0.5 * (np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))))

    def _face_cut_groups(self, grid_type, face_direction, index, lookup):
        """
        Corners and grouped cut points of one grid face.

        ``lookup(axis, edge_index)`` returns ``(alpha, data)`` for a cut edge
        (geometry index along ``axis``) or None. Corners run
        counter-clockwise about the face normal (e_a x e_b = e_n); boundary
        edge m runs from corner m to corner m + 1, so a point at boundary
        parameter m + t lies on it at fraction t. Cut points of two edges at
        their shared corner form one group.

        Returns
        -------
        corner_indices, corners, (a, b), groups
            ``groups`` is sorted by boundary parameter; each holds the
            ``point``, its ``parameter`` and the ``cuts`` with their ``data``.
        """
        normal_axis = "xyz".index(face_direction)
        a, b = (normal_axis + 1) % 3, (normal_axis + 2) % 3
        coordinates = self._get_grid_coordinates(grid_type)

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
            found = lookup(axis, c0 if forward else c1)
            if found is None:
                continue
            alpha, data = found
            start, end = (m, (m + 1) % 4) if forward else ((m + 1) % 4, m)
            cut_points.append(
                {
                    "point": corners[start] + alpha * (corners[end] - corners[start]),
                    "parameter": m + (alpha if forward else 1.0 - alpha),
                    "data": data,
                }
            )

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
        return corner_indices, corners, (a, b), groups

    def _face_polygons(self, grid_type, corner_indices, corners, plane_axes, groups):
        """
        Split a face with two cut-point groups into its two polygons.

        Both sides run from one cut point along the boundary, through the
        corners passed, to the other cut point. The split is valid when
        each side holds corners of one region and the two regions differ.

        Returns
        -------
        list of (region_id, vertices, area_fraction) or None
        """
        a, b = plane_axes
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
                return None
            vertices = np.array([p["point"], *corners[passed], q["point"]])
            area = self._polygon_area(vertices[:, [a, b]])
            if area <= 1e-12 * face_area:
                return None
            polygons.append((regions.pop(), vertices, area / face_area))

        if polygons[0][0] == polygons[1][0]:
            return None
        return polygons

    def _contract_polygons(self, grid_type, face_direction, index):
        """
        Polygons of one face listed in ``face_intersections``, rebuilt from
        the cut edges of the contract (for plotting).
        """
        cuts = self.edge_intersections[grid_type]

        def lookup(axis, edge_index):
            flat = self._flat_field_index(grid_type, "edge", "xyz"[axis], edge_index)
            position = _sparse_lookup(cuts.edge, flat)
            return None if position < 0 else (float(cuts.alpha[position]), None)

        corner_indices, corners, plane_axes, groups = self._face_cut_groups(
            grid_type, face_direction, index, lookup
        )
        if len(groups) != 2:
            return None
        return self._face_polygons(
            grid_type, corner_indices, corners, plane_axes, groups
        )

    # ------------------------------------------------------------------
    # Diagnostics (filled by the backend, sparse geometry indices)
    # ------------------------------------------------------------------

    def _get_edge_diagnostic_masks(self, grid_type, direction):
        """
        Edges on which the conformal geometry is approximated or unreliable.

        The backend stores them in ``conformal_diagnostics[grid_type]
        [direction]`` as geometry indices (n, 3) under ``"thin_feature"``
        (material structure thinner than the mesh) and ``"inconsistent"``
        (STL crossings and node classification disagree, e.g. open
        surfaces). Returns both as boolean edge masks.
        """
        start, _ = self._get_edge_region_ids(grid_type, direction)
        masks = []
        for key in ("thin_feature", "inconsistent"):
            mask = np.zeros(start.shape, dtype=bool)
            index = self.conformal_diagnostics[grid_type][direction][key]
            mask[tuple(index.T)] = True
            masks.append(mask)
        return tuple(masks)

    def _warn_interface_diagnostics(self):
        """
        Warn about edges with geometry thinner than the mesh resolution and
        edges where the STL crossings and point classification disagree.
        """
        counts = {"thin_feature": 0, "inconsistent": 0}
        for grid_type in ("primal", "dual"):
            for direction in ("x", "y", "z"):
                for key in counts:
                    counts[key] += len(
                        self.conformal_diagnostics[grid_type][direction][key]
                    )

        messages = []
        if counts["thin_feature"]:
            messages.append(
                f"{counts['thin_feature']} edges contain material layers thinner "
                "than the mesh resolution; at most one interface per edge is "
                "represented. Refine the mesh there if these layers matter."
            )
        if counts["inconsistent"]:
            messages.append(
                f"{counts['inconsistent']} edges have STL crossings that disagree "
                "with the grid point classification and receive no or an ambiguous "
                "conformal correction. Check that the STL solids are closed "
                "and not tangent to grid points."
            )
        if messages:
            warnings.warn(
                "Conformal geometry diagnostics:\n" + "\n".join(messages),
                UserWarning,
                stacklevel=2,
            )


class IntersectionSolverMixin:
    """Cut metrics and effective materials for ``SolverFIT3D``."""

    # Geometry array -> stored DOF mapping (symmetric FIT stencil)
    #
    # Nx, Ny, Nz count primal nodes. Every field component stores
    # Nx * Ny * Nz values; entries without a geometric counterpart are
    # zero-size ghost DOFs on the high side (L = 0 or iA = 0).
    #
    #   primal x-edge (Nx-1, Ny,   Nz  )  (i,j,k)     -> Ex(i,j,k)
    #   primal x-face (Nx,   Ny-1, Nz-1)  (i,j,k)     -> Hx(i,j,k)
    #   dual   x-edge (Nx,   Ny+1, Nz+1)  (i,j+1,k+1) -> Hx(i,j,k)
    #   dual   x-face (Nx+1, Ny,   Nz  )  (i+1,j,k)   -> Ex(i,j,k)
    #
    # and cyclically for y and z (see GridFIT3D._flat_field_index).

    def _geometry_array_to_field_component(
        self, values, kind, direction, fill_value=None
    ):
        """
        Map a full primal/dual edge or face array to stored DOFs.

        ``kind`` is "primal_edge", "primal_face", "dual_edge" or
        "dual_face". Ghost DOFs without a geometric counterpart get
        ``fill_value``: None (default) gives False for boolean and NaN for
        floating-point arrays, "nearest" copies the nearest stored value
        (used for region IDs), any scalar is used as it is.
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

    def _cut_edge_regions(self, grid_type, edge):
        """Solid IDs at the start and end node of listed cut edges."""
        axis, index = self.grid._flat_index_to_geometry(grid_type, "edge", edge)
        region_ids = self.grid.region_ids[grid_type]
        end = index + np.eye(3, dtype=np.int64)[axis]
        return region_ids[tuple(index.T)], region_ids[tuple(end.T)]

    def _conductor_cuts(self):
        """
        Primal edges and faces cut by a PEC/SIBC wall and the fraction of
        them in the normal material.

        Returns
        -------
        edge, edge_fraction, face, face_fraction, face_normal_region : ndarray
        """
        is_normal, is_conductor = self.grid._region_type_flags()

        cuts = self.grid.edge_intersections["primal"]
        start, end = self._cut_edge_regions("primal", cuts.edge)
        to_conductor = is_normal[start] & is_conductor[end]
        from_conductor = is_conductor[start] & is_normal[end]
        selected = to_conductor | from_conductor
        edge_fraction = np.where(to_conductor, cuts.alpha, 1.0 - cuts.alpha)[selected]

        faces = self.grid.face_intersections["primal"]
        a_normal = is_normal[faces.region_a] & is_conductor[faces.region_b]
        b_normal = is_conductor[faces.region_a] & is_normal[faces.region_b]
        conductor_face = a_normal | b_normal
        face_fraction = np.where(a_normal, faces.fraction_a, 1.0 - faces.fraction_a)
        normal_region = np.where(a_normal, faces.region_a, faces.region_b)
        return (
            cuts.edge[selected],
            edge_fraction,
            faces.face[conductor_face],
            face_fraction[conductor_face],
            normal_region[conductor_face],
        )

    def _build_conformal_metrics(self):
        """
        Cut the primal metrics L and iA at conformal interfaces.

        Edges cut by a PEC/SIBC wall keep only their fraction in the normal
        material; faces cut by it keep the area fraction of their normal
        part. Normal-normal interfaces and edges/faces without a listed cut
        keep their grid metric. The dual metrics are never cut (Dey and
        Mittra). The grid metrics are not modified; ``self.L`` and
        ``self.iA`` become new fields.

        With ``GridFIT3D(conformal_stabilization="area_floor")``, faces cut
        by a PEC/SIBC wall use at least ``conformal_area_floor`` as area
        fraction (a constant-floor form of the area limitation of Benkler et
        al. 2006), which bounds the reduction of the stable time step.
        ``area_floored_faces`` lists the raised faces (flat field indices)
        and ``n_area_floored`` counts them; edges whose fraction is below
        ``conformal_min_edge_fraction`` are eliminated later
        (``_eliminate_small_conformal_edges``).
        """
        edge, edge_fraction, face, face_fraction, _ = self._conductor_cuts()

        L = Field(self.Nx, self.Ny, self.Nz)
        L.fromarray(self.L.toarray().copy())
        L.array[edge] = L.array[edge] * edge_fraction

        floored = face_fraction < self.conformal_area_floor
        face_fraction = np.where(floored, self.conformal_area_floor, face_fraction)
        self.area_floored_faces = face[floored]
        self.n_area_floored = int(np.count_nonzero(floored))

        # The grid stores inverse face areas.
        iA = Field(self.Nx, self.Ny, self.Nz)
        iA.fromarray(self.iA.toarray().copy())
        iA.array[face] = 1.0 / ((1.0 / iA.array[face]) * face_fraction)

        self._small_cut_edges = edge[edge_fraction < self.conformal_min_edge_fraction]
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
        edge = self._small_cut_edges[self.ieps.array[self._small_cut_edges] != 0.0]
        self.ieps.array[edge] = 0.0
        self.eliminated_edges = edge
        self.n_edges_eliminated = len(edge)

    @staticmethod
    def _harmonic_mean(alpha, start_values, end_values):
        """
        Series average along an edge, ``alpha`` of it in the start region.
        A zero value (an insulator for sigma) makes the average zero.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = 1.0 / (alpha / start_values + (1.0 - alpha) / end_values)
        return np.where((start_values == 0.0) | (end_values == 0.0), 0.0, mean)

    def _diagonal_material(self, name, axis, line_grid, face_grid, tol=1e-12):
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

        Returns
        -------
        dof : ndarray of int
            DOF indices within the component (Fortran order) that get a
            value: their edge or face is cut by a normal-normal interface.
        values : ndarray of float
            The diagonal material there; NaN where it cannot be formed.
        """
        N = self.Nx * self.Ny * self.Nz
        is_normal, _ = self.grid._region_type_flags()
        region_values = self.region_material_properties[name]

        # Cut edges of this component (any type: their normal and position
        # also serve an uncut face) and the normal-normal ones among them.
        edges = self.grid.edge_intersections[line_grid]
        in_component = edges.edge // N == axis
        edge_dof = edges.edge[in_component] % N
        edge_alpha = edges.alpha[in_component]
        edge_normal = edges.normal[in_component]
        start, end = self._cut_edge_regions(line_grid, edges.edge[in_component])
        edge_nn = is_normal[start] & is_normal[end]

        # Normal-normal cut faces of this component.
        faces = self.grid.face_intersections[face_grid]
        face_nn = (
            (faces.face // N == axis)
            & is_normal[faces.region_a]
            & is_normal[faces.region_b]
        )
        face_dof = faces.face[face_nn] % N
        fraction_a = faces.fraction_a[face_nn]
        fraction_b = 1.0 - fraction_a
        total = fraction_a + fraction_b
        face_mean = (
            fraction_a / total * region_values[faces.region_a[face_nn]]
            + fraction_b / total * region_values[faces.region_b[face_nn]]
        )
        face_normal = faces.normal[face_nn]

        dof = np.union1d(edge_dof[edge_nn], face_dof)
        at_edge = _sparse_lookup(edge_dof, dof)
        at_face = _sparse_lookup(face_dof, dof)
        has_edge = at_edge >= 0
        line_cut = _gather(edge_nn, at_edge, False).astype(bool)
        face_cut = at_face >= 0

        # The DOF's own edge: its end regions, and whether it exists (high
        # ghost E DOFs have none).
        index = np.column_stack(
            np.unravel_index(dof, (self.Nx, self.Ny, self.Nz), order="F")
        )
        if line_grid == "dual":
            offset = np.ones(3, dtype=np.int64)
            offset[axis] = 0
            index = index + offset
        region_ids = self.grid.region_ids[line_grid]
        exists = index[:, axis] < region_ids.shape[axis] - 1
        index_end = index + np.eye(3, dtype=np.int64)[axis]
        node = np.minimum(index, np.array(region_ids.shape) - 1)
        node_end = np.minimum(index_end, np.array(region_ids.shape) - 1)
        dof_start = region_ids[tuple(node.T)]
        dof_end = region_ids[tuple(node_end.T)]

        alpha = _gather(edge_alpha, at_edge, np.nan)
        normal = np.where(
            has_edge[:, None],
            _gather(edge_normal, at_edge, np.nan),
            _gather(face_normal, at_face, np.nan),
        )
        with np.errstate(invalid="ignore"):
            n2 = (normal[:, axis] / np.linalg.norm(normal, axis=1)) ** 2

        # Where only one of edge and face is cut, the uncut one lies in a
        # single region: an uncut edge in its endpoint region, an uncut face
        # in the region of the edge midpoint, which it passes through.
        perp = np.where(
            line_cut,
            self._harmonic_mean(
                alpha, region_values[dof_start], region_values[dof_end]
            ),
            np.where(dof_start == dof_end, region_values[dof_start], np.nan),
        )
        perp = np.where(exists, perp, np.nan)
        # Crossing before the midpoint (alpha < 0.5): midpoint in the end region.
        # NaN alpha (uncut edge) compares False and selects the start region.
        midpoint_region = np.where(alpha < 0.5, dof_end, dof_start)
        parallel = np.where(
            face_cut,
            _gather(face_mean, at_face, np.nan),
            np.where(exists, region_values[midpoint_region], np.nan),
        )

        cut = np.isfinite(n2)
        n2 = np.clip(n2, 0.0, 1.0)
        values = np.full(len(dof), np.nan)
        both = cut & np.isfinite(perp) & np.isfinite(parallel)
        values[both] = n2[both] * perp[both] + (1.0 - n2[both]) * parallel[both]
        # Purely normal or tangential interfaces need only one of the values.
        pure_normal = cut & np.isfinite(perp) & (1.0 - n2 <= tol)
        values[pure_normal] = perp[pure_normal]
        pure_tangential = cut & np.isfinite(parallel) & (n2 <= tol)
        values[pure_tangential] = parallel[pure_tangential]
        return dof, values

    def _apply_stl_materials_conformal(self):
        """
        Assign STL materials from the conformal region geometry.

        Every DOF first takes the material of its edge endpoints (primal
        edges for E, dual edges for H): the region material if both lie in
        one region, the mean over the normal-material endpoints otherwise,
        and a conductor (ieps = 0) if both lie in PEC/SIBC; at a cut E edge
        this is already the material of its normal part. H DOFs at faces cut
        by a conductor take the material of the normal part of the face, and
        normal-normal interfaces the diagonal effective material
        (``_diagonal_material``), which takes precedence. Only edges without
        a listed crossing, which the conformal diagnostics warn about, keep
        the endpoint value at a material transition.
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
        N = self.Nx * self.Ny * self.Nz
        shape = (self.Nx, self.Ny, self.Nz)

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

        def assign(target, dof, values):
            target[np.unravel_index(dof, shape, order="F")] = values

        # H DOFs at faces cut by a conductor: material of the normal part.
        _, _, conductor_face, _, normal_region = self._conductor_cuts()

        materials = {"eps_r": {}, "mu_r": {}, "sigma": {}}
        for axis, direction in enumerate("xyz"):
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
            materials["mu_r"][direction] = endpoint_material("mu_r", start, end)
            in_component = conductor_face // N == axis
            assign(
                materials["mu_r"][direction],
                conductor_face[in_component] % N,
                region_values["mu_r"][normal_region[in_component]],
            )

            # The diagonal approximation at normal-normal interfaces takes
            # precedence; NaN marks DOFs where it cannot be formed.
            for name, line_grid, face_grid in (
                ("eps_r", "primal", "dual"),
                ("sigma", "primal", "dual"),
                ("mu_r", "dual", "primal"),
            ):
                dof, values = self._diagonal_material(name, axis, line_grid, face_grid)
                valid = np.isfinite(values)
                assign(materials[name][direction], dof[valid], values[valid])

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
