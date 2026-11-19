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

- the geometric split of a face along its cut points (``face_cut_groups``,
  ``face_polygons``, ``contract_polygons``) and the geometry diagnostics
  shared by the backends (``edge_diagnostic_masks``, ``warn_diagnostics``);
- the cut primal metrics (``conformal_metrics``, ``eliminate_small_edges``)
  and the effective materials (``conformal_materials``), used by
  ``SolverFIT3D``.

Edges and faces are given by their flat field index (see
``GridFIT3D._flat_field_index``); the solid IDs on both sides of an edge
are those of its end nodes (``GridFIT3D.region_ids``). Edges and faces that
are not listed are uncut.
"""

import warnings
from dataclasses import dataclass

import numpy as np

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


def normal_normal_edge_mask(grid, grid_type, direction):
    """Edges whose two endpoints lie in different normal-material regions."""
    start, end = grid._get_edge_region_ids(grid_type, direction)
    is_normal, _ = grid._region_type_flags()
    return (start != end) & is_normal[start] & is_normal[end]


def _polygon_area(points_2d):
    """Area of a simple polygon from its vertices in order (shoelace)."""
    u, v = points_2d[:, 0], points_2d[:, 1]
    return float(abs(0.5 * (np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))))


def face_cut_groups(grid, grid_type, face_direction, index, lookup):
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
    coordinates = grid._get_grid_coordinates(grid_type)

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

    tolerance = 1e-8 * np.max(np.linalg.norm(np.roll(corners, -1, 0) - corners, axis=1))
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


def face_polygons(grid, grid_type, corner_indices, corners, plane_axes, groups):
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
    region_ids = grid.region_ids[grid_type]
    face_area = _polygon_area(corners[:, [a, b]])
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
        area = _polygon_area(vertices[:, [a, b]])
        if area <= 1e-12 * face_area:
            return None
        polygons.append((regions.pop(), vertices, area / face_area))

    if polygons[0][0] == polygons[1][0]:
        return None
    return polygons


def contract_polygons(grid, grid_type, face_direction, index):
    """
    Polygons of one face listed in ``face_intersections``, rebuilt from
    the cut edges of the contract (for plotting).
    """
    cuts = grid.edge_intersections[grid_type]

    def lookup(axis, edge_index):
        flat = grid._flat_field_index(grid_type, "edge", "xyz"[axis], edge_index)
        position = _sparse_lookup(cuts.edge, flat)
        return None if position < 0 else (float(cuts.alpha[position]), None)

    corner_indices, corners, plane_axes, groups = face_cut_groups(
        grid, grid_type, face_direction, index, lookup
    )
    if len(groups) != 2:
        return None
    return face_polygons(grid, grid_type, corner_indices, corners, plane_axes, groups)


def edge_diagnostic_masks(grid, grid_type, direction):
    """
    Edges on which the conformal geometry is approximated or unreliable.

    The backend stores them in ``conformal_diagnostics[grid_type]
    [direction]`` as geometry indices (n, 3) under ``"thin_feature"``
    (material structure thinner than the mesh) and ``"inconsistent"``
    (STL crossings and node classification disagree, e.g. open
    surfaces). Returns both as boolean edge masks.
    """
    start, _ = grid._get_edge_region_ids(grid_type, direction)
    masks = []
    for key in ("thin_feature", "inconsistent"):
        mask = np.zeros(start.shape, dtype=bool)
        index = grid.conformal_diagnostics[grid_type][direction][key]
        mask[tuple(index.T)] = True
        masks.append(mask)
    return tuple(masks)


def warn_diagnostics(diagnostics):
    """
    Warn about edges with geometry thinner than the mesh resolution and
    edges where the STL crossings and point classification disagree.

    ``diagnostics[grid_type][direction]`` holds the ``thin_feature`` and
    ``inconsistent`` edges of the backend as geometry indices (n, 3).
    """
    counts = {"thin_feature": 0, "inconsistent": 0}
    for grid_type in ("primal", "dual"):
        for direction in ("x", "y", "z"):
            for key in counts:
                counts[key] += len(diagnostics[grid_type][direction][key])

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


# Geometry array -> stored DOF mapping (symmetric FIT stencil)
#
# Nx, Ny, Nz count primal nodes. Every field component stores Nx * Ny * Nz
# values; entries without a geometric counterpart are zero-size ghost DOFs
# on the high side (L = 0 or iA = 0).
#
#   primal x-edge (Nx-1, Ny,   Nz  )  (i,j,k)     -> Ex(i,j,k)
#   primal x-face (Nx,   Ny-1, Nz-1)  (i,j,k)     -> Hx(i,j,k)
#   dual   x-edge (Nx,   Ny+1, Nz+1)  (i,j+1,k+1) -> Hx(i,j,k)
#   dual   x-face (Nx+1, Ny,   Nz  )  (i+1,j,k)   -> Ex(i,j,k)
#
# and cyclically for y and z (see GridFIT3D._flat_field_index).


def geometry_array_to_field_component(values, kind, direction, shape, fill_value=None):
    """
    Map a full primal/dual edge or face array to stored DOFs.

    ``kind`` is "primal_edge", "primal_face", "dual_edge" or "dual_face",
    ``shape`` the field component shape (Nx, Ny, Nz). Ghost DOFs without a
    geometric counterpart get ``fill_value``: None (default) gives False for
    boolean and NaN for floating-point arrays, "nearest" copies the nearest
    stored value (used for region IDs), any scalar is used as it is.
    """
    values = np.asarray(values)
    offset = [0, 0, 0]
    if kind == "dual_edge":
        offset = [1, 1, 1]
        offset["xyz".index(direction)] = 0
    elif kind == "dual_face":
        offset["xyz".index(direction)] = 1

    oi, oj, ok = offset
    source = values[oi : oi + shape[0], oj : oj + shape[1], ok : ok + shape[2]]

    if source.shape == tuple(shape):
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


def cut_edge_regions(grid, grid_type, edge):
    """Solid IDs at the start and end node of listed cut edges."""
    axis, index = grid._flat_index_to_geometry(grid_type, "edge", edge)
    region_ids = grid.region_ids[grid_type]
    end = index + np.eye(3, dtype=np.int64)[axis]
    return region_ids[tuple(index.T)], region_ids[tuple(end.T)]


def conductor_cuts(grid):
    """
    Primal edges and faces cut by a PEC/SIBC wall.

    Returns
    -------
    edge, edge_fraction : ndarray
        Flat indices of the cut edges and their fraction in the normal
        material.
    face, face_fraction, face_normal_region : ndarray
        Flat indices of the cut faces, the area fraction of their normal
        part and its solid ID.
    """
    is_normal, is_conductor = grid._region_type_flags()

    cuts = grid.edge_intersections["primal"]
    start, end = cut_edge_regions(grid, "primal", cuts.edge)
    to_conductor = is_normal[start] & is_conductor[end]
    from_conductor = is_conductor[start] & is_normal[end]
    selected = to_conductor | from_conductor
    edge_fraction = np.where(to_conductor, cuts.alpha, 1.0 - cuts.alpha)[selected]

    faces = grid.face_intersections["primal"]
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


def conformal_metrics(grid, L, iA, area_floor=0.0, min_edge_fraction=0.0):
    """
    Cut the primal metrics L and iA at conformal interfaces.

    Edges cut by a PEC/SIBC wall keep only their fraction in the normal
    material; faces cut by it keep the area fraction of their normal part.
    Normal-normal interfaces and edges/faces without a listed cut keep their
    grid metric. The dual metrics are never cut (Dey and Mittra).

    With an area floor (``GridFIT3D(conformal_stabilization="area_floor")``),
    faces cut by a PEC/SIBC wall use at least ``area_floor`` as area fraction
    (a constant-floor form of the area limitation of Benkler et al. 2006),
    which bounds the reduction of the stable time step.

    Parameters
    ----------
    grid : GridFIT3D
        Conformal grid with ``edge_intersections`` and ``face_intersections``.
    L, iA : Field
        Uncut primal edge lengths and inverse face areas; not modified.
    area_floor : float, optional
        Minimum area fraction of conductor-cut faces, 0 for none.
    min_edge_fraction : float, optional
        Conductor-cut edges with a smaller fraction are returned for
        elimination (``eliminate_small_edges``), 0 for none.

    Returns
    -------
    L_cut, iA_cut : Field
        The cut metrics.
    floored_faces : ndarray of int
        Flat indices of the faces raised to the area floor.
    small_edges : ndarray of int
        Flat indices of the conductor-cut edges below ``min_edge_fraction``.
    """
    edge, edge_fraction, face, face_fraction, _ = conductor_cuts(grid)

    L_cut = Field(grid.Nx, grid.Ny, grid.Nz)
    L_cut.fromarray(L.toarray().copy())
    L_cut.array[edge] = L_cut.array[edge] * edge_fraction

    floored = face_fraction < area_floor
    face_fraction = np.where(floored, area_floor, face_fraction)

    # The grid stores inverse face areas.
    iA_cut = Field(grid.Nx, grid.Ny, grid.Nz)
    iA_cut.fromarray(iA.toarray().copy())
    iA_cut.array[face] = 1.0 / ((1.0 / iA_cut.array[face]) * face_fraction)

    return L_cut, iA_cut, face[floored], edge[edge_fraction < min_edge_fraction]


def eliminate_small_edges(ieps, small_edges):
    """
    Area-floor treatment: remove nearly covered conductor-cut edges.

    Primal edges cut by a PEC/SIBC wall whose active fraction is below the
    minimum edge fraction are treated as inside the conductor (ieps = 0).
    With raised face areas, such edges would otherwise couple only weakly
    to the field and form spurious low-frequency modes. Modifies ``ieps``
    in place and returns the flat indices of the eliminated edges.
    """
    edge = small_edges[ieps.array[small_edges] != 0.0]
    ieps.array[edge] = 0.0
    return edge


def region_material_table(grid, eps_r_bg, mu_r_bg, sigma_bg):
    """
    Relative permittivity, permeability and conductivity per solid ID
    (index 0: the background).
    """
    table = np.array(
        [[eps_r_bg, mu_r_bg, sigma_bg]]
        + [grid.stl_materials[key] for key in grid.region_key_to_id],
        dtype=float,
    )
    return {"eps_r": table[:, 0], "mu_r": table[:, 1], "sigma": table[:, 2]}


def _harmonic_mean(alpha, start_values, end_values):
    """
    Series average along an edge, ``alpha`` of it in the start region.
    A zero value (an insulator for sigma) makes the average zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = 1.0 / (alpha / start_values + (1.0 - alpha) / end_values)
    return np.where((start_values == 0.0) | (end_values == 0.0), 0.0, mean)


def _diagonal_material(grid, region_values, axis, line_grid, face_grid, tol=1e-12):
    """
    Diagonal effective material of one DOF component at normal-normal
    interfaces.

    For the DOF component along x_i, with n the interface normal,

        m_ii = n_i^2 * m_perp + (1 - n_i^2) * m_parallel,

    where m_perp is the harmonic (series) value along the DOF edge and
    m_parallel the area-weighted arithmetic (parallel) value over its face:

        eps, sigma: primal edge (perp) and dual face (parallel)
        mu:         dual edge (perp) and primal face (parallel)

    Parameters
    ----------
    region_values : ndarray
        The material value per solid ID.

    Returns
    -------
    dof : ndarray of int
        DOF indices within the component (Fortran order) whose edge or face
        is cut by a normal-normal interface.
    values : ndarray of float
        The diagonal material there; NaN where it cannot be formed.
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)
    N = grid.Nx * grid.Ny * grid.Nz
    is_normal, _ = grid._region_type_flags()

    # Cut edges of this component (any type: their normal and position also
    # serve an uncut face) and the normal-normal ones among them.
    edges = grid.edge_intersections[line_grid]
    in_component = edges.edge // N == axis
    edge_dof = edges.edge[in_component] % N
    edge_alpha = edges.alpha[in_component]
    edge_normal = edges.normal[in_component]
    start, end = cut_edge_regions(grid, line_grid, edges.edge[in_component])
    edge_nn = is_normal[start] & is_normal[end]

    # Normal-normal cut faces of this component.
    faces = grid.face_intersections[face_grid]
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

    # The DOF's own edge: its end regions, and whether it exists (high ghost
    # E DOFs have none).
    index = np.column_stack(np.unravel_index(dof, shape, order="F"))
    if line_grid == "dual":
        offset = np.ones(3, dtype=np.int64)
        offset[axis] = 0
        index = index + offset
    region_ids = grid.region_ids[line_grid]
    exists = index[:, axis] < region_ids.shape[axis] - 1
    last = np.array(region_ids.shape) - 1
    node = np.minimum(index, last)
    node_end = np.minimum(index + np.eye(3, dtype=np.int64)[axis], last)
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

    # Where only one of edge and face is cut, the uncut one lies in a single
    # region: an uncut edge in its endpoint region, an uncut face in the
    # region of the edge midpoint, which it passes through.
    perp = np.where(
        line_cut,
        _harmonic_mean(alpha, region_values[dof_start], region_values[dof_end]),
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


def conformal_materials(grid, region_values):
    """
    Effective relative materials of every DOF from the conformal geometry.

    Every DOF first takes the material of its edge endpoints (primal edges
    for E, dual edges for H): the region material if both lie in one region,
    the mean over the normal-material endpoints otherwise, and a conductor
    (eps_r = inf) if both lie in PEC/SIBC; at a cut E edge this is already
    the material of its normal part. H DOFs at faces cut by a conductor take
    the material of the normal part of the face, and normal-normal
    interfaces the diagonal effective material (``_diagonal_material``),
    which takes precedence. Only edges without a listed crossing, which the
    conformal diagnostics warn about, keep the endpoint value at a material
    transition.

    Parameters
    ----------
    grid : GridFIT3D
        Conformal grid with ``edge_intersections`` and ``face_intersections``.
    region_values : dict
        ``eps_r``, ``mu_r`` and ``sigma`` per solid ID
        (``region_material_table``).

    Returns
    -------
    dict
        ``materials[name][direction]``: arrays of shape (Nx, Ny, Nz) for
        ``eps_r``, ``mu_r`` and ``sigma``.
    """
    is_normal, is_conductor = grid._region_type_flags()
    shape = (grid.Nx, grid.Ny, grid.Nz)
    N = grid.Nx * grid.Ny * grid.Nz

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

    def assign(target, dof, values):
        target[np.unravel_index(dof, shape, order="F")] = values

    # H DOFs at faces cut by a conductor: material of the normal part.
    _, _, conductor_face, _, normal_region = conductor_cuts(grid)

    materials = {"eps_r": {}, "mu_r": {}, "sigma": {}}
    for axis, direction in enumerate("xyz"):
        # Electric DOFs: primal edges. Ghost edges copy the nearest region
        # IDs; their length is zero.
        start, end = grid._get_edge_region_ids("primal", direction)
        start = geometry_array_to_field_component(
            start, "primal_edge", direction, shape, "nearest"
        )
        end = geometry_array_to_field_component(
            end, "primal_edge", direction, shape, "nearest"
        )
        fully_conductor = is_conductor[start] & is_conductor[end]
        for name in ("eps_r", "sigma"):
            materials[name][direction] = endpoint_material(name, start, end)

        # Interior PEC/SIBC electric DOFs are eliminated through ieps = 0.
        materials["eps_r"][direction][fully_conductor] = np.inf
        materials["sigma"][direction][fully_conductor] = 0.0

        # Magnetic DOFs: dual edges, conductor cuts through primal faces.
        start, end = grid._get_edge_region_ids("dual", direction)
        start = geometry_array_to_field_component(start, "dual_edge", direction, shape)
        end = geometry_array_to_field_component(end, "dual_edge", direction, shape)
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
            dof, values = _diagonal_material(
                grid, region_values[name], axis, line_grid, face_grid
            )
            valid = np.isfinite(values)
            assign(materials[name][direction], dof[valid], values[valid])

    return materials
