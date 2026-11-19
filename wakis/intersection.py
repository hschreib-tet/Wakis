# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
STL intersection data shared by the conformal geometry backends.

A conformal backend (``conformal.py``: ray tracing) intersects the STL
solids with the primal and dual grid and describes the result with two
sparse lists per grid, one record per cut edge and per cut face. The cut
lengths, cut areas and effective materials are computed from these lists
alone, so the backends are interchangeable.

Edges and faces are given by their flat field index (see
``GridFIT3D._flat_field_index``); the solid IDs on both sides of an edge
are those of its end nodes (``GridFIT3D.region_ids``). Edges and faces that
are not listed are uncut.
"""

from dataclasses import dataclass

import numpy as np


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
        Unit normal of the interface in the face; its sign is not used.
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
