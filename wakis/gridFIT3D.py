# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import time
import warnings

import h5py
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from .conformal import ConformalGridMixin
from .field import Field
from .logger import Logger
from .materials import (
    VALID_MATERIAL_TYPES,
    material_colors,
    material_lib,
)
from .plotting import PlotMixinGrid as PlotMixin

try:
    from mpi4py import MPI

    imported_mpi = True
except ImportError:
    imported_mpi = False


class GridFIT3D(PlotMixin, ConformalGridMixin):
    """
    Store symmetric FIT grid geometry and handle STL solids using PyVista.

    Each axis includes both physical boundary nodes. Computational cell counts
    equal the numbers of primal nodes; the final computational cell has zero
    primal extent along that axis. Thus, an axis with ``Nx`` computational
    cells contains ``Nx - 1`` physical cells.
    """

    def __init__(
        self,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        zmin=None,
        zmax=None,
        Nx=None,
        Ny=None,
        Nz=None,
        x=None,
        y=None,
        z=None,
        use_mpi=False,
        use_mesh_refinement=False,
        refinement_method="insert",
        refinement_tol=1e-8,
        snap_points=None,
        snap_tol=1e-5,
        snap_solids=None,
        stl_solids=None,
        stl_materials=None,
        stl_rotate=[0.0, 0.0, 0.0],
        stl_translate=[0.0, 0.0, 0.0],
        stl_scale=1.0,
        stl_colors=None,
        stl_tol=1e-3,
        stl_method="legacy",
        geometry_mode="legacy",
        bg_material_type=None,
        conformal_stabilization=None,
        conformal_stabilization_fraction=None,
        conformal_min_edge_fraction=None,
        subpixel_smoothing=False,
        subpixel_smoothing_factor=4,
        subpixel_smoothing_threshold=0.3,
        subpixel_smoothing_bool=True,
        load_from_h5=None,
        verbose=2,
        stl_material_types=None,
    ):
        """
        Class holding the grid information and STL importing/handling using PyVista.

        Parameters
        ----------
        xmin, xmax, ymin, ymax, zmin, zmax : float, optional
            Extent of the simulation domain. If None, must provide x, y, z arrays.
        Nx, Ny, Nz : int, optional
            Number of computational cells (primal nodes) per direction,
            including both domain boundaries. The numbers of physical cells
            are Nx - 1, Ny - 1, and Nz - 1. Each count must be at least 2.
            If None, must provide x, y, z arrays.
        x, y, z : array_like, optional
            Primal node coordinates, including both domain boundaries.
            Their lengths define Nx, Ny, and Nz and must each be at least 2.
            Non-uniform grids are supported.
        use_mpi : bool, optional
            Enable MPI domain decomposition in the z direction. Default is False.
            Requires more than one MPI rank. Uniform-grid Nz is rounded up to
            a multiple of the rank count before the z coordinates are built.
            Explicit z arrays must already have a length divisible by the rank
            count; their coordinates are preserved.
        use_mesh_refinement : bool, optional
            Enable mesh refinement based on snap points extracted from the STL solids.
            Default is False.
        refinement_method : str, optional
            Refinement algorithm for mesh refinement. Default is "insert".
        refinement_tol : float, optional
            Tolerance for mesh refinement. Default is 1e-8.
        snap_points : array_like, optional
            Points to snap the mesh to. Default is None.
        snap_tol : float, optional
            Tolerance for snap point detection. Default is 1e-5.
        snap_solids : list or None, optional
            STL solids to use for snap point extraction. Default is None (all).
        stl_solids : dict or str, optional
            STL files to import in the domain. {'Solid 1': stl_1, ...}
        stl_materials : dict, optional
            Material properties associated with STL solids.
            {'Solid 1': [eps1, mu1], ...}
        stl_material_types : dict, str, or None, optional
            Material treatment associated with each STL solid. Valid values are
            "normal", "pec", and "sibc". If None, materials default to "normal".
            Existing PEC material definitions are detected automatically for
            backward compatibility.
        stl_rotate : list or dict, optional
            Angle of rotation to apply to the STL models: [rot_x, rot_y, rot_z].
            If dict, must contain the same keys as stl_solids.
        stl_translate : list or dict, optional
            Translation to apply to the STL models: [dx, dy, dz].
            If dict, must contain the same keys as stl_solids.
        stl_scale : float or dict, optional
            Scaling value to apply to the STL model to convert to [m].
            If dict, must contain the same keys as stl_solids.
        stl_colors : str, list, dict, or None, optional
            Color(s) for STL solids. If None, assigned automatically.
        stl_tol : float, optional
            Tolerance factor for STL import, used in grid.select_enclosed_points.
            Default is 1e-3.
        stl_method : str, optional
            Method for marking cells inside STL solids. Options are "legacy" (default),
            "interior_points", "implicit_distance", or "voxelize_rectilinear".
            Voxelization requires a uniform grid and samples physical cell centres;
            terminal storage planes inherit the nearest physical-cell labels.
        geometry_mode : str, optional
            Geometry-mask representation used after STL classification.
            ``"legacy"`` preserves the existing cell-based mask pipeline.
            ``"conformal"`` classifies primal and dual grid points directly,
            derives the cell mask from the primal point classification, and
            constructs interface topology and geometric STL intersections.
            Default is ``"legacy"``.
        subpixel_smoothing : bool, optional
            Whether to apply subpixel smoothing to physical STL cells after
            voxelization. Terminal labels inherit the smoothed boundary cells.
            Default is False.
        subpixel_smoothing_factor : int, optional
            Factor by which to increase the resolution for subpixel smoothing. Default is 4.
            Memory usage increases with the cube of this factor, so use with caution!
        subpixel_smoothing_threshold : float, optional
            Threshold value (0–1) for classifying smoothed cells as inside/outside the solid.
            Default is 0.3. A lower value includes more boundary cells; typical choice is
            ``1/(subpixel_smoothing_factor**3)``.
        subpixel_smoothing_bool : bool, optional
            If True, convert the smoothed mask to a boolean array after thresholding.
            Default is True.
        load_from_h5 : str, optional
            Load grid from an h5 file previously saved with `save_to_h5`.
        verbose : int or bool, optional
            Enable verbose output on the terminal. Use `verbose=2` for more detail.
        bg_material_type : str, optional
            Material treatment of region 0, used by the conformal geometry
            pipeline to classify interfaces. Default is None.
        conformal_stabilization : {None, "area_floor"}, optional
            Treatment of small cut cells at PEC/SIBC walls in conformal mode,
            which otherwise limit the stable time step. Default is None.

            ``"area_floor"``: the geometry is kept. In the time-stepping
            metric, primal faces cut by a conductor use at least
            ``conformal_stabilization_fraction`` as area fraction (relaxed
            Dey-Mittra), and edges with an active fraction below
            ``conformal_min_edge_fraction`` are treated as inside the
            conductor. Without the edge elimination, raised areas next to
            nearly covered edges create spurious low-frequency modes.
        conformal_stabilization_fraction : float, optional
            Area floor: minimum area fraction of conductor-cut faces,
            0 < value <= 1. Default is 0.1.
        conformal_min_edge_fraction : float, optional
            Area floor only: conductor-cut edges with a smaller active
            fraction are eliminated (E set to zero), 0 <= value < 0.5.
            Default (None) uses the area floor value. With this choice no
            spurious localized modes were found while the mesh resolves the
            wavelength with at least about 6 cells.

        Attributes
        ----------
        x, y, z : ndarray
            Primal node coordinates, including both domain boundaries, with
            lengths Nx, Ny, and Nz.
        Nx, Ny, Nz : int
            Computational cell counts, equal to the primal node counts.
            Physical cell counts are Nx - 1, Ny - 1, and Nz - 1.
        dx, dy, dz : ndarray
            Physical cell widths, with lengths Nx - 1, Ny - 1, and Nz - 1.
        tx, ty, tz : ndarray
            Dual grid coordinates: the domain boundaries and the intervening
            primal cell midpoints, with lengths Nx + 1, Ny + 1, and Nz + 1.
        grid : pyvista.StructuredGrid
            Computational primal grid, including zero-extent high-side cells.
        L, iA : Field
            Primal edge lengths and inverse face areas on the computational
            grid. Edge lengths vanish along the corresponding high-side
            boundary axis; inverse areas are zero for zero-area faces.
        tL, itA : Field
            Dual edge lengths and inverse face areas on the same computational
            index space. Dual edges initially have half-cell lengths at both
            boundaries; a periodic solver combines them at the low boundary.
        stl_solids, stl_materials, stl_colors : dict
            STL solid file paths, materials, and colors.
        (...)
        """

        t0 = time.time()
        self.logger = Logger()
        self.verbose = verbose
        self.use_mpi = use_mpi

        self.geometry_mode = geometry_mode.lower()

        if self.geometry_mode not in ("legacy", "conformal"):
            raise ValueError("[!] geometry_mode must be 'legacy' or 'conformal'.")

        if bg_material_type is None:
            self.bg_material_type = None

        else:
            if not isinstance(
                bg_material_type,
                str,
            ):
                raise TypeError("bg_material_type must be None or a string.")

            self.bg_material_type = bg_material_type.lower()

            if self.bg_material_type not in VALID_MATERIAL_TYPES:
                raise ValueError(
                    "Unknown background material "
                    f"type "
                    f"{self.bg_material_type!r}. "
                    f"Expected one of "
                    f"{VALID_MATERIAL_TYPES}."
                )

        self._set_conformal_stabilization(
            conformal_stabilization,
            conformal_stabilization_fraction,
            conformal_min_edge_fraction,
        )

        # Primary and dual edges whose endpoints belong to different regions
        self.interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        # Geometric STL intersections with primal and dual interface edges
        self.interface_intersections = {
            "primal": {},
            "dual": {},
        }

        self.interior_interface_edge_masks = {
            "primal": {},
            "dual": {},
        }

        # Point-based STL masks on the primal and dual grids
        self.primal_point_masks = {}
        self.dual_point_masks = {}

        # Unique region IDs on primal and dual grid points.
        # Region 0 is always the background.
        self.region_ids = {
            "primal": None,
            "dual": None,
        }

        self.region_id_to_key = {0: None}
        self.region_key_to_id = {}

        self.face_candidate_masks = {
            "primal": {},
            "dual": {},
        }

        self.interface_face_data = {
            "primal": {},
            "dual": {},
        }

        # Grid data
        # generate from file
        if load_from_h5 is not None:
            if self.use_mpi:
                raise NotImplementedError(
                    "MPI decomposition of loaded HDF5 grids is not supported."
                )
            self.load_from_h5(load_from_h5)
            return  # TODO: support MPI decomposition

        # generate from custom x,y,z arrays
        elif x is not None and y is not None and z is not None:
            # allow user to set the grid axis manually
            self.x = np.asarray(x)
            self.y = np.asarray(y)
            self.z = np.asarray(z)
            self.Nx = len(self.x)
            self.Ny = len(self.y)
            self.Nz = len(self.z)
            self.xmin, self.xmax = self.x[0], self.x[-1]
            self.ymin, self.ymax = self.y[0], self.y[-1]
            self.zmin, self.zmax = self.z[0], self.z[-1]
            if self.use_mpi:
                self._setup_mpi()
                if self.Nz % self.size:
                    raise ValueError(
                        f"Custom z has {self.Nz} nodes, which is not divisible "
                        f"by the number of MPI ranks ({self.size})."
                    )

        # generate from domain extents and number of cells [LEGACY]
        elif all(v is not None for v in [xmin, xmax, ymin, ymax, zmin, zmax]):
            # uniform grid from domain extents and number of cells
            self.xmin, self.xmax = xmin, xmax
            self.ymin, self.ymax = ymin, ymax
            self.zmin, self.zmax = zmin, zmax
            self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
            if self.use_mpi:
                self._setup_mpi()
                self.Nz = self._adjust_Nz_for_mpi()

            self.x = np.linspace(self.xmin, self.xmax, self.Nx)
            self.y = np.linspace(self.ymin, self.ymax, self.Ny)
            self.z = np.linspace(self.zmin, self.zmax, self.Nz)

        else:
            raise ValueError(
                "[!] Error initializing GridFIT3D:\n"
                "  - Provide grid axis arrays: x, y, z\n"
                "  - OR domain extents and number of cells: \
                    xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz\n"
                "  - OR load from a HDF5 file using load_from_h5"
            )

        if any(n is None or n < 2 for n in (self.Nx, self.Ny, self.Nz)):
            raise ValueError("Nx, Ny, and Nz must each be at least 2 primal nodes.")

        # TODO: allow non uniform dx, dy, dz from input
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)
        self.update_logger(["Nx", "Ny", "Nz", "dx", "dy", "dz"])
        self.update_logger(["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])

        # stl info
        self.stl_solids = stl_solids
        self.stl_materials = stl_materials
        self.stl_material_types = stl_material_types
        self.stl_rotate = stl_rotate
        self.stl_translate = stl_translate
        self.stl_scale = stl_scale
        self.stl_colors = stl_colors
        self.update_logger(["stl_solids", "stl_materials", "stl_material_types"])
        if stl_rotate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_rotate"])
        if stl_translate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_translate"])
        if stl_scale != 1.0:
            self.update_logger(["stl_scale"])

        if stl_solids is not None:
            self._prepare_stl_dicts()

        # refine self.x, self.y, self.z using snap points
        self.use_mesh_refinement = use_mesh_refinement
        self.refinement_method = refinement_method
        self.snap_points = snap_points
        self.snap_tol = snap_tol
        self.snap_solids = snap_solids  # if None, use all stl_solids
        self.update_logger(["use_mesh_refinement"])

        if self.use_mesh_refinement:
            if verbose:
                print("Applying mesh refinement...")
            if self.snap_points is None and stl_solids is not None:
                self._compute_snap_points(snap_solids=snap_solids, snap_tol=snap_tol)
            self._refine_xyz_axis(method=refinement_method, tol=refinement_tol)
            self.update_logger(["Nx", "Ny", "Nz", "dx", "dy", "dz"])

        print(f"Generating grid with {self.Nx * self.Ny * self.Nz} mesh cells...")
        if verbose > 1:
            print(
                f"    * Simulation domain bounds: \n\
                x:[{self.xmin:.3f}, {self.xmax:.3f}],\n\
                y:[{self.ymin:.3f}, {self.ymax:.3f}],\n\
                z:[{self.zmin:.3f}, {self.zmax:.3f}]"
            )
            print(
                "    * Minimum cell sizes: dx={:.3e}, dy={:.3e}, dz={:.3e}".format(
                    np.min(self.dx), np.min(self.dy), np.min(self.dz)
                )
            )

        # MPI subdivide domain
        if self.use_mpi:
            self.ZMIN = None
            self.ZMAX = None
            self.NZ = None
            self.Z = None
            self._partition_mpi_grid()
            if self.verbose:
                print(f"MPI grid partitioned for rank {self.rank} of {self.size}")

        # grid G and tilde grid ~G, lengths and inverse areas
        self._compute_grid()

        # STL import and mask generation
        if verbose:
            print("Importing STL solids...")
        self.stl_tol = stl_tol
        self.stl_method = stl_method

        self.use_subpixel_smoothing = subpixel_smoothing
        self.subpixel_smoothing_factor = subpixel_smoothing_factor
        self.subpixel_smoothing_threshold = subpixel_smoothing_threshold
        self.subpixel_smoothing_bool = subpixel_smoothing_bool

        if self.subpixel_smoothing_threshold is None:
            self.subpixel_smoothing_threshold = 1 / (self.subpixel_smoothing_factor**3)

        if self.geometry_mode == "conformal" and self.use_subpixel_smoothing:
            warnings.warn(
                "subpixel_smoothing is ignored when geometry_mode='conformal'. "
                "The conformal geometry pipeline derives the masks from "
                "primal grid-point classifications.",
                UserWarning,
                stacklevel=2,
            )

        if stl_solids is not None:
            self._mark_cells_in_stl(method=self.stl_method)

        if verbose:
            print(f"Total grid initialization time: {time.time() - t0} s")

        self.gridInitializationTime = time.time() - t0
        self.update_logger(["gridInitializationTime"])

    def _compute_grid(self):
        """
        Compute the PyVista grid and related geometric quantities.

        Build primal and dual metrics on the same computational index space.

        Primal coordinates are defined at cell vertices.
        The high-side primal coordinate is repeated to retain the boundary node
        in a zero-extent computational cell at the high-boundary (ghost cell).

        Dual coordinates span from the low boundary through primal cell midpoints
        to the high boundary, giving half-cell dual lengths at both ends.
        All metric fields have shape (Nx, Ny, Nz) per component.
        """

        # === Primal grid G ===

        # Extend the grid to account for ghost cell node
        x = np.append(self.x, self.x[-1])
        y = np.append(self.y, self.y[-1])
        z = np.append(self.z, self.z[-1])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        self.grid = pv.StructuredGrid(X.transpose(), Y.transpose(), Z.transpose())

        # Values outside of physical domain are 0
        self.L = Field(self.Nx, self.Ny, self.Nz)
        self.L.field_x = X[1:, 1:, 1:] - X[:-1, :-1, :-1]
        self.L.field_y = Y[1:, 1:, 1:] - Y[:-1, :-1, :-1]
        self.L.field_z = Z[1:, 1:, 1:] - Z[:-1, :-1, :-1]

        self.iA = Field(self.Nx, self.Ny, self.Nz)
        aux = self.L.field_y * self.L.field_z
        self.iA.field_x = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux != 0)
        aux = self.L.field_x * self.L.field_z
        self.iA.field_y = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux != 0)
        aux = self.L.field_x * self.L.field_y
        self.iA.field_z = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux != 0)
        del aux

        # === Tilde grid ~G ===

        # Append domain boundaries to the tilde grid coordinates
        self.tx = np.r_[self.x[0], (self.x[1:] + self.x[:-1]) / 2, self.x[-1]]
        self.ty = np.r_[self.y[0], (self.y[1:] + self.y[:-1]) / 2, self.y[-1]]
        self.tz = np.r_[self.z[0], (self.z[1:] + self.z[:-1]) / 2, self.z[-1]]
        tX, tY, tZ = np.meshgrid(self.tx, self.ty, self.tz, indexing="ij")

        # No values are 0, just snapped to physical domain boundaries
        self.tL = Field(self.Nx, self.Ny, self.Nz)
        self.tL.field_x = tX[1:, 1:, 1:] - tX[:-1, :-1, :-1]
        self.tL.field_y = tY[1:, 1:, 1:] - tY[:-1, :-1, :-1]
        self.tL.field_z = tZ[1:, 1:, 1:] - tZ[:-1, :-1, :-1]

        self.itA = Field(self.Nx, self.Ny, self.Nz)
        self.itA.field_x = np.divide(1.0, self.tL.field_y * self.tL.field_z)
        self.itA.field_y = np.divide(1.0, self.tL.field_x * self.tL.field_z)
        self.itA.field_z = np.divide(1.0, self.tL.field_x * self.tL.field_y)

    def _setup_mpi(self):
        """Initialize and validate the communicator for grid decomposition."""
        if not imported_mpi:
            raise ImportError(
                "[!] mpi4py is required when use_mpi=True but was not found"
            )
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if self.size == 1:
            raise ValueError(
                "use_mpi=True requires more than one MPI rank. "
                "Launch with mpiexec -n 2 (or more), or set use_mpi=False."
            )
        if self.Nz < self.size:
            raise ValueError(
                f"Nz ({self.Nz}) must be greater than or equal to the "
                f"number of MPI processes ({self.size})."
            )

    def _adjust_Nz_for_mpi(self):
        """Round a uniform grid's global node count before creating its z axis."""
        nz = self.Nz
        if nz % self.size == 0:
            return nz

        adjusted_nz = nz + self.size - nz % self.size
        if self.rank == 0:
            print(
                f"\033[33m[!] MPI increased Nz from {nz} to {adjusted_nz} "
                f"for {self.size} ranks; longitudinal spacing changed.\033[0m"
            )
        return adjusted_nz

    def _partition_mpi_grid(self):
        """Partition the global z nodes after the MPI communicator is set up."""
        if self.Nz % self.size:
            raise ValueError(
                "Mesh refinement produced an Nz that is not divisible by "
                f"the number of MPI ranks ({self.size})."
            )

        # global z quantities [ALLCAPS]
        self.ZMIN = self.zmin
        self.ZMAX = self.zmax
        self.NZ = self.Nz
        self.Z = self.z.copy()

        # Partition computational nodes, retaining one neighboring node for
        # each MPI exchange. Only the final rank owns the high boundary node.
        nz = self.NZ // self.size
        self.n_ghosts = 1
        start = self.rank * nz - (self.rank > 0)
        stop = (self.rank + 1) * nz + (self.rank < self.size - 1)
        self.z = self.Z[start:stop].copy()
        self.Nz = len(self.z)
        self.zmin, self.zmax = self.z[0], self.z[-1]
        self.dz = np.diff(self.z)

    def mpi_gather_asGrid(self):
        """
        Gather the global grid from all MPI subdomains and return a new GridFIT3D.

        Returns
        -------
        _grid : GridFIT3D or None
            The global grid object (only on rank 0).
        """
        _grid = None
        if self.rank == 0:
            print(f"Generating global grid from {self.ZMIN} to {self.ZMAX}")
            _grid = GridFIT3D(
                x=self.x,
                y=self.y,
                z=self.Z,
                use_mpi=False,
                stl_solids=self.stl_solids,
                stl_materials=self.stl_materials,
                stl_material_types=self.stl_material_types,
                stl_scale=self.stl_scale,
                stl_rotate=self.stl_rotate,
                stl_translate=self.stl_translate,
                stl_colors=self.stl_colors,
                verbose=self.verbose,
                stl_tol=self.stl_tol,
                geometry_mode=self.geometry_mode,
                stl_method=self.stl_method,
                bg_material_type=self.bg_material_type,
                conformal_stabilization=self.conformal_stabilization,
                conformal_stabilization_fraction=self._get_stabilization_fraction(),
                conformal_min_edge_fraction=self.conformal_min_edge_fraction or None,
            )
        return _grid

    def _prepare_stl_material_types(self):
        """
        Prepare and validate material treatment types for all STL solids.

        Materials default to "normal". Existing PEC definitions are detected
        automatically for backward compatibility.

        Finite conductivity does not imply SIBC. SIBC treatment must always
        be requested explicitly.
        """
        if self.stl_material_types is None:
            material_types = {}

        elif isinstance(self.stl_material_types, str):
            material_types = {
                key: self.stl_material_types.lower() for key in self.stl_solids
            }

        elif isinstance(self.stl_material_types, dict):
            unknown_keys = set(self.stl_material_types) - set(self.stl_solids)

            if unknown_keys:
                raise ValueError(
                    "stl_material_types contains unknown STL solids: "
                    f"{sorted(unknown_keys)}"
                )

            material_types = {}

            for key, material_type in self.stl_material_types.items():
                if not isinstance(material_type, str):
                    raise TypeError(
                        "Each value in stl_material_types must be a string."
                    )

                material_types[key] = material_type.lower()

        else:
            raise TypeError(
                "stl_material_types must be None, a string, or a dictionary."
            )

        # Fill missing entries.
        for key in self.stl_solids:
            if key in material_types:
                continue

            material = self.stl_materials[key]

            # Preserve existing PEC definitions.
            if isinstance(material, str):
                if material.lower() == "pec":
                    material_types[key] = "pec"
                else:
                    material_types[key] = "normal"

            elif np.isinf(material[0]):
                material_types[key] = "pec"

            else:
                material_types[key] = "normal"

        # Validate final material types.
        for key, material_type in material_types.items():
            if material_type not in VALID_MATERIAL_TYPES:
                raise ValueError(
                    f"Unknown material type '{material_type}' "
                    f"for STL solid '{key}'. "
                    f"Expected one of {VALID_MATERIAL_TYPES}."
                )

        self.stl_material_types = material_types

    def _prepare_stl_dicts(self):
        """
        Prepare STL-related dictionaries for rotation, scale, translation, and colors.

        Ensures all STL solids have corresponding entries for rotation, scale,
        translation, and color, converting single values to dicts as needed.
        """
        if type(self.stl_solids) is not dict:
            if type(self.stl_solids) is str:
                self.stl_solids = {"Solid 1": self.stl_solids}
            else:
                raise Exception(
                    "Attribute `stl_solids` must contain a string or a dictionary"
                )

        if type(self.stl_rotate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_rotate = {}
            for key in self.stl_solids.keys():
                stl_rotate[key] = self.stl_rotate
            self.stl_rotate = stl_rotate

        if type(self.stl_scale) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_scale = {}
            for key in self.stl_solids.keys():
                stl_scale[key] = self.stl_scale
            self.stl_scale = stl_scale

        if type(self.stl_translate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_translate = {}
            for key in self.stl_solids.keys():
                stl_translate[key] = self.stl_translate
            self.stl_translate = stl_translate

        if type(self.stl_colors) is not dict:
            if self.stl_colors is None:
                self._assign_colors()
            elif self.stl_colors is str:  # single color for all solids
                stl_colors = {}
                for key in self.stl_solids.keys():
                    stl_colors[key] = self.stl_colors
                self.stl_colors = stl_colors
            elif type(self.stl_colors) is list:
                stl_colors = {}
                try:
                    for i, key in enumerate(self.stl_solids.keys()):
                        stl_colors[key] = self.stl_colors[i]
                    self.stl_colors = stl_colors
                except IndexError:
                    self._assign_colors()
                    print(
                        "[!] If `stl_colors` is a list, it must have \
                        the same length as `stl_solids`."
                    )

        if type(self.stl_materials) is not dict:
            if type(self.stl_materials) is str:
                self.stl_materials = {"Solid 1": self.stl_materials}
            else:
                raise Exception(
                    "Attribute `stl_materials` must contain a string or a dictionary"
                )

        self._prepare_stl_material_types()

        # Material library lookup and conversion to [eps, mu, sigma] format
        for key in self.stl_solids.keys():
            if type(self.stl_materials[key]) is str:
                mat_key = self.stl_materials[key].lower()
                mat = material_lib[mat_key]
                self.stl_materials[key] = [
                    mat[0],
                    mat[1],
                    mat[2] if len(mat) == 3 else 0.0,
                ]
            elif len(self.stl_materials[key]) == 2:
                self.stl_materials[key].append(0.0)

    def _mark_cells_in_stl(self, method):
        if self.geometry_mode == "legacy":
            self._mark_cells_in_stl_legacy(method)

        elif self.geometry_mode == "conformal":
            self._mark_cells_in_stl_conformal(method)

    def _mark_cells_in_stl_legacy(self, method):
        """
        Mark grid cells that are inside each STL solid.

        Uses PyVista's select_interior_points as default (equivalent to the deprecated select_enclosed_points),
        and otherwise input method either compute_implicit_distance or voxelize_rectilinear,
        to create boolean masks for each solid.

        See also the PyVista documentation:
        ----------------------------------
        "interior_points": https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.select_interior_points
        "implicit_distance": https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.compute_implicit_distance
        "voxelize_rectilinear" : https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.voxelize_rectilinear

        """
        # Obtain masks with grid cells inside each stl solid
        stl_tolerance = (
            np.min([np.min(self.dx), np.min(self.dy), np.min(self.dz)]) * self.stl_tol
        )
        progress_bar = False
        if self.verbose > 1:
            progress_bar = True

        for key in self.stl_solids.keys():
            surf = self.read_stl(key)

            if self.verbose:
                print(f" * Marking cells inside STL solid '{key}'...")

            # mark cells in stl [True == in stl, False == out stl]
            if method.lower() == "legacy":
                try:
                    select = self.grid.select_interior_points(
                        surf, method="cell_locator", locator_tolerance=stl_tolerance
                    )
                    self.grid[key] = (
                        select.point_data_to_cell_data()["selected_points"]
                        > stl_tolerance
                    )
                except Exception:
                    select = self.grid.select_interior_points(
                        surf,
                        method="cell_locator",
                        locator_tolerance=stl_tolerance,
                        check_surface=False,
                    )
                    self.grid[key] = (
                        select.point_data_to_cell_data()["selected_points"]
                        > stl_tolerance
                    )
                    if self.verbose > 1:
                        print(
                            f"[!] Warning: stl solid {key} may have issues with closed surfaces. Consider checking the STL file."
                        )

            elif method.lower() == "interior_points":
                try:
                    select = self.grid.select_interior_points(
                        surf, method="cell_locator", locator_tolerance=stl_tolerance
                    )
                    self.grid[key] = (
                        select.point_data_to_cell_data()["selected_points"] > 0.5
                    )
                except Exception:
                    select = self.grid.select_interior_points(
                        surf,
                        method="cell_locator",
                        locator_tolerance=stl_tolerance,
                        check_surface=False,
                    )
                    self.grid[key] = (
                        select.point_data_to_cell_data()["selected_points"] > 0.5
                    )
                    if self.verbose > 1:
                        print(
                            f"[!] Warning: stl solid {key} may have issues with closed surfaces. Consider checking the STL file."
                        )

            elif method.lower() == "implicit_distance":
                # negative distance is inside, positive is outside
                try:
                    select = self.grid.compute_implicit_distance(surf)
                    self.grid[key] = select.point_data_to_cell_data()[
                        "implicit_distance"
                    ] <= 0.5 * np.mean(
                        [np.mean(self.dx), np.mean(self.dy), np.mean(self.dz)]
                    )
                except Exception:
                    print(
                        f"[!] Warning: Implicit distance computation for stl solid {key} failed."
                    )

            elif method.lower() == "voxelize_rectilinear":
                # Check for uniform grid
                widths = (self.dx, self.dy, self.dz)
                if any(not np.allclose(axis, axis[0], atol=0) for axis in widths):
                    raise ValueError(
                        "voxelize_rectilinear requires a uniform grid; "
                        "use interior_points for nonuniform grids."
                    )

                # PyVista reference points become the output voxel centres.
                shape = (self.Nx - 1, self.Ny - 1, self.Nz - 1)
                dx, dy, dz = self.dx[0], self.dy[0], self.dz[0]
                origin = (  # physical cell centers
                    self.xmin + dx / 2,
                    self.ymin + dy / 2,
                    self.zmin + dz / 2,
                )
                reference_vol = pv.ImageData(
                    dimensions=shape, origin=origin, spacing=(dx, dy, dz)
                )
                vox = surf.voxelize_rectilinear(
                    reference_volume=reference_vol, progress_bar=progress_bar
                )

                # Extend material labels to degenerate ghost high-side storage cells.
                # This assumes no material change within the final half-cell.
                mask = np.empty((self.Nx, self.Ny, self.Nz), dtype=bool)
                mask[:-1, :-1, :-1] = np.asarray(vox["mask"]).reshape(shape, order="F")

                mask[-1, :-1, :-1] = mask[-2, :-1, :-1]
                mask[:, -1, :-1] = mask[:, -2, :-1]
                mask[:, :, -1] = mask[:, :, -2]

                self.grid.cell_data[key] = mask.ravel()

                if self.use_subpixel_smoothing:
                    self._apply_subpixel_smoothing(
                        key,
                        factor=self.subpixel_smoothing_factor,
                        threshold=self.subpixel_smoothing_threshold,
                        make_bool=self.subpixel_smoothing_bool,
                    )
            else:
                raise ValueError(
                    f"[!] Error: stl_method {method} not recognized. \
                    Use 'interior_points', 'implicit_distance', or 'voxelize_rectilinear'."
                )

            if self.verbose and np.sum(self.grid[key]) == 0:
                print(
                    f"[!] Warning: no cells were marked inside stl solid {key}. Consider increasing the tolerance factor (currently {self.stl_tol})."
                )

            if self.verbose:
                print(
                    f"    * STL solid {key}: {np.sum(self.grid[key])} cells marked inside the solid."
                )

    def _apply_subpixel_smoothing(
        self,
        key,
        factor=4,
        sigma=0.5,
        make_bool=True,  # default is True
        threshold=None,  # default is 1/(factor**3)
    ):
        """
        Apply subpixel smoothing to the STL mask.

        This method creates a higher resolution reference volume for voxelization,
        applies the voxelization at this higher resolution, and then combines
        the results to create a more accurate mask with the original grid resolution.

        The smoothing is done by averaging the values from the higher resolution grid
        and computing the gradient at mask edges, then applying a Gaussian filter.

        Parameters
        ----------
        key : str
            Key of the STL solid to smooth.
        factor : int, optional
            Factor by which to increase the resolution for subpixel smoothing. Default is 4.
        sigma : float, optional
            Standard deviation for Gaussian filter. Default is 0.5.
        make_bool : bool, optional
            Whether to convert the smoothed mask to boolean. Default is True.
        threshold : float, optional
            Threshold for converting the smoothed mask to boolean. Default is 0.1.
        """

        # Skip for vacuum solids
        if (
            self.stl_materials[key][0] == 1.0
            and self.stl_materials[key][1] == 1.0
            and self.stl_materials[key][2] == 0.0
        ):
            return

        if self.verbose > 1:
            print(f"    * Applying subpixel smoothing with factor {factor}...")

        surface = self.read_stl(key)

        shape = tuple(factor * (n - 1) for n in (self.Nx, self.Ny, self.Nz))
        dx = self.dx[0] / factor
        dy = self.dy[0] / factor
        dz = self.dz[0] / factor
        # Place subcell centres inside each physical cell, including the first.
        origin = (
            self.x[0] + dx / 2,
            self.y[0] + dy / 2,
            self.z[0] + dz / 2,
        )
        reference_vol = pv.ImageData(
            dimensions=shape, origin=origin, spacing=(dx, dy, dz)
        )
        vox = surface.voxelize_rectilinear(reference_volume=reference_vol)
        mask_ref = np.asarray(vox["mask"]).reshape(shape, order="F")

        # Combine the higher resolution mask back to the original grid resolution
        mask = np.zeros((self.Nx - 1, self.Ny - 1, self.Nz - 1))
        for i in range(factor):
            for j in range(factor):
                for k in range(factor):
                    sub_mask = mask_ref[i::factor, j::factor, k::factor].astype(float)
                    # A one-cell axis has no resolved mask gradient.
                    sub_mask += np.sqrt(
                        sum(
                            np.gradient(sub_mask, axis=axis) ** 2
                            for axis, size in enumerate(sub_mask.shape)
                            if size > 1
                        )
                    )
                    mask += gaussian_filter(np.clip(sub_mask, 0, 1), sigma=sigma)

        # Overwrite the previous binary/boolean mask in the PyVista grid
        mask = np.clip(mask, 0, factor**3) / factor**3  # Normalize to [0, 1]
        mask = np.where(mask < threshold, 0 * mask, mask)  # clip below threshold
        if make_bool:
            mask = mask > threshold

        self.grid.cell_data[key] = np.pad(mask, ((0, 1),) * 3, mode="edge").ravel()

    def check_stl_masks_overlap(self):
        """
        Check for overlapping STL masks in the grid
        and print warnings if overlaps are found.

        This method computes the sum of all STL masks and identifies cells
        that are marked as inside more than one solid.
        """
        mask_sum = np.zeros(self.Nx * self.Ny * self.Nz, dtype=int)
        for key in self.stl_solids.keys():
            mask_sum += self.grid[key].astype(int)

        overlap_mask = mask_sum > 1
        num_overlaps = np.sum(overlap_mask)

        if num_overlaps > 0 and self.verbose > 1:
            print(
                f"[!] Warning: {num_overlaps} cells are marked as inside multiple STL solids. \
                Consider checking for overlapping geometries."
            )

    def _get_background_mask(self):
        # Get a mask for the background (cells not inside any STL solid)
        bg_mask = np.ones(self.Nx * self.Ny * self.Nz, dtype=bool)
        for key in self.stl_solids.keys():
            bg_mask &= ~self.grid[key].astype(int).astype(bool)
        return bg_mask

    def _mark_cells_in_surface(self, key):
        # Modify the STL mask to account only for the surface
        # Needed for the SIBC boundary condition when conductivity > 1e3 S/m
        grad = np.array(
            self.grid.compute_derivative(scalars=key, gradient="gradient")["gradient"]
        )

        # Compute normals and transverse cell size dn
        # grad_mag = np.linalg.norm(grad, axis=1)
        # n = grad / (grad_mag[:, None] + 1e-14)
        # dn = np.sqrt((n[mask,0]*self.dx)**2 + (n[mask,1]*self.dy)**2 + (n[mask,2]*self.dz)**2)

        # Get boundary cells via gradient magnitude
        grad = np.sqrt(grad[:, 0] ** 2 + grad[:, 1] ** 2 + grad[:, 2] ** 2)
        mask = grad.astype(bool)  # TODO: subpixel smoothing

        return mask

    def read_stl(self, key):
        """
        Read and transform an STL solid by key.

        Parameters
        ----------
        key : str
            Key of the STL solid to read.

        Returns
        -------
        surf : pyvista.PolyData
            The transformed STL surface.
        """
        # import stl
        surf = pv.read(self.stl_solids[key])

        # rotate
        surf = surf.rotate_x(self.stl_rotate[key][0])
        surf = surf.rotate_y(self.stl_rotate[key][1])
        surf = surf.rotate_z(self.stl_rotate[key][2])

        # translate
        surf = surf.translate(self.stl_translate[key])

        # scale
        surf = surf.scale(self.stl_scale[key])

        return surf

    def _compute_snap_points(self, snap_solids=None, snap_tol=1e-8):
        """
        Compute snap points from STL feature edges for mesh refinement.

        Parameters
        ----------
        snap_solids : list or None, optional
            STL solids to use for snap point extraction. Default is all.
        snap_tol : float, optional
            Tolerance for snap point detection.
        """
        if self.verbose > 1:
            print("    * Calculating snappy points...")
        # Support for user-defined stl_keys as list
        if snap_solids is None:
            snap_solids = self.stl_solids.keys()

        # Union of all the surfaces
        model = None
        for key in snap_solids:
            solid = self.read_stl(key)
            if model is None:
                model = solid
            else:
                model = model + solid

        edges = model.extract_feature_edges(boundary_edges=True, manifold_edges=False)

        # Extract points lying in the X-Z plane (Y ≈ 0)
        xz_plane_points = edges.points[np.abs(edges.points[:, 1]) < snap_tol]
        # Extract points lying in the Y-Z plane (X ≈ 0)
        yz_plane_points = edges.points[np.abs(edges.points[:, 0]) < snap_tol]
        # Extract points lying in the X-Y plane (Z ≈ 0)
        xy_plane_points = edges.points[np.abs(edges.points[:, 2]) < snap_tol]

        self.snap_points = np.r_[xz_plane_points, yz_plane_points, xy_plane_points]

        # get the unique x, y, z coordinates
        x_snaps = np.unique(np.round(self.snap_points[:, 0], 5))
        y_snaps = np.unique(np.round(self.snap_points[:, 1], 5))
        z_snaps = np.unique(np.round(self.snap_points[:, 2], 5))

        # Include simulation domain bounds
        self.x_snaps = np.unique(np.concatenate(([self.xmin], x_snaps, [self.xmax])))
        self.y_snaps = np.unique(np.concatenate(([self.ymin], y_snaps, [self.ymax])))
        self.z_snaps = np.unique(np.concatenate(([self.zmin], z_snaps, [self.zmax])))

    def plot_snap_points(self, snap_solids=None, snap_tol=1e-8):
        """
        Plot snap points extracted from STL feature edges for mesh refinement.

        Parameters
        ----------
        snap_solids : list or None, optional
            STL solids to use for snap point extraction. Default is all.
        snap_tol : float, optional
            Tolerance for snap point detection.
        """
        # Support for user-defined stl_keys as list
        if snap_solids is None:
            snap_solids = self.stl_solids.keys()

        # Union of all the surfaces
        model = None
        for key in snap_solids:
            solid = self.read_stl(key)
            if model is None:
                model = solid
            else:
                model = model + solid

        edges = model.extract_feature_edges(boundary_edges=True, manifold_edges=False)

        # Extract points lying in the X-Z plane (Y ≈ 0)
        xz_plane_points = edges.points[np.abs(edges.points[:, 1]) < snap_tol]
        # Extract points lying in the Y-Z plane (X ≈ 0)
        yz_plane_points = edges.points[np.abs(edges.points[:, 0]) < snap_tol]
        # Extract points lying in the X-Y plane (Z ≈ 0)
        xy_plane_points = edges.points[np.abs(edges.points[:, 2]) < snap_tol]

        xz_cloud = pv.PolyData(xz_plane_points)
        yz_cloud = pv.PolyData(yz_plane_points)
        xy_cloud = pv.PolyData(xy_plane_points)

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter()
        pl.add_mesh(model, color="white", opacity=0.5, label="base STL")
        pl.add_mesh(
            edges,
            color="black",
            line_width=5,
            opacity=0.8,
        )
        pl.add_mesh(
            xz_cloud,
            color="green",
            point_size=20,
            render_points_as_spheres=True,
            label="XZ plane points",
        )
        pl.add_mesh(
            yz_cloud,
            color="orange",
            point_size=20,
            render_points_as_spheres=True,
            label="YZ plane points",
        )
        pl.add_mesh(
            xy_cloud,
            color="magenta",
            point_size=20,
            render_points_as_spheres=True,
            label="XY plane points",
        )
        pl.add_legend()
        pl.show()

    def refine_axis(self, xmin, xmax, Nx, x_snaps, method="insert", tol=1e-12):
        """
        Refine a grid axis using snap points and a chosen method.

        Parameters
        ----------
        xmin, xmax : float
            Axis bounds.
        Nx : int
            Number of grid points.
        x_snaps : array_like
            Snap points to include in the axis.
        method : str, optional
            Refinement algorithm ('insert', 'neighbor', 'subdivision').
        tol : float, optional
            Convergence tolerance for optimization.

        Returns
        -------
        x : ndarray
            Refined axis array.
        """

        # Loss function to minimize cell size spread
        def loss_function(x, x0, is_snap):
            # avoid moving snap points
            penalty_snap = np.sum((x[is_snap] - x0[is_snap]) ** 2) * 1000
            # avoid gaps < uniform gap
            dx = np.diff(x)
            threshold = 1 / (len(x) - 1)  # or a hardcoded `min_spacing`
            penalty_small_gaps = np.sum((threshold - dx[dx < threshold]) ** 2)
            # avoid large spread in gap length
            dx = np.diff(x)
            penalty_variance = np.std(dx) * 10
            # return penalty_snap + penalty_small_gaps + penalty_variance
            return np.hstack([penalty_snap, penalty_small_gaps, penalty_variance])

        # Uniformly distributed points as initial guess
        x_snaps = (x_snaps - xmin) / (xmax - xmin)  # normalize to [0,1]

        if method == "insert":
            x0 = np.unique(np.append(x_snaps, np.linspace(0, 1, Nx - len(x_snaps))))

        elif method == "neighbor":
            x = np.linspace(0, 1, Nx)
            dx = np.diff(x)[0]
            for s in x_snaps:
                m = np.isclose(x, s, rtol=0.0, atol=dx / 2)
                if np.sum(m) > 0:
                    x[np.argmax(m)] = s
            x0 = x.copy()

        elif method == "subdivision":
            # x = snaps
            while len(x) < Nx:
                # idx of segments sorted min -> max
                idx_max_diffs = np.argsort(np.diff(x))[-1]  # take bigger

                # print(f"Bigger segment starts at {x[idx_max_diffs]}")
                # compute new point in the middle of the segment
                val = x[idx_max_diffs] + (x[idx_max_diffs + 1] - x[idx_max_diffs]) / 2

                # insert the new point
                x = np.insert(x, idx_max_diffs + 1, val)
                x = np.unique(x)
                # print(f"Inserted point {val} at index {idx_max_diffs}")
            x0 = x.copy()
        else:
            raise ValueError(
                f"Method {method} not supported. Use 'insert', 'neighbor' or 'subdivision'."
            )

        # minimize segment length spread for the test points
        is_snap = np.isin(x0, x_snaps)
        result = least_squares(
            loss_function,
            x0=x0.copy(),
            bounds=(0, 1),  # (zmin, zmax),
            jac="3-point",
            method="dogbox",
            loss="arctan",
            gtol=tol,
            ftol=tol,
            xtol=tol,
            verbose=0,
            args=(x0.copy(), is_snap.copy()),
        )
        # transform back to [xmin, xmax]
        return result.x * (xmax - xmin) + xmin

    def _refine_xyz_axis(self, method="insert", tol=1e-6):
        """
        Refine grid axes using snap points extracted from STL solids.

        Uses the stored snap points (``self.x_snaps``, ``self.y_snaps``,
        ``self.z_snaps``) to refine the axis arrays ``self.x``, ``self.y``,
        ``self.z``. The refinement method and convergence tolerance control
        how new grid nodes are inserted.

        Parameters
        ----------
        method : {'insert','neighbor','subdivision'}, optional
            Refinement algorithm to use when inserting snap points. Default is
            'insert'.
        tol : float, optional
            Convergence tolerance passed to the refinement routine.
        """

        if self.verbose > 1:
            print(f"    * Refining x axis with {len(self.x_snaps)} snaps...")
        self.x = self.refine_axis(
            self.xmin,
            self.xmax,
            self.Nx,
            self.x_snaps,
            method=method,
            tol=tol,
        )

        if self.verbose > 1:
            print(f"    * Refining y axis with {len(self.y_snaps)} snaps...")
        self.y = self.refine_axis(
            self.ymin,
            self.ymax,
            self.Ny,
            self.y_snaps,
            method=method,
            tol=tol,
        )

        if self.verbose > 1:
            print(f"    * Refining z axis with {len(self.z_snaps)} snaps...")
        self.z = self.refine_axis(
            self.zmin,
            self.zmax,
            self.Nz,
            self.z_snaps,
            method=method,
            tol=tol,
        )

        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.Nz = len(self.z)
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)

        if self.verbose > 1:
            print(f"    * Refined grid: Nx = {self.Nx}, Ny ={self.Ny}, Nz = {self.Nz}")

    def _assign_colors(self):
        """
        Assign colors for each STL solid based on material categories.

        Maps entries in ``self.stl_materials`` to color names using the
        ``material_colors`` lookup. Supports string keys referencing the
        material library or explicit material tuples (eps_r, mu_r[, sigma]).
        The resulting mapping is stored in ``self.stl_colors``.
        """
        self.stl_colors = {}

        for key in self.stl_solids:
            mat = self.stl_materials[key]
            if type(mat) is str:
                _color = material_colors.get(mat, material_colors["other"])
                self.stl_colors[key] = _color
            elif len(mat) == 2:
                if mat[0] is np.inf:  # eps_r
                    self.stl_colors[key] = material_colors["pec"]
                elif mat[0] > 1.0:  # eps_r
                    self.stl_colors[key] = material_colors["dielectric"]
                else:
                    self.stl_colors[key] = material_colors["vacuum"]
            elif len(mat) == 3:
                self.stl_colors[key] = material_colors["lossy metal"]
            else:
                self.stl_colors[key] = material_colors["other"]

    def save_to_h5(self, filename="grid.h5"):
        """
        Save the generated grid and STL metadata to an HDF5 file.

        The file contains primal node arrays and full-size computational-cell
        STL masks, when present. Dual coordinates and FIT metrics are rebuilt
        from the node arrays when loading.

        Parameters
        ----------
        filename : str, optional
            Output filename for the HDF5 file. Default 'grid.h5'.
        """

        if not filename.endswith(".h5"):
            filename += ".h5"

        if self.verbose:
            print("Saving grid to HDF5 file:", filename)

        with h5py.File(filename, "w") as hf:
            hf.create_dataset("x", data=np.array(self.x))
            hf.create_dataset("y", data=np.array(self.y))
            hf.create_dataset("z", data=np.array(self.z))

            # Geometry-pipeline metadata
            hf.attrs["geometry_mode"] = self.geometry_mode

            if hasattr(self, "stl_method"):
                hf.attrs["stl_method"] = self.stl_method

            if hasattr(self, "stl_tol"):
                hf.attrs["stl_tol"] = self.stl_tol

            # Preserve STL ordering explicitly because it defines region
            # priority in overlapping solids.
            hf.create_dataset(
                "stl_key_order",
                data=np.asarray(
                    list(self.stl_solids or {}),
                    dtype="S",
                ),
            )

            # Save stl_ variables as groups
            for attr in [
                "stl_solids",
                "stl_materials",
                "stl_material_types",
                "stl_colors",
                "stl_scale",
                "stl_rotate",
                "stl_translate",
            ]:
                grp = hf.create_group(attr)
                dct = getattr(self, attr)
                if dct is None or not isinstance(dct, dict):
                    dct = {}
                for key, val in dct.items():
                    # Use dtype='S' for strings, otherwise np.array
                    if isinstance(val, str):
                        grp.create_dataset(str(key), data=np.bytes_(val))
                    else:
                        grp.create_dataset(str(key), data=np.array(val))

            for key in self.stl_solids or {}:
                hf.create_dataset(
                    "grid_" + key, data=np.asarray(self.grid.cell_data[key])
                )

                # Conformal point masks (primal and dual grid points)
                if key in self.primal_point_masks:
                    hf.create_dataset(
                        "grid_primal_points_" + key,
                        data=np.asarray(self.primal_point_masks[key], dtype=bool),
                    )

                if key in self.dual_point_masks:
                    hf.create_dataset(
                        "grid_dual_points_" + key,
                        data=np.asarray(self.dual_point_masks[key], dtype=bool),
                    )

    def load_from_h5(self, filename):
        """
        Load grid axis arrays and STL metadata from an HDF5 file.

        Restore primal node arrays and current-format STL masks. Dual
        coordinates and all FIT metrics are recomputed from the node arrays.

        Parameters
        ----------
        filename : str
            HDF5 filename to read. The '.h5' suffix is appended if missing.
        """
        if not filename.endswith(".h5"):
            filename += ".h5"

        if self.verbose:
            print("Loading grid from HDF5 file:", filename)

        with h5py.File(filename, "r") as hf:
            # ----------------------------------------------------------
            # Grid coordinates
            # ----------------------------------------------------------
            self.x = hf["x"][()]
            self.y = hf["y"][()]
            self.z = hf["z"][()]
            self.Nx, self.Ny, self.Nz = len(self.x), len(self.y), len(self.z)

            # ----------------------------------------------------------
            # Geometry-pipeline metadata
            # ----------------------------------------------------------
            if "geometry_mode" in hf.attrs:
                geometry_mode = hf.attrs["geometry_mode"]

                if isinstance(geometry_mode, bytes):
                    geometry_mode = geometry_mode.decode()

                self.geometry_mode = str(geometry_mode).lower()

            else:
                # Backward compatibility with older HDF5 files.
                has_primal_point_masks = any(
                    name.startswith("grid_primal_points_") for name in hf.keys()
                )

                self.geometry_mode = "conformal" if has_primal_point_masks else "legacy"

            if self.geometry_mode not in ("legacy", "conformal"):
                raise ValueError(
                    f"Invalid geometry_mode stored in HDF5 file: {self.geometry_mode}"
                )

            if "stl_method" in hf.attrs:
                stl_method = hf.attrs["stl_method"]

                if isinstance(stl_method, bytes):
                    stl_method = stl_method.decode()

                self.stl_method = str(stl_method)

            if "stl_tol" in hf.attrs:
                self.stl_tol = float(hf.attrs["stl_tol"])

            # ----------------------------------------------------------
            # Preserve STL ordering
            # ----------------------------------------------------------
            if "stl_key_order" in hf:
                stl_key_order = []

                for value in hf["stl_key_order"][()]:
                    if isinstance(value, bytes):
                        value = value.decode()

                    stl_key_order.append(str(value))

            else:
                # Backward compatibility with old files.
                stl_key_order = list(hf["stl_solids"].keys())

            # ----------------------------------------------------------
            # STL dictionaries
            # ----------------------------------------------------------
            for attr in [
                "stl_solids",
                "stl_materials",
                "stl_colors",
                "stl_scale",
                "stl_rotate",
                "stl_translate",
            ]:
                dct = {}
                grp = hf[attr]

                for key in stl_key_order:
                    if key not in grp:
                        continue

                    val = grp[key][()]

                    if isinstance(val, bytes):
                        val = val.decode()

                    dct[key] = val

                setattr(
                    self,
                    attr,
                    dct,
                )

            # ----------------------------------------------------------
            # Material treatment types
            # ----------------------------------------------------------
            if "stl_material_types" in hf:
                self.stl_material_types = {}

                grp = hf["stl_material_types"]

                for key in stl_key_order:
                    if key not in grp:
                        continue

                    value = grp[key][()]

                    if isinstance(value, bytes):
                        value = value.decode()

                    self.stl_material_types[key] = str(value)

            else:
                # Old files did not contain explicit treatment types.
                self.stl_material_types = None

            # Fill missing types and validate the resulting dictionary.
            self._prepare_stl_material_types()

        # recompute dx, dy, dz
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)
        self.xmin, self.xmax = self.x[0], self.x[-1]
        self.ymin, self.ymax = self.y[0], self.y[-1]
        self.zmin, self.zmax = self.z[0], self.z[-1]

        # recommpute grid and L, iA, tL, itA
        self._compute_grid()

        # Current-format masks include the terminal computational planes.
        with h5py.File(filename, "r") as hf:
            for key in self.stl_solids.keys():
                mask = hf["grid_" + key][()]
                if mask.shape != (self.grid.n_cells,):
                    raise ValueError(
                        f"STL mask '{key}' has shape {mask.shape}; "
                        f"the current grid requires ({self.grid.n_cells},) "
                        "including terminal storage cells. Recreate and save "
                        "the grid with the current Wakis version."
                    )
                self.grid.cell_data[key] = mask

                # Conformal point masks (only written in conformal mode).
                for name, masks, shape in (
                    (
                        "grid_primal_points_",
                        self.primal_point_masks,
                        (len(self.x), len(self.y), len(self.z)),
                    ),
                    (
                        "grid_dual_points_",
                        self.dual_point_masks,
                        (len(self.tx), len(self.ty), len(self.tz)),
                    ),
                ):
                    if name + key not in hf:
                        continue
                    point_mask = np.asarray(hf[name + key][()], dtype=bool)
                    if point_mask.shape != shape:
                        raise ValueError(
                            f"Point mask '{name + key}' has shape "
                            f"{point_mask.shape}; the current grid requires "
                            f"{shape}. Recreate and save the grid with the "
                            "current Wakis version."
                        )
                    masks[key] = point_mask

            if all(key in self.primal_point_masks for key in self.stl_solids) and all(
                key in self.dual_point_masks for key in self.stl_solids
            ):
                self._build_interface_topology()

        # add verbosity
        if self.verbose > 1:
            print(f"Loaded grid with {self.Nx * self.Ny * self.Nz} mesh cells:")
            print(f"    * Number of cells: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}")
            print(
                f"    * Simulation domain bounds: \n\
                x:[{self.xmin:.3f}, {self.xmax:.3f}],\n\
                y:[{self.ymin:.3f}, {self.ymax:.3f}],\n\
                z:[{self.zmin:.3f}, {self.zmax:.3f}]"
            )
            print(
                f"    * STL solids imported:\n\
                {list(self.stl_solids.keys())}"
            )
            print(
                f"    * STL solids assigned materials [eps_r, mu_r, sigma]:\n\
                {list(self.stl_materials.values())}"
            )

        # update logger
        self.update_logger(["Nx", "Ny", "Nz", "dx", "dy", "dz"])
        self.update_logger(["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])
        self.update_logger(["stl_solids", "stl_materials"])
        if self.stl_rotate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_rotate"])
        if self.stl_translate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_translate"])
        if self.stl_scale != 1.0:
            self.update_logger(["stl_scale"])

    def update_logger(self, attrs):
        """
        Copy selected Grid attributes into the internal ``Logger``.

        Parameters
        ----------
        attrs : iterable of str
            Names of attributes to copy into ``self.logger.grid``.
        """
        for atr in attrs:
            self.logger.grid[atr] = getattr(self, atr)
