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


class GridFIT3D(PlotMixin):
    """
    Class holding the grid information and
    stl importing handling using PyVista
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
            Number of cells per direction. If None, must provide x, y, z arrays.
        x, y, z : array_like, optional
            Custom grid axis arrays to be used in the meshgrid generation.
            Non-uniform grids are supported.
        use_mpi : bool, optional
            Enable MPI domain decomposition in the z direction. Default is False.
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
        geometry_mode : str, optional
            Geometry-mask representation used after STL classification.
            ``"legacy"`` preserves the existing cell-based mask pipeline.
            ``"conformal"`` classifies primal and dual grid points directly,
            derives the cell mask from the primal point classification, and
            constructs interface topology and geometric STL intersections.
            Default is ``"legacy"``.
        subpixel_smoothing : bool, optional
            Whether to apply subpixel smoothing to the STL masks after voxelization. Default is False.
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

        Attributes
        ----------
        x, y, z : ndarray
            Grid axis arrays.
        Nx, Ny, Nz : int
            Number of cells in each direction.
        dx, dy, dz : ndarray
            Cell sizes in each direction.
        grid : pyvista.StructuredGrid
            PyVista grid object.
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

        # Grid data
        # generate from file
        if load_from_h5 is not None:
            self.load_from_h5(load_from_h5)
            return  # TODO: support MPI decomposition

        # generate from custom x,y,z arrays
        elif x is not None and y is not None and z is not None:
            # allow user to set the grid axis manually
            self.x = x
            self.y = y
            self.z = z
            self.Nx = len(self.x) - 1
            self.Ny = len(self.y) - 1
            self.Nz = len(self.z) - 1
            self.xmin, self.xmax = self.x[0], self.x[-1]
            self.ymin, self.ymax = self.y[0], self.y[-1]
            self.zmin, self.zmax = self.z[0], self.z[-1]
            if self.use_mpi:
                raise ValueError(
                    "[!] Error: use_mpi=True is not compatible with custom x,y,z arrays."
                )

        # generate from domain extents and number of cells [LEGACY]
        elif all(v is not None for v in [xmin, xmax, ymin, ymax, zmin, zmax]):
            # uniform grid from domain extents and number of cells
            self.xmin, self.xmax = xmin, xmax
            self.ymin, self.ymax = ymin, ymax
            self.zmin, self.zmax = zmin, zmax
            self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
            self.x = np.linspace(self.xmin, self.xmax, self.Nx + 1)
            self.y = np.linspace(self.ymin, self.ymax, self.Ny + 1)
            self.z = np.linspace(self.zmin, self.zmax, self.Nz + 1)

        else:
            raise ValueError(
                "[!] Error initializing GridFIT3D:\n"
                "  - Provide grid axis arrays: x, y, z\n"
                "  - OR domain extents and number of cells: \
                    xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz\n"
                "  - OR load from a HDF5 file using load_from_h5"
            )

        # TODO: allow non uniform dx, dy, dz
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

        print(f"Generating grid with {self.Nx * self.Ny * self.Nz} mesh cells...")
        if verbose > 1:
            print(
                f"    * Simulation domain bounds: \n\
                x:[{xmin:.3f}, {xmax:.3f}],\n\
                y:[{ymin:.3f}, {ymax:.3f}],\n\
                z:[{zmin:.3f}, {zmax:.3f}]"
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
            if imported_mpi:
                self._mpi_initialize()
                if self.verbose:
                    print(f"MPI initialized for {self.rank} of {self.size}")
            else:
                raise ImportError(
                    "[!] mpi4py is required when use_mpi=True but was not found"
                )

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

        Sets up the structured grid, cell lengths, inverse areas, and tilde grid.
        """
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.grid = pv.StructuredGrid(X.transpose(), Y.transpose(), Z.transpose())

        self.L = Field(self.Nx, self.Ny, self.Nz)
        self.L.field_x = X[1:, 1:, 1:] - X[:-1, :-1, :-1]
        self.L.field_y = Y[1:, 1:, 1:] - Y[:-1, :-1, :-1]
        self.L.field_z = Z[1:, 1:, 1:] - Z[:-1, :-1, :-1]

        self.iA = Field(self.Nx, self.Ny, self.Nz)
        self.iA.field_x = np.divide(1.0, self.L.field_y * self.L.field_z)
        self.iA.field_y = np.divide(1.0, self.L.field_x * self.L.field_z)
        self.iA.field_z = np.divide(1.0, self.L.field_x * self.L.field_y)

        # tilde grid ~G
        self.tx = (self.x[1:] + self.x[:-1]) / 2
        self.ty = (self.y[1:] + self.y[:-1]) / 2
        self.tz = (self.z[1:] + self.z[:-1]) / 2

        self.tx = np.append(self.tx, self.x[-1])
        self.ty = np.append(self.ty, self.y[-1])
        self.tz = np.append(self.tz, self.z[-1])

        tX, tY, tZ = np.meshgrid(self.tx, self.ty, self.tz, indexing="ij")

        self.tL = Field(self.Nx, self.Ny, self.Nz)
        self.tL.field_x = tX[1:, 1:, 1:] - tX[:-1, :-1, :-1]
        self.tL.field_y = tY[1:, 1:, 1:] - tY[:-1, :-1, :-1]
        self.tL.field_z = tZ[1:, 1:, 1:] - tZ[:-1, :-1, :-1]

        self.itA = Field(self.Nx, self.Ny, self.Nz)
        aux = self.tL.field_y * self.tL.field_z
        self.itA.field_x = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux != 0)
        aux = self.tL.field_x * self.tL.field_z
        self.itA.field_y = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux != 0)
        aux = self.tL.field_x * self.tL.field_y
        self.itA.field_z = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux != 0)
        del aux

    def _mpi_initialize(self):
        """
        Initialize MPI domain decomposition in the z direction.

        Sets up MPI communicator, rank, size, and subdomain grid quantities.
        """
        comm = MPI.COMM_WORLD  # Get MPI communicator
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Error handling for Nz < size
        if self.Nz < self.size:
            raise ValueError(
                f"Nz ({self.Nz}) must be greater than or equal to the number of \
                MPI processes ({self.size})."
            )

        # global z quantities [ALLCAPS]
        self.ZMIN = self.zmin
        self.ZMAX = self.zmax
        self.NZ = self.Nz - self.Nz % (self.size)  # ensure multiple of MPI size
        self.Z = np.linspace(self.ZMIN, self.ZMAX, self.NZ + 1)[:-1]
        self.Z += (self.ZMAX - self.ZMIN) / (2 * self.NZ)

        if self.verbose and self.rank == 0:
            print(f" * Global grid ZMIN={self.ZMIN}, ZMAX={self.ZMAX}, NZ={self.NZ}")

        # MPI subdomain quantities [TODO: support non-uniform dz with MPI]
        self.Nz = self.NZ // (self.size)
        self.dz = (self.ZMAX - self.ZMIN) / self.NZ
        self.zmin = self.rank * self.Nz * self.dz + self.ZMIN
        self.zmax = (self.rank + 1) * self.Nz * self.dz + self.ZMIN

        if self.verbose:
            print(
                f"MPI rank {self.rank} of {self.size} initialized with \
                        zmin={self.zmin}, zmax={self.zmax}, Nz={self.Nz}"
            )
        # Add ghost cells
        self.n_ghosts = 1
        if self.rank > 0:
            self.zmin += -self.n_ghosts * self.dz
            self.Nz += self.n_ghosts
        if self.rank < (self.size - 1):
            self.zmax += self.n_ghosts * self.dz
            self.Nz += self.n_ghosts

        # Support for single core
        if self.rank == 0 and self.size == 1:
            self.zmax += self.n_ghosts * self.dz
            self.Nz += self.n_ghosts

        self.z = np.linspace(self.zmin, self.zmax, self.Nz + 1)
        self.dz = np.diff(self.z)  # only uniform grid possible with MPI

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
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax,
                self.ZMIN,
                self.ZMAX,
                self.Nx,
                self.Ny,
                self.NZ,
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

        spacing = np.abs(np.diff(axis_coordinates))

        positive_spacing = spacing[spacing > 0.0]

        if positive_spacing.size == 0:
            raise RuntimeError(
                f"No finite {direction}-edge length available on the {grid_type} grid."
            )

        line_extension = np.min(positive_spacing)

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

            points, cell_ids = surface.ray_trace(
                start_point,
                end_point,
                first_point=False,
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

    def _build_interface_intersections(
        self,
        alpha_tolerance=1e-8,
    ):
        """
        Compute all STL intersections with primal and dual grid edges.

        Intersections are obtained by tracing complete grid lines through all
        STL surfaces. Therefore, intersections are also detected when an STL
        region is thinner than one grid edge and both edge endpoints belong
        to the same effective region.
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
                    "crossing_groups": {},
                    "contact_groups": {},
                }

                raw_hits = {}

                for transverse_index in transverse_indices:
                    line_hits = self._trace_grid_line(
                        grid_type,
                        direction,
                        transverse_index,
                        surface_cache,
                        alpha_tolerance=alpha_tolerance,
                    )

                    for edge_index, hits in line_hits.items():
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

                        intersections["crossing_groups"][edge_index] = crossing_groups

                        intersections["multiple_crossing_positions"][edge_index] = (
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
                        selected_group = self._select_crossing_group(crossing_groups)

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
        Build the interface masks used by the current conformal FIT
        discretization.

        An interface edge is considered resolved if

        1. its endpoint region IDs indicate a material transition, and
        2. at least one geometric STL crossing was found.

        Hidden crossings are retained in the geometric intersection data but
        are intentionally ignored by the current discretization. Endpoint
        interfaces without a crossing are also excluded.
        """
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

                self.resolved_interface_edge_masks[grid_type][direction] = (
                    endpoint_mask & has_crossing
                )

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

                resolved_mask = self.resolved_interface_edge_masks[grid_type][direction]

                data = self.interface_intersections[grid_type][direction]

                has_crossing = data["has_crossing"]

                multiple = data["multiple_crossing_positions"]

                missing = endpoint_mask & ~has_crossing

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
                dx, dy, dz = (
                    (self.xmax - self.xmin) / (self.Nx),
                    (self.ymax - self.ymin) / (self.Ny),
                    (self.zmax - self.zmin) / (self.Nz),
                )
                reference_vol = pv.ImageData(
                    dimensions=(self.Nx, self.Ny, self.Nz),
                    origin=(self.xmin, self.ymin, self.zmin),
                    spacing=(dx, dy, dz),
                )

                try:
                    vox = surf.voxelize_rectilinear(
                        reference_volume=reference_vol, progress_bar=progress_bar
                    )
                    mask = np.reshape(
                        vox["mask"], (self.Nx, self.Ny, self.Nz), order="F"
                    ).astype(bool)

                    self.grid[key] = np.reshape(mask, (self.Nx * self.Ny * self.Nz))

                    # Apply subpixel smoothing if enabled
                    if self.use_subpixel_smoothing:
                        self._apply_subpixel_smoothing(
                            key,
                            factor=self.subpixel_smoothing_factor,
                            threshold=self.subpixel_smoothing_threshold,
                            make_bool=self.subpixel_smoothing_bool,
                        )

                except Exception:
                    print(
                        f"[!] Warning: voxelization for stl solid {key} failed. Consider checking if the grid is uniform or using a different method."
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

        # Produce Warnings for ignored interface cases (hidden crossings, missing crossings)
        self._warn_interface_diagnostics()

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

        # Create a higher resolution reference volume for voxelization
        Nx, Ny, Nz = self.Nx * factor, self.Ny * factor, self.Nz * factor
        dx = (self.xmax - self.xmin) / (Nx)
        dy = (self.ymax - self.ymin) / (Ny)
        dz = (self.zmax - self.zmin) / (Nz)

        reference_vol = pv.ImageData(
            dimensions=(Nx, Ny, Nz),
            origin=(self.xmin, self.ymin, self.zmin),
            spacing=(dx, dy, dz),
        )
        vox = surface.voxelize_rectilinear(
            reference_volume=reference_vol,
        )
        mask_ref = np.reshape(vox["mask"], (Nx, Ny, Nz), order="F")

        # Combine the higher resolution mask back to the original grid resolution
        mask = np.zeros((self.Nx, self.Ny, self.Nz)).astype(float)
        for i in range(factor):
            for j in range(factor):
                for k in range(factor):
                    sub_mask = mask_ref[i::factor, j::factor, k::factor].astype(float)
                    sub_mask += np.sqrt(sum(g**2 for g in np.gradient(sub_mask)))
                    mask += gaussian_filter(np.clip(sub_mask, 0, 1), sigma=sigma)

        # Overwrite the previous binary/boolean mask in the PyVista grid
        mask = np.clip(mask, 0, factor**3) / factor**3  # Normalize to [0, 1]
        mask = np.where(mask < threshold, 0 * mask, mask)  # clip below threshold
        if make_bool:
            mask = mask > threshold

        self.grid[key] = np.reshape(mask, (self.Nx * self.Ny * self.Nz))

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
            self.Nx + 1,
            self.x_snaps,
            method=method,
            tol=tol,
        )

        if self.verbose > 1:
            print(f"    * Refining y axis with {len(self.y_snaps)} snaps...")
        self.y = self.refine_axis(
            self.ymin,
            self.ymax,
            self.Ny + 1,
            self.y_snaps,
            method=method,
            tol=tol,
        )

        if self.verbose > 1:
            print(f"    * Refining z axis with {len(self.z_snaps)} snaps...")
        self.z = self.refine_axis(
            self.zmin,
            self.zmax,
            self.Nz + 1,
            self.z_snaps,
            method=method,
            tol=tol,
        )

        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1
        self.Nz = len(self.z) - 1
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

        The file contains axis arrays, STL masks suitable for ``grid.cell_data``,
        and all ``stl_`` related variables (materials, colors, transforms).

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
                    list(self.stl_solids.keys()),
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
                for key, val in dct.items():
                    # Use dtype='S' for strings, otherwise np.array
                    if isinstance(val, str):
                        grp.create_dataset(str(key), data=np.bytes_(val))
                    else:
                        grp.create_dataset(str(key), data=np.array(val))

            for key in self.stl_solids.keys():
                # Existing cell mask
                hf.create_dataset(
                    "grid_" + key,
                    data=np.array(self.grid[key]),
                )

                # New primal point mask
                if key in self.primal_point_masks:
                    hf.create_dataset(
                        "grid_primal_points_" + key,
                        data=np.asarray(
                            self.primal_point_masks[key],
                            dtype=bool,
                        ),
                    )

                if key in self.dual_point_masks:
                    hf.create_dataset(
                        "grid_dual_points_" + key,
                        data=np.asarray(
                            self.dual_point_masks[key],
                            dtype=bool,
                        ),
                    )

    def load_from_h5(self, filename):
        """
        Load grid axis arrays and STL metadata from an HDF5 file.

        The function restores axis arrays, recomputes grid metrics and fills
        ``self.grid`` cell_data with imported STL masks saved previously by
        ``save_to_h5``.

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

            # ----------------------------------------------------------
            # Geometry-pipeline metadata
            # ----------------------------------------------------------
            if "geometry_mode" in hf.attrs:
                geometry_mode = hf.attrs["geometry_mode"]

                if isinstance(geometry_mode, bytes):
                    geometry_mode = geometry_mode.decode()

                self.geometry_mode = str(geometry_mode).lower()

            else:
                # Backward compatibility:
                # Files containing primal point masks were produced by
                # the conformal geometry pipeline.
                has_primal_point_masks = any(
                    name.startswith("grid_primal_points_") for name in hf.keys()
                )

                if has_primal_point_masks:
                    self.geometry_mode = "conformal"

            if self.geometry_mode not in (
                "legacy",
                "conformal",
            ):
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

        # recompute dx, dy, dz, Nx, Ny, Nz
        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1
        self.Nz = len(self.z) - 1
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)
        self.xmin, self.xmax = self.x[0], self.x[-1]
        self.ymin, self.ymax = self.y[0], self.y[-1]
        self.zmin, self.zmax = self.z[0], self.z[-1]

        # recommpute grid and L, iA, tL, itA
        self._compute_grid()

        # asign masks to grid.cell_data
        with h5py.File(filename, "r") as hf:
            for key in self.stl_solids.keys():
                # Existing cell mask
                self.grid[key] = hf["grid_" + key][()]

                # Newer files contain the primal point mask.
                point_mask_name = "grid_primal_points_" + key

                if point_mask_name in hf:
                    primal_point_mask = np.asarray(
                        hf[point_mask_name][()],
                        dtype=bool,
                    )

                    self.primal_point_masks[key] = primal_point_mask

                # Newest files also contain the directly
                # classified dual point mask.
                dual_point_mask_name = "grid_dual_points_" + key

                if dual_point_mask_name in hf:
                    self.dual_point_masks[key] = np.asarray(
                        hf[dual_point_mask_name][()],
                        dtype=bool,
                    )

                elif point_mask_name in hf:
                    # Backward compatibility for HDF5 files
                    # written after primal point masks were
                    # introduced but before direct dual-point
                    # classification was introduced.
                    cell_center_mask = self._point_mask_to_cell_mask(
                        primal_point_mask,
                        threshold=0.5,
                    )

                    self.dual_point_masks[key] = self._cell_mask_to_dual_point_mask(
                        cell_center_mask
                    )

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
