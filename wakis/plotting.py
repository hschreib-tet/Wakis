# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


class PlotMixinSolver:
    """Plotting mixin providing 1D/2D/3D visualization helpers for SolverFIT3D."""

    def plot3D(
        self,
        field="E",
        component="z",
        clim=None,
        hide_solids=None,
        show_solids=None,
        add_stl=None,
        stl_opacity=0.1,
        stl_colors="white",
        title=None,
        cmap="jet",
        clip_interactive=True,
        clip_normal="-y",
        clip_box=False,
        clip_bounds=None,
        off_screen=False,
        zoom=0.5,
        camera_position=None,
        nan_opacity=1.0,
        n=None,
    ):
        """Built-in 3D plotting using PyVista.

        Displays a cell-centered field (E, H or J) on the structured grid using
        PyVista. Optionally overlays STL solids, clipping widgets or a static
        clip box. When rendered off-screen the plot can be saved to a PNG file
        for later composition into animations.

        Parameters
        ----------
        field : str, optional
            3D field magnitude to plot: 'E', 'H', or 'J'. A two-letter value
            (e.g., 'Ex') is accepted to plot a single component; in that case
            ``component`` is overridden. Default 'E'.
        component : str, optional
            Component of the field to plot: 'x', 'y', 'z', or 'Abs'. Default
            'z'. Overridden when ``field`` contains a component.
        clim : sequence, optional
            Colorbar limits for the field plot as ``[min, max]``.
        hide_solids : str, list, or None, optional
            If provided, mask values inside the named STL solids to ``np.nan``.
            NaNs are shown in grey due to a known PyVista opacity limitation.
        show_solids : str, list, or None, optional
            If provided, mask values outside the named STL solids to ``np.nan``.
        add_stl : str or list, optional
            STL solid key or list of keys to add to the scene via ``pv.add_mesh``.
        stl_opacity : float, optional
            Opacity for STL surfaces (0 transparent, 1 opaque). Default 0.1.
        stl_colors : str or list, optional
            Color or list of colors for STL surfaces (default 'white').
        title : str, optional
            Basename used to save screenshots when ``off_screen=True``.
        cmap : str, optional
            Colormap name for the field display (default 'jet').
        clip_interactive : bool, optional
            Enable an interactive clipping plane widget (default False).
        clip_normal : str, optional
            Normal direction for the interactive clip plane (default '-y').
        clip_box : bool, optional
            Enable a static box clipping of the domain (default False).
        clip_bounds : sequence, optional
            Box bounds as ``[xmin,xmax,ymin,ymax,zmin,zmax]`` when ``clip_box``
            is active.
        off_screen : bool, optional
            Render off-screen and save screenshots instead of opening an
            interactive window (default False).
        zoom : float, optional
            Camera zoom factor (default 0.5).
        camera_position : str, sequence, or None, optional
            Camera position preset or coordinates (default None).
        nan_opacity : float, optional
            Opacity for NaN values produced by masking (default 1.0).
        n : int, optional
            Timestep index appended to saved screenshot filenames.

        Returns
        -------
        None

        Notes
        -----
        - When running with MPI (`use_mpi=True`), plotting is not supported.
        - The method modifies the PyVista plotter state and may save screenshots if
          `off_screen=True`.
        """
        if self.use_mpi:
            print("[!] plot3D is not supported when `use_mpi=True`")
            return

        if len(field) == 2:  # support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component + "3d"

        if self.plotter_active and not off_screen:
            self.plotter_active = False

        if not self.plotter_active:
            pl = pv.Plotter(off_screen=off_screen)

            # Plot stl surface(s)
            if add_stl is not None:
                if type(add_stl) is str:
                    key = add_stl
                    surf = self.grid.read_stl(key)
                    pl.add_mesh(
                        surf,
                        color=stl_colors,
                        opacity=stl_opacity,
                        smooth_shading=True,
                    )

                elif type(add_stl) is list:
                    for i, key in enumerate(add_stl):
                        surf = self.grid.read_stl(key)
                        if type(stl_colors) is list:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[i],
                                opacity=stl_opacity,
                                smooth_shading=True,
                            )
                        else:
                            pl.add_mesh(
                                surf,
                                color=stl_colors,
                                opacity=stl_opacity,
                                smooth_shading=True,
                            )
                else:
                    key = self.grid.stl_solids.keys()[0]
                    surf = self.grid.read_stl(key)
                    pl.add_mesh(
                        surf,
                        color=stl_colors,
                        opacity=stl_opacity,
                        smooth_shading=True,
                    )

            if camera_position is None:
                pl.camera_position = "zx"
                pl.camera.azimuth += 30
                pl.camera.elevation += 30
            else:
                pl.camera_position = camera_position

            pl.set_background("mistyrose", top="white")
            self._add_logo_widget(pl)
            pl.camera.zoom(zoom)
            pl.add_axes()
            pl.enable_3_lights()

            if off_screen:
                self.plotter_active = True
        else:
            pl = self.pl

        # Plot field
        if field == "E":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.E[:, :, :, component], self.N
            )
        elif field == "H":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.H[:, :, :, component], self.N
            )
        elif field == "J":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.J[:, :, :, component], self.N
            )
        else:
            print("`field` value not valid")

        points = self.grid.grid.cell_data_to_point_data()  # interpolate

        # Mask the values inside solid to np.nan
        if hide_solids is not None:
            tol = np.min([self.dx, self.dy, self.dz]) * 1e-3
            if type(hide_solids) is str:
                surf = self.grid.read_stl(hide_solids)
                select = self.grid.grid.select_enclosed_points(surf, tolerance=tol)
                mask = select["SelectedPoints"] > 0

            elif type(hide_solids) is list:
                for i, solid in enumerate(hide_solids):
                    surf = self.grid.read_stl(solid)
                    select = self.grid.grid.select_enclosed_points(surf, tolerance=tol)
                    if i == 0:
                        mask = select["SelectedPoints"] > 0
                    else:
                        mask += select["SelectedPoints"] > 0

            points[field + component][mask] = np.nan

        # Mask the values outside solid to np.nan
        if show_solids is not None:
            tol = np.min([self.dx, self.dy, self.dz]) * 1e-3
            if type(show_solids) is str:
                surf = self.grid.read_stl(show_solids)
                select = self.grid.grid.select_enclosed_points(surf, tolerance=tol)
                mask = select["SelectedPoints"] > 0

            elif type(show_solids) is list:
                for i, solid in enumerate(show_solids):
                    surf = self.grid.read_stl(solid)
                    select = self.grid.grid.select_enclosed_points(surf, tolerance=tol)
                    if i == 0:
                        mask = select["SelectedPoints"] > 0
                    else:
                        mask += select["SelectedPoints"] > 0

            points[field + component][np.logical_not(mask)] = np.nan

        # Clip a rectangle of the domain
        if clip_box:
            if clip_bounds is None:
                Lx, Ly = (
                    (self.grid.xmax - self.grid.xmin),
                    (self.grid.ymax - self.grid.ymin),
                )
                clip_bounds = [
                    self.grid.xmax - Lx / 2,
                    self.grid.xmax,
                    self.grid.ymax - Ly / 2,
                    self.grid.ymax,
                    self.grid.zmin,
                    self.grid.zmax,
                ]

            ac1 = pl.add_mesh(
                points.clip_box(bounds=clip_bounds),
                opacity=nan_opacity,
                scalars=field + component,
                cmap=cmap,
                clim=clim,
            )

        # Enable an interactive widget to clip out part of the domain with a plane, with clip_normal
        elif clip_interactive:
            ac1 = pl.add_mesh_clip_plane(
                points,
                normal=clip_normal,
                opacity=1.0,
                scalars=field + component,
                cmap=cmap,
                clim=clim,
                normal_rotation=False,
                nan_opacity=nan_opacity,
            )
        else:
            print("Plotting option inconsistent")

        # Save
        if n is not None:
            pl.add_title(
                field + component + f" field, timestep={n}",
                font="times",
                font_size=12,
            )
            title += "_" + str(n).zfill(6)
        if off_screen:
            pl.screenshot(title + ".png")
            try:
                pl.remove_actor(ac1)
            except Exception:
                pass
            self.pl = pl
        else:
            pl.show(full_screen=False, auto_close=False, interactive_update=True)

    def plot3DonSTL(
        self,
        field="E",
        component="z",
        clim=None,
        cmap="jet",
        log_scale=False,
        stl_with_field=None,
        field_opacity=1.0,
        tolerance=None,
        stl_transparent=None,
        stl_opacity=0.1,
        stl_colors="white",
        clip_plane=False,
        clip_interactive=False,
        clip_normal="-x",
        clip_origin=[0, 0, 0],
        clip_box=False,
        clip_bounds=None,
        title=None,
        off_screen=False,
        n=None,
        zoom=0.5,
        camera_position=None,
        **kwargs,
    ):
        """Built-in 3D plotting sampling fields onto STL surfaces.

        Samples a cell-centered field onto STL surfaces and visualizes the
        sampled field. Supports clipping with a plane or a static box and
        rendering the STL surfaces with or without sampled field data.

        Parameters
        ----------
        field : str, optional
            Field to sample and plot: 'E', 'H', or 'J'. Two-letter values like
            'Ex' are accepted and override ``component``. Default 'E'.
        component : str, optional
            Component to plot ('x', 'y', 'z', 'Abs'). Default 'z'.
        clim : sequence, optional
            Color limits for the plotted field.
        cmap : str, Colormap, optional
            Colormap for field rendering (default 'jet').
        log_scale : bool, optional
            Use logarithmic color scaling when True (default False).
        stl_with_field : str or list, optional
            STL key or list of keys whose surfaces will be sampled with the
            selected field.
        field_opacity : float, optional
            Opacity for the sampled field rendering on STL surfaces.
        tolerance : float, optional
            Sampling tolerance passed to ``pyvista.PolyData.sample``.
        stl_transparent : str or list, optional
            STL solids to add to the scene with transparent rendering.
        stl_opacity : float, optional
            Opacity for non-sampled STL solids (default 0.1).
        stl_colors : str or list, optional
            Colors used for STL solids (default 'white').
        clip_interactive : bool, optional
            Enable an interactive clipping plane for sampled surfaces.
        clip_plane : bool, optional
            Clip the STL surface with a plane and show the field on the plane.
        clip_normal : str, optional
            Normal direction for clipping operations (default '-x').
        clip_origin : sequence, optional
            Origin for planar clipping when ``clip_plane`` is True.
        clip_box : bool, optional
            Clip sampled surfaces using a static box (default False).
        clip_bounds : sequence, optional
            Bounds for the static clip box as
            ``[xmin,xmax,ymin,ymax,zmin,zmax]``.
        title : str, optional
            Basename used to save screenshots when ``off_screen=True``.
        off_screen : bool, optional
            Render off-screen and save screenshots when True.
        n : int, optional
            Timestep index appended to saved filenames.
        zoom : float, optional
            Camera zoom factor (default 0.5).
        camera_position : str, sequence, or None, optional
            Camera position preset or coordinates (default None).
        **kwargs : dict
            Extra keyword arguments forwarded to ``pyvista.Plotter.add_mesh``.

        Returns
        -------
        None

        Notes
        -----
        - When running with MPI (`use_mpi=True`), plotting is not supported.
        - The method modifies the PyVista plotter state and may save screenshots if
          `off_screen=True`.
        """
        if self.use_mpi:
            print("[!] plot3D is not supported when `use_mpi=True`")
            return

        if len(field) == 2:  # support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component + "3d"

        if self.plotter_active and not off_screen:
            self.plotter_active = False

        if not self.plotter_active:
            pl = pv.Plotter(off_screen=off_screen, lighting="none")
            light = pv.Light(light_type="headlight")
            pl.add_light(light)

            # Plot stl surface(s)
            if stl_transparent is not None:
                if type(stl_transparent) is str:
                    key = stl_transparent
                    surf = self.grid.read_stl(key)
                    pl.add_mesh(
                        surf,
                        color=stl_colors,
                        opacity=stl_opacity,
                        smooth_shading=True,
                    )

                elif type(stl_transparent) is list:
                    for i, key in enumerate(stl_transparent):
                        surf = self.grid.read_stl(key)
                        if type(stl_colors) is list:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[i],
                                opacity=stl_opacity,
                                smooth_shading=True,
                            )
                        else:
                            pl.add_mesh(
                                surf,
                                color=stl_colors,
                                opacity=stl_opacity,
                                smooth_shading=True,
                            )

            if off_screen:
                self.plotter_active = True
        else:
            pl = self.pl

        # Plot field
        if field == "E":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.E[:, :, :, component], self.N
            )

        elif field == "H":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.H[:, :, :, component], self.N
            )

        elif field == "J":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.J[:, :, :, component], self.N
            )
        else:
            print("`field` value not valid")

        points = self.grid.grid.cell_data_to_point_data()  # interpolate

        # Interpolate fields on stl
        if stl_with_field is not None:
            if type(stl_with_field) is str:
                key = stl_with_field
                surf = self.grid.read_stl(key)
                if clip_plane:
                    try:
                        surf = surf.clip_closed_surface(
                            normal=clip_normal, origin=clip_origin
                        ).subdivide_adaptive(max_edge_len=3 * np.min(self.dz))
                    except Exception as e:
                        surf = surf.clip(
                            normal=clip_normal,
                            origin=clip_origin,
                            invert=False,
                        ).subdivide_adaptive(max_edge_len=3 * np.min(self.dz))
                        if self.verbose > 1:
                            print(f"[!] '{key}' surface non-manifold: {e}")

                fieldonsurf = surf.sample(points, tolerance=tolerance)

                if clip_interactive:  # interactive plotting with a plane
                    ac1 = pl.add_mesh_clip_plane(
                        fieldonsurf,
                        normal=clip_normal,
                        normal_rotation=False,
                        scalars=field + component,
                        opacity=field_opacity,
                        cmap=cmap,
                        clim=clim,
                        log_scale=log_scale,
                        **kwargs,
                    )

                elif clip_box:  # Clip a rectangle of the domain
                    if clip_bounds is None:
                        Lx, Ly = (
                            (self.grid.xmax - self.grid.xmin),
                            (self.grid.ymax - self.grid.ymin),
                        )
                        clip_bounds = [
                            self.grid.xmax - Lx / 2,
                            self.grid.xmax,
                            self.grid.ymax - Ly / 2,
                            self.grid.ymax,
                            self.grid.zmin,
                            self.grid.zmax,
                        ]

                    ac1 = pl.add_mesh(
                        fieldonsurf.clip_box(bounds=clip_bounds),
                        cmap=cmap,
                        clim=clim,
                        scalars=field + component,
                        opacity=field_opacity,
                        log_scale=log_scale,
                        **kwargs,
                    )

                else:
                    ac1 = pl.add_mesh(
                        fieldonsurf,
                        cmap=cmap,
                        clim=clim,
                        scalars=field + component,
                        opacity=field_opacity,
                        log_scale=log_scale,
                        **kwargs,
                    )

            elif type(stl_with_field) is list:
                for i, key in enumerate(stl_with_field):
                    surf = self.grid.read_stl(key)
                    if clip_plane:
                        try:
                            surf = surf.clip_closed_surface(
                                normal=clip_normal, origin=clip_origin
                            ).subdivide_adaptive(max_edge_len=3 * np.min(self.dz))
                        except Exception as e:
                            surf = surf.clip(
                                normal=clip_normal,
                                origin=clip_origin,
                                invert=False,
                            ).subdivide_adaptive(max_edge_len=3 * np.min(self.dz))
                            if self.verbose > 1:
                                print(f"[!] '{key}' surface non-manifold: {e}")

                    fieldonsurf = surf.sample(points, tolerance=tolerance)

                    if clip_interactive:  # interactive plotting with a plane
                        ac1 = pl.add_mesh_clip_plane(
                            fieldonsurf,
                            normal=clip_normal,
                            normal_rotation=False,
                            scalars=field + component,
                            opacity=field_opacity,
                            cmap=cmap,
                            clim=clim,
                            log_scale=log_scale,
                            **kwargs,
                        )
                    elif clip_box:  # Clip a rectangle of the domain
                        if clip_bounds is None:
                            Lx, Ly = (
                                (self.grid.xmax - self.grid.xmin),
                                (self.grid.ymax - self.grid.ymin),
                            )
                            clip_bounds = [
                                self.grid.xmax - Lx / 2,
                                self.grid.xmax,
                                self.grid.ymax - Ly / 2,
                                self.grid.ymax,
                                self.grid.zmin,
                                self.grid.zmax,
                            ]
                        ac1 = pl.add_mesh(
                            fieldonsurf.clip_box(bounds=clip_bounds),
                            cmap=cmap,
                            clim=clim,
                            scalars=field + component,
                            opacity=field_opacity,
                            log_scale=log_scale,
                            **kwargs,
                        )
                    else:
                        ac1 = pl.add_mesh(
                            fieldonsurf,
                            cmap=cmap,
                            clim=clim,
                            scalars=field + component,
                            opacity=field_opacity,
                            log_scale=log_scale,
                            **kwargs,
                        )
        if camera_position is None:
            pl.camera_position = "zx"
            pl.camera.azimuth += 30
            pl.camera.elevation += 30
        else:
            pl.camera_position = camera_position

        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.camera.zoom(zoom)
        pl.add_axes()
        pl.enable_anti_aliasing()
        # pl.enable_3_lights()

        if n is not None:
            pl.add_title(
                field + component + f" field, timestep={n}",
                font="times",
                font_size=12,
            )
            title += "_" + str(n).zfill(6)

        # Save
        if off_screen:
            pl.screenshot(title + ".png")
            pl.remove_actor(ac1)
            self.pl = pl
        else:
            pl.show(full_screen=False, auto_close=False, interactive_update=True)

    def plot2D(
        self,
        field="E",
        component="z",
        plane="ZY",
        pos=0.5,
        norm=None,
        vmin=None,
        vmax=None,
        figsize=[8, 4],
        cmap="jet",
        patch_alpha=0.1,
        patch_reverse=False,
        add_patch=False,
        title=None,
        off_screen=False,
        n=None,
        interpolation="antialiased",
        dpi=100,
        return_handles=False,
        backend="matplotlib",
    ):
        """Built-in 2D plotting of a field slice.

        Renders a 2D cut through the structured grid using either a matplotlib
        (``imshow``) or PyVista (static slice) backend.

        Parameters
        ----------
        field : str, optional
            Field to plot: 'E', 'H', or 'J'. Two-letter values (e.g. 'Ex') are
            accepted and override ``component``. Default 'E'.
        component : str, optional
            Component to plot ('x','y','z','Abs'). Default 'z'.
        plane : str or sequence, optional
            Cut plane specified as 'XY', 'ZY', 'ZX' or a sequence of slices
            and an integer index ``[x,y,z]``. Default 'ZY'.
        pos : float, optional
            Relative position in the normal direction for the cut (0-1).
            Default 0.5 (center).
        norm : str or None, optional
            Color normalization for ``imshow`` ('linear','log','symlog').
            Matplotlib backend only.
        vmin, vmax : scalar, optional
            Color limits for the plot.
        figsize : sequence, optional
            Figure size for the matplotlib figure (default [8,4]).
        cmap : str, optional
            Colormap for the field plot (default 'jet').
        patch_alpha : float, optional
            Alpha value for STL overlays (default 0.1).
        patch_reverse : bool, optional
            If True, reverse the mask for the patch overlay. Matplotlib only.
        add_patch : str or list, optional
            STL key or list of keys to overlay.
            Matplotlib: drawn as a filled mask patch.
            PyVista: drawn as STL contour lines on the slice.
        title : str, optional
            Basename used to save screenshots when ``off_screen=True``.
        off_screen : bool, optional
            Matplotlib: save figure to disk. PyVista: return plotter object.
        n : int, optional
            Timestep index appended to saved filenames. Matplotlib only.
        interpolation : str, optional
            Interpolation method for ``imshow`` (default 'antialiased').
            Matplotlib only.
        dpi : int, optional
            Dots per inch for the figure (default 100). Matplotlib only.
        return_handles : bool, optional
            If True return the ``(fig, ax)`` handles instead of showing.
            Matplotlib only.
        backend : {'matplotlib', 'pyvista'}, optional
            Visualization backend. Default 'matplotlib'.

        Returns
        -------
        None or tuple or pyvista.Plotter
            - ``(fig, ax)`` when ``backend='matplotlib'`` and
              ``return_handles=True`` or ``off_screen=True``.
            - ``pyvista.Plotter`` when ``backend='pyvista'`` and
              ``off_screen=True``.
            - ``None`` otherwise.

        Notes
        -----
        - When running with MPI (`use_mpi=True`), only rank 0 will display or
          save the figure (matplotlib backend only).
        - STL patch overlays are not supported when running under MPI.
        """
        # ------------------------------------------------------------------ #
        # PyVista backend                                                      #
        # ------------------------------------------------------------------ #
        if backend.lower() == "pyvista":
            if self.use_mpi:
                print("[!] plot2D pyvista backend is not supported with `use_mpi=True`")
                return

            if len(field) == 2:
                component = field[1]
                field = field[0]

            pv = __import__("pyvista")

            # Populate the grid cell data with the requested field component
            scalar_key = field + component
            if field == "E":
                self.grid.grid.cell_data[scalar_key] = np.reshape(
                    self.E[:, :, :, component], self.N
                )
            elif field == "H":
                self.grid.grid.cell_data[scalar_key] = np.reshape(
                    self.H[:, :, :, component], self.N
                )
            elif field == "J":
                self.grid.grid.cell_data[scalar_key] = np.reshape(
                    self.J[:, :, :, component], self.N
                )
            else:
                print("`field` value not valid")
                return

            pv_grid = self.grid.grid.cell_data_to_point_data()
            pv_grid.set_active_scalars(scalar_key)

            # Plane → normal mapping (same conventions as matplotlib branch)
            plane_up = plane.upper() if isinstance(plane, str) else ""
            plane_to_normal = {
                "XY": "z",
                "YX": "z",
                "ZY": "x",
                "YZ": "x",
                "ZX": "y",
                "XZ": "y",
            }
            if plane_up not in plane_to_normal:
                raise ValueError(
                    f"plane must be one of 'XY','XZ','YZ','ZX','ZY'; got '{plane}'"
                )
            normal = plane_to_normal[plane_up]

            xlo, xhi = self.grid.xmin, self.grid.xmax
            ylo, yhi = self.grid.ymin, self.grid.ymax
            zlo, zhi = self.grid.zmin, self.grid.zmax

            if normal == "x":
                axis_min, axis_max = xlo, xhi
                _cy, _cz = (ylo + yhi) / 2, (zlo + zhi) / 2
                origin = (axis_min + pos * (axis_max - axis_min), _cy, _cz)
            elif normal == "y":
                axis_min, axis_max = ylo, yhi
                _cx, _cz = (xlo + xhi) / 2, (zlo + zhi) / 2
                origin = (_cx, axis_min + pos * (axis_max - axis_min), _cz)
            else:  # z
                axis_min, axis_max = zlo, zhi
                _cx, _cy = (xlo + xhi) / 2, (ylo + yhi) / 2
                origin = (_cx, _cy, axis_min + pos * (axis_max - axis_min))

            pv.global_theme.allow_empty_mesh = True
            pl = pv.Plotter(off_screen=off_screen)

            # Field slice
            field_slice = pv_grid.slice(normal=normal, origin=origin)
            pl.add_mesh(
                field_slice,
                cmap=cmap,
                clim=[vmin, vmax] if (vmin is not None and vmax is not None) else None,
                name="slice",
            )

            # STL outlines on the slice plane (add_patch)
            if add_patch:
                _stl_colors = getattr(self.grid, "stl_colors", {})
                solids = [add_patch] if isinstance(add_patch, str) else list(add_patch)
                for key in solids:
                    surf = self.grid.read_stl(key)
                    if surf is not None:
                        outline = surf.slice(normal=normal, origin=origin)
                        color = (
                            _stl_colors[key]
                            if isinstance(_stl_colors, dict) and key in _stl_colors
                            else "black"
                        )
                        pl.add_mesh(outline, color=color, name=f"outline_{key}")

            # Camera — enforce correct 2-D view
            _view_fns = {
                "XY": pl.view_xy,
                "YX": pl.view_xy,
                "XZ": pl.view_xz,
                "ZX": pl.view_zx,
                "YZ": pl.view_yz,
                "ZY": pl.view_zy,
            }
            _view_fns.get(plane_up, pl.view_yz)()

            pl.set_background("mistyrose", top="white")
            self._add_logo_widget(pl)
            pl.add_axes()
            pl.enable_anti_aliasing()

            _title = title if title is not None else field + component + "2d"
            if n is not None:
                pl.add_title(
                    field + component + f" field, timestep={n}",
                    font="times",
                    font_size=12,
                )
                _title += "_" + str(n).zfill(6)

            if off_screen:
                pl.screenshot(_title + ".png")
                return None
            elif return_handles:
                return pl
            else:
                pl.show()
                return None

        # ------------------------------------------------------------------ #
        # Matplotlib backend                                                   #
        # ------------------------------------------------------------------ #
        elif backend.lower() == "matplotlib":
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
            xmin, xmax = self.grid.xmin, self.grid.xmax
            ymin, ymax = self.grid.ymin, self.grid.ymax
            zmin, zmax = self.grid.zmin, self.grid.zmax
            _z = self.z

            if self.use_mpi:
                zmin, zmax = self.grid.ZMIN, self.grid.ZMAX
                Nz = self.grid.NZ
                _z = self.grid.Z

            if type(field) is str:
                if len(field) == 2:  # support for e.g. field='Ex'
                    component = field[1]
                    field = field[0]
                elif len(field) == 4:  # support for e.g. field='EAbs'
                    component = field[1:]
                    field = field[0]

            if title is None:
                title = field + component + "2d"

            if type(plane) is not str and len(plane) == 3:
                x, y, z = plane[0], plane[1], plane[2]

                if type(plane[2]) is int:
                    cut = f"(x,y,a) a={round(self.z[z], 3)}"
                    xax, yax = "y", "x"
                    extent = [
                        self.y[y].min(),
                        self.y[y].max(),
                        self.x[x].min(),
                        self.x[x].max(),
                    ]

                if type(plane[0]) is int:
                    cut = f"(a,y,z) a={round(self.x[x], 3)}"
                    xax, yax = "z", "y"
                    extent = [
                        _z[z].min(),
                        _z[z].max(),
                        self.y[y].min(),
                        self.y[y].max(),
                    ]

                if type(plane[1]) is int:
                    cut = f"(x,a,z) a={round(self.y[y], 3)}"
                    xax, yax = "z", "x"
                    extent = [
                        _z[z].min(),
                        _z[z].max(),
                        self.x[x].min(),
                        self.x[x].max(),
                    ]

            elif plane == "XY":
                x, y, z = slice(0, Nx), slice(0, Ny), int(Nz * pos)  # plane XY
                cut = f"(x,y,a) a={round(pos * (zmax - zmin) + zmin, 3)}"
                xax, yax = "y", "x"
                extent = [ymin, ymax, xmin, xmax]

            elif plane == "ZY" or plane == "YZ":
                x, y, z = int(Nx * pos), slice(0, Ny), slice(0, Nz)  # plane ZY
                cut = f"(a,y,z) a={round(pos * (xmax - xmin) + xmin, 3)}"
                xax, yax = "z", "y"
                extent = [zmin, zmax, ymin, ymax]

            elif plane == "ZX" or plane == "XZ":
                x, y, z = slice(0, Nx), int(Ny * pos), slice(0, Nz)  # plane XZ
                cut = f"(x,a,z) a={round(pos * (ymax - ymin) + ymin, 3)}"
                xax, yax = "z", "x"
                extent = [zmin, zmax, xmin, xmax]

            else:
                print(
                    "Plane needs to be an array of slices [x,y,z] or a str 'XY', 'ZY', 'ZX'"
                )

            if self.use_mpi:  # only in rank=0
                _field = self.mpi_gather(field, x=x, y=y, z=z, component=component)

                if self.rank == 0:
                    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
                    im = ax.imshow(
                        _field,
                        cmap=cmap,
                        norm=norm,
                        extent=extent,
                        origin="lower",
                        vmin=vmin,
                        vmax=vmax,
                        interpolation=interpolation,
                    )

                    fig.colorbar(
                        im,
                        cax=make_axes_locatable(ax).append_axes(
                            "right", size="5%", pad=0.05
                        ),
                    )
                    ax.set_title(f"Wakis {field}{component}{cut}")
                    ax.set_xlabel(xax)
                    ax.set_ylabel(yax)

                    if n is not None:
                        fig.suptitle(
                            "$"
                            + str(field)
                            + "_{"
                            + str(component)
                            + "}$ field, timestep="
                            + str(n)
                        )
                        title += "_" + str(n).zfill(6)

                    fig.tight_layout()

                    if off_screen:
                        fig.savefig(title + ".png")
                        plt.clf()
                        plt.close(fig)
                    elif return_handles:
                        return fig, ax
                    else:
                        plt.show(block=False)
            else:
                fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
                if field == "E":
                    _field = self.E[x, y, z, component]
                if field == "H":
                    _field = self.H[x, y, z, component]
                if field == "J":
                    _field = self.J[x, y, z, component]

                im = ax.imshow(
                    _field,
                    cmap=cmap,
                    norm=norm,
                    extent=extent,
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation=interpolation,
                )

                fig.colorbar(
                    im,
                    cax=make_axes_locatable(ax).append_axes(
                        "right", size="5%", pad=0.05
                    ),
                )
                ax.set_title(f"Wakis {field}{component}{cut}")
                ax.set_xlabel(xax)
                ax.set_ylabel(yax)

                if add_patch:
                    # All solids are merged into a single combined mask.
                    # Two-pass approach:
                    #   Pass 1: OR in all non-vacuum solids to build the base mask.
                    #   Pass 2: AND-NOT all vacuum solids [eps=1, mu=1, sigma=0]
                    #           to carve out vacuum regions from the result.
                    # A single imshow overlay is produced for the combined mask.
                    solids = (
                        [add_patch] if isinstance(add_patch, str) else list(add_patch)
                    )
                    combined = np.zeros((Nx, Ny, Nz), dtype=bool)
                    vacuum_masks = []
                    for solid in solids:
                        solid_mask = np.reshape(
                            self.grid.grid[solid], (Nx, Ny, Nz)
                        ).astype(bool)
                        mat = self.grid.stl_materials.get(solid, [0.0, 0.0, 0.0])
                        is_vacuum = (
                            len(mat) >= 3
                            and mat[0] == 1.0
                            and mat[1] == 1.0
                            and mat[2] == 0.0
                        )
                        if is_vacuum:
                            vacuum_masks.append(solid_mask)
                        else:
                            combined |= solid_mask  # pass 1: add non-vacuum regions
                    for solid_mask in vacuum_masks:
                        combined &= ~solid_mask  # pass 2: subtract vacuum regions

                    patch = np.ones((Nx, Ny, Nz))
                    if patch_reverse:
                        patch[combined] = np.nan
                    else:
                        patch[~combined] = np.nan
                    ax.imshow(
                        patch[x, y, z],
                        cmap="Greys",
                        extent=extent,
                        origin="lower",
                        alpha=patch_alpha,
                    )

                if n is not None:
                    fig.suptitle(
                        "$"
                        + str(field)
                        + "_{"
                        + str(component)
                        + "}$ field, timestep="
                        + str(n)
                    )
                    title += "_" + str(n).zfill(6)

                fig.tight_layout()

                if off_screen:
                    fig.savefig(title + ".png")
                    plt.clf()
                    plt.close(fig)
                elif return_handles:
                    return fig, ax
                else:
                    plt.show(block=False)

        else:
            raise ValueError(
                "Invalid plotting backend specified. Choose 'matplotlib' or 'pyvista'."
            )

    def plot1D(
        self,
        field="E",
        component="z",
        line="z",
        pos=[0.5],
        xscale="linear",
        yscale="linear",
        xlim=None,
        ylim=None,
        figsize=[8, 4],
        title=None,
        off_screen=False,
        n=None,
        colors=None,
        **kwargs,
    ):
        """Built-in 1D plotting of a field line using matplotlib.

        Plots 1D cuts along a grid line (x, y or z) for the selected field
        component. Supports plotting multiple positions and MPI gathering when
        running in parallel.

        Parameters
        ----------
        field : str, optional
            Field to plot: 'E', 'H', or 'J'. Two-letter values (e.g. 'Ex') are
            accepted and override ``component``. Default 'E'.
        component : str, optional
            Component to plot ('x','y','z','Abs'). Default 'z'.
        line : str or sequence, optional
            Line specification as 'x','y','z' or a sequence of three indices
            or slices (e.g. [0, slice(10,Ny-10), 0]). Default 'z'.
        pos : float or sequence, optional
            Relative position(s) (0-1) used when ``line`` is a single-axis
            specifier. Default 0.5.
        xscale, yscale : str, optional
            Axis scaling passed to matplotlib (default 'linear').
        xlim, ylim : sequence, optional
            Axis limits for the plot.
        figsize : sequence, optional
            Figure size for the matplotlib figure (default [8,4]).
        title : str, optional
            Basename used to save screenshots when ``off_screen=True``.
        off_screen : bool, optional
            If True save the figure to disk instead of showing it interactively.
        n : int, optional
            Timestep index appended to saved filenames.
        colors : sequence, optional
            Colors used for each plotted line when multiple positions are
            requested.
        **kwargs : dict
            Extra keyword arguments forwarded to ``matplotlib.axes.Axes.plot``.

        Returns
        -------
        None

        Notes
        -----
        - When running with MPI (`use_mpi=True`), only rank 0 will display or save the
          figure.
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        xmin, xmax = self.grid.xmin, self.grid.xmax
        ymin, ymax = self.grid.ymin, self.grid.ymax
        zmin, zmax = self.grid.zmin, self.grid.zmax
        _z = self.z

        if self.use_mpi:
            zmin, zmax = self.grid.ZMIN, self.grid.ZMAX
            Nz = self.grid.NZ
            _z = self.grid.Z
            if self.rank == 0:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plotkw = {"lw": 1.2, "ls": "-"}

        if colors is None:
            colors = [
                "k",
                "#5ccfe6",
                "#bae67e",
                "#ffae57",
                "#fdb6d0",
                "#ffd580",
                "#a2aabc",
            ]
        plotkw.update(kwargs)

        if len(field) == 2:  # support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component + "1d"

        if type(pos) is not list:  # support for a list of cut positions
            pos_arr = [pos]
        else:
            pos_arr = pos

        for i, pos in enumerate(pos_arr):
            if type(line) is not str and len(line) == 3:
                x, y, z = line[0], line[1], line[2]

                # z-axis
                if type(line[2]) is slice:
                    cut = f"(a,b,z) a={round(self.x[x], 3)}, b={round(self.y[y], 3)}"
                    xax = "z"
                    xx = _z[z]
                    xlims = (_z[z].min(), _z[z].max())

                # x-axis
                elif type(line[0]) is slice:
                    cut = f"(x,a,b) a={round(self.y[y], 3)}, b={round(_z[z], 3)}"
                    xax = "x"
                    xx = self.x[x]
                    xlims = (self.x[x].min(), self.x[x].max())

                # y-axis
                elif type(line[1]) is slice:
                    cut = f"(a,y,b) a={round(self.x[x], 3)}, b={round(_z[z], 3)}"
                    xax = "y"
                    xx = self.y[y]
                    xlims = (self.y[y].min(), self.y[y].max())

            elif line.lower() == "x":
                x, y, z = slice(0, Nx), int(Ny * pos), int(Nz * pos)  # x-axis
                cut = f"(x,a,b) a={round(self.y[y], 3)}, b={round(_z[z], 3)}"
                xax = "x"
                xx = self.x[x]
                xlims = (xmin, xmax)

            elif line.lower() == "y":
                x, y, z = int(Nx * pos), slice(0, Ny), int(Nz * pos)  # y-axis
                cut = f"(a,y,b) a={round(self.x[x], 3)}, b={round(_z[z], 3)}"
                xax = "y"
                xx = self.y[y]
                xlims = (ymin, ymax)

            elif line.lower() == "z":
                x, y, z = int(Nx * pos), int(Ny * pos), slice(0, Nz)  # z-axis
                cut = f"(a,b,z) a={round(self.x[x], 3)}, b={round(self.y[y], 3)}"
                xax = "z"
                xx = _z[z]
                xlims = (zmin, zmax)

            else:
                print(
                    "line needs to be an array of slices [x,y,z] or a str 'x', 'y', 'z'"
                )

            if i == 0:  # first one on top
                zorder = 10
            else:
                zorder = i

            if self.use_mpi:  # only in rank=0
                _field = self.mpi_gather(field, x=x, y=y, z=z, component=component)

                if self.rank == 0:
                    ax.plot(
                        xx,
                        _field,
                        color=colors[i],
                        zorder=zorder,
                        label=f"{field}{component}{cut}",
                        **plotkw,
                    )
            else:
                if field == "E":
                    _field = self.E[x, y, z, component]
                if field == "H":
                    _field = self.H[x, y, z, component]
                if field == "J":
                    _field = self.J[x, y, z, component]

                ax.plot(
                    xx,
                    _field,
                    color=colors[i],
                    zorder=zorder,
                    label=f"{field}{component}{cut}",
                    **plotkw,
                )

        if not self.use_mpi:
            yax = f"{field}{component} amplitude"

            ax.set_title(f"Wakis {field}{component}" + (len(pos_arr) == 1) * f"{cut}")
            ax.set_xlabel(xax)
            ax.set_ylabel(yax, color=colors[0])
            ax.set_xlim(xlims)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            if len(pos_arr) > 1:
                ax.legend(loc=1)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if n is not None:
                fig.suptitle(
                    "$" + field + "_{" + component + "}$ field, timestep=" + str(n)
                )
                title += "_" + str(n).zfill(6)

            fig.tight_layout()

            if off_screen:
                fig.savefig(title + ".png")
                plt.clf()
                plt.close(fig)

            else:
                plt.show()

        elif self.use_mpi and self.rank == 0:
            yax = f"{field}{component} amplitude"

            ax.set_title(f"Wakis {field}{component}" + (len(pos_arr) == 1) * f"{cut}")
            ax.set_xlabel(xax)
            ax.set_ylabel(yax, color=colors[0])
            ax.set_xlim(xlims)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            if len(pos_arr) > 1:
                ax.legend(loc=1)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if n is not None:
                fig.suptitle(
                    "$" + field + "_{" + component + "}$ field, timestep=" + str(n)
                )
                title += "_" + str(n).zfill(6)

            fig.tight_layout()

            if off_screen:
                fig.savefig(title + ".png")
                plt.clf()
                plt.close(fig)

            else:
                plt.show()

    def inspect(
        self,
        wake=None,
        window_size=None,
        off_screen=False,
        opacity=1.0,
        inactive_opacity=0.1,
        add_silhouette=False,
        specular=0.5,
        smooth_shading=False,
    ):
        """Interactive inspection of STL solids and optional wake geometry.

        Provides a PyVista-based UI to toggle visibility and highlighting for
        the package STL solids. When a ``wake`` object is provided the method
        also creates a beam and an integration-path actor and exposes check
        boxes to toggle them. A master checkbox toggles all solids at once.

        Parameters
        ----------
        wake : object or None, optional
            Optional wake-like object (for example a ``WakeSolver`` instance)
            that provides ``xsource``, ``ysource``, ``xtest`` and ``ytest``
            attributes used to place a beam and an integration path. When
            provided the object is stored as ``self.wake``. Default ``None``.
        window_size : sequence or None, optional
            Tuple ``(width, height)`` passed to ``pyvista.Plotter`` to set
            the window size. Default ``None`` (use PyVista defaults).
        off_screen : bool, optional
            If True the plotter is not shown interactively and the method
            returns the created ``pyvista.Plotter`` instance for programmatic
            control and screenshot capture. Default ``False``.
        opacity : float, optional
            Opacity applied to solids when they are active/highlighted
            (default 1.0).
        inactive_opacity : float, optional
            Opacity applied to solids when inactive (default 0.1).
        add_silhouette : bool, optional
            When True add a silhouette actor behind each solid for emphasis.
        specular : float, optional
            Specular lighting value forwarded to ``pyvista.Plotter.add_mesh``
            for STL actors (default 0.5).
        smooth_shading : bool, optional
            Enable smooth shading for STL meshes (default False).

        Returns
        -------
        pyvista.Plotter or None
            Returns the created ``pyvista.Plotter`` when ``off_screen`` is
            True so the caller can save screenshots or control rendering
            programmatically. Otherwise shows the interactive window and
            returns ``None``.

        Notes
        -----
        - The method expects ``self.stl_solids`` and ``self.stl_colors`` to
          be populated and uses ``self.grid.read_stl(key)`` to load surfaces.
        - When running under MPI the method prints a warning and returns
          immediately (interactive plotting is not supported under MPI).
        """
        if self.use_mpi:
            print("[!] plot3D is not supported when `use_mpi=True`")
            return
        if wake is not None:
            self.wake = wake

        # Initialize plotter
        pl = pv.Plotter(window_size=window_size, off_screen=off_screen)
        solid_state = {}
        for key, path in self.stl_solids.items():
            surf = self.grid.read_stl(key)
            color = self.stl_colors.get(key, "lightgray")
            actor = pl.add_mesh(
                surf,
                color=color,
                name=key,
                opacity=inactive_opacity,
                silhouette=False,
                smooth_shading=smooth_shading,
                specular=specular,
            )
            sil = None
            if add_silhouette:
                sil = pl.add_silhouette(surf, color="black", line_width=3.0)
                sil.SetVisibility(False)

            solid_state[key] = {
                "actor": actor,
                "silhouette": sil,
                "active_opacity": opacity,
                "inactive_opacity": inactive_opacity,
                "checked": False,
                "highlight": False,
                "button": None,
            }

        # UI scale and solid checkboxes positioning
        w, h = pl.window_size
        ui = h / 800.0
        box = max(16, int(20 * ui))
        font = max(8, int(12 * ui))
        pad = int(10 * ui)
        dy = box + pad
        cx = int(10 * ui)
        cy = h // 2

        # checkboxes callbacks for solids and master (all On/Off)
        color_on = "green"
        color_off = "white"

        def apply(state):
            if state["highlight"]:
                state["actor"].GetProperty().SetOpacity(state["active_opacity"])
                if add_silhouette:
                    state["silhouette"].SetVisibility(True)
            else:
                state["actor"].GetProperty().SetOpacity(state["inactive_opacity"])
                if add_silhouette:
                    state["silhouette"].SetVisibility(False)

        def make_cb(name):
            def _cb(v):
                s = solid_state[name]
                s["checked"] = bool(v)
                s["highlight"] = s["checked"]
                apply(s)

            return _cb

        master_on = True

        def master_cb(v):
            nonlocal master_on
            master_on = bool(v)
            for s in solid_state.values():
                s["checked"] = master_on
                s["highlight"] = master_on
                btn = s["button"]
                if btn:
                    rep = btn.GetRepresentation()
                    if hasattr(rep, "SetState"):
                        rep.SetState(1 if master_on else 0)
                apply(s)

        pl.add_checkbox_button_widget(
            master_cb,
            value=False,
            color_on=color_on,
            color_off=color_off,
            position=(cx, cy),
            size=box,
        )
        pl.add_text("All on", position=(cx + box + pad, cy), font_size=font)

        for i, name in enumerate(solid_state):
            y = cy - (i + 2) * dy
            _color_on = self.stl_colors.get(name, color_on)
            if str(_color_on) == "white" or list(_color_on) == [1.0, 1.0, 1.0]:
                _color_on = "gray"
            btn = pl.add_checkbox_button_widget(
                make_cb(name),
                value=False,
                color_on=_color_on,
                color_off=color_off,
                position=(cx, y),
                size=box,
            )
            solid_state[name]["button"] = btn
            pl.add_text(name, position=(cx + box + pad, y), font_size=font)

        # Add beam & integration path checkboxes if wake object is passed
        if self.wake is not None:
            z_center = 0.5 * (self.grid.zmin + self.grid.zmax)
            z_height = self.grid.zmax - self.grid.zmin
            radius = 0.005 * max(
                self.grid.xmax - self.grid.xmin,
                self.grid.ymax - self.grid.ymin,
            )

            beam = pv.Cylinder(
                center=(self.wake.xsource, self.wake.ysource, z_center),
                direction=(0, 0, 1),
                height=z_height,
                radius=radius * 1.1,
            )
            path = pv.Cylinder(
                center=(self.wake.xtest, self.wake.ytest, z_center),
                direction=(0, 0, 1),
                height=z_height,
                radius=radius,
            )

            beam_actor = pl.add_mesh(beam, color="orange", name="beam", opacity=1.0)
            beam_actor.SetVisibility(True)
            path_actor = pl.add_mesh(
                path, color="blue", name="integration_path", opacity=1.0
            )
            path_actor.SetVisibility(True)

            bx = int(w - box - 200 * ui)
            to = box + pad

            def beam_cb(v):
                beam_actor.SetVisibility(bool(v))

            def path_cb(v):
                path_actor.SetVisibility(bool(v))

            pl.add_checkbox_button_widget(
                path_cb,
                value=True,
                color_off=color_off,
                color_on="blue",
                position=(bx, cy + dy),
                size=box,
            )
            pl.add_text("Integration path", position=(bx + to, cy + dy), font_size=font)

            pl.add_checkbox_button_widget(
                beam_cb,
                value=True,
                color_off=color_off,
                color_on="orange",
                position=(bx, cy),
                size=box,
            )
            pl.add_text("Beam", position=(bx + to, cy), font_size=font)

        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.add_axes()

        # Save
        if off_screen:
            return pl
        else:
            pl.show(auto_close=False, interactive_update=True)
            return None

    def _add_logo_widget(self, pl):
        """Add packaged logo via importlib.resources (Python 3.9+)."""
        try:
            from importlib import resources

            # resource inside the installed package (use current package)
            logo_res = resources.files(__package__).joinpath(
                "static", "img", "wakis-logo-pink.png"
            )
            with resources.as_file(logo_res) as logo_path:
                pl.add_logo_widget(str(logo_path))
                return
        except Exception as e:
            # fallback to the legacy relative path for dev installs
            try:
                pl.add_logo_widget("../docs/img/wakis-logo-pink.png")
            except Exception:
                if self.verbose > 1:
                    print(f"[!] Could not add logo widget: {e}")

    @staticmethod
    def _plotting_plane_axes(normal):
        """
        Return the two coordinate directions lying in a slicing plane.

        normal='x' -> y-z plane
        normal='y' -> x-z plane
        normal='z' -> x-y plane
        """
        normal = normal.lower()

        if normal == "x":
            return (1, 2), ("y", "z")

        if normal == "y":
            return (0, 2), ("x", "z")

        if normal == "z":
            return (0, 1), ("x", "y")

        raise ValueError(f"Unknown plane normal '{normal}'. Expected 'x', 'y' or 'z'.")

    @staticmethod
    def _polydata_lines_to_segments(poly):
        """
        Convert PyVista PolyData lines into a list of point arrays.

        Each returned array contains the points of one polyline.
        """
        if poly is None or poly.n_points == 0 or poly.n_cells == 0:
            return []

        points = np.asarray(poly.points)
        lines = np.asarray(poly.lines)

        segments = []

        offset = 0

        while offset < len(lines):
            n_points = int(lines[offset])

            point_ids = lines[offset + 1 : offset + 1 + n_points]

            segments.append(points[point_ids])

            offset += n_points + 1

        return segments

    @staticmethod
    def _point_on_edge(
        p0,
        p1,
        alpha,
    ):
        """
        Return the point

            p(alpha) = p0 + alpha * (p1 - p0)

        on an edge.
        """
        return p0 + alpha * (p1 - p0)

    def _conformal_region_colors(self):
        """
        Build plotting colors for all conformal material regions.

        STL colors explicitly supplied through ``grid.stl_colors`` are
        preserved. All remaining STL regions receive automatically assigned
        colors from Matplotlib's tab20 colormap.

        Region 0 is the background and receives a neutral color.
        """
        grid = self.grid

        region_colors = {
            0: "0.55",
        }

        stl_colors = getattr(
            grid,
            "stl_colors",
            {},
        )

        if stl_colors is None:
            stl_colors = {}

        cmap = plt.get_cmap("tab20")

        color_index = 0

        for key in grid.stl_solids.keys():
            region_id = int(grid.region_key_to_id[key])

            requested_color = stl_colors.get(
                key,
                None,
            )

            if requested_color is not None:
                region_colors[region_id] = requested_color

            else:
                region_colors[region_id] = cmap(color_index % cmap.N)

                color_index += 1

        # Fallback for any additional region IDs which are not
        # represented by an STL key.
        for region_id in self.region_material_types:
            region_id = int(region_id)

            if region_id in region_colors:
                continue

            region_colors[region_id] = cmap(color_index % cmap.N)

            color_index += 1

        return region_colors

    def _conformal_region_label(
        self,
        region_id,
    ):
        """
        Return a human-readable label for a conformal material region.
        """
        region_id = int(region_id)

        key = self.grid.region_id_to_key.get(
            region_id,
            None,
        )

        material_type = self.region_material_types.get(
            region_id,
            "unknown",
        )

        if key is None:
            name = "background"

        else:
            name = str(key)

        if material_type == "normal":
            return name

        return f"{name} ({material_type.upper()})"

    def _conformal_endpoint_alpha(
        self,
        direction,
        edge_index,
        tol=1e-8,
    ):
        """
        Detect whether an interface lies exactly on one endpoint of a
        primal edge.

        Geometric hits located exactly on a grid node are assigned by
        GridFIT3D to one unique edge. Because _locate_intersection_edge()
        uses searchsorted(..., side="right"), an interface at the end of
        the current edge can therefore appear as alpha=0 on the following
        edge.

        Returns
        -------
        0.0
            Interface lies at the start point of the current edge.

        1.0
            Interface lies at the end point of the current edge.

        None
            No endpoint interface could be identified unambiguously.
        """
        grid = self.grid

        intersection_data = grid.interface_intersections["primal"][direction]

        hit_groups = intersection_data.get(
            "hit_groups",
            {},
        )

        def group_alphas(index):
            groups = hit_groups.get(
                index,
                [],
            )

            alphas = []

            for group in groups:
                if not group:
                    continue

                alpha = float(np.mean([hit["alpha"] for hit in group]))

                if np.isfinite(alpha):
                    alphas.append(alpha)

            return alphas

        # ------------------------------------------------------
        # 1) Check the current edge itself.
        # ------------------------------------------------------
        current_alphas = group_alphas(edge_index)

        if any(alpha <= tol for alpha in current_alphas):
            return 0.0

        if any(alpha >= 1.0 - tol for alpha in current_alphas):
            return 1.0

        # ------------------------------------------------------
        # 2) A hit at the end point of the current edge is often
        #    stored as alpha=0 on the following edge.
        # ------------------------------------------------------
        i, j, k = edge_index

        if direction == "x":
            next_index = (
                i + 1,
                j,
                k,
            )

            valid_next = i + 1 < grid.Nx

        elif direction == "y":
            next_index = (
                i,
                j + 1,
                k,
            )

            valid_next = j + 1 < grid.Ny

        elif direction == "z":
            next_index = (
                i,
                j,
                k + 1,
            )

            valid_next = k + 1 < grid.Nz

        else:
            raise ValueError(f"Unknown edge direction '{direction}'.")

        if valid_next:
            next_alphas = group_alphas(next_index)

            if any(alpha <= tol for alpha in next_alphas):
                return 1.0

        return None

    def _conformal_edge_display_segments(
        self,
        direction,
        line_data,
        edge_index,
        p0,
        p1,
        tol=1e-8,
    ):
        """
        Return the active drawable parts of one primal conformal edge.

        Each returned entry is

            (q0, q1, region_id)

        where q0 and q1 are the endpoints of one active edge segment.

        Plotting rules
        --------------
        Edge entirely inside one normal region:
            Draw the complete edge.

        Edge entirely inside PEC/SIBC:
            Draw nothing.

        normal <-> normal with 0 < alpha < 1:
            Split the edge at alpha and draw both material portions.

        normal <-> PEC/SIBC:
            Draw only the normal-material portion.

        alpha ~= 0:
            The interface lies at the start node. Apart from that single
            point, the complete edge belongs to the end region.

        alpha ~= 1:
            The interface lies at the end node. Apart from that single
            point, the complete edge belongs to the start region.

        Unresolved transitions:
            Endpoint contacts are recovered from the geometric
            contact-group information. Other unresolved cases are omitted.
        """
        start_region = int(line_data["start_region"][edge_index])

        end_region = int(line_data["end_region"][edge_index])

        start_type = self.region_material_types[start_region]

        end_type = self.region_material_types[end_region]

        def full_normal_edge(region_id, material_type):
            if material_type == "normal":
                return [
                    (
                        p0,
                        p1,
                        region_id,
                    )
                ]

            return []

        # ----------------------------------------------------------
        # No material transition on this edge.
        # ----------------------------------------------------------
        if start_region == end_region:
            return full_normal_edge(
                start_region,
                start_type,
            )

        # ----------------------------------------------------------
        # First try the representative solver intersection.
        # ----------------------------------------------------------
        resolved = bool(line_data["resolved"][edge_index])

        alpha = line_data["alpha"][edge_index]

        if resolved and np.isfinite(alpha):
            alpha = float(
                np.clip(
                    alpha,
                    0.0,
                    1.0,
                )
            )

            # ------------------------------------------------------
            # Interface exactly at the start node:
            #
            #     interface
            #         |
            #         x-------------------x
            #        p0                  p1
            #
            # The finite-length edge belongs to the end region.
            # ------------------------------------------------------
            if alpha <= tol:
                return full_normal_edge(
                    end_region,
                    end_type,
                )

            # ------------------------------------------------------
            # Interface exactly at the end node:
            #
            #                             interface
            #                                 |
            #         x-----------------------x
            #        p0                      p1
            #
            # The finite-length edge belongs to the start region.
            # ------------------------------------------------------
            if alpha >= 1.0 - tol:
                return full_normal_edge(
                    start_region,
                    start_type,
                )

            # ------------------------------------------------------
            # Genuine interior interface.
            # ------------------------------------------------------
            interface_point = self._point_on_edge(
                p0,
                p1,
                alpha,
            )

            segments = []

            if start_type == "normal":
                segments.append(
                    (
                        p0,
                        interface_point,
                        start_region,
                    )
                )

            if end_type == "normal":
                segments.append(
                    (
                        interface_point,
                        p1,
                        end_region,
                    )
                )

            return segments

        # ----------------------------------------------------------
        # No resolved crossing.
        #
        # A common valid case is that the material transition lies
        # exactly on one of the edge endpoints. These are stored by
        # the geometry pipeline as contact groups rather than ordinary
        # crossing groups.
        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # No resolved interior crossing.
        #
        # The transition may lie exactly on a grid node. Such a hit
        # can be stored on the neighbouring edge rather than this one.
        # ----------------------------------------------------------
        endpoint_alpha = self._conformal_endpoint_alpha(
            direction=direction,
            edge_index=edge_index,
            tol=tol,
        )

        # Interface at p0:
        #
        #     interface
        #         |
        #         x-------------------x
        #        p0                  p1
        #
        # The complete finite-length edge belongs to the end region.
        if endpoint_alpha == 0.0:
            return full_normal_edge(
                end_region,
                end_type,
            )

        # Interface at p1:
        #
        #                             interface
        #                                 |
        #         x-----------------------x
        #        p0                      p1
        #
        # The complete finite-length edge belongs to the start region.
        if endpoint_alpha == 1.0:
            return full_normal_edge(
                start_region,
                start_type,
            )

        # Truly unresolved case.
        return []

    def plot_conformal_primal_edge_slice(
        self,
        normal="z",
        index=0,
        ax=None,
        figsize=(7, 7),
        show_full_grid=False,
        show_stl=True,
        show_intersection_marker=True,
        show_legend=True,
        full_grid_color="0.85",
        full_grid_lw=0.6,
        edge_lw=2.0,
        stl_lw=1.0,
        marker_size=14,
        tol=1e-8,
    ):
        """
        Plot the active primal conformal edge geometry in a 2D slice.

        The plot represents the material regions actually used by the
        conformal solver.

        Normal-material edge portions are drawn in the color assigned to
        their material region. PEC and SIBC edge portions are omitted.

        At normal-normal interfaces the edge is split at the geometric
        intersection position alpha and both portions are drawn in their
        corresponding region colors.

        At normal-conductor interfaces only the normal-material portion
        is drawn.

        Parameters
        ----------
        normal : {'x', 'y', 'z'}
            Normal direction of the slicing plane.

        index : int
            Index of the primal grid plane.

        ax : matplotlib.axes.Axes, optional
            Existing Matplotlib axes.

        figsize : tuple
            Figure size used when no axes are supplied.

        show_full_grid : bool
            Draw the complete Cartesian grid as a faint reference mesh.

            This reference grid is purely geometric and therefore also
            passes through PEC/SIBC regions. Set False when visualizing
            only solver-active edge portions.

        show_stl : bool
            Draw the exact STL intersection contours with the slicing
            plane.

        show_intersection_marker : bool
            Mark resolved interface intersections that lie strictly
            inside an edge.

        show_legend : bool
            Show the region-color legend.

        full_grid_color : matplotlib color
            Color of the optional reference grid.

        full_grid_lw : float
            Line width of the optional reference grid.

        edge_lw : float
            Line width of active conformal edge segments.

        stl_lw : float
            Line width of the STL intersection contours.

        marker_size : float
            Size of interface intersection markers.

        tol : float
            Numerical tolerance used to suppress zero-length segments and
            markers located directly on grid nodes.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the conformal slice plot.
        """
        if not hasattr(
            self,
            "conformal_line_data",
        ):
            raise RuntimeError("No conformal line data available.")

        if not self.conformal_line_data.get(
            "primal",
            {},
        ):
            raise RuntimeError("No primal conformal line data available.")

        if not getattr(
            self,
            "region_material_types",
            None,
        ):
            raise RuntimeError("No conformal region material types available.")

        grid = self.grid

        # Use the actual primal grid coordinates rather than rebuilding
        # them from the domain bounds.
        x = np.asarray(
            grid.x,
            dtype=float,
        )

        y = np.asarray(
            grid.y,
            dtype=float,
        )

        z = np.asarray(
            grid.z,
            dtype=float,
        )

        normal = normal.lower()

        center = np.array(
            [
                0.5 * (x[0] + x[-1]),
                0.5 * (y[0] + y[-1]),
                0.5 * (z[0] + z[-1]),
            ],
            dtype=float,
        )

        if normal == "x":
            if index < 0 or index >= len(x):
                raise IndexError("Slice index out of bounds for x-slice.")

            in_plane_dirs = (
                "y",
                "z",
            )

            plane_origin = center.copy()

            plane_origin[0] = x[index]

            plane_coordinate = x[index]

        elif normal == "y":
            if index < 0 or index >= len(y):
                raise IndexError("Slice index out of bounds for y-slice.")

            in_plane_dirs = (
                "x",
                "z",
            )

            plane_origin = center.copy()

            plane_origin[1] = y[index]

            plane_coordinate = y[index]

        elif normal == "z":
            if index < 0 or index >= len(z):
                raise IndexError("Slice index out of bounds for z-slice.")

            in_plane_dirs = (
                "x",
                "y",
            )

            plane_origin = center.copy()

            plane_origin[2] = z[index]

            plane_coordinate = z[index]

        else:
            raise ValueError(
                f"Unknown plane normal '{normal}'. Expected 'x', 'y' or 'z'."
            )

        (
            plane_axes,
            axis_labels,
        ) = self._plotting_plane_axes(normal)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        region_colors = self._conformal_region_colors()

        drawn_region_ids = set()

        # ======================================================
        # Draw one primal edge.
        # ======================================================
        def plot_edge(
            direction,
            edge_index,
            p0,
            p1,
        ):
            p0_2d = p0[list(plane_axes)]

            p1_2d = p1[list(plane_axes)]

            # Optional reference Cartesian grid.
            if show_full_grid:
                ax.plot(
                    [
                        p0_2d[0],
                        p1_2d[0],
                    ],
                    [
                        p0_2d[1],
                        p1_2d[1],
                    ],
                    color=full_grid_color,
                    lw=full_grid_lw,
                    zorder=1,
                )

            line_data = self.conformal_line_data["primal"][direction]

            segments = self._conformal_edge_display_segments(
                direction=direction,
                line_data=line_data,
                edge_index=edge_index,
                p0=p0,
                p1=p1,
                tol=tol,
            )

            for (
                q0,
                q1,
                region_id,
            ) in segments:
                q0_2d = q0[list(plane_axes)]

                q1_2d = q1[list(plane_axes)]

                ax.plot(
                    [
                        q0_2d[0],
                        q1_2d[0],
                    ],
                    [
                        q0_2d[1],
                        q1_2d[1],
                    ],
                    color=region_colors[region_id],
                    lw=edge_lw,
                    zorder=3,
                )

                drawn_region_ids.add(int(region_id))

            # ----------------------------------------------
            # Optional marker for a resolved intersection.
            #
            # Mark only intersections lying strictly inside
            # an edge. Intersections exactly on a grid node
            # are intentionally suppressed.
            # ----------------------------------------------
            if not show_intersection_marker:
                return

            start_region = int(line_data["start_region"][edge_index])

            end_region = int(line_data["end_region"][edge_index])

            if start_region == end_region:
                return

            if not bool(line_data["resolved"][edge_index]):
                return

            alpha = float(line_data["alpha"][edge_index])

            if not np.isfinite(alpha):
                return

            if not (tol < alpha < 1.0 - tol):
                return

            interface_point = self._point_on_edge(
                p0,
                p1,
                alpha,
            )

            interface_2d = interface_point[list(plane_axes)]

            ax.scatter(
                interface_2d[0],
                interface_2d[1],
                s=marker_size,
                facecolors="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )

        # ======================================================
        # Draw x-directed primal edges.
        #
        # Shape:
        #     (Nx, Ny + 1, Nz + 1)
        # ======================================================
        if "x" in in_plane_dirs:
            for i in range(grid.Nx):
                for j in range(grid.Ny + 1):
                    for k in range(grid.Nz + 1):
                        if normal == "y" and j != index:
                            continue

                        if normal == "z" and k != index:
                            continue

                        p0 = np.array(
                            [
                                x[i],
                                y[j],
                                z[k],
                            ],
                            dtype=float,
                        )

                        p1 = np.array(
                            [
                                x[i + 1],
                                y[j],
                                z[k],
                            ],
                            dtype=float,
                        )

                        plot_edge(
                            "x",
                            (i, j, k),
                            p0,
                            p1,
                        )

        # ======================================================
        # Draw y-directed primal edges.
        #
        # Shape:
        #     (Nx + 1, Ny, Nz + 1)
        # ======================================================
        if "y" in in_plane_dirs:
            for i in range(grid.Nx + 1):
                for j in range(grid.Ny):
                    for k in range(grid.Nz + 1):
                        if normal == "x" and i != index:
                            continue

                        if normal == "z" and k != index:
                            continue

                        p0 = np.array(
                            [
                                x[i],
                                y[j],
                                z[k],
                            ],
                            dtype=float,
                        )

                        p1 = np.array(
                            [
                                x[i],
                                y[j + 1],
                                z[k],
                            ],
                            dtype=float,
                        )

                        plot_edge(
                            "y",
                            (i, j, k),
                            p0,
                            p1,
                        )

        # ======================================================
        # Draw z-directed primal edges.
        #
        # Shape:
        #     (Nx + 1, Ny + 1, Nz)
        # ======================================================
        if "z" in in_plane_dirs:
            for i in range(grid.Nx + 1):
                for j in range(grid.Ny + 1):
                    for k in range(grid.Nz):
                        if normal == "x" and i != index:
                            continue

                        if normal == "y" and j != index:
                            continue

                        p0 = np.array(
                            [
                                x[i],
                                y[j],
                                z[k],
                            ],
                            dtype=float,
                        )

                        p1 = np.array(
                            [
                                x[i],
                                y[j],
                                z[k + 1],
                            ],
                            dtype=float,
                        )

                        plot_edge(
                            "z",
                            (i, j, k),
                            p0,
                            p1,
                        )

        # ======================================================
        # Draw exact STL intersections with the slice.
        #
        # Use grid.read_stl() so that scale / rotation /
        # translation applied by GridFIT3D are respected.
        # ======================================================
        if show_stl:
            normal_vector = {
                "x": (
                    1.0,
                    0.0,
                    0.0,
                ),
                "y": (
                    0.0,
                    1.0,
                    0.0,
                ),
                "z": (
                    0.0,
                    0.0,
                    1.0,
                ),
            }[normal]

            for key in grid.stl_solids.keys():
                region_id = int(grid.region_key_to_id[key])

                surface = grid.read_stl(key)

                slice_poly = surface.slice(
                    normal=normal_vector,
                    origin=plane_origin,
                )

                for segment in self._polydata_lines_to_segments(slice_poly):
                    segment_2d = segment[
                        :,
                        list(plane_axes),
                    ]

                    ax.plot(
                        segment_2d[:, 0],
                        segment_2d[:, 1],
                        color=region_colors[region_id],
                        lw=stl_lw,
                        zorder=4,
                    )

        # ======================================================
        # Legend
        # ======================================================
        if show_legend:
            from matplotlib.lines import Line2D

            handles = []

            # Background only needs a legend entry if it is
            # actually visible in the active edge plot.
            if 0 in drawn_region_ids:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=region_colors[0],
                        lw=edge_lw,
                        label=self._conformal_region_label(0),
                    )
                )

            # Every STL geometry gets its own legend entry.
            for key in grid.stl_solids.keys():
                region_id = int(grid.region_key_to_id[key])

                material_type = self.region_material_types[region_id]

                legend_lw = edge_lw if material_type == "normal" else stl_lw

                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=region_colors[region_id],
                        lw=legend_lw,
                        label=self._conformal_region_label(region_id),
                    )
                )

            if handles:
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0,
                )

                ax.figure.subplots_adjust(right=0.78)

        # ======================================================
        # Final plot formatting
        # ======================================================
        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.set_xlabel(axis_labels[0])

        ax.set_ylabel(axis_labels[1])

        ax.set_title(
            "Conformal primal edge slice, "
            f"{normal} = "
            f"{plane_coordinate:.6g} "
            f"(index {index})"
        )

        coordinate_arrays = {
            "x": x,
            "y": y,
            "z": z,
        }

        horizontal_coordinates = coordinate_arrays[axis_labels[0]]

        vertical_coordinates = coordinate_arrays[axis_labels[1]]

        ax.set_xlim(
            horizontal_coordinates[0],
            horizontal_coordinates[-1],
        )

        ax.set_ylim(
            vertical_coordinates[0],
            vertical_coordinates[-1],
        )

        return ax


class PlotMixinGrid:
    """Plotting mixin providing 1D/2D/3D visualization helpers for GridFIT3D."""

    def _add_logo_widget(self, pl):
        """
        Add packaged logo to a PyVista plotter via importlib.resources.

        Attempts to load a packaged logo image from the installed package
        resources. Falls back to a local development path when resources are
        not available (typical in editable/dev installs).
        """
        try:
            from importlib import resources

            # resource inside the installed package (use current package)
            logo_res = resources.files(__package__).joinpath(
                "static", "img", "wakis-logo-pink.png"
            )
            with resources.as_file(logo_res) as logo_path:
                pl.add_logo_widget(str(logo_path))
                return
        except Exception as e:
            # fallback to the legacy relative path for dev installs
            try:
                pl.add_logo_widget("../docs/img/wakis-logo-pink.png")
            except Exception:
                if self.verbose > 2:
                    print(f"[!] Could not add logo widget: {e}")

    def plot_solids(
        self,
        bounding_box=False,
        show_grid=False,
        anti_aliasing=None,
        opacity=1.0,
        specular=0.5,
        smooth_shading=False,
        off_screen=False,
        **kwargs,
    ):
        """
        Generate a 3D visualization of imported STL geometries using PyVista.

        Parameters
        ----------
        bounding_box : bool, optional
            If True, adds a bounding box around the plotted geometry (default False).
        show_grid : bool, optional
            If True, overlays the grid wireframe on the scene (default False).
        anti_aliasing : str or None, optional
            Anti-aliasing mode passed to PyVista (default: None).
        opacity : float, optional
            Opacity for solids (1.0 opaque, 0.0 transparent). Default 1.0.
        specular : float, optional
            Specular lighting strength, higher is shinier (default 0.5).
        smooth_shading : bool, optional
            Enable smooth shading for surface rendering (default False).
        off_screen : bool, optional
            If True, export to HTML instead of opening an interactive window.
        **kwargs : dict
            Additional keyword args forwarded to ``pyvista.add_mesh``.

        Notes
        -----
        - Colors come from ``self.stl_colors`` when available.
        - Solids labeled 'vacuum' are rendered with reduced opacity by default.
        """
        pl = pv.Plotter(off_screen=off_screen)
        pl.add_mesh(self.grid, opacity=0.0, name="grid", show_scalar_bar=False)
        for key in self.stl_solids:
            color = self.stl_colors[key]
            if self.stl_materials[key] == [1.0, 1.0, 0.0] or self.stl_materials[
                key
            ] == [1.0, 1.0]:
                _opacity = 0.3  # vacuum
            else:
                _opacity = opacity
            pl.add_mesh(
                self.read_stl(key),
                color=color,
                opacity=_opacity,
                specular=specular,
                smooth_shading=smooth_shading,
                **kwargs,
            )

        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.camera_position = "zx"
        pl.camera.azimuth += 30
        pl.camera.elevation += 30
        pl.add_axes()

        if anti_aliasing is not None:
            pl.enable_anti_aliasing(anti_aliasing)

        if bounding_box:
            pl.add_bounding_box()

        if show_grid:
            pl.add_mesh(
                self.grid,
                style="wireframe",
                color="grey",
                opacity=0.3,
                name="grid",
            )

        if off_screen:
            return pl
            # pl.export_html("grid_plot_solids.html")
        else:
            pl.show()

    def plot_stl_mask(
        self,
        stl_solid,
        cmap="viridis",
        bounding_box=True,
        clip_plane="x",
        add_stl=True,
        stl_opacity=0.1,
        stl_colors=None,
        anti_aliasing="ssaa",
        smooth_shading=False,
        off_screen=False,
    ):
        """
        Interactive 3D visualization of the structured grid mask and imported STL
        geometries.

        This routine uses PyVista to display the grid scalar field corresponding to a
        chosen STL mask. It provides interactive slider widgets to clip the domain
        along the X, Y, and Z directions. At each slider position, the clipped
        scalar field is shown with a colormap while the grid structure is shown
        as a 2D slice in wireframe. Optionally, one or more STL geometries can
        be added to the scene, along with a bounding box of the simulation domain.

        Parameters
        ----------
        stl_solid : str
            Key name of the `stl_solids` dictionary to retrieve the mask for
            visualization (used as the scalar field).
        cmap : str, optional
            Colormap used to visualize the clipped scalar values. Default 'viridis'.
        bounding_box : bool, optional
            If True, add a static wireframe bounding box of the simulation domain.
        show_grid : bool, optional
            If True, adds the computational grid overlay on the clipped slice.
        add_stl : {True, 'all', str, list[str]}, optional
            STL geometries to add. Default True adds the STL solid corresponding to the mask.
             If 'all', add all STL solids. If a string, add the STL solid with that key.
             If a list of strings, add the STL solids with those keys.
             False or None will not add any STL solids.
        stl_opacity : float, optional
            Opacity of the STL surfaces (0 = fully transparent, 1 = fully opaque).
        stl_colors : str, list[str], dict, or None, optional
            Color(s) of the STL surfaces.
        xmax, ymax, zmax : float, optional
            Initial clipping positions along each axis. If None, use the maximum domain
            extent.
        anti_aliasing : {'ssaa', 'fxaa', None}, optional
            Anti-aliasing mode passed to `pl.enable_anti_aliasing`.
        smooth_shading : bool, optional
            Enable smooth shading for STL surfaces. Default False.
        off_screen : bool, optional
            If True, render off-screen and export the scene to HTML.

        Notes
        -----
        - Three sliders (X, Y, Z) control clipping of the scalar field by a box.
        - STL solids can be visualized in transparent mode.
        - A static domain bounding box can be added for reference.
        """
        if stl_colors is None:
            stl_colors = self.stl_colors

        # change the sign of clip plane if y or z
        if clip_plane.lower() in ["-y", "y"]:
            clip_plane = "-y"
        elif clip_plane.lower() in ["-z", "z"]:
            clip_plane = "-z"

        # Default threshold value for extracting cells inside the mask
        value = 0.1
        if getattr(self, "subpixel_smoothing_threshold", None) is not None:
            value = self.subpixel_smoothing_threshold

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter(off_screen=off_screen)

        # Threshold to extract only the cells inside the mask
        self.grid.set_active_scalars(stl_solid)
        cells_inside = self.grid.threshold(value=value, scalars=stl_solid)

        # Interactive clip-plane widget over the thresholded cells
        pl.add_mesh_clip_plane(
            cells_inside,
            cmap=cmap,
            show_edges=True,
            # scalars=stl_solid,
            label="cells mask",
            normal=clip_plane,
            widget_color="white",
        )

        # Plot stl surface(s)
        if add_stl is not None and add_stl is not False:
            if add_stl is True:  # default: add the STL solid matching the mask
                key = stl_solid
                surf = self.read_stl(key)
                pl.add_mesh(
                    surf,
                    color=stl_colors[key],
                    opacity=stl_opacity,
                    smooth_shading=smooth_shading,
                    label="STL surface",
                )
            elif type(add_stl) is str:
                if add_stl.lower() == "all":
                    for i, key in enumerate(self.stl_solids):
                        surf = self.read_stl(key)
                        color = (
                            stl_colors[key]
                            if isinstance(stl_colors, dict)
                            else (
                                stl_colors[i]
                                if isinstance(stl_colors, list)
                                else "white"
                            )
                        )
                        pl.add_mesh(
                            surf,
                            color=color,
                            opacity=stl_opacity,
                            smooth_shading=smooth_shading,
                            name=key,
                        )
                else:  # single named solid
                    key = add_stl
                    surf = self.read_stl(key)
                    color = stl_colors[key] if isinstance(stl_colors, dict) else "white"
                    pl.add_mesh(
                        surf,
                        color=color,
                        opacity=stl_opacity,
                        smooth_shading=smooth_shading,
                        label="STL surface",
                    )
            elif type(add_stl) is list:
                for i, key in enumerate(add_stl):
                    surf = self.read_stl(key)
                    color = (
                        stl_colors[key]
                        if isinstance(stl_colors, dict)
                        else (
                            stl_colors[i] if isinstance(stl_colors, list) else "white"
                        )
                    )
                    pl.add_mesh(
                        surf,
                        color=color,
                        opacity=stl_opacity,
                        smooth_shading=smooth_shading,
                        name=key,
                    )

        # Camera orientation — face the cut plane, then tilt +30°/+30° to show 3D structure
        _clip_axis = clip_plane.lstrip("-").lower()
        {"x": pl.view_zy, "y": pl.view_zx, "z": pl.view_xy}.get(
            _clip_axis, pl.view_zy
        )()
        pl.camera.azimuth += 30
        pl.camera.elevation += 30

        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.add_axes()
        pl.enable_3_lights()

        if bounding_box:
            pl.add_mesh(
                pv.Box(
                    bounds=(
                        self.xmin,
                        self.xmax,
                        self.ymin,
                        self.ymax,
                        self.zmin,
                        self.zmax,
                    )
                ),
                style="wireframe",
                color="black",
                line_width=2,
                name="domain_box",
            )

        if off_screen:
            return pl
        else:
            pl.show()

    def plot_stl_mask_slice(
        self,
        stl_solid,
        plane="ZX",
        position=None,
        cmap="viridis",
        bounding_box=False,
        show_grid=True,
        add_stl=False,
        stl_opacity=0.1,
        stl_colors=None,
        smooth_shading=False,
        off_screen=False,
    ):
        """
        Interactive 2D slice visualization of the structured grid STL mask.

        Lighter alternative to :meth:`plot_stl_mask` for heavy grids (>10M cells).
        Instead of clipping the full 3D domain, only a single 2D slice is rendered
        at a time. A slider widget controls the slice position along the axis normal
        to the chosen plane.

        Parameters
        ----------
        stl_solid : str
            Key name of the `stl_solids` dictionary to retrieve the mask for
            visualization (used as the scalar field).
        plane : {'XY', 'YX', 'ZY', 'YZ', 'ZX', 'XZ'}, optional
            Plane of the slice. Default 'ZX'.

            - 'XY' / 'YX' → normal along Z, slider controls Z position.
            - 'ZY' / 'YZ' → normal along X, slider controls X position.
            - 'ZX' / 'XZ' → normal along Y, slider controls Y position.
        position : float or None, optional
            Initial position of the slice along the normal axis. If None, uses
            the center of the domain along that axis.
        cmap : str, optional
            Colormap used to visualize the scalar values. Default 'viridis'.
        bounding_box : bool, optional
            If True, add a static wireframe bounding box of the simulation domain.
        add_stl : {True, 'all', str, list[str]}, optional
            STL geometries to add. Default True adds the STL solid corresponding to the mask.
             If 'all', add all STL solids. If a string, add the STL solid with that key.
             If a list of strings, add the STL solids with those keys.
             False or None will not add any STL solids.
        stl_opacity : float, optional
            Opacity of the STL surfaces (0 = fully transparent, 1 = fully opaque).
        stl_colors : str, list[str], dict, or None, optional
            Color(s) of the STL surfaces.
        smooth_shading : bool, optional
            Enable smooth shading for STL surfaces. Default False.
        off_screen : bool, optional
            If True, render off-screen and return the plotter object.

        Notes
        -----
        - Only a single 2D slice of the domain is rendered at a time, making this
          much lighter than :meth:`plot_stl_mask` for large grids.
        - The slider controls the slice position along the normal to the chosen plane.
                - The selected ``stl_solid`` intersection with the plane is overlaid as a
                    contour line for easier visual alignment.
        """
        self.grid.set_active_scalars(stl_solid)

        if stl_colors is None:
            stl_colors = self.stl_colors

        plane = plane.upper()
        plane_to_normal = {
            "XY": "z",
            "YX": "z",
            "ZY": "x",
            "YZ": "x",
            "ZX": "y",
            "XZ": "y",
        }
        if plane not in plane_to_normal:
            raise ValueError(
                f"plane must be one of 'XY', 'YX', 'ZY', 'YZ', 'ZX', 'XZ', got '{plane}'"
            )

        normal = plane_to_normal[plane]
        if normal == "x":
            axis_min, axis_max = self.xmin, self.xmax
            slider_title = "X Position"
            center = ((self.ymin + self.ymax) / 2, (self.zmin + self.zmax) / 2)

            def origin_fn(v):
                return (v, center[0], center[1])

        elif normal == "y":
            axis_min, axis_max = self.ymin, self.ymax
            slider_title = "Y Position"
            center = ((self.xmin + self.xmax) / 2, (self.zmin + self.zmax) / 2)

            def origin_fn(v):
                return (center[0], v, center[1])

        else:  # z
            axis_min, axis_max = self.zmin, self.zmax
            slider_title = "Z Position"
            center = ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

            def origin_fn(v):
                return (center[0], center[1], v)

        if position is None:
            position = (axis_min + axis_max) / 2

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter(off_screen=off_screen)

        # Surface used to draw the selected STL/plane intersection contour
        outline_surf = self.read_stl(stl_solid)
        outline_color = stl_colors.get(stl_solid, "white")

        # --- Initial slice ---
        initial_slice = self.grid.slice(normal=normal, origin=origin_fn(position))
        slice_actor = pl.add_mesh(
            initial_slice,
            cmap=cmap,
            name="slice",
        )

        outline_actor = None
        if outline_surf is not None:
            initial_outline = outline_surf.slice(
                normal=normal, origin=origin_fn(position)
            )
            outline_actor = pl.add_mesh(
                initial_outline,
                color=outline_color,
                # line_width=3,
                name="stl_outline",
            )

        # --- Update function ---
        def update_slice(val):
            new_slice = self.grid.slice(normal=normal, origin=origin_fn(val))
            slice_actor.mapper.SetInputData(new_slice)
            if outline_actor is not None:
                new_outline = outline_surf.slice(normal=normal, origin=origin_fn(val))
                outline_actor.mapper.SetInputData(new_outline)

            # add slice wireframe (grid structure)
            if show_grid:
                pl.add_mesh(
                    new_slice, style="wireframe", color="grey", opacity=0.3, name="grid"
                )

            pl.render()

        # --- Slider ---
        pl.add_slider_widget(
            update_slice,
            [axis_min, axis_max],
            value=position,
            title=slider_title,
            pointa=(0.8, 0.6),
            pointb=(0.95, 0.6),
            style="modern",
        )

        # Plot stl surface(s)
        if add_stl is not None:
            if add_stl:  # Default, add the stl solid corresponding to the mask
                key = stl_solid
                surf = self.read_stl(key)
                pl.add_mesh(
                    surf,
                    color=stl_colors[key],
                    opacity=stl_opacity,
                    silhouette=dict(color=stl_colors[key]),
                    smooth_shading=smooth_shading,
                    name=key,
                )
            if type(add_stl) is str:
                if add_stl.lower() == "all":
                    for i, key in enumerate(self.stl_solids):
                        surf = self.read_stl(key)
                        if type(stl_colors) is dict:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[key],
                                opacity=stl_opacity,
                                silhouette=dict(color=stl_colors[key]),
                                smooth_shading=smooth_shading,
                                name=key,
                            )
                        elif type(stl_colors) is list:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[i],
                                opacity=stl_opacity,
                                silhouette=dict(color=stl_colors[i]),
                                smooth_shading=smooth_shading,
                                name=key,
                            )
                        else:
                            pl.add_mesh(
                                surf,
                                color="white",
                                opacity=stl_opacity,
                                silhouette=True,
                                smooth_shading=smooth_shading,
                                name=key,
                            )
                else:  # add 1 selected stl solid
                    key = add_stl
                    surf = self.read_stl(key)
                    pl.add_mesh(
                        surf,
                        color=stl_colors[key],
                        opacity=stl_opacity,
                        silhouette=dict(color=stl_colors[key]),
                        smooth_shading=smooth_shading,
                        name=key,
                    )

            elif type(add_stl) is list:
                for i, key in enumerate(add_stl):
                    surf = self.read_stl(key)
                    if type(stl_colors) is dict:
                        pl.add_mesh(
                            surf,
                            color=stl_colors[key],
                            opacity=stl_opacity,
                            silhouette=dict(color=stl_colors[key]),
                            smooth_shading=smooth_shading,
                            name=key,
                        )
                    elif type(stl_colors) is list:
                        pl.add_mesh(
                            surf,
                            color=stl_colors[i],
                            opacity=stl_opacity,
                            silhouette=dict(color=stl_colors[i]),
                            smooth_shading=smooth_shading,
                            name=key,
                        )
                    else:
                        pl.add_mesh(
                            surf,
                            color="white",
                            opacity=stl_opacity,
                            silhouette=True,
                            smooth_shading=smooth_shading,
                            name=key,
                        )

        # Camera orientation — use view_*() to enforce correct horizontal/vertical axes
        _view_fns = {
            "XY": pl.view_xy,
            "YX": pl.view_xy,
            "XZ": pl.view_xz,
            "YZ": pl.view_yz,
            "ZX": pl.view_zx,
            "ZY": pl.view_zy,
        }
        _view_fns.get(plane, pl.view_yz)()
        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.add_axes()

        if bounding_box:
            pl.add_mesh(
                pv.Box(
                    bounds=(
                        self.xmin,
                        self.xmax,
                        self.ymin,
                        self.ymax,
                        self.zmin,
                        self.zmax,
                    )
                ),
                style="wireframe",
                color="black",
                line_width=2,
                name="domain_box",
            )

        if off_screen:
            return pl
        else:
            pl.show()

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

    def inspect(
        self,
        add_stl=None,
        stl_opacity=0.5,
        stl_colors=None,
        anti_aliasing="ssaa",
        smooth_shading=True,
        off_screen=False,
    ):
        """
        Interactive 3D inspector showing grid and STL geometries.

        Parameters
        ----------
        add_stl : str or list, optional
            Key or list of keys of STL solids to include. If None, all solids are shown.
        stl_opacity : float, optional
            Opacity for STL surfaces (0 transparent, 1 opaque). Default 0.5.
        stl_colors : list or dict, optional
            Colors for STL surfaces. Use a list in the same order as ``add_stl``
            or a dictionary keyed by solid name. Defaults to ``self.stl_colors``.
        anti_aliasing : str or None, optional
            Anti-aliasing mode to enable in the plotter (default 'ssaa').
        smooth_shading : bool, optional
            Enable smooth shading for surfaces (default True).
        off_screen : bool, optional
            If True, return the off-screen plotter object instead of showing an
            interactive window.

        Returns
        -------
        pl : pyvista.Plotter or None
            The plotter object if off_screen is True, otherwise None.
        """
        if stl_colors is None:
            stl_colors = self.stl_colors

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter(off_screen=off_screen)
        pl.add_mesh(self.grid, show_edges=True, cmap=["white", "white"], name="grid")

        def clip(widget):
            # Plot structured grid
            b = widget.bounds
            x = self.x[np.logical_and(self.x >= b[0], self.x <= b[1])]
            y = self.y[np.logical_and(self.y >= b[2], self.y <= b[3])]
            z = self.z[np.logical_and(self.z >= b[4], self.z <= b[5])]
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            grid = pv.StructuredGrid(X.transpose(), Y.transpose(), Z.transpose())

            pl.add_mesh(grid, show_edges=True, cmap=["white", "white"], name="grid")
            # Plot stl surface(s)
            if add_stl is not None:  # add 1 selected stl solid
                if type(add_stl) is str:
                    key = add_stl
                    surf = self.read_stl(key)
                    surf = surf.clip_box(widget.bounds, invert=False)
                    pl.add_mesh(
                        surf,
                        color=stl_colors[key],
                        opacity=stl_opacity,
                        silhouette=True,
                        smooth_shading=smooth_shading,
                        name=key,
                    )

                elif type(add_stl) is list:  # add selected list of stl solids
                    for i, key in enumerate(add_stl):
                        surf = self.read_stl(key)
                        surf = surf.clip_box(widget.bounds, invert=False)
                        if type(stl_colors) is dict:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[key],
                                opacity=stl_opacity,
                                silhouette=True,
                                smooth_shading=smooth_shading,
                                name=key,
                            )
                        elif type(stl_colors) is list:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[i],
                                opacity=stl_opacity,
                                silhouette=True,
                                smooth_shading=smooth_shading,
                                name=key,
                            )
                        else:
                            pl.add_mesh(
                                surf,
                                color="white",
                                opacity=stl_opacity,
                                silhouette=True,
                                smooth_shading=smooth_shading,
                                name=key,
                            )

            else:  # add all stl solids
                for i, key in enumerate(self.stl_solids):
                    surf = self.read_stl(key)
                    surf = surf.clip_box(widget.bounds, invert=False)
                    if type(stl_colors) is dict:
                        pl.add_mesh(
                            surf,
                            color=stl_colors[key],
                            opacity=stl_opacity,
                            silhouette=True,
                            smooth_shading=smooth_shading,
                            name=key,
                        )
                    elif type(stl_colors) is list:
                        pl.add_mesh(
                            surf,
                            color=stl_colors[i],
                            opacity=stl_opacity,
                            silhouette=True,
                            smooth_shading=smooth_shading,
                            name=key,
                        )
                    else:
                        pl.add_mesh(
                            surf,
                            color="white",
                            opacity=stl_opacity,
                            silhouette=True,
                            smooth_shading=smooth_shading,
                            name=key,
                        )

        _ = pl.add_box_widget(callback=clip, rotation_enabled=False)

        # Camera orientation
        pl.camera_position = "zx"
        pl.camera.azimuth += 30
        pl.camera.elevation += 30
        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.add_axes()
        pl.enable_3_lights()
        pl.enable_anti_aliasing(anti_aliasing)

        if off_screen:
            return pl
            # pl.export_html('grid_inspect.html')
        else:
            pl.show()
            return None
