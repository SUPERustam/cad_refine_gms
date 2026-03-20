from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dataset-prep dependency
    from vis_for_norm_parts import Plotter as _BasePlotter

    _VIS_FOR_NORM_PARTS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via explicit failure test
    _BasePlotter = None
    _VIS_FOR_NORM_PARTS_IMPORT_ERROR = exc


def _raise_missing_vis_dependency() -> None:
    message = (
        "Dataset preparation visualization requires the optional "
        "'vis_for_norm_parts' dependency. Install it or run the dataset-prep "
        "pipeline in an environment that provides it."
    )
    raise RuntimeError(message) from _VIS_FOR_NORM_PARTS_IMPORT_ERROR


if _BasePlotter is not None:

    class Plotter1_1(_BasePlotter):
        def _get_img(
            self,
            mesh_path,
            cmap,
            apply_augs=False,
            color=None,
        ):
            import numpy as np
            import pyvista as pv
            from PIL import Image

            mesh = pv.read(mesh_path)
            b = np.array(mesh.bounds)
            mins = b[::2]
            maxs = b[1::2]
            center = (mins + maxs) / 2.0
            extents = maxs - mins
            s = 1.0 / max(extents.max(), 1e-12)
            mesh.points = (mesh.points - center) * s + 0.5

            mesh.point_data.update(self.get_scalars(mesh))
            mesh_actor = self.plotter.add_mesh(
                mesh,
                reset_camera=False,
                color=None,
                scalars=None,
                cmap=cmap,
                show_scalar_bar=False,
            )
            mesh_actor.use_bounds = False

            view_images = []
            for view_name, set_view_func in self.views.items():
                set_view_func(mesh)
                if view_name in ("Iso", "-Iso"):
                    self.plotter.disable_parallel_projection()
                else:
                    self.plotter.enable_parallel_projection()
                    self.plotter.zoom_camera(1.7)

                img_array = self.plotter.screenshot(return_img=True)
                pil_img = Image.fromarray(img_array)
                pil_img.thumbnail(
                    (self.view_img_size, self.view_img_size),
                    resample=Image.Resampling.BILINEAR,
                )
                if self.align_coordinates and view_name in ("-Z", "+Y", "+X", "Iso"):
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                view_images.append(pil_img)

            self.remove_meshes(mesh_actor)

            if apply_augs:
                try:
                    view_images = self.apply_augs(
                        {
                            view_name: view_image
                            for view_name, view_image in zip(self.views, view_images)
                        }
                    )
                except Exception as ex:  # pragma: no cover - rendering fallback
                    print("Exception in augs:", ex)

            mesh_actor = self.iso_plotter.add_mesh(
                mesh, reset_camera=False, color=color
            )
            mesh_actor.use_bounds = False

            for view_name in ("Iso", "-Iso"):
                if view_name == "Iso":
                    self.iso_plotter.view_isometric()
                else:
                    self.iso_plotter.view_isometric(negative=True)
                self.iso_plotter.zoom_camera(1.1)

                img_array = self.iso_plotter.screenshot(return_img=True)
                pil_img = Image.fromarray(img_array)
                pil_img.thumbnail(
                    (self.view_img_size, self.view_img_size),
                    resample=Image.Resampling.BILINEAR,
                )
                if self.align_coordinates and view_name in ("-Z", "+Y", "+X", "Iso"):
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                view_images.append(pil_img)

            _success = self.iso_plotter.remove_actor(
                mesh_actor, reset_camera=False, render=False
            )
            if not _success:
                self.reload()

            padding = 0
            total_width = round(
                self.cols * self.view_img_size + (self.cols - 1) * padding
            )
            total_height = round(
                self.rows * self.view_img_size + (self.rows - 1) * padding
            )
            collage = Image.new("RGB", (total_width, total_height), color="white")
            for i, img in enumerate(view_images):
                row = i // self.cols
                col = i % self.cols
                x_offset = col * (img.width + padding)
                y_offset = row * (img.height + padding)
                collage.paste(img, (x_offset, y_offset))

            return collage

else:

    class Plotter1_1:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_missing_vis_dependency()
