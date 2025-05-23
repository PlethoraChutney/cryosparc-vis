from typing import TYPE_CHECKING, Optional, Literal, Any
from cryosparc.tools import lowpass2
from cryosparc.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable, Sequence
from ..utils import vmin_vmax_percentile

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    from .vis_dataset import VisDataset
    from numpy.typing import NDArray, ArrayLike
    from collections.abc import Callable
    from cryosparc.row import Row
    from cryosparc.mrc import Header

class Particles:
    def __init__(self, parent:"VisDataset") -> None:
        self.parent = parent
        spec = self.parent.particles_spec
        self._particles:"Dataset|None"
        self._particle_mrcs:"dict[str, tuple[Header, NDArray[np.floating]]]" = {}
        self._particle_images:"dict[int, tuple[Header, NDArray[np.floating]]]" = {}
        self.bool_masks:dict[str, "NDArray[np.bool_]"] = {}
        self.function_masks:dict[str, "Callable[[Row], bool]"] = {}
        self.filter_by_mic = True
        self.filter_by_crop = True
        self.allow_masked_assignment = False

        if spec[0] is None:
            self.job = None
            self.particles = None
        else:
            self.job = self.parent.project.find_job(spec[0])
            self.particles = self.job.load_output("particles" if spec[1] is None else spec[1], version = self.parent.iteration)

    def remove_mask(self, mask_name:str) -> None:
        if mask_name in self.bool_masks:
            del self.bool_masks[mask_name]
        else:
            print(f"{mask_name} not found")

    @property
    def unmasked_particles(self) -> "Dataset":
        if self._particles is None:
            raise AttributeError("Particles are not loaded")
        return self._particles

    @property
    def bool_mask(self) -> "NDArray[np.bool_]":
        masks = list(self.bool_masks.values())
        if self.filter_by_mic:
            masks.append(self.unmasked_particles["location/micrograph_uid"] == self.parent.mic_uid)
        if self.filter_by_crop:
            masks.append(self.unmasked_particles["location/center_x_frac"] >= self.parent.crop_slice[0])
            masks.append(self.unmasked_particles["location/center_x_frac"] < self.parent.crop_slice[1])
            masks.append(self.unmasked_particles["location/center_y_frac"] >= self.parent.crop_slice[2])
            masks.append(self.unmasked_particles["location/center_y_frac"] < self.parent.crop_slice[3])

        return np.all(masks, axis = 0)

    @property
    def particles(self) -> "Dataset":
        p = self.unmasked_particles.mask(self.bool_mask)
        for f_mask in self.function_masks.values():
            p = p.query(f_mask)

        return p
    
    @particles.setter
    def particles(self, dset:"None|Dataset") -> None:
        if dset is not None and not isinstance(dset, Dataset):
            raise ValueError("Particles must be a CryoSPARC Dataset")
        self._particles = dset

    @property
    def fields(self) -> list[str]:
        return self.particles.fields()
    
    def fields_prefix(self, prefix:str) -> list[str]:
        return list(f for f in self.fields if f.startswith(prefix))
    
    # dataset passthrough

    def __getitem__(self, key):
        return self.particles[key]
    
    def __setitem__(self, key, value):
        if not self.allow_masked_assignment:
            raise ValueError("You are attempting to assign to masked particles. If you mean to do this, set self.allow_masked_assignment to True")
        else:
            self.particles[key] = value
    
    def __delitem__(self, key):
        self.unmasked_particles.drop_fields(key)

    def query(self, query:"dict[str, ArrayLike] | Callable[[Row], bool]") -> Dataset:
        return self.particles.query(query)
    


    # plotting

    @property
    def particles_x(self) -> "NDArray[np.floating]":
        crop_offset = (self.particles["location/center_x_frac"] - self.parent.crop_slice[0]) / self.parent.crop_slice[1]
        return crop_offset * self.parent.base_mic.shape[1]
    
    @property
    def particles_y(self) -> "NDArray[np.floating]":
        crop_offset = (self.particles["location/center_y_frac"] - self.parent.crop_slice[2]) / self.parent.crop_slice[3]
        return crop_offset * self.parent.base_mic.shape[0]
    
    plot_defaults = {
        "s": 40,
        "facecolor": "#FAB06E",
        "edgecolor": "black",
        "linewidth": 1
    }

    def plot(
            self,
            ax:Optional["matplotlib.axes.Axes"] = None,
            figsize:Optional[tuple[float, float]] = None,
            color_by:Optional[str] = None,
            legend:None | Literal["colorbar"] | Literal["legend"] = "colorbar",
            **kwargs,
            ) -> tuple["matplotlib.figure.Figure | None", "matplotlib.axes.Axes"]:
        if ax is None:
            fig, ax = plt.subplots(
                1,
                1,
                frameon = False,
                layout = "constrained",
                figsize = figsize if figsize else self.parent.figsize
            )
            ax.margins(0)
        else:
            fig = ax.get_figure()

        defaults = self.__class__.plot_defaults.copy()
        defaults["c"] = self.particles[color_by] if color_by is not None else None
        defaults.update(**kwargs)

        if color_by is not None:
            del defaults["facecolor"]

        scatter = ax.scatter(
            self.particles_x,
            self.particles_y,
            **defaults
        )
        if color_by is not None and fig is not None:
            if legend == "colorbar":
                fig.colorbar(scatter, ax = ax, location = "right")
            elif legend == "legend":
                ax.add_artist(
                    ax.legend(*scatter.legend_elements(), loc = "top right", title = color_by)
                )
        ax.set_aspect("equal")
        ax.axis("off")
        
        return fig, ax
    
    # single particle ---

    def particle_image(self, particle_uid:int) -> "tuple[Header, NDArray[np.floating]]":
        if particle_uid in self._particle_images:
            return self._particle_images[particle_uid]
        
        row = self.unmasked_particles.query({"uid": particle_uid})[0]
        if len(row) == 0:
            raise ValueError(f"Particle {particle_uid} not in particle dataset")
        
        mrc_path = row["blob/path"]
        particle_idx = row["blob/idx"]
        if mrc_path in self._particle_mrcs:
            hdr, pstack = self._particle_mrcs[mrc_path]
        else:
            hdr, pstack = self.parent.project.download_mrc(mrc_path)
            self._particle_mrcs[mrc_path] = (hdr, pstack)

        particle_image = pstack[particle_idx]
        self._particle_images[particle_uid] = (hdr, particle_image)

        return hdr, particle_image
    
    single_particle_imshow_defaults:dict[str, Any] = {
        "cmap": "Grays",
        "origin": "lower",
        "interpolation": "nearest"
    }

        
    def single_particle_imshow(
            self,
            img:Optional["np.ndarray[np.floating, np.dtype[np.floating]]"] = None,
            hdr:Optional["Header"] = None,
            ax:Optional["matplotlib.axes.Axes"] = None,
            figsize:tuple[float, float] | Literal["pixel_perfect"] = (2, 2),
            particle_selection:Optional[int] = None,
            selection_type:Literal["uid", "unmasked_index", "masked_index"] = "uid",
            lowpass:Optional[float] = 20,
            percentile:Optional[float] = None,
            **kwargs,
            ) -> tuple["matplotlib.figure.Figure | None", "matplotlib.axes.Axes"]:

        if any(x is None for x in (img, hdr)) and not all(x is None for x in (img, hdr)):
            raise ValueError("Must provide either both or neither of img, hdr")
        
        if img is None or hdr is None:
            if particle_selection is None:
                raise ValueError("Must provide img and hdr or a particle selection")
            if selection_type == "uid":
                hdr, img = self.particle_image(particle_selection)
            elif selection_type == "masked_index":
                hdr, img = self.particle_image(self.particles["uid"][particle_selection])
            elif selection_type == "unmasked_index":
                hdr, img = self.particle_image(self.unmasked_particles["uid"][particle_selection])
            else:
                raise ValueError("Invalid value for selection_type")

        if lowpass is not None:
            img = lowpass2(img, hdr.xlen / hdr.nx, lowpass)

        defaults = self.single_particle_imshow_defaults.copy()
        defaults.update(**kwargs)

        if percentile is not None:
            vmin, vmax = vmin_vmax_percentile(img, percentile)
            defaults["vmin"] = vmin
            defaults["vmax"] = vmax

        if ax is None:
            dpi = plt.rcParams["figure.dpi"]
            if figsize == "pixel_perfect":
                figsize = (img.shape[1] / dpi, img.shape[0] / dpi)
            fig, ax = plt.subplots(1, 1, figsize = figsize, frameon = False, layout = "constrained")
        else:
            fig = ax.get_figure()

        ax.imshow(
            img,
            **defaults
        )
        ax.axis("off")
        ax.set_aspect("equal")

        return fig, ax
    
    def multiple_particle_imshow(
            self,
            images:Sequence[int]|int = 6,
            image_selection:Optional[Literal["uid", "index", "number_of_random", "number_in_order"]] = None,
            axs:Optional[np.ndarray["matplotlib.axes.Axes", np.dtype[np.object_]]] = None,
            im_layout:Optional[tuple[int, int]] = None,
            imsize:Optional[float] = 2.0,
            figsize:Optional[tuple[float, float]] = None,
            cmap_range:None|float|Iterable[float]|Sequence[Iterable[float]] = 0.01,
            percentile_per_class:Literal["overall", "per_class"] = "overall",
            lowpass:None|float = 20.0,
            **kwargs
    ) -> tuple["matplotlib.figure.Figure", "np.ndarray[matplotlib.axes.Axes, np.dtype[np.object_]]"]:
        
        if image_selection is None:
            if isinstance(images, int):
                image_selection = "number_in_order"
            elif isinstance(images, Sequence):
                image_selection = "uid"
        
        if image_selection == "uid":
            plot_p = self.particles.query({"uid": image_selection})
        elif image_selection == "index":
            if isinstance(images, int):
                images = [images]
            plot_p = self.particles.take(images) # type: ignore - cs-tools requires specifically a list
        elif image_selection == "number_of_random":
            if isinstance(images, int):
                plot_p = self.particles.take(np.random.default_rng().choice(len(self.particles), images, replace = False))
            else:
                raise ValueError("Provide a single int if selecting a number of particles")
        elif image_selection == "number_in_order":
            if isinstance(images, int):
                plot_p = self.particles.take(list(range(images)))
            else:
                raise ValueError("Provide a single int if selecting a number of particles")
        else:
            raise ValueError("Invalid image_selection option")
        
        plot_imgs = []
        plot_hdrs = []
        for im in plot_p.rows():
            hdr, img = self.particle_image(im["uid"])
            if lowpass is not None:
                img = lowpass2(img, hdr.xlen / hdr.nx, lowpass)
            plot_imgs.append(img)
            plot_hdrs.append(hdr)

        if percentile_per_class == "overall" and not isinstance(cmap_range, Iterable) and cmap_range is not None:
            all_pixels = np.dstack(plot_imgs)
            vmin, vmax = vmin_vmax_percentile(all_pixels, cmap_range) # type: ignore
        elif isinstance(cmap_range, Sequence) and not isinstance(cmap_range[0], Iterable):
            vmin = cmap_range[0]
            vmax = cmap_range[1]
        else:
            vmin = None
            vmax = None

        if axs is None:
            if im_layout is None:
                xims = int(np.ceil(np.sqrt(len(plot_p))))
                yims = len(plot_p) // xims
            else:
                xims, yims = im_layout
            while xims * yims < len(plot_p):
                yims += 1

            if figsize is None:
                if imsize is not None:
                    figsize = (imsize * xims, imsize * yims)
                else:
                    figsize = self.parent.figsize

            fig, axs = plt.subplots(
                yims,
                xims,
                frameon = False,
                figsize = figsize,
                layout = "constrained",
                squeeze = False
            )
        else:
            fig = axs[0].get_figure()

        defaults = self.single_particle_imshow_defaults.copy()
        for i, ax in enumerate(axs.flatten()):
            try:
                img = plot_imgs[i]
                hdr = plot_hdrs[i]
            except IndexError:
                ax.axis("off")
                continue
            
            defaults.update(**kwargs)

            if not isinstance(cmap_range, Iterable) and cmap_range is not None and percentile_per_class == "per_class":
                defaults["percentile"] = cmap_range
            elif isinstance(cmap_range, Sequence) and isinstance(cmap_range[0], Sequence):
                defaults["percentile"] = None
                vmin = cmap_range[i][0] # type: ignore
                vmax = cmap_range[i][1] # type: ignore
                defaults["vmin"] = vmin
                defaults["vmax"] = vmax
            else:
                defaults["vmin"] = vmin
                defaults["vmax"] = vmax

            self.single_particle_imshow(
                img,
                hdr,
                ax,
                lowpass = None,
                **defaults
            )

        return fig, axs
