from typing import TYPE_CHECKING, Optional, Literal, Any
from cryosparc.tools import lowpass2
from cryosparc.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt

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
    
    plot_single_particle_defaults:dict[str, Any] = {
        "cmap": "Grays_r",
        "origin": "lower",
        "interpolation": "nearest"
    }

        
    def plot_single_particle(
            self,
            ax:Optional["matplotlib.axes.Axes"] = None,
            figsize:tuple[float, float] | Literal["pixel_perfect"] = (2, 2),
            particle_uid:Optional[int] = None,
            particle_index:Optional[int] = None,
            index_from:Optional[Literal["unmasked", "masked"]] = "masked",
            lowpass:Optional[float] = 20,
            **kwargs,
            ) -> tuple["matplotlib.figure.Figure | None", "matplotlib.axes.Axes"]:

        if particle_uid is None:
            if particle_index is None:
                raise ValueError("Must specify a UID or a particle index")
            if index_from == "masked":
                particle_uid = self.particles["uid"][particle_index]
            elif index_from == "unmasked":
                particle_uid = self.unmasked_particles["uid"][particle_index]
            else:
                raise ValueError("Invalid value for index_from")
        
        hdr, img = self.particle_image(particle_uid) # type: ignore idk why it can't figure out the block above

        if lowpass is not None:
            img = lowpass2(img, hdr.xlen / hdr.nx, lowpass)

        defaults = self.plot_single_particle_defaults.copy()
        defaults.update(**kwargs)


        if ax is None:
            if figsize == "pixel_perfect":
                figsize = (img.shape[1] / 100, img.shape[0] / 100)
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
