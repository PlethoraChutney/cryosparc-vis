from typing import TYPE_CHECKING, Optional
from cryosparc.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    from .vis_dataset import VisDataset
    from numpy.typing import NDArray
    from collections.abc import Callable
    from cryosparc.row import Row
    from cryosparc.column import Column

class Particles:
    def __init__(self, parent:"VisDataset") -> None:
        self.parent = parent
        spec = self.parent.particles_spec
        self._particles:"Dataset|None"
        self.bool_masks:dict[str, "NDArray[np.bool_]"] = {}
        self.function_masks:dict[str, "Callable[[Row], bool]"] = {}

        if spec[0] is None:
            self.job = None
            self.particles = None
        else:
            self.job = self.parent.project.find_job(spec[0])
            self.particles = self.job.load_output("particles" if spec[1] is None else spec[1])
            self.bool_masks["mic_uid"] = self.unmasked_particles["location/micrograph_uid"] == self.parent.mic_uid

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
    def particles(self) -> "Dataset":
        bool_mask = np.all(list(self.bool_masks.values()), axis = 0)
        p = self.unmasked_particles.mask(bool_mask)
        for f_mask in self.function_masks.values():
            p = p.query(f_mask)

        return p
    
    @particles.setter
    def particles(self, dset:"None|Dataset") -> None:
        if dset is not None and not isinstance(dset, Dataset):
            raise ValueError("Particles must be a CryoSPARC Dataset")
        self._particles = dset

    def update_mic_uid_filter(self) -> None:
        if self.parent.mic_uid is None or self._particles is None:
            self.remove_mask("mic_uid")
        else:
            self.bool_masks["mic_uid"] = self.unmasked_particles["location/micrograph_uid"] == self.parent.mic_uid

    @property
    def fields(self) -> list[str]:
        return self.particles.fields()
    
    def fields_prefix(self, prefix:str) -> list[str]:
        return list(f for f in self.fields if f.startswith(prefix))

    # plotting

    @property
    def particles_x(self) -> "Column":
        return self.particles["location/center_x_frac"]
    
    @property
    def particles_y(self) -> "Column":
        return self.particles["location/center_y_frac"]
    
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
        legend:Optional[bool] = True,
        **kwargs,
    ) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
        if ax is None:
            fig, ax = plt.subplots(
                1,
                1,
                frameon = False,
                layout = "constrained",
                figsize = figsize if figsize else self.parent.figsize
            )
        else:
            fig = ax.get_figure()

        defaults = self.__class__.plot_defaults.copy()
        defaults.update(**kwargs)

        if color_by is not None:
            del defaults["facecolor"]

        scatter = ax.scatter(
            self.particles_x,
            self.particles_y,
            c = self.particles[color_by] if color_by is not None else None,
            **defaults
        )
        if color_by is not None and legend:
            ax.add_artist(
                ax.legend(*scatter.legend_elements(), loc = "upper right", title = color_by)
            )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0)

        
        
        return fig, ax # type: ignore
        


