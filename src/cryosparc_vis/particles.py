from typing import TYPE_CHECKING
from cryosparc.dataset import Dataset
import numpy as np

if TYPE_CHECKING:
    from .vis_dataset import VisDataset
    from numpy.typing import NDArray
    from collections.abc import Callable
    from cryosparc.row import Row

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
    def particles(self) -> "None|Dataset":
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
