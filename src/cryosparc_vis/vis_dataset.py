import cryosparc
from cryosparc.tools import CryoSPARC, lowpass2
import cryosparc.mrc as cmrc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .config import VisConfig
from .micrograph import RawMicrograph, DenoisedMicrograph, JunkAnnotations
from .particles import Particles
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cryosparc.dataset


class VisDataset:
    """
    A collection of datasets and micrograph functions to make direct visualization of raw cryoEM
    data easier.

    Attributes
    ----------
    cs: cryosparc.tools.CryoSPARC
        A CryoSPARC object. Used to fetch data from a CS instance.
    project_uid: str
        A project UID. Prefix with P, like "P123".
    """

    def __init__(self, config:VisConfig) -> None:
        self.cs = config.cs

        self.figsize = config.figsize

        self.project_uid = config.project_uid
        self.project = self.cs.find_project(self.project_uid)

        # micrographs

        self._mic_uid = None
        self.download_mic = config.download_mic
        self._downsample_size = None
        self.downsample_size = config.downsample_size

        self.crop_slice = config.crop_slice

        self._base_mic_spec = (None, None)
        self.base_mic = RawMicrograph(self, None)
        self.base_mic_results = None
        self.base_mic_spec = config.base_mic_spec if config.base_mic_spec is not None else (None, None)

        self._denoised_mic_spec = (None, None)
        self.denoised_mic = DenoisedMicrograph(self, None)
        self.denoised_mic_results = None
        self.denoised_mic_spec = config.denoised_mic_spec if config.denoised_mic_spec else (None, None)

        self._junk_annotation_spec = (None, None)
        self.junk_annotations = JunkAnnotations(self, None)
        self.junk_annotations_results = None
        self.junk_annotation_spec = config.junk_annotation_spec if config.junk_annotation_spec else (None, None)


        if config.mic_uid is not None:
            self.mic_uid = config.mic_uid
            if config.mic_index is not None:
                print("WARNING: Ignoring mic index because mic UID was provided")
        elif config.mic_index is not None:
            self.select_mic_index(config.mic_index)

        # particles

        self._particles:Particles = Particles(self)
        self.particles_spec = config.particles_spec

    # ===========================
    #           CONFIG
    # ===========================

    @property
    def config(self) -> VisConfig:
        return VisConfig(
            cs = self.cs,
            project_uid = self.project_uid,
            base_mic_spec = self.base_mic_spec,
            denoised_mic_spec = self.denoised_mic_spec,
            junk_annotation_spec = self.junk_annotation_spec,
            mic_uid = self.mic_uid,
            downsample_size = self.downsample_size,
            crop_slice = None,
            download_mic = self.download_mic,
            particles_spec = self.particles_spec
        )
    
    def copy(self, updates:dict[str,Any] = {}) -> "VisDataset":
        config = self.config
        config.update(updates)
        return VisDataset(config)

    # ===========================
    #         MICROGRAPHS
    # ===========================

    # downsample and crop

    @property
    def downsample_size(self) -> int:
        if isinstance(self._downsample_size, int):
            return self._downsample_size
        elif self.base_mic.full_image is not None:
            return np.max(self.base_mic.full_image.shape)

        raise AttributeError("No downsample size set and base micrograph not loaded.")
        
    @downsample_size.setter
    def downsample_size(self, size:int|None) -> None:
        if size is None:
            self._downsample_size = None
        elif isinstance(size, int):
            self._downsample_size = size
        else:
            raise ValueError("Downsample size must be int or None")

    @property
    def crop_slice(self) -> tuple[float, float, float, float]:
        if self._crop_slice is None:
            return (0., 1., 0., 1.)
        xs, xe, ys, ye = self._crop_slice
        proper_slice = (
            0. if xs is None else xs,
            1. if xe is None else xe,
            0. if ys is None else ys,
            1. if ye is None else ye
        )
        return proper_slice
    
    @crop_slice.setter
    def crop_slice(self, new_slice: None | tuple[float|None, float|None, float|None, float|None] | list[float|None]) -> None:
        if new_slice is None:
            self._crop_slice = None
            return
        if not all(x is None or isinstance(x, (float, int)) for x in new_slice) or len(new_slice) != 4:
            raise ValueError("Slice must be None, or four values, each a float or None")
        if all(x is not None for x in new_slice[:2]) and new_slice[0] > new_slice[1]: # type: ignore
            raise ValueError("First slice component must be less than second slice component")
        if all(x is not None for x in new_slice[2:]) and new_slice[2] > new_slice[3]: # type: ignore
            raise ValueError("Third slice component must be less than fourth slice component")
        self._crop_slice = new_slice


    # mic selection

    @property
    def mic_uid(self) -> int | None:
        return self._mic_uid
    
    @mic_uid.setter
    def mic_uid(self, muid:int|str|None) -> None:
        if self.base_mic_results is None:
            raise AttributeError("Select a base micrograph job before setting the micrograph UID")
        if muid is None:
            self._mic_uid = None
            self.base_mic = RawMicrograph(self, None)
            return
        
        self._mic_uid = int(muid)

        self.load_micrographs()
        try:
            self.particles.update_mic_uid_filter()
        except AttributeError:
            pass
        

    def select_mic_index(self, midx:str|int) -> int:
        if self.base_mic_results is None:
            raise AttributeError("Must load base micrographs to select by index")
        mic_uid = self.base_mic_results[int(midx)]["uid"]
        self.mic_uid = mic_uid
        return mic_uid
    
    def load_micrographs(self, mic_load_list:list[str] = ["base", "denoised", "junk"]) -> None:
        if self.mic_uid is None:
            return
        
        if self.base_mic_results is not None and "base" in mic_load_list:
            self.base_mic = RawMicrograph(
                self,
                self.base_mic_results.query({"uid": self.mic_uid})[0],
            )

        if self.denoised_mic_results is not None and "denoised" in mic_load_list:
            self.denoised_mic = DenoisedMicrograph(
                self,
                self.denoised_mic_results.query({"uid": self.mic_uid})[0],
            )

        if self.junk_annotations_results is not None and "junk" in mic_load_list:
            self.junk_annotations = JunkAnnotations(
                self,
                self.junk_annotations_results.query({"uid": self.mic_uid})[0],
            )

    # mic spec getters and setters

    @property
    def base_mic_spec(self) -> tuple[str|None, str|None]:
        return self._base_mic_spec
    
    @base_mic_spec.setter
    def base_mic_spec(self, mic_spec:tuple[str|None, str|None]) -> None:
        if not isinstance(mic_spec, tuple):
            raise ValueError("Micrograph spec must be a tuple")
        
        self._base_mic_spec = mic_spec
        job_uid, title = mic_spec
        
        if job_uid is None:
            self.base_mic = RawMicrograph(self, None)
            self.base_mic_results = None
            return
        
        if title is None:
            raise ValueError("If loading micrographs, specify the result title")
        
        self.base_mic_results = self.project.find_job(job_uid).load_output(title)
        if self.mic_uid is not None:
            self.load_micrographs(["base"])


    @property
    def denoised_mic_spec(self) -> tuple[str|None, str|None]:
        return self._denoised_mic_spec
    
    @denoised_mic_spec.setter
    def denoised_mic_spec(self, mic_spec:tuple[str|None, str|None]) -> None:
        if not isinstance(mic_spec, tuple):
            raise ValueError("Micrograph spec must be a tuple")
        
        self._denoised_mic_spec = mic_spec
        job_uid, title = mic_spec

        if job_uid is None:
            self.denoised_mic = DenoisedMicrograph(self, None)
            self.denoised_mic_results = None
            return
        
        if title is None:
            if self.base_mic_spec[1] is None:
                raise ValueError("Specify a base or denoised micrograph title")
            else:
                print("INFO: Using base mic title for denoised mics")
                title = self.base_mic_spec[1]
        
        self.denoised_mic_results = self.project.find_job(job_uid).load_output(title)
        if self.mic_uid is not None:
            self.load_micrographs(["denoised"])

    @property
    def junk_annotation_spec(self) -> tuple[str|None, str|None]:
        return self._junk_annotation_spec
    
    @junk_annotation_spec.setter
    def junk_annotation_spec(self, junk_spec:tuple[str|None, str|None]) -> None:
        if not isinstance(junk_spec, tuple):
            raise ValueError("Micrograph spec must be a tuple")
        
        self._denoised_mic_spec = junk_spec
        job_uid, title = junk_spec

        if job_uid is None:
            self.denoised_mic = DenoisedMicrograph(self, None)
            self.denoised_mic_results = None
            return
        
        if title is None:
            if self.base_mic_spec[1] is None:
                raise ValueError("Specify a base or denoised micrograph title")
            else:
                print("INFO: Using base mic title for denoised mics")
                title = self.base_mic_spec[1]
        
        self.junk_annotations_results = self.project.find_job(job_uid).load_output(title)
        if self.mic_uid is not None:
            self.load_micrographs(["junk"])

    # ===========================
    #          PARTICLES
    # ===========================

    @property
    def particles_spec(self) -> tuple[str|None, str|None]:
        try:
            return self._particles_spec
        except AttributeError:
            return (None, None)
    
    @particles_spec.setter
    def particles_spec(self, new_spec:None|tuple[str|None, str|None]) -> None:
        if new_spec is None:
            new_spec = (None, None)
        self._particles_spec = new_spec
        self.particles_job = None if new_spec[0] is None else self.project.find_job(new_spec[0])
        self.particles = Particles(self)

    @property
    def particles(self) -> Particles:
        return self._particles
    
    @particles.setter
    def particles(self, new_particles) -> None:
        if not isinstance(new_particles, Particles):
            raise ValueError("New particles must be a Particles object. Use particles_spec to replace it.")
        self._particles = new_particles