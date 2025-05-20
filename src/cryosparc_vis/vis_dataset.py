import cryosparc
from cryosparc.tools import CryoSPARC, lowpass2
import cryosparc.mrc as cmrc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .config import VisConfig
from .micrograph import RawMicrograph, DenoisedMicrograph, JunkAnnotations
from typing import TYPE_CHECKING

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

        self.all_mics = [self.base_mic]

    # ===========================
    #         MICROGRAPHS
    # ===========================

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
            self.base_mic = UnloadedMicrograph("Base")
            return
        
        self._mic_uid = int(muid)

        self.load_micrographs()
        

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