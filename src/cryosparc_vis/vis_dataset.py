import cryosparc
from cryosparc.tools import CryoSPARC, lowpass2
import cryosparc.mrc as cmrc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .config import VisConfig
from .micrograph import UnloadedMicrograph, RawMicrograph, DenoisedMicrograph, JunkAnnotationMicrograph
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
        self.config = config
        self.cs = config.cs

        self.project_uid = config.project_uid
        self.project = self.cs.find_project(self.project_uid)

        self.download_mic = config.download_mic
        self.base_mic_juid, self.base_mic_title = config.base_mic if config.base_mic is not None else (None, None)
        self.base_mic = UnloadedMicrograph("Base")
        self.denoised_mic_juid, self.denoised_mic_title = config.denoised_mic if config.denoised_mic is not None else (None, None)
        self.denoised_mic = UnloadedMicrograph("Denoised")
        self.junk_annotation_juid, self.junk_annotation_title = config.junk_annotation_mic if config.junk_annotation_mic is not None else (None, None)
        self.junk_annotation_mic = UnloadedMicrograph("Junk Annotations")
        self.mic_uid = config.mic_uid
        self.mic_index = config.mic_index
        self.downsample_size = config.downsample_size
        
        self.load_mics()

    def load_mics(self) -> None:
        if all(x is None for x in [self.base_mic_juid, self.denoised_mic_juid, self.junk_annotation_juid]):
            return
        
        if self.base_mic_juid is None or self.base_mic_title is None:
            raise AttributeError("If loading micrographs, must specify base mic JUID")
        
        base_job = self.project.find_job(self.base_mic_juid)
        base_mic_info = base_job.load_output(self.base_mic_title)

        if self.mic_uid is not None:
            if self.mic_uid not in base_mic_info["uid"]:
                raise ValueError("UID not in base micrograph dataset")
            base_mic_info = base_mic_info.query({"uid": self.mic_uid})[0]
        else:
            if self.mic_index is None:
                raise AttributeError("Specify micrograph UID or index")
            base_mic_info = base_mic_info[self.mic_index]
            self.mic_uid = int(base_mic_info["uid"])
        
        self.base_mic = RawMicrograph(self.project, base_mic_info, self.downsample_size, self.download_mic)

        if self.denoised_mic_juid is not None:
            denoised_job = self.project.find_job(self.denoised_mic_juid)

            denoised_mic_info = denoised_job.load_output(self.denoised_mic_title).query({"uid": self.mic_uid})[0] #type: ignore
            self.denoised_mic = DenoisedMicrograph(self.project, denoised_mic_info, self.downsample_size, self.download_mic)
        