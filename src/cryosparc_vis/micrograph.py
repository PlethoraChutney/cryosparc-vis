import numpy as np
from skimage.transform import resize
from cryosparc.tools import lowpass2
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryosparc.project import Project
    from cryosparc.row import Row
    from numpy.typing import NDArray

class UnloadedMicrograph:
    def __init__(self, mic_type) -> None:
        self.mic_type = mic_type

    @property
    def image(self) -> "NDArray":
        raise AttributeError(f"{self.mic_type} not loaded")

    @property
    def shape(self) -> "NDArray":
        raise AttributeError(f"{self.mic_type} not loaded")
    
    def lp_image(self, res) -> "NDArray":
        raise AttributeError(f"{self.mic_type} not loaded")

class MicrographBase:
    """
    Handle loading, filtering, scaling of a micrograph.
    """

    def __init__(
        self,
        project:"Project",
        micrograph_row:"Row",
        downsample_size:int|tuple[int, int]|None = None,
        download_mic:bool = True,
    ) -> None:
        self.project = project
        self.micrograph_row = micrograph_row
        self.blob_field = None
        self.apix = None
        self.downsample_size = downsample_size
        self.download_mic = download_mic
        self.full_image = None

    def load_mic(self):
        self.hdr, self.full_image = self.project.download_mrc(self.project.dir() / self.micrograph_row[self.blob_field]) # type: ignore
        self.full_image = np.squeeze(self.full_image)

    @property
    def image(self) -> "NDArray":
        if self.full_image is not None:
            im = self.full_image.copy()
        else:
            raise AttributeError("Micrograph is not loaded")

        if self.downsample_size is None:
            return im
        
        return resize(im, self.shape)
    
    def lp_image(self, lowpass_cutoff:float = 20.0, order:int = 6) -> "NDArray":
        return lowpass2(self.image, self.apix * self.scaling_factor, lowpass_cutoff, order) # type: ignore

    @image.setter
    def image(self, im:"NDArray") -> None:
        if not isinstance(im, np.ndarray):
            raise ValueError("Image must be a numpy array")
        self._image = np.squeeze(im)

    @property
    def shape(self) -> "NDArray":
        s = self.full_image.shape if self.full_image is not None else self.micrograph_row[self.shape_field] # type: ignore

        if self.downsample_size is None:
            return np.array(s)
        
        aspect_ratio = np.array(s) / np.max(s)
        downsampled_shape = (np.array([self.downsample_size]*2) * aspect_ratio).astype(int)
        return downsampled_shape
    
    @property
    def scaling_factor(self) -> float:
        if self.full_image is None:
            raise AttributeError("Micrograph is not loaded")
        return (np.array(self.full_image.shape) / self.shape)[1]

class RawMicrograph(MicrographBase):
    def __init__(
        self,
        project: "Project",
        micrograph_row: "Row",
        downsample_size: int | tuple[int, int] | None = None,
        download_mic:bool = True,
    ) -> None:
        
        super().__init__(project, micrograph_row, downsample_size, download_mic)
        self.blob_field = "micrograph_blob/path"
        self.shape_field = "micrograph_blob/shape"
        self.apix = micrograph_row["micrograph_blob/psize_A"]
        if self.download_mic:
            self.load_mic()

class DenoisedMicrograph(MicrographBase):
    def __init__(
        self,
        project: "Project",
        micrograph_row: "Row",
        downsample_size: int | tuple[int, int] | None = None,
        download_mic:bool = True,
    ) -> None:
        
        super().__init__(project, micrograph_row, downsample_size, download_mic)
        self.blob_field = "micrograph_blob_denoised/path"
        self.shape_field = "micrograph_blob_denoised/shape"
        self.apix = micrograph_row["micrograph_blob_denoised/psize_A"]
        if self.download_mic:
            self.load_mic()

class JunkAnnotationMicrograph(MicrographBase):
    def __init__(
        self,
        project: "Project",
        micrograph_row: "Row",
        downsample_size: int | tuple[int, int] | None = None,
        download_mic:bool = True,
    ) -> None:
        
        super().__init__(project, micrograph_row, downsample_size, download_mic)
        self.blob_field = "annotation_blob/path"
        self.shape_field = "annotation_blob/shape"
        self.apix = micrograph_row["annotation_blob/psize_A"]
        if self.download_mic:
            self.load_mic()

    def load_mic(self) -> None:
        im = self._image

        if im is None:
            raise AttributeError("Micrograph not loaded")
        
        with tempfile.NamedTemporaryFile(suffix = ".npy") as tmp:
            self.project.download_file(self.micrograph_row[self.blob_field], tmp)
            self.image = np.load(tmp.name)
    