import numpy as np
from skimage.transform import resize
from cryosparc.tools import lowpass2
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Optional, Any
from cryosparc.row import Row

if TYPE_CHECKING:
    from cryosparc.project import Project
    from numpy.typing import NDArray
    import matplotlib.axes
    import matplotlib.figure
    from .vis_dataset import VisDataset

class MicrographBase:
    """
    Handle loading, filtering, scaling of a micrograph.
    """

    def __init__(
        self,
        parent: "VisDataset",
        micrograph_row:"Row|None",
    ) -> None:
        
        self.parent = parent
        self.project = parent.project
        self._micrograph_row = None
        self.micrograph_row = micrograph_row
        self.blob_field:str|None = None
        self.apix_field:str|None = None
        self.full_image:"NDArray[np.float64]|None" = None

    @property
    def micrograph_row(self) -> "Row":
        if self._micrograph_row is None:
            raise AttributeError("Micrograph is not loaded")
        return self._micrograph_row
    
    @micrograph_row.setter
    def micrograph_row(self, r:"Row|None") -> None:
        if r is None or isinstance(r, Row):
            self._micrograph_row = r
        else:
            raise ValueError("Micrograph row must be None or a CryoSPARC Row")
        
    @property
    def apix(self) -> float:
        return self.micrograph_row[self.apix_field] # type: ignore

    def load_mic(self):
        blob_path = self.micrograph_row[self.blob_field] # type: ignore
        self.hdr, self.full_image = self.project.download_mrc(self.project.dir() / blob_path) # type: ignore
        self.full_image = np.squeeze(self.full_image)

    @property
    def image(self) -> "NDArray[np.float64]":
        if self.full_image is not None:
            im = self.full_image.copy()
        else:
            raise AttributeError("Micrograph is not loaded")

        if self.parent.downsample_size is None:
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

        if self.parent.downsample_size is None:
            return np.array(s)
        
        aspect_ratio = np.array(s) / np.max(s)
        downsampled_shape = (np.array([self.parent.downsample_size]*2) * aspect_ratio).astype(int)
        return downsampled_shape
    
    @property
    def scaling_factor(self) -> float:
        if self.full_image is None:
            raise AttributeError("Micrograph is not loaded")
        return (np.array(self.full_image.shape) / self.shape)[1]
    

    # plots

    def plot(
            self,
            ax:Optional["matplotlib.axes.Axes"] = None,
            figsize:Optional[tuple[float, float]] = None,
            percentile:Optional[float] = None,
            lowpass:Optional[float] = None,
            **kwargs
            ) -> None | tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
        
        if ax is None:
            fig, ax = plt.subplots(
                1,
                1,
                frameon = False,
                layout = "constrained",
                figsize = figsize if figsize else self.parent.figsize
            )
        else:
            fig = None

        im = self.image if lowpass is None else self.lp_image(lowpass)
        if percentile is not None:
            p = np.percentile(im, [percentile, 100 - percentile])
            kwargs["vmin"] = np.min(p)
            kwargs["vmax"] = np.max(p)


        defaults:dict[str,Any] = {
            "cmap": "Greys_r",
            "origin": "lower",
            "interpolation": "nearest"
        }
        defaults.update(kwargs)

        ax.imshow(
            im,
            **defaults
        )

        ax.axis("off")
        ax.set_aspect("equal")

        return (fig, ax) if fig is not None else None

class RawMicrograph(MicrographBase):
    def __init__(
        self,
        parent: "VisDataset",
        micrograph_row:"Row|None",
    ) -> None:
        
        super().__init__(parent, micrograph_row)
        self.blob_field = "micrograph_blob/path"
        self.shape_field = "micrograph_blob/shape"
        self.apix_field = "micrograph_blob/psize_A"
        if self.parent.download_mic:
            try:
                self.load_mic()
            except AttributeError:
                pass

class DenoisedMicrograph(MicrographBase):
    def __init__(
        self,
        parent: "VisDataset",
        micrograph_row:"Row|None",
    ) -> None:
        
        super().__init__(parent, micrograph_row)
        self.blob_field = "micrograph_blob_denoised/path"
        self.shape_field = "micrograph_blob_denoised/shape"
        self.apix_field = "micrograph_blob_denoised/psize_A"
        if self.parent.download_mic:
            try:
                self.load_mic()
            except AttributeError:
                pass


    