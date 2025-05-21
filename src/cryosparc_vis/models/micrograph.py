import numpy as np
from skimage.transform import resize
from cryosparc.tools import lowpass2
import tempfile
import matplotlib.colors
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Optional, Any
from cryosparc.row import Row

if TYPE_CHECKING:
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
        self.blob_field:str = None # type: ignore
        self.apix_field:str = None # type: ignore
        self.full_image:"NDArray[np.float64]|None" = None
        self.anti_alias_resize = True
        self.mic_type:str = "Base"

        self.plot_defaults:dict[str,Any] = {
            "cmap": "Greys_r",
            "origin": "lower",
            "interpolation": "nearest"
        }

    def __repr__(self) -> str:
        if self._micrograph_row is None:
            return f"Unloaded {self.mic_type} micrograph"
        else:
            return f"{self.parent.project_uid} {self.mic_type} micrograph UID {self.parent.mic_uid}"

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
        return self.micrograph_row[self.apix_field]

    def load_mic(self):
        blob_path = self.micrograph_row[self.blob_field]
        self.hdr, self.full_image = self.project.download_mrc(self.project.dir() / blob_path)
        self.image = np.squeeze(self.full_image)

    @property
    def image(self) -> "NDArray[np.float64]":
        if self.full_image is not None:
            im = self.full_image.copy()
        else:
            raise AttributeError("Micrograph is not loaded")
        
        im = resize(im, self.shape, anti_aliasing = self.anti_alias_resize, order = 1 if self.anti_alias_resize else 0)

        ysize, xsize = im.shape
        xs, xe, ys, ye = self.parent.crop_slice
        c = [
            int(ys * ysize),
            int(-1 + ye * ysize),
            int(xs * xsize),
            int(-1 + xe * xsize),
        ]
        return im[c[0]:c[1], c[2]:c[3]]
    
    def lp_image(self, lowpass_cutoff:float = 20.0, order:int = 6) -> "NDArray":
        return lowpass2(self.image, self.apix * self.scaling_factor, lowpass_cutoff, order) # type: ignore

    @image.setter
    def image(self, im:"NDArray") -> None:
        if not isinstance(im, np.ndarray):
            raise ValueError("Image must be a numpy array")
        self.full_image = np.squeeze(im)

    @property
    def shape(self) -> "NDArray":
        s = self.full_image.shape if self.full_image is not None else self.micrograph_row[self.shape_field] # type: ignore        
        aspect_ratio = np.array(s) / np.max(s)
        downsampled_shape = (np.array([self.parent.downsample_size]*2) * aspect_ratio).astype(int)
        return downsampled_shape
    
    @property
    def cropped_shape(self) -> "NDArray":
        s = self.shape
        xs, xe, ys, ye = self.parent.crop_slice
        xscale = xe - xs
        yscale = ye - ys
        return np.array([s[0] * yscale, s[1] * xscale])
    
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
            ) -> tuple["matplotlib.figure.Figure | None", "matplotlib.axes.Axes"]:
        
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

        im = self.image if lowpass is None else self.lp_image(lowpass)
        if percentile is not None:
            p = np.percentile(im, [percentile, 100 - percentile])
            kwargs["vmin"] = np.min(p)
            kwargs["vmax"] = np.max(p)

        defaults = self.plot_defaults.copy()
        defaults.update(kwargs)

        ax.imshow(
            im,
            **defaults
        )

        ax.axis("off")
        ax.set_aspect("equal")

        return fig, ax

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
        self.mic_type = "Raw"
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
        self.mic_type = "Denoised"
        if self.parent.download_mic:
            try:
                self.load_mic()
            except AttributeError:
                pass


class JunkAnnotations(MicrographBase):
    def __init__(
        self,
        parent: "VisDataset",
        micrograph_row:"Row|None"
    ) -> None:
        
        super().__init__(parent, micrograph_row)
        self.blob_field = "annotation_blob/path"
        self.shape_field = "annotation_blob/shape"
        self.apix_field = "annotation_blob/psize_A"
        self.mic_type = "Junk Annotations"
        self.anti_alias_resize = False
        self.plot_defaults.update({
            "alpha": 0.4,
            "cmap": self.annotation_cmap,
            "vmin": 0,
            "vmax": 5
        })
        if self.parent.download_mic:
            try:
                self.load_mic()
            except AttributeError:
                pass

    annotation_cmap = matplotlib.colors.ListedColormap(
        colors = [
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0]
        ],
        name = "cs_junk",
    )
        
    def load_mic(self) -> None:
        try:
            blob_path = self.micrograph_row["annotation_blob/path"]
        except AttributeError:
            self.mic_annotations = None
            return
        
        with tempfile.NamedTemporaryFile(suffix = ".npy") as tmp:
            self.parent.project.download_file(blob_path, tmp)
            self.image = np.squeeze(np.load(tmp.name))