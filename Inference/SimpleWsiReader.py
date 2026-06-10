# SimpleWsiReader.py
import os
import numpy as np
from PIL import Image

try:
    from cucim import CuImage
    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

import tifffile  # <<< robust TIFF reader (uses imagecodecs if available)

class BaseWSI:
    def read_region(self, location, size, level=0):
        raise NotImplementedError
    def get_thumbnail(self, thumb_size):
        raise NotImplementedError

from typing import Optional, Literal
WSIReaderType = Literal['openslide', 'image', 'cucim', 'tifffile']

OPENSLIDE_EXTENSIONS = {'.svs', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs'}  # WSI formats
TIFF_EXTENSIONS      = {'.tif', '.tiff'}
CUCIM_EXTENSIONS     = {'.svs', '.tif', '.tiff'}

class CuCIMWSI(BaseWSI):
    def __init__(self, path):
        self.img = CuImage(path)
        infos = self.img.resolutions
        self.resolutions = {
            "level_count": infos["level_count"],
            "level_dimensions": infos["level_dimensions"]
        }
    def read_region(self, location, size, level=0):
        arr = self.img.read_region(location=location, size=size, level=level, mode="rgb", dtype="uint8")
        arr = np.asarray(arr)
        return arr[..., :3] if arr.shape[-1] == 4 else arr
    def get_thumbnail(self, thumb_size):
        lvl = self.img.resolutions["level_count"] - 1
        W, H = self.img.resolutions["level_dimensions"][lvl]
        arr = self.img.read_region(location=(0, 0), size=(W, H), level=lvl, mode="rgb", dtype="uint8")
        pil = Image.fromarray(np.asarray(arr))
        pil.thumbnail(thumb_size, Image.BILINEAR)
        return pil

class OpenSlideWSI(BaseWSI):
    def __init__(self, path):
        self.img = openslide.OpenSlide(path)
    def read_region(self, location, size, level=0):
        patch = self.img.read_region(location, level, size)  # PIL Image RGBA
        arr = np.asarray(patch)
        return arr[..., :3] if arr.shape[-1] == 4 else arr
    def get_thumbnail(self, thumb_size):
        return self.img.get_thumbnail(thumb_size)

class TiffFileWSI(BaseWSI):
    """
    Robust small-TIFF reader:
    - loads the whole image once with tifffile (no Pillow/libtiff)
    - implements read_region by numpy slicing
    """
    def __init__(self, path):
        arr = tifffile.imread(path)  # H×W×C or H×W
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)           # grayscale → RGB
        if arr.shape[-1] == 4:
            arr = arr[..., :3]                         # drop alpha
        self.arr = arr.astype(np.uint8, copy=False)
        H, W, _ = self.arr.shape
        self.resolutions = {"level_count": 1, "level_dimensions": [(W, H)]}
    def read_region(self, location, size, level=0):
        x, y = int(location[0]), int(location[1])      # location = (x,y)
        w, h = int(size[0]), int(size[1])
        H, W, _ = self.arr.shape
        # clamp box to image bounds
        x0 = max(0, min(x, W))
        y0 = max(0, min(y, H))
        x1 = max(x0, min(x + w, W))
        y1 = max(y0, min(y + h, H))
        patch = np.zeros((h, w, 3), dtype=self.arr.dtype)
        src = self.arr[y0:y1, x0:x1]
        ph, pw = src.shape[:2]
        patch[:ph, :pw] = src
        return patch
    def get_thumbnail(self, thumb_size):
        pil = Image.fromarray(self.arr)
        pil.thumbnail(thumb_size, Image.BILINEAR)
        return pil

class PILWSI(BaseWSI):
    """Fallback for random flat images (PNG/JPEG). Avoid for TIFF."""
    def __init__(self, path):
        self.path = path
        self.img = Image.open(self.path).convert("RGB")
        self.arr = np.array(self.img)
        H, W, _ = self.arr.shape
        self.resolutions = {"level_count": 1, "level_dimensions": [(W, H)]}
    def read_region(self, location, size, level=0):
        x, y = int(location[0]), int(location[1])
        w, h = int(size[0]), int(size[1])
        H, W, _ = self.arr.shape
        x0 = max(0, min(x, W)); y0 = max(0, min(y, H))
        x1 = max(x0, min(x + w, W)); y1 = max(y0, min(y + h, H))
        patch = np.zeros((h, w, 3), dtype=self.arr.dtype)
        src = self.arr[y0:y1, x0:x1]
        ph, pw = src.shape[:2]
        patch[:ph, :pw] = src
        return patch
    def get_thumbnail(self, thumb_size):
        pil = Image.fromarray(self.arr)
        pil.thumbnail(thumb_size, Image.BILINEAR)
        return pil

def _is_tiff_with_pyramid(path: str) -> bool:
    """Quick probe: returns True if TIFF has multiple levels (WSI-like)."""
    try:
        with tifffile.TiffFile(path) as tf:
            # pyramidal tiffs usually have multiple pages or subifds
            return len(tf.pages) > 1 or any(len(p.subifds) > 0 for p in tf.pages if hasattr(p, "subifds"))
    except Exception:
        return False

def make_wsi(path: str, reader_type: Optional[WSIReaderType] = None) -> BaseWSI:
    ext = os.path.splitext(path)[1].lower()

    # explicit selection
    if reader_type is not None:
        if reader_type == 'cucim':
            if not CUCIM_AVAILABLE: raise ValueError("cuCIM requested but not installed")
            return CuCIMWSI(path)
        if reader_type == 'openslide':
            if not OPENSLIDE_AVAILABLE: raise ValueError("OpenSlide requested but not installed")
            return OpenSlideWSI(path)
        if reader_type == 'tifffile':
            return TiffFileWSI(path)
        if reader_type == 'image':
            return PILWSI(path)
        raise ValueError(f"Unknown reader_type: {reader_type!r}")

    # auto-detect
    if ext in OPENSLIDE_EXTENSIONS and OPENSLIDE_AVAILABLE:
        return OpenSlideWSI(path)

    if ext in TIFF_EXTENSIONS:
        # prefer cuCIM for pyramidal tiffs, else TiffFile for flat tiffs
        if CUCIM_AVAILABLE and _is_tiff_with_pyramid(path):
            return CuCIMWSI(path)
        return TiffFileWSI(path)   # <<< avoids Pillow/libtiff errors

    # fallback for PNG/JPG, etc.
    return PILWSI(path)
