
################################################

#!/usr/bin/env python
#!/usr/bin/env python
import types, os, sys, toml, numpy as np, pandas as pd

# ————————————————————————————————————————————————————————————————
# 1) Monkey-patch napari so patho_sam can import it
# ————————————————————————————————————————————————————————————————
def _dummy_progress(it, **kw): return it
_napari = types.ModuleType("napari")
_utils  = types.ModuleType("napari.utils")
_utils.progress = _dummy_progress
_napari.utils   = _utils
sys.modules["napari"]       = _napari
sys.modules["napari.utils"] = _utils

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

# ————————————————————————————————————————————————————————————————
# 2) Monkey-patch micro_sam to drop any 4th channel
# ————————————————————————————————————————————————————————————————
try:
    import micro_sam.util as _msu
    _orig_to_image = _msu._to_image
    def _to_image_strip_alpha(arr):
        # if RGBA, just take RGB
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        return _orig_to_image(arr)
    _msu._to_image = _to_image_strip_alpha
except ModuleNotFoundError:
    # micro_sam not yet installed – will hit the next error below
    pass

# ————————————————————————————————————————————————————————————————
# 3) Now import PathoSAM and run segmentation as before
# ————————————————————————————————————————————————————————————————
from patho_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
import imageio.v3 as iio
# from Utils.SAM_utils import get_bbox_from_mask
import openslide
# from Utils.SAM_utils import get_bbox_from_mask, remove_edge_masks
# import torch
# from torchvision.ops import nms
import time
# ————————————————————————————————————————————————————————————————

import tifffile
from PIL import Image
import numpy as np
import cv2
import openslide


def get_thumbnail_image(wsi_path,out_prefix, thumb_size):
    try:
        slide = openslide.OpenSlide(wsi_path)
        img = slide.get_thumbnail(thumb_size)
        
    except openslide.OpenSlideUnsupportedFormatError:
        # fallback for plain TIFFs or other unsupported formats
        img = Image.open(wsi_path)
        img = img.convert("RGB")
        img.thumbnail(thumb_size, resample=Image.BILINEAR)
   
    img.save(f"{out_prefix}_thumb.png")
    print(f"✅ Saved thumbnail → {out_prefix}_thumb.png")
    return img

def save_masked_thumbnail(wsi_path, inst_map, out_prefix,
                           thumb_size=(1024,1024), alpha=0.5, seed=42):
     slide_thumb = get_thumbnail_image(wsi_path, out_prefix, thumb_size)
     bg = np.array(slide_thumb, dtype=np.uint8)

     # ensure mask is downsampled to bg’s true shape, not always thumb_size
     h, w = bg.shape[:2]
     inst_thumb = cv2.resize(
         inst_map.astype(np.int32),
        dsize=(w, h),
         interpolation=cv2.INTER_NEAREST
     )

     rng = np.random.RandomState(seed)
     pids = np.unique(inst_thumb)
     colormap = {int(pid): rng.randint(0,256,3, dtype=np.uint8)
                 for pid in pids if pid != 0}
     mask_rgb = np.zeros_like(bg)
     for pid, color in colormap.items():
         mask_rgb[inst_thumb == pid] = color

     overlaid = (bg*(1-alpha) + mask_rgb*alpha).astype(np.uint8)
     Image.fromarray(overlaid).save(f"{out_prefix}_thumbnail.png")
     print("✅ Saved thumbnail →", f"{out_prefix}_thumbnail.png")

def segment_wsi(cfg_path, wsi_path, out_prefix):
    cfg = toml.load(cfg_path)
    
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"Input WSI not found: {wsi_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    if os.path.exists(f"{out_prefix}_instances.tif"):
        print(f"→ Instance map already exists: {out_prefix}_instances.tif")
        # inst_map= np.load(f"{out_prefix}_instances.npy")
        inst_map = tifffile.imread(f"{out_prefix}_instances.tif")
        print("→ Skipping instance segmentation, using existing map.")

    else:
        # 1) build PathoSAM once
        predictor, segmenter = get_predictor_and_segmenter(
            model_type = cfg['SAM_MODEL']['Model_Type'],
            checkpoint =  None,
            device     = cfg['SAM_MODEL'].get('device', 'cuda'),
            amg        = False,
            is_tiled   = True,
        )
        min_a = cfg['SAM_MODEL']['min_mask_region_area']
        halo   = tuple(cfg['SAM_MODEL']['halo'])
        tile_shape = tuple(cfg['SAM_MODEL']['tile_shape'])
        batch_size     = cfg['SAM_MODEL']['batch_size']
       
        print("→ Loaded PathoSAM predictor+segmenter")
       
        start_time = time.time()
        # 2) run full-slide instance segmentation
        inst_map = automatic_instance_segmentation(
            predictor      = predictor,
            segmenter      = segmenter,
            input_path     = wsi_path,
            tile_shape     = tile_shape,
            halo           = halo,
            batch_size     = batch_size,
            ndim           = 2,
            output_mode    = None,
            verbose        = cfg['SAM_MODEL'].get('verbose', False),
            min_size       = min_a,
       
            return_embeddings=False
        )
        print(f"→ Instance segmentation done in {time.time()- start_time} seconds, with shape: {inst_map.shape}")
       
              
        # 4) save the (filtered) raw integer map
        np.save(f"{out_prefix}_instances.npy", inst_map)
        print("✅ Saved instance map →", f"{out_prefix}_instances.npy")
        # optional: also write a GeoTIFF or .tif with the same array
        # keep the raw inst_map, even if it has IDs >2³²−1
        # 2) write out a tiled, zlib-compressed TIFF
        tifffile.imwrite(
            f"{out_prefix}_instances.tif",
            inst_map.astype(np.uint32),         # save as unsigned 32-bit
            tile=(256, 256),                    # break into 256×256 tiles on disk
            compression='zlib',                 # other options: 'jpeg', 'lzw', etc.
            photometric='minisblack',           # grayscale “palette”
            metadata={'axes': 'YX'}             # for clarity, optional
        )
        print(f"✅ Saved tiled GeoTIFF → {out_prefix}_instances.tif")
        save_masked_thumbnail(wsi_path, inst_map, out_prefix,
                              thumb_size=(1024,1024), alpha=0.5)
 
    if os.path.exists(f"{out_prefix}_instances.csv"):
        print(f"→ Instance table already exists: {out_prefix}_instances.csv")
        pass
    else:

        
        from skimage.measure import regionprops_table
        
        props = regionprops_table(
            inst_map,
            properties=('label','bbox','centroid'),
            cache=True
        )
        df = pd.DataFrame({
            'pid':        props['label'],
            'xmin':       props['bbox-1'],
            'ymin':       props['bbox-0'],
            'xmax':       props['bbox-3'],
            'ymax':       props['bbox-2'],
            'centroid_x': props['centroid-1'],
            'centroid_y': props['centroid-0'],
        })
        df.to_csv(f"{out_prefix}_instances.csv", index=False)
      
        print("✅ Saved instance table →", f"{out_prefix}_instances.csv")
    
    save_masked_thumbnail(wsi_path, inst_map, out_prefix,
                              thumb_size=(2048,2048), alpha=0.5)

    # get_thumbnail_image(wsi_path,out_prefix, thumb_size=(1024,1024))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: segment_wsi.py <config.toml> <input.svs> <output_prefix>")
        
        sys.exit(1)
    _, cfg, wsi, prefix = sys.argv
    segment_wsi(cfg, wsi, prefix)
