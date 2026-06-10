

#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import toml
import numpy as np
import pandas as pd
import torch
import openslide
from PIL import Image, ImageDraw
from Models.Classifier import Classifier
from torch.utils.data import DataLoader
import lightning as L
from torch.utils.data import Dataset
import torch.nn.functional as F


from PIL import Image
from scipy.spatial import cKDTree
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


import logging
logging.getLogger("cucim").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message="In the future `np.bool` will be defined as the corresponding NumPy scalar.")

# at top of your script
from monai.data.wsi_reader import WSIReader
import pandas as pd

import tifffile
# cache for all your slide’s patch origins
_ANNOT_DF = None

import os
import nrrd
import torchvision.transforms as transforms
from Utils.constants import get_constants
import torchvision.transforms as T



#################################

def set_precision(config,  backbone: str) -> torch.dtype:
    """
    Given the name of a backbone (e.g. config['BASEMODEL']['Backbone']), 
    return the torch.dtype it expects for its forward pass.
    """
    bb = backbone.lower()
    # everything here comes from the `precision = ...` lines in each _build()
    if bb in ("conch_v1",):
        return torch.float32
    if bb in ("conch_v15",):
        return torch.float16

    if bb in ("uni_v1",):
        return torch.float16
    if bb in ("uni_v2",):
        return torch.bfloat16

    if bb in ("ctranspath",):
        return torch.float32

    if bb in ("phikon",):
        return torch.float32
    if bb in ("phikon_v2",):
        return torch.float32

    if bb in ("resnet50",):
        return torch.float32

    if bb in ("gigapath",):
        return torch.float32

    if bb in ("virchow", "virchow2"):
        return torch.float16

    if bb in ("hoptimus0", "hoptimus1"):
        return torch.float16

    if bb in ("musk",):
        return torch.float16

    if bb in ("hibou_l",):
        return torch.float32

    if bb.startswith("kaiko-"):
        # all Kaiko variants use float32
        return torch.float32

    if bb in ("lunit-vits8",):
        return torch.float32

    if bb in ("midnight12k",):
        return torch.float32
    
    else:
        return config["BASEMODEL"].get("precision", torch.float16)

def get_transforms(config):
    bb = config['BASEMODEL']['Backbone'].lower()

    # ------------------------------------------------------------------ #
    # model-specific **normalisation / resizing**
    # ------------------------------------------------------------------ #
    if bb == "conch_v1":      
        # # ---- CONCH ----
        # print("Using conch's preprocessing Pipeline! ")
        # tf_train = _conch_preprocess()
        # tf_val   = tf_train


        print("Using costumized preprocessing conch_v1 pipeline!")
        # size = tuple(config['DATA']['Input_Size'])
        size = tuple((224, 224))
        mean = (0.48145466, 0.4578275, 0.40821073)
        std  = (0.26862954, 0.26130258, 0.27577711)
       
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return  tf_val
    
    elif bb.startswith("conch_v15"):
        print("Using costumized preprocessing for conch_v15 pipeline!")
        # size = tuple(config['DATA']['Input_Size'])
        size = tuple((448, 448))
        mean, std = get_constants('imagenet')
       
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return  tf_val
    
    elif bb.startswith("virchow"):
        print("Using Virchow/Virchow2 preprocessing pipeline")
        size = (224, 224)
        # ImageNet normalization (same as your eval_transform)
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        # train transforms 

        # val / test transforms
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return  tf_val

    # ——— UNI v1 branch ———
    if bb.startswith("uni"):
        print("Using UNI‐v1/UNI-v2 preprocessing pipeline")
        size = (224, 224)
        # ImageNet normalization (same as your eval_transform)
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        # val / test transforms
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return  tf_val
    
    if bb.startswith("hoptimus0"):
        # ---- Hoptimus0 ----
        print("Using Hoptimus0 preprocessing pipeline")
        size = (224, 224)
        mean = (0.707223, 0.578729, 0.703617)
        std  = (0.211883, 0.230117, 0.177517)

        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return tf_val
 
    if bb.startswith("hoptimus1"):
        # ---- Hoptimus1 ----
        print("Using Hoptimus1 preprocessing pipeline")
        size = (224, 224)
        mean = (0.707223, 0.578729, 0.703617)
        std  = (0.211883, 0.230117, 0.177517)

        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return  tf_val
    
    elif bb.startswith("hibou_l"):
        # ---- HIBOU-L ----
        print("Using HIBOU-L preprocessing pipeline")
        size = (224, 224)
        mean, std = get_constants('hibou')

        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC,max_size=None, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return tf_val    
    
    elif bb.startswith("midnight12k"):
        # ---- Midnight12k ----
        print("Using Midnight12k preprocessing pipeline")
        from Utils.constants import KAIKO_MEAN, KAIKO_STD
        size = (224, 224)

        mean, std = KAIKO_MEAN, KAIKO_STD

        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return tf_val
    

######################


def get_patch_origin(nrrd_file: str, config: dict) -> (int,int):
    """
    Given a patch filename like "MIDOG_006_3.nrrd", read its NRRD header
    (via the same custom_field_map you used at segmentation time)
    and return its top-left coordinate in the WSI reference frame.
    """
    # where your NRRD patches live
    nrrd_dir = config['DATA']['nrrd_path']
    full_path = os.path.join(nrrd_dir, nrrd_file)
    # read header only; data can be large so we drop it afterwards
    data, header = nrrd.read(full_path, custom_field_map={
        'top_left':'int list',
        'center':'int list',
        'dim':'int list',
        'vis_level':'int',
        'diagnosis':'string',
        'annotation_label':'string',
        'mask':'double matrix'
    })
    # header['top_left'] is e.g. [x0, y0]
    x0, y0 = header['top_left']
    return int(x0), int(y0)


def read_patch(reader, x0, y0, size, wsi_path):
    """
    Read a size×size RGB patch from a WSI using the provided reader.
    Never falls back to PIL on WSIs (prevents libtiff TIFFFillStrip errors).
    """
    x0 = int(x0); y0 = int(y0); size = int(size)

    # Try common .read_region signatures
    for call in (
        lambda: reader.read_region((x0, y0), (size, size)),
        lambda: reader.read_region(location=(x0, y0), size=(size, size)),
    ):
        try:
            arr = call()
            if arr is None:
                continue
            # cuCIM returns ndarray; OpenSlide returns PIL.Image
            if isinstance(arr, Image.Image):
                img = np.asarray(arr.convert("RGB"))
            else:
                img = np.asarray(arr)[..., :3]
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            return img
        except Exception:
            pass

    # Last resort: use OpenSlide explicitly (if not already)
    try:
        slide = openslide.OpenSlide(wsi_path)
        pil = slide.read_region((x0, y0), 0, (size, size)).convert("RGB")
        return np.asarray(pil)
    except Exception as e:
        # Only *now* consider PIL for truly small, non‑WSI files
        try:
            pil = Image.open(wsi_path)
            if max(pil.size) <= 8192:   # heuristic: small non‑WSI
                pil = pil.convert("RGB").crop((x0, y0, x0 + size, y0 + size))
                return np.asarray(pil)
        except Exception:
            pass
        raise RuntimeError(f"Could not read patch ({x0},{y0},{size}) from {wsi_path}") from e


def get_thumbnail_image(wsi_path, out_prefix, thumb_size):
    # Prefer OpenSlide; if that fails, try cuCIM; avoid PIL for WSIs
    try:
        slide = openslide.OpenSlide(wsi_path)
        img = slide.get_thumbnail(thumb_size).convert("RGB")
    except Exception:
        try:
            from cucim import CuImage
            cu = CuImage(wsi_path)
            # Get a downsampled view using cuCIM (level selection)
            # Use the largest level whose size fits thumb_size aspect
            levels = cu.resolutions["level_dimensions"]
            # pick the coarsest level
            lvl = len(levels) - 1
            arr = cu.read_region(location=(0, 0), size=levels[lvl], level=lvl)
            pil = Image.fromarray(np.asarray(arr)[..., :3])
            pil.thumbnail(thumb_size, resample=Image.BILINEAR)
            img = pil
        except Exception:
            # Only for small, non‑WSI images
            pil = Image.open(wsi_path).convert("RGB")
            pil.thumbnail(thumb_size, resample=Image.BILINEAR)
            img = pil

    img.save(f"{out_prefix}_thumb.png")
    print(f"✅ Saved thumbnail → {out_prefix}_thumb.png")
    return img

class NucleusDataset(Dataset):
    def __init__(self, reader, svs_path, size, df, inst_map, transform):
        self.cu_reader = reader
        self.svs_path  = svs_path
        self.df       = df
        self.map      = inst_map
        self.tform    = transform
        self.size     = size
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # use integer centroids (make sure df.centroid_x/y are ints, or cast here)
        cx0, cy0 = int(row.centroid_x), int(row.centroid_y)
        p        = self.size // 2
        W, H     = self.map.shape[1], self.map.shape[0]
    
        # clamp to [0 .. W-size] and [0 .. H-size]
        x0 = int(max(0, min(cx0 - p, W - self.size)))
        y0 = int(max(0, min(cy0 - p, H - self.size)))
        
        # print(f"Extracting patch {i+1}/{len(self.df)}: ({x0}, {y0}) size={self.size}x{self.size} for PID {row.pid}")
        # read RGB patch
        rgb = read_patch(self.cu_reader, x0, y0, self.size, self.svs_path)
    
        # extract the binary mask
        msk = (self.map[y0 : y0 + self.size, x0 : x0 + self.size] == row.pid).astype(np.float32)
    
        img_t = self.tform(Image.fromarray(rgb))

        # msk_t = torch.from_numpy(msk)[None]

        # --------------------------------------------------------
        # resize mask to match transformed image spatial size
        msk_t = torch.as_tensor(msk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        msk_t = F.interpolate(msk_t, size=img_t.shape[1:], mode="nearest")  # keep binary
        msk_t = msk_t.squeeze(0)  # (1, H, W)

        # --------------------------------------------------------


        nothing=0
        return {"img": img_t, "msk": msk_t}, nothing
    
def load_classifier(checkpoint_path: str, config: dict, device: torch.device):
    """
    Instantiate your Lightning Classifier by inspecting its __init__ signature,
    then load the checkpoint’s state_dict manually.
    """
    import inspect
    sig = inspect.signature(Classifier.__init__)
    init_kwargs = {}

    # 1. Pass TOML config if accepted
    if "config" in sig.parameters:
        init_kwargs["config"] = config

    # 2. Handle class_counts
    if "class_counts" in sig.parameters:
        cc = config.get("DATA", {}).get("class_counts", None)
        if cc is None:
            # fallback to uniform 1:1 dict for binary classification
            num_classes = config["DATA"]["Num_of_Classes"]
            cc = {i: 1 for i in range(num_classes)}
        elif isinstance(cc, list):
            cc = {i: v for i, v in enumerate(cc)}
        init_kwargs["class_counts"] = cc

    # 3. Create model
    model = Classifier(**init_kwargs).to(device)

    # 4. Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def prob_to_colour(p: float, alpha_int: int):
    """
    p in [0,1] → colour from yellow (255,255,0) at p=0
                        through orange at p~0.5
                        to red   (255,  0,0) at p=1
    Returns an (R,G,B,A) tuple.
    """
    # Keep R at 255, B at 0, linearly interpolate G from 255→0
    r = 255
    g = int(255 * (1.0 - p))
    b = 0
    return (r, g, b, alpha_int)

def classify_and_overlay(
    slide_reader,
    seg_cfg: str,
    wsi_path: str,
    out_prefix: str,
    classify_ckpt: str,
    final_dataset_csv_path: str = None,
    show_real_gt: bool = False,
    thumb_size=(1024,1024),
    alpha=0.3,
    circle_radius=2 
):
    # 1) SEGMENTATION
    # # if outputs exist, segment_wsi will print and skip
    # segment_wsi(seg_cfg, wsi_path, out_prefix)

    # 2) LOAD SEGMENTATION OUTPUTS
    if not os.path.exists(f"{out_prefix}_instances.tif"):
        print(f"❌ Segmentation outputs not found. Please run segmentation first.")
        sys.exit(1)

    # inst_map = np.load(f"{out_prefix}_instances.npy")        # shape (H, W)
    inst_map = tifffile.imread(f"{out_prefix}_instances.tif")
    df = pd.read_csv(f"{out_prefix}_instances.csv")          # cols: pid, xmin,ymin,xmax,ymax,centroid_x,centroid_y
    print (f"Loaded {len(df)} instances from {out_prefix}_instances.csv")

    # 3) SET UP

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config   = toml.load(seg_cfg)  # or load a separate Inference_Config.ini if you need it
    if not os.path.exists(f"{out_prefix}_classification.csv"):
    #    # 4) CLASSIFY EACH NUCLEUS
    #    model = load_classifier(classify_ckpt, config, device)
    #    model.eval().to(device)

    #    classif_transform = _virchow_preprocess()

    #    # 4) EXTRACT PATCHES, RUN CLASSIFICATION
    #    H_full, W_full = inst_map.shape

    #    print (f"Classifying {len(df)} nuclei...")
    #    start_time = pd.Timestamp.now()
    #    size = config["BASEMODEL"]["Input_Size"][0]

    #    half = size//2
    #    valid = (
    #      (df.centroid_x >= half) &
    #      (df.centroid_x <= W_full-half) &
    #      (df.centroid_y >= half) &
    #      (df.centroid_y <= H_full-half)
    #    )
    #    df = df[valid].reset_index(drop=True)
    #    print(f"✅ Valid nuclei for classification: {len(df)}")
    #    ds = NucleusDataset(slide_reader,wsi_path, size, df, inst_map, classif_transform)
    #    dl = DataLoader(ds, batch_size= config['BASEMODEL']['Batch_Size_Classification'], num_workers=config['BASEMODEL']['Num_of_Worker'], pin_memory=True)
       
    #    devices     = [0]
    #    L.seed_everything(42, workers=True)
    #    torch.set_float32_matmul_precision('medium')
    #    trainer = L.Trainer(devices=devices,
    #                        accelerator="gpu",
    #                        strategy="ddp",
    #                        logger=False,
    #                        precision=config['BASEMODEL']['Precision'],
    #                        use_distributed_sampler = False,
    #                        benchmark=False,)

    #    # this will yield a list of (probs, coords, idx) for each batch
    #    outputs = trainer.predict(model, dl)
       


       ##############################
       # 4) CLASSIFY EACH NUCLEUS
       model = load_classifier(classify_ckpt, config, device)
       model.eval().to(device)
       classif_transform = get_transforms(config)

       # 4) EXTRACT PATCHES, RUN CLASSIFICATION
       H_full, W_full = inst_map.shape

       print (f"Classifying {len(df)} nuclei...")
       start_time = pd.Timestamp.now()
       size = config["BASEMODEL"]["Input_Size"][0]

       half = size//2
       valid = (
         (df.centroid_x >= half) &
         (df.centroid_x <= W_full-half) &
         (df.centroid_y >= half) &
         (df.centroid_y <= H_full-half)
       )
       df = df[valid].reset_index(drop=True)
       print(f"✅ Valid nuclei for classification: {len(df)}")
       ds = NucleusDataset(slide_reader,wsi_path, size, df, inst_map, classif_transform)
       dl = DataLoader(ds, batch_size= config['BASEMODEL']['Batch_Size_Classification'], num_workers=config['BASEMODEL']['Num_of_Worker'], pin_memory=True)
       
       # map from our dtypes into Trainer-friendly values
       _dtype2trainer = {
           torch.float32: 32,
           torch.float16: "16-mixed",
           torch.bfloat16: "bf16-mixed",
       }
       
       bb = config['BASEMODEL']['Backbone']
       model_dtype = set_precision(config, bb)
       # pick the trainer‐precision kwarg
       trainer_prec = _dtype2trainer[model_dtype]
   
       L.seed_everything(config['BASEMODEL']['Random_Seed'], workers=True)
       torch.set_float32_matmul_precision('medium')
       num_gpus = torch.cuda.device_count()
       strategy = "ddp" if num_gpus and num_gpus > 1 else "auto"
       devices = "auto" if num_gpus else 1  # 1 device on CPU
       
       trainer = L.Trainer(
                           accelerator="gpu" if num_gpus else "cpu",
                           devices=devices,
                           strategy="ddp",
                           logger=False,
                           precision=trainer_prec,
                           use_distributed_sampler = False,
                           benchmark=False,)

       # this will yield a list of (probs, coords, idx) for each batch
       outputs = trainer.predict(model, dl)
       flat = []
       for o in outputs:
           if isinstance(o, dict) and 'logits' in o:
               flat.append(o['logits'])
           else:
               flat.append(o)
       all_probs = torch.cat(flat, dim=0)
       
       if all_probs.ndim == 2 and all_probs.size(1) == 2:
           probs = torch.softmax(all_probs, dim=1)[:, 1]
       else:
           raise RuntimeError(f"Unexpected predict() output shape: {tuple(all_probs.shape)}")
       
       df['prob_mitosis'] = probs.cpu().numpy()
       thr = float(config.get("BASEMODEL", {}).get("Classification_Threshold", 0.5))
       df['pred_class'] = (df['prob_mitosis'] >= thr).astype(int)
       # reorder back to original df order if needed:    
       print(f"✅ Classified {len(all_probs)} nuclei.")

       print("number of mitotic figures:", (df['pred_class'] == 1).sum())
       print("number of non-mitotic figures:", (df['pred_class'] == 0).sum())


       df.to_csv(f"{out_prefix}_classification.csv", index=False)
       print("✅ Saved classification table →", f"{out_prefix}_classification.csv")
       print(f"Time taken for classification: {pd.Timestamp.now() - start_time}")

    else:
        df = pd.read_csv(f"{out_prefix}_classification.csv")
        print(f"✅ Loaded classification table → {out_prefix}_classification.csv")
        print("number of mitotic figures:", (df['pred_class'] == 1).sum())
        print("number of non-mitotic figures:", (df['pred_class'] == 0).sum())
    

    if df.empty:
        print("❌ No nuclei classified. Exiting.")
        sys.exit(1)

    # --- 5) COMPARISON & OVERLAY ------------------------------------------

    # 5.1) Gather *all* patch‐level GT for this slide
    do_gt    = False
    gt_pts   = []
    gt_class = []

    if final_dataset_csv_path and os.path.exists(final_dataset_csv_path):
        gt_all = pd.read_csv(final_dataset_csv_path)
        slide_base   = os.path.splitext(os.path.basename(wsi_path))[0]    # "006"
        dataset_name = gt_all.iloc[0]['nrrd_file'].split("_")[0]         # e.g. "MIDOG"
        prefix       = f"{dataset_name}_{slide_base}_"

        # select every patch of this slide
        gt_sub = gt_all[gt_all['nrrd_file'].str.startswith(prefix)].copy()
        if not gt_sub.empty:
            for _, row_gt in gt_sub.iterrows():
                nrrd      = row_gt['nrrd_file']        # "MIDOG_006_3.nrrd"
                masks_npy = row_gt['masks_file']       # "masks_MIDOG_006_3.npy"
                mask_id   = int(row_gt['mask_id'])
                cls       = int(row_gt['class'])

                # 1) load the patch's mask and get its local centroid
                patch_map = np.load(os.path.join(config['DATA']['masks_path'], masks_npy))
                ys, xs = np.where(patch_map == mask_id)
                if len(xs)==0:
                    continue
                cx_loc, cy_loc = xs.mean(), ys.mean()

                # 2) look up that patch's (x0,y0) in slide coords
                x0_patch, y0_patch = get_patch_origin(nrrd, config)

                # 3) slide‐level centroid
                gt_pts.append([x0_patch + cx_loc,
                               y0_patch + cy_loc])
                gt_class.append(cls)

            if gt_pts:
                gt_pts   = np.array(gt_pts)
                gt_class = np.array(gt_class)
                do_gt    = True

    if not do_gt:
        print("❗ no GT patches found for this slide → skipping comparison")

    # 5.2) match predictions ↔ GT, compute metrics
    matches = None
    if do_gt:
        pred_pts = df[['centroid_x','centroid_y']].to_numpy()
        tree      = cKDTree(gt_pts)
        dists, idxs = tree.query(pred_pts, distance_upper_bound=32)
        keep      = np.where(dists < 32)[0]

        # enforce one-to-one
        unique = {}
        for pi, gi, dist in zip(keep, idxs[keep], dists[keep]):
            if gi not in unique or dist < unique[gi][1]:
                unique[gi] = (pi, dist)

        rows = []
        for gi, (pi, _) in unique.items():
            P = df.iloc[pi].add_prefix('pred_')
            G = pd.Series({
                'gt_centroid_x': gt_pts[gi,0],
                'gt_centroid_y': gt_pts[gi,1],
                'gt_class':      gt_class[gi]
            })
            rows.append(pd.concat([P, G]))
        matches = pd.DataFrame(rows)

        # detection metrics
        TP = len(matches)
        FP = len(df)    - TP
        FN = len(gt_pts)- TP
        prec = TP/(TP+FP+1e-8)
        rec  = TP/(TP+FN+1e-8)
        f1_det = 2*prec*rec/(prec+rec+1e-8)
        print(f"▶ Detection — Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1_det:.3f}")

        # classification on matched
        y_t = matches['gt_class']
        y_p = matches['pred_pred_class']
        acc_cls = balanced_accuracy_score(y_t, y_p)
        f1_cls  = f1_score(y_t, y_p, average="binary", zero_division=0)
        prec_cls= precision_score(y_t, y_p)
        recall_cls= recall_score(y_t, y_p)
        print(f"▶ Classification — Bal. Acc: {acc_cls:.3f}, F1 binary: {f1_cls:.3f}, Prec.: {prec_cls:.3f}, Recall: {recall_cls:.3f}")

    # ─── 5.3) draw two-pane overlay with actual masks ───────────────────────────────

    if show_real_gt :
        thumb     = get_thumbnail_image(wsi_path,out_prefix, thumb_size)
        w_t, h_t  = thumb.size
        H_full, W_full = inst_map.shape
        
        # scale from level-0 → thumbnail
        sx, sy    = w_t / W_full, h_t / H_full
        
        alpha_i   = int(255 * alpha)
        # prepare two RGBA layers
        gt_layer  = Image.new("RGBA", (w_t, h_t), (0,0,0,0))
        pd_layer  = Image.new("RGBA", (w_t, h_t), (0,0,0,0))
        
        # 1) ground-truth masks
        if do_gt:
            for (_, row_gt), cls in zip(gt_sub.iterrows(), gt_class):
                # load patch‐local mask from .npy
                masks_npy = row_gt["masks_file"]
                mask_map  = np.load(os.path.join(config["DATA"]["masks_path"], masks_npy))
                inst_id   = int(row_gt["mask_id"])
        
                # binary patch mask
                patch_mask = (mask_map == inst_id).astype("uint8") * 255
        
                # get patch origin in slide coords
                x0, y0 = get_patch_origin(row_gt["nrrd_file"], config)
                ph, pw  = patch_mask.shape
        
                H_full, W_full = inst_map.shape
                # compute target region in slide coords
                x1 = min(x0 + pw, W_full)
                y1 = min(y0 + ph, H_full)
                x0_t = max(x0, 0)
                y0_t = max(y0, 0)
        
                # compute corresponding source region
                sx0 = x0_t - x0              # if x0<0, skip left cols
                sy0 = y0_t - y0              # if y0<0, skip top rows
                sx1 = sx0 + (x1 - x0_t)
                sy1 = sy0 + (y1 - y0_t)
        
                # now patch
                full_mask = np.zeros((H_full, W_full), dtype="uint8")
                full_mask[y0_t:y1, x0_t:x1] = patch_mask[int(sy0):int(sy1), int(sx0):int(sx1)]
        
                # convert to PIL and downsample to thumb
                pil_mask = Image.fromarray(full_mask, mode="L")
                pil_mask = pil_mask.resize((w_t, h_t), resample=Image.NEAREST)
        
                # choose colour
                colour = (255,0,0,alpha_i) if cls==1 else (255,255,0,alpha_i)
                colour_img = Image.new("RGBA", (w_t, h_t), colour)
        
                # composite onto gt_layer
                gt_layer.paste(colour_img, (0,0), mask=pil_mask)
        
        # 2) predicted masks
        for pid, p in zip(df["pid"], df["prob_mitosis"]):
            # binary slide‐level mask for this pid
            mask_map = (inst_map == pid).astype("uint8") * 255
            pil_mask = Image.fromarray(mask_map, mode="L")
            pil_mask = pil_mask.resize((w_t, h_t), resample=Image.NEAREST)
        
            # heatmap colour by probability
            colour = prob_to_colour(float(p), alpha_i)
            colour_img = Image.new("RGBA", (w_t, h_t), colour)
        
            pd_layer.paste(colour_img, (0,0), mask=pil_mask)
        
        # 3) composite onto thumbnail
        gt_overlay = thumb.convert("RGBA")
        gt_overlay = Image.alpha_composite(gt_overlay, gt_layer)
        pred_overlay = thumb.convert("RGBA")
        pred_overlay = Image.alpha_composite(pred_overlay, pd_layer)
        
        # 4) stitch & save
        if do_gt:
            combo = Image.new("RGBA", (w_t*2, h_t))
            combo.paste(gt_overlay,   (0,0))
            combo.paste(pred_overlay, (w_t,0))
            combo.save(f"{out_prefix}_comparison.png")
            print(f"✅ Saved comparison → {out_prefix}_comparison.png")
        else:
            pred_overlay.save(f"{out_prefix}_classification_thumbnail.png")
            print(f"✅ Saved prediction → {out_prefix}_classification_thumbnail.png")
        
    else:
        # 5.3) draw two-pane overlay
        thumb     = get_thumbnail_image(wsi_path,out_prefix, thumb_size)
        w_t, h_t  = thumb.size
        H_full, W_full = inst_map.shape
    
        sx, sy    = w_t/W_full, h_t/H_full
        a_i       = int(255*alpha)
        r         = circle_radius
    
        gt_can    = thumb.convert("RGBA")
        pd_can    = thumb.convert("RGBA")
        draw_gt   = ImageDraw.Draw(gt_can, "RGBA")
        draw_pd   = ImageDraw.Draw(pd_can, "RGBA")
    
        if do_gt:
            # all GT
            for (cx,cy), cls in zip(gt_pts, gt_class):
                x, y = cx*sx, cy*sy
                c = (255,0,0,a_i) if cls==1 else (255,255,0,a_i)
                draw_gt.ellipse([x-r,y-r, x+r,y+r], fill=c)
            # highlight matched
            for _, m in matches.iterrows():
                x, y = m['gt_centroid_x']*sx, m['gt_centroid_y']*sy
                draw_gt.ellipse([x-(r+1),y-(r+1), x+(r+1),y+(r+1)],
                                outline=(255,255,255,a_i), width=2)
    
        # all preds as heatmap
        for _, row in df.iterrows():
            x, y = row.centroid_x*sx, row.centroid_y*sy
            draw_pd.ellipse([x-r,y-r, x+r,y+r],
                            fill=prob_to_colour(row.prob_mitosis, a_i))
        if matches is not None:
            for _, m in matches.iterrows():
                x, y = m['pred_centroid_x']*sx, m['pred_centroid_y']*sy
                draw_pd.ellipse([x-(r+1),y-(r+1), x+(r+1),y+(r+1)],
                                outline=(255,255,255,a_i), width=2)
    
        if do_gt:
            combo = Image.new("RGBA",(w_t*2,h_t))
            combo.paste(gt_can,   (0,0))
            combo.paste(pd_can,   (w_t,0))
            combo.save(f"{out_prefix}_comparison.png")
            print(f"✅ Saved comparison → {out_prefix}_comparison.png")
        else:
            pd_can.save(f"{out_prefix}_classification_thumbnail.png")
            print(f"✅ Saved prediction → {out_prefix}_classification_thumbnail.png")
            
    
    
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python segment_and_classify.py <seg_config.toml> <input.svs> <output_prefix> <classifier.ckpt> <final_dataset.csv>")
        sys.exit(1)
    config_path, wsi_path, out_prefix, ckpt_path,  final_dataset_csv_path= sys.argv[1:]
    
    # if CUCIM_AVAILABLE:
    #     reader = CuImage(wsi_path)              # single cuCIM instance → one warning only
    # else:
    #     reader = openslide.OpenSlide(wsi_path)  # single OpenSlide instance
    from SimpleWsiReader import make_wsi
    
    # automatic:
    reader = make_wsi(wsi_path)
    
    # # force OpenSlide even if cuCIM is present:
    # reader = make_wsi(wsi_path, reader_type="openslide")
    
    # # force PIL (flat‐file mode):
    # reader = make_wsi(wsi_path, reader_type="image")
        
    classify_and_overlay(
        reader,     
        config_path,
        wsi_path,
        out_prefix,
        ckpt_path,
        final_dataset_csv_path,
        show_real_gt=False,
    )



