import toml
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import lightning as L
import torch
torch.set_float32_matmul_precision("medium")

import albumentations as A
import torchvision.transforms as T
from Utils import ColourAugment
from Models.Classifier import Classifier
from Dataloader.SAM import DataModule
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import nrrd
import time
from datetime import datetime
from Utils.SAM_utils import *
from torchvision.ops import nms
from torchvision.transforms import v2
import time
from huggingface_hub import login
import timm
from timm.layers import SwiGLUPacked
from conch.open_clip_custom import create_model_from_pretrained
from timm.data.transforms_factory   import create_transform
from timm.data                      import resolve_data_config
import torchvision.transforms as transforms
from Utils.constants import get_constants


def load_config(config_file):
    cfg = toml.load(config_file)
    cfg['SEGMENTATION']['backend'] = cfg['SEGMENTATION']['backend'].lower()

    return cfg

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

    if bb in ("phikon",):
        return torch.float32
    if bb in ("phikon_v2",):
        return torch.float32


    if bb in ("gigapath",):
        return torch.float32

    if bb in ("virchow", "virchow2"):
        return torch.float16

    if bb in ("hoptimus0", "hoptimus1"):
        return torch.float16


    if bb in ("hibou_l",):
        return torch.float32


    if bb in ("midnight12k",):
        return torch.float32
    
    else:
        return config["BASEMODEL"].get("precision", torch.float16)


def get_logger(config, timestamp):
    return TensorBoardLogger(os.path.join(config['CHECKPOINT']['logger_folder'],
                                          config['CHECKPOINT']['model_name'],
                                          config['BASEMODEL']['Backbone'],
                                          ),
                             name="Mask_Input_" + str(config['BASEMODEL']['Mask_Input']), version=timestamp)
def get_callbacks(config, save_dir):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath     = save_dir,
        monitor     = config['CHECKPOINT']['Monitor'],
        filename    = config['CHECKPOINT']['filename'],
        save_top_k  = config['CHECKPOINT']['save_top_k'],
        mode        = config['CHECKPOINT']['Mode'])

    return [lr_monitor, checkpoint_callback]

def get_transforms(config):
    bb = config['BASEMODEL']['Backbone'].lower()

    # ------------------------------------------------------------------ #
    # generic geometric / colour augmentations (work for every model)
    # ------------------------------------------------------------------ #
    augmentation = A.Compose(
        [
            # A.RandomCrop(width=config['DATA']['Input_Size'][0],
            #              height=config['DATA']['Input_Size'][1]),
            A.HorizontalFlip(p=config['AUGMENTATION']['horizontalflip']),
            # A.VerticalFlip(p=0.2),
            # A.RandomRotate90(p=0.2),
            A.RandomBrightnessContrast(
                p=config['AUGMENTATION']['randombrightnesscontrast']),
        ]
    )

    # ------------------------------------------------------------------ #
    # model-specific **normalisation / resizing**
    # ------------------------------------------------------------------ #
    if bb == "conch_v1":      
        print("Using costumized preprocessing conch_v1 pipeline!")
        # size = tuple(config['DATA']['Input_Size'])
        size = tuple((224, 224))
        mean = (0.48145466, 0.4578275, 0.40821073)
        std  = (0.26862954, 0.26130258, 0.27577711)
       
        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            # ensure 3-channel RGB ↴
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode = config['AUGMENTATION']['Colour_Mode']
            ),
            T.Normalize(mean=mean, std=std),
        ])
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val
    
    elif bb.startswith("conch_v15"):
        print("Using costumized preprocessing for conch_v15 pipeline!")
        # size = tuple(config['DATA']['Input_Size'])
        size = tuple((448, 448))
        mean, std = get_constants('imagenet')
       
        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(size),
            # ensure 3-channel RGB ↴
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode = config['AUGMENTATION']['Colour_Mode']
            ),
            T.Normalize(mean=mean, std=std),
        ])
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val
    
    elif bb.startswith("virchow"):
        print("Using Virchow/Virchow2 preprocessing pipeline")
        size = (224, 224)
        # ImageNet normalization (same as your eval_transform)
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        # train transforms 
        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            # ensure 3-channel RGB
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode = config['AUGMENTATION']['Colour_Mode']
            ),

            T.Normalize(mean=mean, std=std),
        ])

        # val / test transforms
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val

    # ——— UNI v1 branch ———
    if bb.startswith("uni"):
        print("Using UNI‐v1/UNI-v2 preprocessing pipeline")
        size = (224, 224)
        # ImageNet normalization (same as your eval_transform)
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        # train transforms 
        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            # ensure 3-channel RGB
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode = config['AUGMENTATION']['Colour_Mode']
            ),

            T.Normalize(mean=mean, std=std),
        ])

        # val / test transforms
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val
    
    elif bb.startswith("phikon"):
        size = (224, 224)
        # ImageNet normalization (same as your eval_transform)
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        # train transforms
        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=True),
            T.CenterCrop(size),
            # ensure 3-channel RGB
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode = config['AUGMENTATION']['Colour_Mode']
            ),

            T.Normalize(mean=mean, std=std),
        ])

        # val / test transforms
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val


    if bb.startswith("hoptimus0"):
        # ---- Hoptimus0 ----
        print("Using Hoptimus0 preprocessing pipeline")
        size = (224, 224)
        mean = (0.707223, 0.578729, 0.703617)
        std  = (0.211883, 0.230117, 0.177517)

        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            # ensure 3-channel RGB
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode=config['AUGMENTATION']['Colour_Mode']
            ),
            T.Normalize(mean=mean, std=std),
        ])
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val
 
    if bb.startswith("hoptimus1"):
        # ---- Hoptimus1 ----
        print("Using Hoptimus1 preprocessing pipeline")
        size = (224, 224)
        mean = (0.707223, 0.578729, 0.703617)
        std  = (0.211883, 0.230117, 0.177517)

        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            # ensure 3-channel RGB
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode=config['AUGMENTATION']['Colour_Mode']
            ),
            T.Normalize(mean=mean, std=std),
        ])
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val
    
    elif bb.startswith("gigapath"):
        # ---- GigaPath ----
        print("Using GigaPath preprocessing pipeline")
        size = (256, 256)
        mean, std = get_constants('imagenet')

        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop((224, 224)),  # GigaPath uses 224x224 for training	
            # ensure 3-channel RGB
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode=config['AUGMENTATION']['Colour_Mode']
            ),
            T.Normalize(mean=mean, std=std),
        ])
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop((224, 224)),  # GigaPath uses 224x224 for validation
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val
    
    
    elif bb.startswith("hibou_l"):
        # ---- HIBOU-L ----
        print("Using HIBOU-L preprocessing pipeline")
        size = (224, 224)
        mean, std = get_constants('hibou')

        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC,max_size=None, antialias=True),
            T.CenterCrop(size),
            # ensure 3-channel RGB
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode=config['AUGMENTATION']['Colour_Mode']
            ),
            T.Normalize(mean=mean, std=std),
        ])
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC,max_size=None, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val
        
    elif bb.startswith("midnight12k"):
        # ---- Midnight12k ----
        print("Using Midnight12k preprocessing pipeline")
        from Utils.constants import KAIKO_MEAN, KAIKO_STD
        size = (224, 224)

        mean, std = KAIKO_MEAN, KAIKO_STD

        tf_train = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            # ensure 3-channel RGB
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            ColourAugment.ColourAugment(
                sigma=config['AUGMENTATION']['Colour_Sigma'],
                mode=config['AUGMENTATION']['Colour_Mode']
            ),
            T.Normalize(mean=mean, std=std),
        ])
        tf_val = T.Compose([
            T.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return augmentation, tf_train, tf_val
    

def get_SAM_masks(config, df, csv_file):
    # prepare output dirs
    base           = config['DATA']['output_path']
    std_mask_dir   = os.path.join(base, "masks")
    std_fig_dir    = os.path.join(base, "figures")
    histo_mask_dir = "/projects/wispermed_rp18/mitosisDetect/Data/New_masks_pathosam"
    histo_fig_dir  = "/projects/wispermed_rp18/mitosisDetect/Data/New_figures_pathosam"
    os.makedirs(std_mask_dir,    exist_ok=True)
    os.makedirs(std_fig_dir,     exist_ok=True)
    os.makedirs(histo_mask_dir,  exist_ok=True)
    os.makedirs(histo_fig_dir,   exist_ok=True)

    mtype = config['SAM_MODEL']['Model_Type']
    patho_post   = config['SAM_MODEL'].get("pathosam_postprocess", True)
    use_pathosam = mtype.endswith("_histopathology")


    if use_pathosam:
        mask_dir, fig_dir = histo_mask_dir, histo_fig_dir
        print(f"→ Using PathoSAM {mtype}")
         
    
    else:
        raise ValueError(f"Unknown SAM_MODEL.Model_Type '{mtype}'")

    custom_field_map = {
                'SVS_ID': 'string',
                'top_left': 'int list',
                'center': 'int list',
                'dim': 'int list',
                'vis_level': 'int',
                'diagnosis': 'string',
                'annotation_label': 'string',
                'mask': 'double matrix'}

    masks_datasets = []

    for i in range(len(df)):
        image_name = df['nrrd_file'][i].split(".")[0]
        label = df['class'][i]
        nrrd_data = os.path.join(config['DATA']['nrrd_path'], df['nrrd_file'][i])
        img, header = nrrd.read(nrrd_data, custom_field_map=custom_field_map)
        img = img[:, :, :3]
        gt = np.array(header['mask'])

        out_mask_file = os.path.join(mask_dir, f"masks_{image_name}.npy")
        # print(f"→ Now: {image_name} mask file: {out_mask_file}")


        if use_pathosam:
            if not os.path.isfile(out_mask_file):
                raise FileNotFoundError(
                    f"[{i+1}/{len(df)}] Missing PathoSAM precomputed mask: {out_mask_file}"
                )
            # load the integer‐instance map
            instance_map = np.load(out_mask_file, allow_pickle=True)
            print(f"[{i+1}/{len(df)}] Found existing mask (Pathosam): {out_mask_file}")
            
        # now build one row per mask‐instance exactly as you had it
        df_masks = pd.DataFrame()
       # now build one row per mask‐instance
        mask_ids = range(1, instance_map.max()+1)
        df_masks['mask_id'] = mask_ids
        df_masks['image_id'] = df['image_id'][i]
        df_masks['nrrd_file'] = df['nrrd_file'][i]
        df_masks['masks_file'] = "masks_{}.npy".format(image_name)
        
         # last mask is GT-positive, all others negative
        df_masks['class'] = [0] * (instance_map.max() - 1) + [label]

        # drop any mask_id that really has no pixels
        valid_ids = set(np.unique(instance_map)) - {0}
        dropped = set(list(mask_ids)) - valid_ids
        if dropped:
            for mid in sorted(dropped):
                print(f"Deleting empty mask_id={mid} for {image_name}")
            df_masks = df_masks[df_masks['mask_id'].isin(valid_ids)].reset_index(drop=True)

        df_masks.reset_index(drop=True, inplace=True)
        masks_datasets.append(df_masks)
        print(f"→ masks_dataset of Image No.{i+1}/{len(df)} added")

    # concat + save
    masks_dataset = pd.concat(masks_datasets, axis=0).reset_index(drop=True)
    save_to = os.path.join(base, "dataset", csv_file)
    masks_dataset.to_csv(save_to, index=False)
    print(f"✅ Saved {len(masks_dataset)} masks → {save_to}")

def get_datasets(config):
    original_df = pd.read_csv(config['DATA']['Dataframe'], low_memory=False,)
    # df = df[(df['image_quality'] == 1) & (df['class'] != 999)]

    ## Changed by Mohamed:
    original_df = original_df[(original_df['class'] != 999)] 

    original_df = original_df[original_df['species'].isin(config['DATA']['species'])]
    original_df = original_df[original_df['source'].isin(config['DATA']['datasets'])]
    original_df.reset_index(drop=True, inplace=True)
    print(original_df)
    print("Total number of annoteted images in the dataframe:", len(original_df))

    if not os.path.isdir(config['DATA']['output_path']):
        os.mkdir(config['DATA']['output_path'])

    if not os.path.isfile(os.path.join(config['DATA']['output_path'], "dataset" ,"masks_dataset_with_source.csv")):
        print("No masks found, Generating SAM masks...")
        get_SAM_masks(config, df, csv_file="masks_dataset_with_source.csv")

    print ("Loading SAM masks...")
    df = pd.read_csv(os.path.join(config['DATA']['output_path'], "dataset" , "masks_dataset_with_source.csv"))
    # df = df[df['image_id'].isin(image_ids)]
    print(df)
    # print("Total number of annoteted images in the SAM masks dataframe:", len(df['nrrd_file'].unique()))

    ########### changes by Mohamed:

    # carve out the “MIDOG_006” prefix
    df['image_str'] = df['nrrd_file']\
                           .str.rsplit('_', n=1).str[0]   # → "MIDOG_006"

    if config["DATA"]["DomainGen_study"]: 
        # split by source: train+val on one dataset, test on all others ──────────
        train_source  = config['DATA']['train_dataset']

        print("Domain Gen. Study. Train Dataset is: ", str(train_source))
        # all rows whose 'source' == train_source go into train+val
        df_train_val = df[df['source'] == train_source].reset_index(drop=True)
        print("len(train dataset) in this settings: ", len(df_train_val))

        # the rest go into test
        masks_dataset_test     = df[df['source'] != train_source].reset_index(drop=True)
         
    else:
    
        mask = df['image_str'].isin(config['DATA']['filenames_test'])
        # test set
        masks_dataset_test = df[mask].reset_index(drop=True)
        print("Unique test prefixes actually found in mask DataFrame:", len(masks_dataset_test['image_str'].unique()))
        # df_train_val = df.drop(masks_dataset_test.index).reset_index(drop=True)
        df_train_val      = df[~mask].reset_index(drop=True)


    print("len(df_train_val['image_str'].Unique):", len(df_train_val['image_str'].unique()))
    print("len(test):",      len(masks_dataset_test))
    print("len(train_val):", len(df_train_val))

    filenames = list(df_train_val['image_str'].unique())
    train_idx, val_idx = train_test_split(filenames, test_size=config['DATA']['val_size'], random_state=42)

    masks_dataset_train = df_train_val[df_train_val['image_str'].isin(train_idx)].copy()
    masks_dataset_train.reset_index(drop=True, inplace=True)

    masks_dataset_val = df_train_val[df_train_val['image_str'].isin(val_idx)].copy()
    masks_dataset_val.reset_index(drop=True, inplace=True)
    print("Total raw number of images: ", len(masks_dataset_train) + len(masks_dataset_val) + len(masks_dataset_test))

    # ——————— 1) Tag “original” GT masks ———————
    # every row has a masks_file (one per image); the GT mask is the one whose mask_id == max per file
    for split_df in (masks_dataset_train, masks_dataset_val, masks_dataset_test):
        split_df['is_original'] = (
            split_df
            .groupby('masks_file')['mask_id']
            .transform(lambda mids: mids == mids.max())
        )

    # ——————— 2) Subsample only the *proposal* negatives ———————
    frac = config['DATA'].get('dataset_negatives_subsample', 1.0)
    
    # sanity-check
    if not (0.0 <= frac <= 1.0):
        raise ValueError(
            f"`dataset_negatives_subsample` must be between 0 and 1 "
            f"(inclusive), got {frac!r}"
        )
    
    def _subsample_proposals(df_split: pd.DataFrame) -> pd.DataFrame:
        """Keep all annotated cells but randomly down-sample PathoSAM proposals."""
        orig  = df_split[df_split['is_original']]           # always keep
        props = df_split[~df_split['is_original']]          # PathoSAM negatives
        props_sub = props.sample(
            frac=frac,
            random_state=config['BASEMODEL']['Random_Seed']
        )
        out = pd.concat([orig, props_sub])
        # shuffle result for robustness
        return out.sample(frac=1.0,
                          random_state=config['BASEMODEL']['Random_Seed']) \
                  .reset_index(drop=True)
    
    # do the actual work only if 0 < frac < 1
    if frac == 0.0:
       masks_dataset_train = masks_dataset_train[masks_dataset_train['is_original']].reset_index(drop=True)
       masks_dataset_val = masks_dataset_val[masks_dataset_val['is_original']].reset_index(drop=True)
       masks_dataset_test = masks_dataset_test[masks_dataset_test['is_original']].reset_index(drop=True)

       print("→ subsampling *proposals* to 0% "
             "(keeping only original GT masks)")
       print("Number of original GT masks in test set:")
       print(len(masks_dataset_test[masks_dataset_test['is_original']]))

    elif 0.0 < frac < 1.0:

        print(f"→ subsampling *proposals* to {frac*100:.1f}% "
              f"(train: {len(masks_dataset_train)} → ", end="")
        masks_dataset_train = _subsample_proposals(masks_dataset_train)
        print(f"{len(masks_dataset_train)} train samples)")
    
        masks_dataset_val = _subsample_proposals(masks_dataset_val)
        print(f"{len(masks_dataset_val)} val samples)")
        
        masks_dataset_test = masks_dataset_test[masks_dataset_test['is_original']].reset_index(drop=True)


        print("Number of proposals in train and val (negatives):")
        print( len(masks_dataset_train[~masks_dataset_train['is_original']]) +len(masks_dataset_val[~masks_dataset_val['is_original']]))

    else:
        # frac == 1.0  ➜ keep all proposals (explicit full-keep)
        print("→ no subsampling of PathoSAM proposals "
              f"(dataset_negatives_subsample={frac})")
            
    # —————————————————————————————————————————————————————


    print("Total number of images after negatives subsampling: ",
          len(masks_dataset_train) + len(masks_dataset_val) + len(masks_dataset_test))


    def balancing(df, target_label):
        N_min = min([len(group) for label, group in df.groupby(target_label)])
        return (df.
                groupby(target_label).
                apply(lambda group: group.sample(N_min, replace=False))
                .reset_index(drop=True))
                
    def oversample_positives(df, target_label):
        counts = df[target_label].value_counts()
        max_n  = counts.max()
        return (
            df
            .groupby(target_label)
            .apply(lambda g: g.sample(max_n, replace=(len(g) < max_n)))
            .reset_index(drop=True)
        )
        
    def downsample_negatives(df, target_label, neg_to_pos=3):
        pos = df[df[target_label] == 1]
        neg = df[df[target_label] == 0]
        # sample at most neg_to_pos × #positives
        n_neg = min(len(neg), len(pos) * neg_to_pos)
        neg = neg.sample(n_neg, replace=False)
        return pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)

    # in get_datasets:
    if config['DATA']['balancing_strategy'] == "downsample_negatives":
        print("using downsample_negatives strategy")
        masks_dataset_train = downsample_negatives(masks_dataset_train, 'class', neg_to_pos=3)
        masks_dataset_val   = downsample_negatives(masks_dataset_val,   'class', neg_to_pos=3)
    
    elif config['DATA']['balancing_strategy'] == "oversample_positives":
        print("using oversample_positives strategy") 
        masks_dataset_train = oversample_positives(masks_dataset_train, 'class')
        masks_dataset_val = oversample_positives(masks_dataset_val, 'class')

    elif config['DATA']['balancing_strategy'] == "same":
        print("using same strategy")
        masks_dataset_train = balancing(masks_dataset_train, 'class')
        masks_dataset_val = balancing(masks_dataset_val, 'class')
    else: 
        pass 

    
    class_counts = masks_dataset_train['class'].value_counts().to_dict()

    N = len(masks_dataset_train) + len(masks_dataset_val) + len(masks_dataset_test)

    print('Training Size: {}/{}({}) Positive Rate: {}'.format(len(masks_dataset_train), N, len(masks_dataset_train) / N,
                                                              list(masks_dataset_train['class'].value_counts(normalize=True))[1]))
    print('Validation Size: {}/{}({}) Positive Rate: {}'.format(len(masks_dataset_val), N, len(masks_dataset_val) / N,
                                                                list(masks_dataset_val['class'].value_counts(normalize=True))[1]))
    print('Testing Size: {}/{}({}) Positive Rate: {}'.format(len(masks_dataset_test), N, len(masks_dataset_test) / N,
                                                             list(masks_dataset_test['class'].value_counts(normalize=True))[1]))

    return masks_dataset_train, masks_dataset_val, masks_dataset_test, class_counts

def main(config_file):
    start = time.time()
    config = toml.load(config_file)

    import random, numpy as np, torch
    random.seed(config['BASEMODEL']['Random_Seed'])
    np.random.seed(config['BASEMODEL']['Random_Seed'])
    torch.manual_seed(config['BASEMODEL']['Random_Seed'])
    torch.cuda.manual_seed_all(config['BASEMODEL']['Random_Seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    augmentation, train_normalization, val_normalization = get_transforms(config)
    masks_dataset_train, masks_dataset_val, masks_dataset_test,class_counts = get_datasets(config)
    print("-" * 50 + "Time Elapsed: {}".format(time.time() - start) + "-" * 50)

    data = DataModule(df_train = masks_dataset_train,
                      df_val = masks_dataset_val,
                      df_test = masks_dataset_test,
                      config = config,
                      train_normalization=train_normalization,
                      val_normalization=val_normalization,
                      augmentation=augmentation,
                      inference=False,
                      )

    logger = get_logger(config, timestamp)
    print(logger.log_dir)
    callbacks = get_callbacks(config, logger.log_dir)

    L.seed_everything(config['BASEMODEL']['Random_Seed'], workers=True)

    # map from our dtypes into Trainer-friendly values
    _dtype2trainer = {
        torch.float32: 32,
        torch.float16: "16-mixed",
        torch.bfloat16: "bf16-mixed",
    }
    
    bb = config['BASEMODEL']['Backbone']
    model_dtype = set_precision(config, bb)
    
    # put your model on GPU *and* in the right dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Classifier(config, class_counts, logger.log_dir)
    model = model.to(device)
    
    # pick the trainer‐precision kwarg
    trainer_prec = _dtype2trainer[model_dtype]
    # Fall back to 1 if SLURM_NNODES is not set

    trainer = L.Trainer(devices="auto",
                        accelerator="gpu",
                        strategy="ddp_find_unused_parameters_true",         # recommended for multi‑GPU
                        benchmark=False,
                        max_epochs=config['BASEMODEL']['Max_Epochs'],
                        callbacks=callbacks,
                        precision=trainer_prec,
                        logger=logger,
                        accumulate_grad_batches=8,
                        num_nodes=int(os.environ.get("SLURM_NNODES", 1))
                        )

    trainer.fit(model,datamodule=data)
    print("-" * 50 + "Time Elapsed: {}".format(time.time() - start) + "-" * 50)
    print("Best Model Path: ", callbacks[1].best_model_path)

    #### Test on one gpu only 
    single_gpu_trainer = L.Trainer(devices=1,
                        accelerator="gpu",
                        strategy="auto",      
                        benchmark=False,
                        max_epochs=config['BASEMODEL']['Max_Epochs'],
                        callbacks=callbacks,
                        precision=trainer_prec,
                        logger=logger,
                        # accumulate_grad_batches=2,
                        # num_nodes=int(os.environ.get("SLURM_NNODES", 1))
    )
   
    print("Testing on single GPU")
    single_gpu_trainer.test(model, dataloaders=data.test_dataloader(), ckpt_path='best')


if __name__ == "__main__":
    main(sys.argv[1])
