import json
import os
import gc
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import openslide
import sqlite3
import matplotlib.pyplot as plt
import nrrd
import torch
from segment_anything import sam_model_registry, SamPredictor

def get_img_msk(
        filename,
        center_x, 
        center_y, 
        label, 
        nrrd_file, 
        image_folder, 
        nrrd_folder, 
        predictor, 
        dim=[256, 256], 
        vis_level=0,
        box_dim=[32, 32],
        ):
    
    slide = openslide.open_slide(os.path.join(image_folder, filename))
    top_left = (int(center_x - dim[0]/2), int(center_y - dim[1]/2))
    center_x = int(center_x - top_left[0])
    center_y = int(center_y - top_left[1])
    img = np.array(slide.read_region(top_left, level=vis_level, size=(dim[0], dim[1])))[:, :, :3]
    bbox = [center_x - int(box_dim[0]/2), center_y - int(box_dim[1]/2),
            center_x + int(box_dim[0]/2), center_y + int(box_dim[1]/2), ]

    header = {
        'filename': filename,
        'top_left': top_left,
        'center': (center_x, center_y),
        'dim': (dim[0], dim[1]),
        'vis_level': vis_level,
        'annotation_label': 1 if label == 'mitotic figure' else 0,
        'mask': np.zeros_like(img)
        }

    predictor.set_image(img)
    masks, _, _ = predictor.predict(
        box=np.array([bbox[0], bbox[1], bbox[2], bbox[3],])[None, :],
        multimask_output=False,
    )

    header['mask'] = np.array(masks[0]).astype('float32')
    nrrd.write(os.path.join(nrrd_folder, nrrd_file), img, header, custom_field_map=custom_field_map)

def save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim=[256, 256], vis_level=0, box_dim=[32, 32]):
    for i in range(len(df)):
        get_img_msk(df['filename'][i], df.coordinateX[i], df.coordinateY[i], 
                    df['annotation_label'][i], df['nrrd_file'][i], 
                    image_folder, nrrd_folder, predictor, 
                    dim, vis_level, box_dim)

def table2df(cursor, table_name):
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    return df

#Configure Paths
nrrd_folder        = "/projects/wispermed_rp18/mitosisDetect/Data/nrrd"                #Path to save the NRRD files
final_dataset      = "/projects/wispermed_rp18/mitosisDetect/Data/final_dataset"       #Path to save the CSV file of the final dataset
midog_folder       = "/projects/wispermed_rp18/mitosisDetect/Data/MIDOG_PLUS/"         #Path to MIDOG_PLUS
cmc_folder         = "/projects/wispermed_rp18/mitosisDetect/Data/MITOS_WSI_CMC/"      #Path to MITOS_WSI_CMC
ccmct_folder       = "/projects/wispermed_rp18/mitosisDetect/Data/MITOS_WSI_CCMT/"    #Path to MITOS_WSI_CCMCT
tupac_folder       = "/projects/wispermed_rp18/mitosisDetect/Data/TUPAC/"              #Path to TUPAC

#Load SAM mask generator
sam_checkpoint     = "/projects/wispermed_rp18/mitosisDetect/OMG-Net/Models/sam_vit_h_4b8939.pth" 
model_type         = "vit_h"
device             = "cuda"
dim                = [256, 256]
vis_level          = 0
box_dim            = [32, 32]
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=torch.device(device))
predictor = SamPredictor(sam)

#Configure NRRD files
custom_field_map   = {
    'SVS_ID': 'string',
    'top_left': 'int list',
    'center': 'int list',
    'dim': 'int list',
    'vis_level': 'int',
    'diagnosis': 'string',
    'annotation_label': 'string',
    'mask': 'double matrix'
    }

datasets = []

#Process the MIDOG dataset-----------------------------------------------------------------------------------------------
# print("Processing MIDOG dataset")
# annotation_file = os.path.join(midog_folder, "MIDOG++.json")
# image_folder    = os.path.join(midog_folder, "images")
# image_files     = [fn.split("\\")[-1] for fn in glob.glob(image_folder + "/*.tiff")]
# slides          = pd.read_csv(os.path.join(midog_folder, "datasets_xvalidation.csv"), delimiter=";")
# dataframe       = os.path.join(midog_folder, "MIDOG.csv")

# print("Creating dataframe from annotation")
# rows = []
# with open(annotation_file) as f:
#     data = json.load(f)
#     categories = {1: 'mitotic figure', 2: 'hard negative'}
#     for row in data["images"]:
#         file_name = row["file_name"]
#         image_id = row["id"]
#         width = row["width"]
#         height = row["height"]
#         for ann_id, annotation in enumerate([anno for anno in data['annotations'] if anno["image_id"] == image_id]):
#             box = annotation["bbox"]
#             xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#             cat = categories[annotation["category_id"]]
#             slide = slides.loc[slides['Slide'] == image_id, 'Tumor':]
#             tumour = slide['Tumor'].array[0]
#             scanner = slide['Scanner'].array[0]
#             origin = slide['Origin'].array[0]
#             species = slide['Species'].array[0]
#             nrrd_file = 'MIDOG_{}_{}.nrrd'.format(f"{image_id:03d}", ann_id)
#             rows.append(
#                 [file_name, image_id, ann_id, width, height, xmin, ymin, xmax, ymax, cat, tumour, scanner, origin,
#                     species, nrrd_file])

# df = pd.DataFrame(rows, columns=["filename", "image_id", "ann_id", "width", "height",
#                                  "xmin", "ymin", "xmax", "ymax",
#                                  "annotation_label", "tumour", "scanner", "origin", "species", "nrrd_file"])
# df['coordinateX'] = (df['xmin'] + df['xmax']) / 2
# df['coordinateY'] = (df['ymin'] + df['ymax']) / 2
# df.to_csv(dataframe, index=False)

# save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim, vis_level, box_dim)

df=pd.read_csv("/projects/wispermed_rp18/mitosisDetect/Data/MIDOG_PLUS/MIDOG.csv")
datasets.append(df)

# #Process the MITOS_WSI_CMC/MITOS_WSI_CCMCT dataset-----------------------------------------------------------------------------------------------
# print("Processing MITOS_WSI_CMC/MITOS_WSI_CCMCT dataset")
# for dataset in ['MITOS_WSI_CMC', 'MITOS_WSI_CCMCT']:
#     if dataset == "MITOS_WSI_CCMCT":
#         annotation_file = os.path.join(ccmct_folder, "databases", "MITOS_WSI_CCMCT_ODAEL.sqlite")
#         image_folder    = os.path.join(ccmct_folder, "WSI")
#         dataframe       = os.path.join(ccmct_folder, "MITOS_WSI_CCMCT.csv")
#         mitotic_label   = 'mitotic figure'

#     elif dataset == "MITOS_WSI_CMC":
#         annotation_file = os.path.join(cmc_folder, "databases", "MITOS_WSI_CMC_CODAEL_TR_ROI.sqlite")
#         image_folder    = os.path.join(cmc_folder, "WSI")
#         dataframe       = os.path.join(cmc_folder, "MITOS_WSI_CMC.csv")
#         mitotic_label   = 'Mitotic figure'
    
#     print("Creating dataframe from annotation")
#     con = sqlite3.connect(annotation_file)
#     cur = con.cursor()
#     cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

#     df_Annotations = table2df(cur, 'Annotations')
#     df_sqlite_sequence = table2df(cur, 'sqlite_sequence')
#     df_Annotations_coordinates = table2df(cur, 'Annotations_coordinates')
#     df_Annotations_label = table2df(cur, 'Annotations_label')
#     df_Classes = table2df(cur, 'Classes')
#     df_Log = table2df(cur, 'Log')
#     df_Persons = table2df(cur, 'Persons')
#     df_Slides = table2df(cur, 'Slides')
#     con.close()

#     df_Annotations = df_Annotations[df_Annotations['agreedClass'].isin([1, 2])]
#     df_Annotations_coordinates = df_Annotations_coordinates[df_Annotations_coordinates['orderIdx'] == 1]
#     df_Annotations_coordinates.drop(columns=['slide'], inplace=True)
#     df_Annotations = df_Annotations.rename(columns={'uid': 'annoId'})

#     df_Slides = df_Slides.rename(columns={'uid': 'slide'})
#     df_Classes = df_Classes.rename(columns={'uid': 'agreedClass', 'name': 'annotation_label'})

#     df = df_Annotations.merge(df_Slides, on='slide', how='inner')
#     df = df.merge(df_Annotations_coordinates, on='annoId', how='inner')
#     df = df.merge(df_Classes, on='agreedClass', how='inner')
#     df = df.replace(mitotic_label, 'mitotic figure')
#     df.reset_index(drop=True, inplace=True)
#     df['nrrd_file'] = ['{}_{}.nrrd'.format(df['filename'][i].split(".")[0], df['annoId'][i]) for i in range(len(df))]
#     df.to_csv(dataframe, index=False)

#     save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim, vis_level, box_dim)
#     datasets.append(df)


#### Already Done, only need to concatenate now
df=pd.read_csv("/projects/wispermed_rp18/mitosisDetect/Data/MITOS_WSI_CCMT/MITOS_WSI_CCMCT.csv")
datasets.append(df)

df=pd.read_csv("/projects/wispermed_rp18/mitosisDetect/Data/MITOS_WSI_CMC/MITOS_WSI_CMC.csv")
datasets.append(df)


#Process the TUPAC16 dataset-----------------------------------------------------------------------------------------------
print("Processing TUPAC16 dataset")
annotation_file = os.path.join(tupac_folder, "databases", "TUPAC_alternativeLabels_augmented_training.sqlite")
image_folder    = os.path.join(tupac_folder, "WSI")
dataframe       = os.path.join(tupac_folder, "TUPAC16.csv")

print("Creating dataframe from annotation")
con = sqlite3.connect(annotation_file)
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

df_Annotations = table2df(cur, 'Annotations')
df_sqlite_sequence = table2df(cur, 'sqlite_sequence')
df_Annotations_coordinates = table2df(cur, 'Annotations_coordinates')
df_Annotations_label = table2df(cur, 'Annotations_label')
df_Classes = table2df(cur, 'Classes')
df_Log = table2df(cur, 'Log')
df_Persons = table2df(cur, 'Persons')
df_Slides = table2df(cur, 'Slides')

print(df_Annotations.agreedClass.value_counts())
print(df_Annotations.uid.unique().shape)
print(df_Annotations_label.annoId.unique().shape)
print(df_Annotations_coordinates.annoId.unique().shape)
con.close()

df_Annotations_coordinates.drop(columns=['slide'], inplace=True)
df_Annotations = df_Annotations.rename(columns={'uid': 'annoId'})
df_Slides = df_Slides.rename(columns={'uid': 'slide'})
df_Classes = df_Classes.rename(columns={'uid': 'agreedClass', 'name': 'annotation_label'})

df = df_Annotations.merge(df_Slides, on='slide', how='inner')
df = df.merge(df_Annotations_coordinates, on='annoId', how='inner')
df = df.merge(df_Classes, on='agreedClass', how='inner')
df.reset_index(drop=True, inplace=True)
df.drop(columns=['guid', 'lastModified', 'deleted', 'type',
                    'description', 'directory', 'uuid', 'exactImageID',
                    'EXACTUSER', 'uid', 'orderIdx', 'coordinateZ', 'color'], inplace=True)
df = df.replace('Mitose', 'mitotic figure')
df['nrrd_file'] = ['{}_{}.nrrd'.format(df['filename'][i].split(".")[0], df['annoId'][i]) for i in range(len(df))]
df.to_csv(dataframe, index=False)

save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim, vis_level, box_dim)
datasets.append(df)

#Merge all datasets-----------------------------------------------------------------------------------------------
print("Merging datasets")
df = pd.concat(datasets)
df.to_csv(os.path.join(final_dataset, "final_dataset.csv"), index=False)
print("Done")





###############################   Use this at the end 
#%%

import pandas as pd

# MIDOG dataset
df_midog = pd.read_csv("/projects/wispermed_rp18/mitosisDetect/Data/MIDOG_PLUS/MIDOG.csv")
df_midog['annotation_label'] = df_midog['annotation_label'].apply(lambda x: 1 if x.lower() == 'mitotic figure' else 0)
df_midog['source'] = 'MIDOG'
df_midog.to_csv("/projects/wispermed_rp18/mitosisDetect/Data/MIDOG_PLUS/MIDOG_numeric_labels.csv", index=False)

# MITOS_WSI_CCMCT dataset
df_ccmct = pd.read_csv("/projects/wispermed_rp18/mitosisDetect/Data/MITOS_WSI_CCMT/MITOS_WSI_CCMCT.csv")
df_ccmct['annotation_label'] = df_ccmct['annotation_label'].apply(lambda x: 1 if x.lower() == 'mitotic figure' else 0)
df_ccmct['source'] = 'MITOS_WSI_CCMCT'
df_ccmct['species'] = 'Canine'
df_ccmct.to_csv("/projects/wispermed_rp18/mitosisDetect/Data/MITOS_WSI_CCMT/MITOS_WSI_CCMCT_numeric_labels.csv", index=False)

# MITOS_WSI_CMC dataset
df_cmc = pd.read_csv("/projects/wispermed_rp18/mitosisDetect/Data/MITOS_WSI_CMC/MITOS_WSI_CMC.csv")
df_cmc['annotation_label'] = df_cmc['annotation_label'].apply(lambda x: 1 if x.lower() == 'mitotic figure' else 0)
df_cmc['source'] = 'MITOS_WSI_CMC'
df_cmc['species'] = 'Canine'
df_cmc.to_csv("/projects/wispermed_rp18/mitosisDetect/Data/MITOS_WSI_CMC/MITOS_WSI_CMC_numeric_labels.csv", index=False)

# TUPAC16 dataset
df_tupac = pd.read_csv("/projects/wispermed_rp18/mitosisDetect/Data/TUPAC/TUPAC16.csv")

columns_to_drop = [
    'guid', 'lastModified', 'deleted', 'type',
    'description', 'directory', 'uuid', 'exactImageID',
    'EXACTUSER', 'uid', 'orderIdx', 'coordinateZ', 'color'
]
df_tupac.drop(columns=columns_to_drop, inplace=True, errors='ignore')

df_tupac['annotation_label'] = df_tupac['annotation_label'].apply(
    lambda x: 1 if str(x).lower() in ['mitotic figure', 'mitose'] else 0
)
df_tupac['species'] = 'Human'
df_tupac['source'] = 'TUPAC16'
df_tupac.to_csv("/projects/wispermed_rp18/mitosisDetect/Data/TUPAC/TUPAC16_numeric_labels.csv", index=False)

# Concatenate datasets
datasets = [
    df_midog,
    df_ccmct,
    df_cmc,
    df_tupac
]

final_df = pd.concat(datasets, ignore_index=True)
final_df = final_df.rename(columns={
    "annotation_label": "class"
})

final_df.to_csv("/projects/wispermed_rp18/mitosisDetect/Data/final_dataset/final_dataset.csv", index=False)

print("Merged datasets successfully with numeric labels and source column.")

# %%
