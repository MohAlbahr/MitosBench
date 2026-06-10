export CUDA_VISIBLE_DEVICES=4

# try to avoid WandB timeout issues
export WANDB__SERVICE_WAIT=300
 

# record start
START_TS=$(date +%s)


# paths — edit these to match your environment
Config="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/Bech-Mitos/Inference/Inference_Config.ini"

# Image="/projects/wispermed_rp18/mitosisDetect/Data/MIDOG_PLUS/images/150.tiff"
# # Image="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/Inference/350.tiff"
# Output="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/outputs/Inference_Masks/150"
# FinalCSV="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/testing_dataset/masks_dataset.csv"


# #### For Schaller dataset
Image="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/outputs/Inference_Masks/H-21.245243he.ndpi"
Output="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/Bech-Mitos/outputs/Inference_Masks/H-21.245243he"
FinalCSV=""
 #### Segmentation time: 120:23 (MM:SS)
 #### Instance segmentation done in 7026.825754642487 seconds, with shape: (26496, 103680)



#Checkpoint="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/lightning_logs/Classifier/conch/Mask_Input_True/2025_06_11_10_55_50/epoch=25-val_loss_epoch=0.0924_val_f1_epoch=0.6836_Classifier.ckpt"

## Uses LoRa (good results)
# Checkpoint="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/lightning_logs/Classifier/virchow2/Mask_Input_True/2025_07_11_23_14_54/epoch=18-val_f1_tuned=0.9701_val_f1_epoch=0.9680_Classifier.ckpt"

### Virchow2 using lora on all 4.8 million
Checkpoint="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/lightning_logs/Classifier/virchow2/Mask_Input_True/2025_08_25_11_04_17/epoch=06-val_loss_epoch=0.0908_val_f1_epoch=0.8815_Classifier.ckpt"

# lora film masks
# Checkpoint="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/lightning_logs/Classifier/virchow2/Mask_Input_True/2025_08_16_11_50_38/epoch=11-val_loss_epoch=0.1476_val_f1_epoch=0.9688_Classifier.ckpt"

### No mask
# Checkpoint="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/lightning_logs/Classifier/virchow2/Mask_Input_True/2025_07_06_11_07_05/epoch=21-val_f1_tuned=0.9162_val_f1_epoch=0.9123_Classifier.ckpt"

#### UNI2 with Lora + mask
# Checkpoint="/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/lightning_logs/Classifier/uni_v2/Mask_Input_True/2025_07_11_23_32_18/epoch=08-val_f1_tuned=0.9681_val_f1_epoch=0.9650_Classifier.ckpt"



##################  Normal  


echo "→ Running segmentation in pathoSam1 env"
conda run -n pathoSam1 --live-stream python \
    /projects/wispermed_rp18/mitosisDetect/OMG-Net-test/Bench-Mitos/Inference/Segment_PathoSam_Inference.py \
    "$Config" \
    "$Image" \
    "$Output"

END_TS=$(date +%s)

# compute elapsed
ELAPSED_Seg=$(( END_TS - START_TS ))

printf "Segmentation time: %02d:%02d (MM:SS)\n" $((ELAPSED_Seg/60)) $((ELAPSED_Seg%60))

# record start
START_TS=$(date +%s)
echo "→ Running classification & overlay in omgNet2 env"
conda run -n omgNet2 --live-stream python \
    /projects/wispermed_rp18/mitosisDetect/OMG-Net-test/Bench-Mitos/Inference/PathoSam_Inference.py \
    "$Config" \
    "$Image" \
    "$Output" \
    "$Checkpoint" \
    "$FinalCSV"
 

##################### MIDOG

# echo "→ Running segmentation in pathoSam1 env"
# conda run -n pathoSam1 --live-stream python \
#     /projects/wispermed_rp18/mitosisDetect/OMG-Net-test/Inference/midog_segment.py \
#     "$Config" \
#     "$Image" \
#     "$Output"

# END_TS=$(date +%s)

# # compute elapsed
# ELAPSED_Seg=$(( END_TS - START_TS ))

# printf "Segmentation time: %02d:%02d (MM:SS)\n" $((ELAPSED_Seg/60)) $((ELAPSED_Seg%60))

# # record start
# START_TS=$(date +%s)
# echo "→ Running classification & overlay in omgNet2 env"
# conda run -n omgNet2 --live-stream python \
#     /projects/wispermed_rp18/mitosisDetect/OMG-Net-test/Inference/midog_inference.py \
#   --config "$Config" \
#   --wsi "$Image" \
#   --prefix "$Output" \
#   --checkpoint "$Checkpoint" \
#   --output_json "/projects/wispermed_rp18/mitosisDetect/OMG-Net-test/outputs/Inference_Masks/mitotic-figures.json"



# record end
END_TS=$(date +%s)

# compute elapsed
ELAPSED_Class=$(( END_TS - START_TS ))
printf "Classification time: %02d:%02d (MM:SS)\n" $((ELAPSED_Class/60)) $((ELAPSED_Class%60))

printf "Total time: %02d:%02d (MM:SS)\n" $(((ELAPSED_Seg + ELAPSED_Class)/60)) $(((ELAPSED_Seg + ELAPSED_Class)%60))