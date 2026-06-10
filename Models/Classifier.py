# Models/Classifier.py
# -------------------------------------------------------------
# Unified classifier for ResNet, Virchow2 and CONCH with optional mask→image & mask→text fusion
# -------------------------------------------------------------
import os
import torch
import torch.nn as nn
import lightning as L
import torchvision
from torch.nn.functional import softmax
from torchmetrics.functional import accuracy, f1_score
import sklearn.metrics as skm
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import traceback
from typing import Literal, Optional

# ---------- optional dependencies ----------
try:
    from timm.data.mixup import Mixup
    from timm.layers import SwiGLUPacked
except ImportError:
    Mixup = None

import timm 

# CONCH model loader
try:
    from conch.open_clip_custom import create_model_from_pretrained
except ImportError:
    print("CONCH model loading failed.")
    create_model_from_pretrained = None

# Optional tokenization for text fusion
from conch.open_clip_custom import tokenize, get_tokenizer
# -------------------------------------------

# ────────────────────────────────────────────────────────────────────────────
# near your other imports
import numpy as np
from skimage.measure import regionprops
from skimage.color   import rgb2hed



class Classifier(L.LightningModule):
    """
    Backbones
    ---------
    • torchvision.* (e.g. resnet18)          → cfg['Backbone']="resnet18"
    • Virchow2                               → "virchow"
    • CONCH ViT-B/16                         → "conch"

    Mask & Text options (in [BASEMODEL]):
      Use_Mask:       bool
      Mask_Fusion:    "concat" | "add" | "film"
      Use_Text_Mask:  bool           # fuse text encoder features
      Text_Prefix_Len:int            # length of learned mask prefix tokens
    """

    def __init__(self, config,class_counts, log_dir=None):
        super().__init__()
        self.save_hyperparameters(config)
        cfg = self.hparams
        self.config = config
        # internally track best validation F1 & its threshold
        self._best_thr    = 0.5
        self._best_val_f1 = 0.0
        self.log_dir = log_dir
        # ----------------------------------------
        # Loss
        # ----------------------------------------

        if config['DATA']['balancing_strategy'] == "weighted_loss":
           print("using weighted_loss strategy")
           total = sum(class_counts.values())
           weights = {
             0: total / (2 * class_counts[0]), 
             1: total / (2 * class_counts[1])
           }
           self.cls_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([weights[0], weights[1]]),
           label_smoothing=cfg['REGULARIZATION']['Label_Smoothing'] )

        elif config['DATA']['use_focal_loss']:
            print("using focal_loss ")
            from Utils.losses import ASLSingleLabel
            self.cls_loss = ASLSingleLabel(
                gamma_pos=1,
                gamma_neg=4,
                reduction='mean'
            )
        else:
            cls_w = cfg['DATA'].get('weights', torch.ones(self.config['DATA']['Num_of_Classes'], dtype=torch.float32))
            self.cls_loss = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(cls_w, dtype=torch.float32),
                label_smoothing=cfg['REGULARIZATION']['Label_Smoothing']
            )
        

        # ---------------------------------
        # ----------------------------------------
        # Backbone + mask→image + mask→text setup
        # ----------------------------------------
        self.activation = getattr(torch.nn, self.config["BASEMODEL"]["Activation"])()
        bb = cfg['BASEMODEL']['Backbone'].lower()
        use_mask       = cfg['BASEMODEL'].get('Use_Mask', False)
        fusion         = cfg['BASEMODEL'].get('Mask_Fusion', 'concat')
        use_text_mask  = cfg['BASEMODEL'].get('Use_Text_Mask', False)
        prefix_len     = cfg['BASEMODEL'].get('Text_Prefix_Len', 16)
        num_classes    = cfg['DATA']['Num_of_Classes']
        token          = os.getenv("HF_HUB_TOKEN")

        if bb == "conch_v1":
            try:
                from conch.open_clip_custom import create_model_from_pretrained
            except ImportError:
                print("CONCH model loading failed.")
                create_model_from_pretrained = None

            # load CONCH vision+text encoders
            self.backbone, self.preprocess = create_model_from_pretrained(
                "conch_ViT-B-16","hf_hub:MahmoodLab/conch", hf_auth_token=token
            )
            # print("Preprocessing:", self.preprocess)
            # get image embed dim
            _dummy = torch.empty(1,3,224,224)
            with torch.inference_mode():
                self.embed_dim = self.backbone.encode_image(_dummy, proj_contrast=False, normalize=False).shape[-1]
            
            # CoCa’s text context length (should be 77 by default)
            if use_text_mask:
                # grab Conch’s tokenizer (it knows the right context length)
                self.conch_tokenizer = get_tokenizer()
                # print(f"[CONCH] using tokenizer with context length {self.conch_tokenizer}")
                # print("tokenizer type:", type(self.conch_tokenizer))


            # freeze/unfreeze last k blocks of vision
            k = cfg['BASEMODEL'].get('Unfreeze_last',0)
            vit = self.backbone.visual.trunk

        elif bb.startswith("conch_v15"):
            from Utils.model_zoo.conchv1_5.conchv1_5 import create_model_from_pretrained

            # load CONCH ViT-B/16 vision encoder
            if create_model_from_pretrained is None:
                raise ImportError("Please install CONCH with `pip install conch` to use the CONCH backbone")
            try:
                self.backbone, eval_transform = create_model_from_pretrained(checkpoint_path="hf_hub:MahmoodLab/conchv1_5", img_size=448)
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1.5 model, make sure that you were granted access and that you correctly registered your token")
            # print("Preprocessing:", self.preprocess)
            # get image embed dim
            _dummy = torch.empty(1,3,448,448)
            with torch.inference_mode():
                self.embed_dim = self.backbone(_dummy).shape[-1]
            
            # CoCa’s text context length (should be 77 by default)
            if use_text_mask:
                # grab Conch’s tokenizer (it knows the right context length)
                self.conch_tokenizer = get_tokenizer()
                # print(f"[CONCH] using tokenizer with context length {self.conch_tokenizer}")
                # print("tokenizer type:", type(self.conch_tokenizer))

            # freeze/unfreeze last k blocks of vision
            k = cfg['BASEMODEL'].get('Unfreeze_last',0)
            vit = self.backbone

        elif bb.startswith("virchow2"):
            import timm
            if timm is None:
                raise ImportError("Please `pip install timm` to use the Virchow2 backbone")

            # load Virchow2 (Paige AI) vision transformer
            # note: SwiGLUPacked and SiLU give the same architecture they used
            self.backbone = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU
            )
            # Virchow outputs a sequence of tokens (one CLS + patch tokens)
            self.embed_dim = self.backbone.embed_dim  # e.g. 1280
            ### only CLS token

            ### head: concat CLS token + mean‐pooled patch tokens → 2×embed_dim
            # self.classifier = nn.Linear(embed_dim * 2, num_classes)


        elif bb =="virchow":
            import timm            
            try:
                # load Virchow (Paige AI) vision transformer
                self.backbone = timm.create_model(
                    "hf-hub:paige-ai/Virchow",
                    pretrained=True,
                    mlp_layer=SwiGLUPacked,
                    act_layer=torch.nn.SiLU
                )
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow model, make sure that you were granted access and that you correctly registered your token")
            self.embed_dim = self.backbone.embed_dim


        elif bb.startswith("uni_v1"):
            import timm

            timm_kwargs = {
                    'img_size': 224,
                    'patch_size': 16,
                    'init_values': 1e-5,
                    'num_classes': 0,
                    'dynamic_img_size': True,
                }

            if timm is None:
                raise ImportError("Please `pip install timm` to use the UNI backbone")

            # load UNI v1 (MahmoodLab) vision transformer
            try:
                self.backbone = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")

            # UNI outputs a sequence of tokens (one CLS + patch tokens)
            self.embed_dim = self.backbone.embed_dim

        elif bb.startswith("uni_v2"):
            import timm
            timm_kwargs = {
                 'img_size': 224,
                 'patch_size': 14,
                 'depth': 24,
                 'num_heads': 24,
                 'init_values': 1e-5,
                 'embed_dim': 1536,
                 'mlp_ratio': 2.66667 * 2,
                 'num_classes': 0,
                 'no_embed_class': True,
                 'mlp_layer': timm.layers.SwiGLUPacked,
                 'act_layer': torch.nn.SiLU,
                 'reg_tokens': 8,
                 'dynamic_img_size': True
             }

            if timm is None:
                raise ImportError("Please `pip install timm` to use the UNI backbone")

            # load UNI v1 (MahmoodLab) vision transformer
            try:
                self.backbone = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")

            # UNI outputs a sequence of tokens (one CLS + patch tokens)
            self.embed_dim = self.backbone.embed_dim
         
        elif bb.startswith("phikon_v1"):
            from transformers import ViTModel
            from torchvision.transforms import InterpolationMode

            try:
                self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Phikon model, make sure that you were granted access and that you correctly registered your token")
            
            self.embed_dim = self.backbone.config.hidden_size
        
        elif bb.startswith("phikon_v2"):

            # if not has_internet_connection():
            #     raise Exception("No internet connection available. Please check your connection or the HF_ENDPOINT environment variable.")
            
            from transformers import AutoModel

            try:
                self.backbone = AutoModel.from_pretrained("owkin/phikon-v2", use_auth_token=True, trust_remote_code=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Phikon v2 model, make sure that you were granted access and that you correctly registered your token")

            self.embed_dim = self.backbone.config.hidden_size
        
        elif bb.startswith("hoptimus0"):
            timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
            import timm
            assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
            try:
                self.backbone = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download HOptimus-0 model, make sure that you were granted access and that you correctly registered your token")
            self.embed_dim = self.backbone.embed_dim

        elif bb.startswith("hoptimus1"):
            timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
            import timm
            assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
            try:
                self.backbone = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download HOptimus-1 model, make sure that you were granted access and that you correctly registered your token")
            self.embed_dim = self.backbone.embed_dim

        elif bb.startswith("gigapath"):
            import timm
            assert timm.__version__ == '0.9.16', f"Gigapath requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
            from torchvision import transforms
            try:
                self.backbone = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download GigaPath model, make sure that you were granted access and that you correctly registered your token")
            self.embed_dim = self.backbone.embed_dim

        elif bb.startswith("hibou_l"):
            from transformers import AutoModel
            try:
                self.backbone = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Hibou-L model, make sure that you were granted access and that you correctly registered your token")
        
            self.embed_dim = self.backbone.config.hidden_size
        
        elif bb.startswith("midnight12k"):
            from transformers import AutoModel
            from typing import Literal, Optional

            try:
                self.backbone = AutoModel.from_pretrained("kaiko-ai/midnight")
            except:
                traceback.print_exc()
                raise Exception("Failed to download Midnight-12k model")
            self.embed_dim = self.backbone.config.hidden_size

 
        # mask→image fusion head
        if use_mask:
            self.mask_encoder = torch.nn.Conv2d(1,64,7,2,3,bias=True)
            print(f"Using Mask_Fusion={fusion} for {bb} model")

            if fusion=="concat":
                self.classifier = torch.nn.Linear(self.embed_dim+64, num_classes)
            elif fusion=="add":
                self.mask_proj = torch.nn.Linear(64, self.embed_dim)
                self.classifier = torch.nn.Linear(self.embed_dim, num_classes)
            elif fusion=="film":
                self.mask_gamma = torch.nn.Linear(64, self.embed_dim)
                self.mask_beta  = torch.nn.Linear(64, self.embed_dim)
                self.classifier = torch.nn.Linear(self.embed_dim, num_classes)
            else:
                raise ValueError(f"Unknown Mask_Fusion={fusion}")
        else:
            self.classifier = torch.nn.Linear(self.embed_dim, num_classes)


        if cfg['BASEMODEL'].get('full_finetune', False):
           for p in self.backbone.parameters():
               p.requires_grad = True
        else:
            for p in self.backbone.parameters():
               p.requires_grad = False

            ################ LoRA setup ################
            # LoRA config
    
            if cfg['BASEMODEL'].get('LoRA', False):
                print(f"Using LoRA for {bb} model")
                from peft import get_peft_model, LoraConfig
    
                # LoRA config for CONCH
                # r: rank of the low-rank decomposition
                # lora_alpha: scaling factor for the LoRA weights
                # target_modules: list of modules to apply LoRA to
                # lora_dropout: dropout rate for the LoRA layers
                # bias: bias handling, can be "none", "lora_only", or "all"
    
                if bb.startswith("phikon") or bb.startswith("hibou_l") or bb.startswith("midnight12k"):
                    lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=[
                        "attention.attention.query",
                        "attention.attention.key",
                        "attention.attention.value",
                        "attention.output.dense",
                        "intermediate.dense",
                        "output.dense",
                    ],
                    lora_dropout=0.3,
                    bias="none",
                    modules_to_save=["head"]
                      )
                else:
    
                    lora_config = LoraConfig(
                        r=8,
                        lora_alpha=16,
                        target_modules=["qkv", "proj", "fc1", "fc2"],
                        lora_dropout=0.3,
                        bias="none",
                        modules_to_save=["head"]
                    )
        
                self.backbone = get_peft_model(self.backbone, lora_config)
                self.backbone.print_trainable_parameters()

        # always-ready convs for other mask→image uses
        # self.mask_encoder = nn.Conv2d(1,64,7,2,3,bias=True)
        # self.encoder_4d   = nn.Conv2d(4,64,7,2,3,bias=False)


    def forward(self, img, msk):
        bb = self.config['BASEMODEL']['Backbone'].lower()
        use_mask      = self.config['BASEMODEL'].get('Use_Mask',False)
        fusion        = self.config['BASEMODEL'].get('Mask_Fusion','concat')
        use_text_mask = self.config['BASEMODEL'].get('Use_Text_Mask',False)
        # --- helper for mask fusion ---
        def fuse_feats(img_feat):
            if not use_mask:
                return img_feat
            # 1) encode & pool mask → 64-d vector
            m64 = self.mask_encoder(msk)                     # (B,64,H',W')
            mf = m64.mean(dim=[2,3])                         # (B,64)
            # 2) fuse
            if fusion == "concat":
                return torch.cat([img_feat, mf], dim=-1)     # (B, D+64)
            elif fusion == "add":
                return img_feat + self.mask_proj(mf)         # (B, D)
            elif fusion == "film":
                return img_feat * (1 + self.mask_gamma(mf)) + self.mask_beta(mf)
            else:
                raise ValueError(f"Unknown Mask_Fusion={fusion}")


        if bb=="conch_v1":
            # print("img.shape: ", img.shape)
            # encode image features
            img_feat = self.backbone.encode_image(img, proj_contrast=False, normalize=False)
            # mask→image fusion
           
            fused    = fuse_feats(img_feat)
            # mask→text

            logits = self.classifier(fused)
            return self.activation(logits)

        if bb.startswith("conch_v15"):
            # encode image features
            img_feat = self.backbone(img)
            fused    = fuse_feats(img_feat)

            logits = self.classifier(fused)
            return self.activation(logits)


        # ---- Virchow ----
        if bb.startswith("virchow2"):
            return_cls=True
            import timm
            output = self.backbone(img)
        
            class_token = output[:, 0]
            fused     = fuse_feats(class_token)

            if return_cls:
                return self.activation(self.classifier(fused))
            
            patch_tokens = output[:, 5:]
            fused     = fuse_feats(patch_tokens)
            embedding = torch.cat([class_token, fused.mean(1)], dim=-1)

            return self.activation(self.classifier(embedding))
        
        # 3) UNI-v1 / UNI-v2
        if bb.startswith("uni"):
            img_feat = self.backbone(img)               # (B, D)
            fused    = fuse_feats(img_feat)
            return self.activation(self.classifier(fused))
    
        # 4) Phikon-v1
        if bb.startswith("phikon_v1"):
            out     = self.backbone(pixel_values=img)   # HF ViTModelOutput
            cls_tok = out.last_hidden_state[:, 0, :]    # (B, D)
            fused   = fuse_feats(cls_tok)
            return self.activation(self.classifier(fused))
    
        # 5) Phikon-v2
        if bb.startswith("phikon_v2"):
            out     = self.backbone(img)                # HF BaseModelOutput
            cls_tok = out.last_hidden_state[:, 0, :]
            fused   = fuse_feats(cls_tok)
            return self.activation(self.classifier(fused))
    
        # 6) H-Optimus 0 & 1
        if bb.startswith("hoptimus0") or bb.startswith("hoptimus1"):
            img_feat = self.backbone(img)               # (B, D)
            fused    = fuse_feats(img_feat)
            return self.activation(self.classifier(fused))
    
        # 7) GigaPath
        if bb.startswith("gigapath"):
            img_feat = self.backbone(img)               # (B, D)
            fused    = fuse_feats(img_feat)
            return self.activation(self.classifier(fused))
        
        if bb == "virchow":
            return_cls=True

            output = self.backbone(img)
            class_token = output[:, 0]
            fused    = fuse_feats(class_token)

            if return_cls:
               return self.activation(self.classifier(fused))
            else:
                patch_tokens = output[:, 1:]
                fused    = fuse_feats(patch_tokens)
                embeddings = torch.cat([class_token, fused.mean(1)], dim=-1)
                return self.activation(self.classifier(embeddings))
        

        elif bb.startswith("hibou_l"):

            # pass the image through the Hugging-Face AutoModel
            outputs = self.backbone(pixel_values=img)
            # grab the pooled [CLS] token
            x = outputs.pooler_output
            # then your classification head + activation
            fused = fuse_feats(x)
            logits = self.classifier(fused)
            return self.activation(logits)
        
        if bb.startswith("midnight12k"):
            return_type: Literal["cls_token", "cls+mean"] = "cls_token"

            out = self.backbone(img).last_hidden_state
            cls_token = out[:, 0, :]
            if return_type == "cls_token":
                fused = fuse_feats(cls_token)
                return self.activation(self.classifier(fused))
            
            elif return_type == "cls+mean":
                patch_embeddings = out[:, 1:, :]
                features= torch.cat([cls_token, patch_embeddings.mean(1)], dim=-1)
                fused      = fuse_feats(features)
                return self.activation(self.classifier(fused))

            else:
                raise ValueError(
                    f"expected return_type to be one of 'cls_token' or 'cls+mean', but got '{self.return_type}'"
                )
            

    # ======================================================
    #  Lightning hooks
    # ======================================================
    def training_step(self, batch, _):
        data, y = batch
        x, m = data['img'], data['msk']

        logits = self(x,m)
        loss   = self.cls_loss(logits,y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

 #####################  Finetuning the f1 score threshold  #####################
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Lightning will insert these into the .ckpt dictionary
        checkpoint['best_thr']    = self._best_thr
        checkpoint['best_val_f1'] = self._best_val_f1

    # 2) Load them back when reloading
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # pull them back into your module
        self._best_thr    = checkpoint.get('best_thr',    self._best_thr)
        self._best_val_f1 = checkpoint.get('best_val_f1', self._best_val_f1)

    def on_validation_epoch_start(self):
        self.val_logits = []
        self.val_labels = []
        self.val_losses = []

    def validation_step(self, batch, batch_idx):
        data, y    = batch
        logits     = self(data['img'], data['msk'])
        loss       = self.cls_loss(logits, y)

        # stash for epoch
        self.val_logits.append(logits.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        self.val_losses.append(loss.detach().cpu())

    def on_validation_epoch_end(self):
        # stack
        logits = torch.cat(self.val_logits, dim=0)
        logits = logits.to(torch.float32)            # cast to FP32

        labels = torch.cat(self.val_labels, dim=0).numpy()
        mean_loss = torch.stack(self.val_losses).mean().item()

        # 1) your existing argmax‐based metrics
        preds_arg = logits.argmax(dim=1).numpy()
        self.log("val_loss_epoch", mean_loss, prog_bar=True, sync_dist=True)
        self.log("val_f1_epoch", skm.f1_score(labels, preds_arg, average="binary"),
                 prog_bar=True, sync_dist=True)
        self.log("val_acc", skm.accuracy_score(labels, preds_arg),
                 prog_bar=True, sync_dist=True)
        self.log("val_prec", skm.precision_score(labels, preds_arg, average="binary", zero_division=0),
                 sync_dist=True)
        self.log("val_rec", skm.recall_score(labels, preds_arg, average="binary"),
                 sync_dist=True)

        # 2) sweep thresholds on the positive‐class probability
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        prec, rec, ths = skm.precision_recall_curve(labels, probs)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.nanargmax(f1s)
        best_thr = ths[best_idx] if best_idx < len(ths) else 0.5
        best_f1  = f1s[best_idx]

        # log the tuned threshold + its F1
        self.log("val_best_thr", best_thr,    prog_bar=False, sync_dist=True)
        self.log("val_f1_tuned", best_f1,     prog_bar=True,  sync_dist=True)

        # if it’s the best so far, remember it
        if best_f1 > self._best_val_f1:
            self._best_val_f1 = best_f1
            self._best_thr    = best_thr

        # clear for next epoch
        self.val_logits.clear()
        self.val_labels.clear()
        self.val_losses.clear()

    # ————— Test —————
    def on_test_start(self):
        self.test_logits = []
        self.test_labels = []
        self.test_preds_argmax=[]

    def test_step(self, batch, batch_idx):
        data, y    = batch
        logits     = self(data['img'], data['msk'])
        logits= logits.to(torch.float32) 
        self.test_logits.append(logits.detach().cpu())
        self.test_labels.append(y.detach().cpu())

        ###### For argmax-based metrics ######
        
        preds_argmax  = logits.argmax(dim=1)
        self.test_preds_argmax.append(preds_argmax.detach().cpu())

    def on_test_epoch_end(self):
        logits = torch.cat(self.test_logits, dim=0)
        labels = torch.cat(self.test_labels, dim=0).numpy()
        probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
       
        if self.log_dir is None:
            self.log_dir = self.trainer.log_dir
        

        # apply the best validation threshold
        preds = (probs >= self._best_thr).astype(int)
        
        print(f"Best threshold found for f1-score ={self._best_thr:.3f}, best f1 validation={self._best_val_f1:.3f}")
        # log *all* your existing metrics — now on thresholded preds
        self.log("sklearn test_f1_binary",      skm.f1_score(labels, preds, average="binary"),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_precision",      skm.precision_score(labels, preds, average="binary",zero_division=0),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_recall",         skm.recall_score(labels, preds, average="binary"),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_balanced_acc",   skm.balanced_accuracy_score(labels, preds),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_acc",            skm.accuracy_score(labels, preds),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_f1 macro",       skm.f1_score(labels, preds, average="macro"),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_f1 micro",       skm.f1_score(labels, preds, average="micro"),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_f1 weighted",    skm.f1_score(labels, preds, average="weighted"),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_auc macro",    skm.roc_auc_score(labels, probs, average="macro"),
                 prog_bar=True, sync_dist=True)

        self.log("sklearn test_auc weighted",    skm.roc_auc_score(labels, probs, average="weighted"),
                 prog_bar=True, sync_dist=True)
        
        ###### For argmax-based metrics ######
        preds_argmax = torch.cat(self.test_preds_argmax).numpy()
        self.log("sklearn test_f1_binary using argmax and default f1-threshold",      skm.f1_score(labels, preds_argmax, average="binary"),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_precision using argmax and default f1-threshold",      skm.precision_score(labels, preds_argmax, average="binary", zero_division=0),
                 prog_bar=True, sync_dist=True)
        self.log("sklearn test_recall using argmax and default f1-threshold",         skm.recall_score(labels, preds_argmax, average="binary"),
                 prog_bar=True, sync_dist=True)

        # save out for later
        np.savez( 
            os.path.join(self.log_dir, "test_preds_labels.npz"),
            probs=probs,
            logits=logits,
            labels=labels
        )
        print(f"Saved test logits, probs, and labels to {self.log_dir}/test_preds_labels.npz")

        # clean up
        self.test_logits.clear()
        self.test_labels.clear()
        self.test_preds_argmax.clear()

    def predict_step(self, batch, _):
        data,_ = batch
        return softmax(self(data['img'],data['msk']),dim=1)

    # ======================================================
    #  Optimizer & Scheduler
    # ======================================================
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['OPTIMIZER']['lr'],
            eps=self.config['OPTIMIZER']['eps'],
            weight_decay=self.config['REGULARIZATION']['Weight_Decay']
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.config['BASEMODEL']['Max_Epochs']
        )
        return [opt], [sched]

