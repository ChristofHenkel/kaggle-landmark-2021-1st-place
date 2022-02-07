from default_config import basic_cfg
import albumentations as A
import os
import pandas as pd
import cv2

cfg = basic_cfg
cfg.debug = True

# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.data_dir = "/raid/landmark-recognition-2019/"

cfg.train
cfg.data_folder = cfg.data_dir + "train/"
cfg.train_df = "/mount/glr2021/data/2021/train_gldv2x.csv"

cfg.val_df = '/raid/landmark-recognition-2019/' + "recognition_solution_v2.1.csv"
cfg.output_dir = f"/mount/glr2021/models/{os.path.basename(__file__).split('.')[0]}"
cfg.val_data_folder = "/raid/landmark-recognition-2019/" + "test/"

cfg.test = False
cfg.test_data_folder = cfg.data_dir + "test/"
# cfg.test_df = cfg.data_dir + "sample_submission_v1.csv"

cfg.eval_retrieval = True
cfg.query_data_folder = "/raid/landmark-recognition-2019/" + "test/"
cfg.index_data_folder = "/raid/landmark-recognition-2019/" + "index/"
cfg.query_df = '/mount/glr2021/data/2019/query_v2.csv'
cfg.index_df = '/mount/glr2021/data/2019/index_v2.csv'

#logging
cfg.neptune_project = "christofhenkel/glr2021"
cfg.neptune_connection_mode = "async"
cfg.tags = "debug"





# MODEL
cfg.model = "ch_mdl_hybrid_transformer_2x"
cfg.stride = (1,1)
cfg.embedder = "tf_efficientnet_b5_ns"
cfg.backbone = "swin_base_patch4_window7_224"
cfg.freeze_backbone_head = False
cfg.find_unused_parameters = True
cfg.neck = "option-D"
cfg.embedding_size = 512
cfg.pool = "gem"
cfg.gem_p_trainable = True
cfg.pretrained_weights ='/mount/glr2021/models/cfg_ch_hybrid_swin224_2x_b5_cutout_s2x/fold0/checkpoint_last_seed248126.pth'
cfg.pretrained_weights_strict = False
cfg.pretrained=True
cfg.pop_weights = ['patch_embed.proj.weight']
# DATASET
cfg.dataset = "ch_ds_1"
cfg.normalization = 'imagenet'
cfg.landmark_id2class_id = pd.read_csv('./assets/landmark_id2class.csv')
cfg.num_workers = 8
# cfg.data_sample = 100000
cfg.loss = 'adaptive_arcface'
cfg.arcface_s = 45
cfg.arcface_m = 0.3


# OPTIMIZATION & SCHEDULE

# cfg.fold = 0
cfg.lr = 0.00005
# cfg.optimizer = "adam"
# cfg.weight_decay = 1e-4
cfg.warmup = 1
cfg.epochs = 40
cfg.stop_at = 16
cfg.batch_size = 8
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 1.

#inference
cfg.train = True
cfg.val = True
cfg.test = False
cfg.save_val_data = True
cfg.train_val = False
cfg.save_only_last_ckpt = False
cfg.eval_ddp =True
cfg.save_headless = False
# AUGS

cfg.img_size = (448,448)
# AUGS

image_size = cfg.img_size[0]

cfg.train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.Resize(image_size, image_size),
        A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
    ])

cfg.val_aug = A.Compose([
        A.Resize(image_size, image_size),
    ])

