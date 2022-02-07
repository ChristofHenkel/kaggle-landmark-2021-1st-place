from types import SimpleNamespace
from copy import deepcopy
import numpy as np

cfg = SimpleNamespace(**{})

# dataset
cfg.dataset = "base_ds"
cfg.suffix = ".jpg"
cfg.n_classes = 81313
cfg.batch_size = 32
cfg.val_df = None
cfg.test = False
cfg.test_df = None
cfg.batch_size_val = None
cfg.normalization = "none"
cfg.train_aug = False
cfg.val_aug = False
cfg.test_augs = False
cfg.img_aug = False
cfg.scale = 1


# img model
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.stride = None
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.pretrained_weights_strict = True
cfg.pop_weights = None
cfg.freeze_backbone_head = False
cfg.resume_from = None
cfg.pool = "avg"
cfg.train = True
cfg.val = True
cfg.in_channels = 3
cfg.epoch_weights = None
cfg.calc_loss = True
cfg.gem_p_trainable = False
cfg.study_weight = 1
cfg.backbone_kwargs = {}
cfg.headless = False
cfg.pretrained_convhead = False
cfg.loss = 'adaptive_arcface'
cfg.arcface_m_x =  0.45
cfg.arcface_m_y = 0.05
cfg.dilations = [6,12,18]
cfg.stride = (2,2)
cfg.alpha = 1
cfg.cls_loss_pos_weight = None
cfg.train_val = True
cfg.eval_epochs = 1
cfg.eval_train_epochs = 5
cfg.drop_path_rate = 0.
cfg.drop_rate = 0.
cfg.dropout = 0.
cfg.attn_drop_rate=0.
cfg.warmup = 0
cfg.label_smoothing = 0
cfg.return_local_features = False

# training
cfg.fold = 0
cfg.val_fold = -1
cfg.lr = 1e-4
cfg.schedule = "cosine"
cfg.weight_decay = 0
cfg.optimizer = "Adam"  # "Adam", "fused_Adam", "SGD", "fused_SGD", "SAM"
cfg.sam_momentum = 0.9
cfg.epochs = 10
cfg.seed = -1
cfg.resume_training = False
cfg.do_test = True
cfg.stop_at = -1

cfg.clip_grad = 0
cfg.debug = False

#eval
cfg.eval_ddp = True
cfg.eval_retrieval = False
cfg.query_data_folder = None
cfg.index_data_folder = None
cfg.pre_train_val = 'tr_val'
cfg.reload_train_loader = False

# ressources
cfg.find_unused_parameters = False
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.syncbn = False
cfg.gpu = 0
cfg.dp = False
cfg.num_workers = 4
cfg.drop_last = True
cfg.pin_memory = False

# logging,
cfg.neptune_project = None
cfg.neptune_connection_mode = "async"
cfg.tags = None
cfg.calc_metric =True
cfg.sgd_nesterov = True
cfg.sgd_momentum = 0.9
cfg.clip_mode = "norm"
cfg.data_sample = -1

#saving
cfg.save_only_last_ckpt = True
cfg.save_headless = False
cfg.save_val_data = True
cfg.save_first_batch = False
cfg.save_first_batch_preds = False
cfg.save_checkpoint = True


cfg.mixup = 0
cfg.cutmix = 0
cfg.mosaic = 0
cfg.boxmix = 0

cfg.mixup_beta = 0.5
cfg.mixadd = False




cfg.on_only = False

cfg.stride = None

cfg.loss = "bce"

cfg.class_weights = None

cfg.tta = []


basic_cfg = cfg
