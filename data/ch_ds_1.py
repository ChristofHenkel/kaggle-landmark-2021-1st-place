from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch 
import albumentations as A
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import pandas as pd
from tqdm import tqdm

def batch_to_device(batch,device):
    batch_dict = {key:batch[key].to(device) for key in batch}
    return batch_dict


tr_collate_fn = None
val_collate_fn = None

def fix_row(row):
    if len(str(row).split()) > 1:
        row = int(str(row).split()[0])
    return row

class CustomDataset(Dataset):

    def __init__(self, df, cfg, aug, mode='train', allowed_targets=None):

        self.cfg = cfg
        self.df = df.copy()

        if mode == 'val':       
            self.df['landmark_id'] = self.df['landmarks'].apply(lambda x:fix_row(x))
            self.df['landmark_id'] = self.df['landmark_id'].fillna(-1)
            self.df['landmark_id'] = self.df['landmark_id'].astype(int)
            self.df['landmarks'].fillna('',inplace=True)                       

        self.landmark_id2class_id = dict(zip(cfg.landmark_id2class_id['landmark_id'].astype(int),
                                             cfg.landmark_id2class_id['class_id'].astype(int)))
        self.landmark_id2class_id[-1] = self.cfg.n_classes

        if mode == 'index':
            pass
               
        if mode == 'query':
            pass
        
        if mode in ['test','query','index']:
            self.df['target'] = 0
        else:
            self.df['target'] = self.df['landmark_id'].map(self.landmark_id2class_id)
            self.df['target'] = self.df['target'].fillna(-1).astype(int)
        if allowed_targets is not None:
            self.df = self.df[self.df['target'].isin(allowed_targets)]
        self.labels = self.df['target'].values

        self.image_ids = self.df['id'].values
        self.mode = mode
        self.aug = aug
        self.normalization = cfg.normalization
        tmp = np.sqrt(1 / np.sqrt(self.df['target'].value_counts().sort_index().values))
        self.margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * cfg.arcface_m_x + cfg.arcface_m_y
        
        if mode == 'test':
            self.data_folder = cfg.test_data_folder
        elif mode == 'val':
            self.data_folder = cfg.val_data_folder
        elif mode == 'query':
            self.data_folder = cfg.query_data_folder
        elif mode == 'index':
            self.data_folder = cfg.index_data_folder
        else:
            self.data_folder = cfg.data_folder
        

    def __getitem__(self, idx):
        
        image_id = self.image_ids[idx]
        label = self.labels[idx]


        img = self.load_one(image_id)   
        
        if self.aug:
            img = self.augment(img)

        img = img.astype(np.float32)
        if self.normalization:
            img = self.normalize_img(img)


        img = self.to_torch_tensor(img)
        feature_dict = {'input':img,
                       'target':torch.tensor(label).float(),
                        'image_idx':torch.tensor(idx)
                       }
        return feature_dict   
    
    def __len__(self):
        return len(self.image_ids)

    def load_one(self, id_):
        id_ = str(id_)
        fp = f'{self.data_folder}{id_[0]}/{id_[1]}/{id_[2]}/{id_}{self.cfg.suffix}'
        try:
            img = cv2.imread(fp)[:,:,::-1]
        except:
            print("FAIL READING img", id_)
        return img
    
    def augment(self,img):
        img_aug = self.aug(image=img)['image']
        return img_aug

    def normalize_img(self,img):
        
        if self.normalization == 'channel':
            #print(img.shape)
            pixel_mean = img.mean((0,1))
            pixel_std = img.std((0,1)) + 1e-4
            img = (img - pixel_mean[None,None,:]) / pixel_std[None,None,:]
            img = img.clip(-20,20)
           
        elif self.normalization == 'image':
            img = (img - img.mean()) / (img.std() + 1e-4)
            img = img.clip(-20,20)
            
        elif self.normalization == 'simple':
            img = img/255
            
        elif self.normalization == 'inception':
            mean = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img/255.
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        elif self.normalization == 'imagenet':
            mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
            std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        elif self.normalization == 'min_max':
            img = img - np.min(img)
            img = img / np.max(img)
            return img
        
        else:
            pass
        
        return img
    
    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1)))
