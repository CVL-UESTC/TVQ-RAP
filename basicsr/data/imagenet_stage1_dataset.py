import cv2
import math
import random
import numpy as np
import os.path as osp
from scipy.io import loadmat
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, 
                                        adjust_hue, adjust_saturation, normalize)
from basicsr.data import gaussian_kernels as gaussian_kernels
from basicsr.data.transforms import augment
from basicsr.data.data_util import paths_from_folder, brush_stroke_mask, random_ff_mask
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Imagenet_stage1_Dataset(data.Dataset):

    def __init__(self, opt):
        super(Imagenet_stage1_Dataset, self).__init__()
        logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.gt_size = opt.get('gt_size', 256)
      
        
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])


        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = paths_from_folder(self.gt_folder)

        self.is_val = opt.get('is_val', False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]

        img_bytes = self.file_client.get(gt_path)
        if img_bytes is None:
            print(f"[Warning] Failed to load image bytes from {self.paths[index]}")
            return self.__getitem__((index + 1) % len(self.paths))
        img_gt = imfrombytes(img_bytes, float32=True)
        if img_gt is None:
            return self.__getitem__((index + 1) % len(self.paths))


        if self.is_val is not True:
    
            # random horizontal flip
            img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)

            h, w = img_gt.shape[0:2]
            crop_pad_size = max(min(h, w), self.gt_size)
        
            while h < crop_pad_size or w < crop_pad_size:
                pad_h = min(max(0, crop_pad_size - h), h)
                pad_w = min(max(0, crop_pad_size - w), w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                h, w = img_gt.shape[0:2]
            # crop
            crop_pad_size = self.gt_size
            if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
                h, w = img_gt.shape[0:2]
                # randomly choose top and left coordinates
                top = random.randint(0, h - crop_pad_size)
                left = random.randint(0, w - crop_pad_size)
                img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]


        # generate in image
        img_in = img_gt
        scale = 8
        img_in = cv2.resize(img_in, (int(self.gt_size // scale), int(self.gt_size // scale)), interpolation=cv2.INTER_AREA)

        # BGR to RGB, HWC to CHW, numpy to tensor
       
        img_in, img_gt = img2tensor([img_in, img_gt], bgr2rgb=True, float32=True)



        # round and clip
        img_in = np.clip((img_in * 255.0).round(), 0, 255) / 255.

        # Set vgg range_norm=True if use the normalization here
        # normalize
        normalize(img_in, self.mean, self.std, inplace=True)
        normalize(img_gt, self.mean, self.std, inplace=True)

        return_dict = {'in': img_in, 'gt': img_gt, 'gt_path': gt_path}
       

        return return_dict


    def __len__(self):
        return len(self.paths)