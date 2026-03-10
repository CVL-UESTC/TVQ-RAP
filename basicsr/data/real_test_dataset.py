import os
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
import os.path as osp
import cv2
from basicsr.data.data_util import paths_from_folder
import torchvision as thv
import torchvision.transforms as transforms
from utils import img_util
@DATASET_REGISTRY.register()
class Real_test_Dataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Real_test_Dataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.lq_folder =  opt['dataroot_lq']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        self.paths = paths_from_folder(self.lq_folder)


        self.transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=0.5, std=0.5),
        ])
       
    def __getitem__(self, index):
        lq_path = self.paths[index]
        img_lq = img_util.imread(lq_path, chn='rgb', dtype='float32')
     
        h, w = img_lq.shape[0:2]
        h_origin = h
        w_origin = w
        pad_h = (64 - h % 64) if h % 64 != 0 else 0
        pad_w = (64 - w % 64) if w % 64 != 0 else 0
        
        img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        img_lq = self.transform(img_lq)
      

        return {'lq': img_lq, 'h':h_origin, 'w':w_origin, 'lq_path': lq_path }

    def __len__(self):
        return len(self.paths)
