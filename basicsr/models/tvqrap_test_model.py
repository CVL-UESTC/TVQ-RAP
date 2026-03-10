import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm


from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .sr_model import SRModel

from basicsr.losses import build_loss
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .sr_model import SRModel
import numpy as np
import random

from basicsr.utils.util_image import ImageSpliterTh

@MODEL_REGISTRY.register()
class TVQRAP_test_Model(SRModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True),  'params_ema')
        self.net_g.eval()
        
       

    def feed_data(self, data):
        # for paired training or validation
        self.lq = data['lq'].to(self.device)
        self.hq = None
   
        self.h = data['h']
        self.w = data['w']



    def test(self):

        with torch.no_grad():
            self.p = []
            self.srp = []
            context = torch.cuda.amp.autocast 

            crop_size = 768
            stride = 688
            if self.lq.shape[2] > crop_size or self.lq.shape[3] > crop_size:
                im_spliter = ImageSpliterTh(
                        self.lq,
                        crop_size,
                        stride=stride,
                        sf=4,
                        extra_bs=1,
                        )
                for im_lq_pch, index_infos in im_spliter:
                    with context():
                        self.p.append(im_lq_pch)
                        im_sr_pch =  self.net_g(im_lq_pch)
                        self.srp.append(im_sr_pch)
                    im_spliter.update(im_sr_pch, index_infos)
                self.output = im_spliter.gather()
            else:
                with context():
                    self.output = self.net_g(self.lq)
        


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        
        pbar = tqdm(total=len(dataloader), unit='image')

        l_g_total = 0
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
           
            self.feed_data(val_data)
            self.test()
          

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], min_max=[-1,1])
            sr_img = sr_img[0:self.h*4 ,0:self.w*4, ...]

            # tentative for out of GPU memory
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_out_path = osp.join(self.opt['path']['visualization'],"current_iter","fake",
                                             f'{img_name}_out.png')
               
                imwrite(sr_img, save_img_out_path)

             
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()


    def get_current_visuals(self):
        out_dict = OrderedDict()
   
        out_dict['result'] = self.output.detach().cpu()
        return out_dict

