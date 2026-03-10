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

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.img_process_util import  USMSharp
from basicsr.utils.diffjpeg import DiffJPEG

from torchvision.transforms.functional import normalize

from basicsr.losses import build_loss

@MODEL_REGISTRY.register()
class TVQRAP_stage2_Model(SRModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
        self.usm_sharpener = USMSharp().cuda()
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def reset_queue(self):
        """Reset the training pair pool (queue).
        Useful when changing training stages or datasets.
        """
        if hasattr(self, 'queue_lr'):
            del self.queue_lr
            del self.queue_gt
            del self.queue_ptr
            torch.cuda.empty_cache()
            
        logger = get_root_logger()
        logger.info(f'Degradation queue has been reset.')

    def feed_data(self, data,is_train = 1):
        if is_train == 1:
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if random.random() < self.opt['second_order_prob']:
                # blur
                if np.random.uniform() < self.opt['second_blur_prob']:
                    out = filter2D(out, self.kernel2)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range2'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                    out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
                # add noise
                gray_noise_prob = self.opt['gray_noise_prob2']
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            # self.gt = torch.clamp((self.gt * 255.0).round(), 0, 255) / 255.
            
         
            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            

            mean= [0.5, 0.5, 0.5]
            std= [0.5, 0.5, 0.5]
            normalize(self.lq, mean, std, inplace=True)
            normalize(self.gt, mean, std, inplace=True)

            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
            self.input = self.lq

            self.b = self.gt.shape[0]
           

        else:
           
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            # if 'gt' in data:
            self.gt = data['gt'].to(self.device)
                
            mean= [0.5, 0.5, 0.5]
            std= [0.5, 0.5, 0.5]
            normalize(self.lq, mean, std, inplace=True)
            normalize(self.gt, mean, std, inplace=True)
            self.input = self.lq
          
            

                
    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if self.opt['datasets']['train'].get('latent_gt_path', None) is not None:
            self.generate_idx_gt = False
        elif self.opt.get('network_vqgan', None) is not None:
            self.hq_vqgan_fix = build_network(self.opt['network_vqgan']).to(self.device)

            load_path = self.opt['network_vqgan'].get('model_path', None)
            self.load_network(self.hq_vqgan_fix, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')

            self.hq_vqgan_fix.eval()
            self.generate_idx_gt = True
            for param in self.hq_vqgan_fix.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError(f'Shoule have network_vqgan config or pre-calculated latent code.')
        
        logger.info(f'Need to generate latent GT code: {self.generate_idx_gt}')

        self.hq_feat_loss = train_opt.get('use_hq_feat_loss', True)
        self.feat_loss_weight = train_opt.get('feat_loss_weight', 1.0)
        self.cross_entropy_loss = train_opt.get('cross_entropy_loss', True)
        self.entropy_loss_weight = train_opt.get('entropy_loss_weight', 0.5)

        self.net_g.train()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

      
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


    def optimize_parameters(self, current_iter):

        logger = get_root_logger()
        # optimize net_g
        self.optimizer_g.zero_grad()

        x, x_low = self.hq_vqgan_fix.encoder(self.gt)
     
        _, x_fuse = self.hq_vqgan_fix.generator(x,x_low,None)
        _, _, quant_stats = self.hq_vqgan_fix.quantize(x_fuse)

        min_encoding_indices = quant_stats['min_encoding_indices']
        self.idx_gt = min_encoding_indices.view(self.b, -1)

    
        lq_feat,logits = self.net_g(self.input, code_only=True)


        l_g_total = 0
        loss_dict = OrderedDict()
        # hq_feat_loss
        if self.hq_feat_loss: # codebook loss 
            l_lq = torch.mean( (x.detach()-lq_feat)**2) * self.feat_loss_weight
            l_g_total += l_lq
            loss_dict['l_feat_encoder'] = l_lq

            loss_dict['distance'] =  torch.mean( torch.sqrt((x-lq_feat)**2) ) 
            loss_dict['value'] = torch.mean(torch.sqrt(x**2)) 


        # cross_entropy_loss
        if self.cross_entropy_loss:
            # b(hw)n -> bn(hw)
            cross_entropy_loss = F.cross_entropy(logits.permute(0, 2, 1), self.idx_gt) * self.entropy_loss_weight
            l_g_total += cross_entropy_loss
            loss_dict['cross_entropy_loss'] = cross_entropy_loss

        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.output = self.net_g_ema(self.input)
               
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                self.output = self.net_g(self.input)
                self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data,0)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']],min_max=[-1,1])
            in_img = tensor2img([visuals['in']],min_max=[-1,1])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']],min_max=[-1,1])
               
            if save_img:
                save_img_out_path = osp.join(self.opt['path']['visualization'],str(current_iter),"fake",
                                            f'{img_name}_out.png')
                imwrite(sr_img, save_img_out_path)     
          
            del self.gt
            del self.output
            torch.cuda.empty_cache()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['in'] = self.input.detach().cpu()
        return out_dict


    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
