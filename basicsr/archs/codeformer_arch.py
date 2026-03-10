import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

from basicsr.archs.tokenizer_arch import *
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.network_swinir import RSTB,RSTB_AR


class SwinLayers_ar(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                n_layers=9,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(n_layers):
            layer = RSTB_AR(embed_dim, input_resolution, blk_depth, num_heads, window_size, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x,shape):
      
        b, c, h, w = shape
        for m in self.swin_blks: #b,  h*w , c
            x = m(x, (h, w)) 
        return x

class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                n_layers=2,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(n_layers):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x, is_skip_shift = 0):
      
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        if is_skip_shift == 0:
            for m in self.swin_blks:
                x = m(x, (h, w))
        else:
            for m in self.swin_blks:
                x = m(x, (h, w),1)
          
        x = x.transpose(1, 2).reshape(b, c, h, w) 
      
        return x


class LqEncoder(nn.Module):
    def __init__(self, in_channels=3, nf=64):
        super().__init__()
    
        blocks = []
        # initial convultion  #64x64 3
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)) #64x64 64
        blocks.append(ResBlock(64, 128))
        blocks.append(ResBlock(128, 128))
        blocks.append(ResBlock(128, 128))  #64x64 128

        blocks.append(Downsample(128))  #32x32 256
        blocks.append(ResBlock(256, 256))
        blocks.append(ResBlock(256, 256))

        blocks.append(SwinLayers(input_resolution=(32, 32), embed_dim=256))         
        blocks.append(ResBlock(256, 256))
        blocks.append(ResBlock(256, 256))
        
        blocks.append(Downsample(256)) #16x16 256
        blocks.append(ResBlock(512, 256))
        blocks.append(ResBlock(256, 128))

        blocks.append(Downsample(128)) #8x8 256
        blocks.append(ResBlock(256, 256))
        blocks.append(ResBlock(256, 128))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):     
        i = 0
        for block in self.blocks:
            x = block(x)

            if i == 9:
                x_low = x.clone()
            i = i + 1
        
        return x , x_low

class Exactor(nn.Module):
    def __init__(self):
        super().__init__()
    
        blocks = []
        blocks.append(ResBlock(128, 128))
        blocks.append(ResBlock(128, 64))
        blocks.append(ResBlock(64, 64))
        blocks.append(ResBlock(64, 64))
        
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
      

        for block in self.blocks:
            x = block(x)
        
        return x 



@ARCH_REGISTRY.register()
class CodeFormer(VQAutoEncoder):
    def __init__(self, dim_embd=512, 
                codebook_size=1024,
                fix_modules=['quantize','generator'], vqgan_path=None):
        super(CodeFormer, self).__init__([1, 2, 2, 4, 2,1],2, codebook_size)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'])

  
        self.ft_layers_low =  SwinLayers_ar(input_resolution=(32, 32) , n_layers =2) 


        self.idx_pred_layer_low = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))
        
        self.lqencoder = LqEncoder()
        self.exactor = Exactor()

        blocks_fuse = []
        blocks_fuse.append(ResBlock(512,256))
        blocks_fuse.append(SwinLayers(input_resolution=(32, 32),n_layers=1)) 
        blocks_fuse.append(ResBlock(256,256))
        self.blocks_fuse = nn.ModuleList(blocks_fuse)
    

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, code_only=False):
        # ################### encoder of stage2 #####################
        x , x_low  = self.lqencoder(x)
        x = self.exactor(x)

        lq_feat = x
        lq_feat_low = x_low
       
        # ################# fixed trained decoder of stage1 ###################
        for i, block in enumerate(self.generator.blocks_1):
            x = block(x) 
        
        ################# predictor ###################
        x_low_d = x.detach()
        lq_feat_low_fuse = torch.cat([x_low_d,lq_feat_low],dim=1)

        for block in self.blocks_fuse:
                lq_feat_low_fuse =  block(lq_feat_low_fuse)
        
        feat_emb_low = lq_feat_low_fuse.permute(2,3,0,1)
        query_emb_low = feat_emb_low.view(-1,feat_emb_low.shape[2],feat_emb_low.shape[3])
        query_emb_low = query_emb_low.transpose(0,1)
        query_emb_low = self.ft_layers_low(query_emb_low,x_low.shape)
        query_emb_low = query_emb_low.transpose(0,1)
        
        logits_low = self.idx_pred_layer_low(query_emb_low) # (hw)bn
        logits_low = logits_low.permute(1,0,2) # (hw)bn -> b(hw)n


        if code_only: # for training stage 2
            return lq_feat ,logits_low
        
        # ################# fixed trained quantizer of stage1 ###################
        soft_one_hot_low = F.softmax(logits_low, dim=2)
        _, top_idx_low = torch.topk(soft_one_hot_low, 1, dim=2)

        quant_feat_low = self.quantize.get_codebook_feat(top_idx_low, [x_low.shape[0],x_low.shape[2],x_low.shape[3],256],soft_one_hot_low)

        # ################# fixed trained decoder of stage1 ###################
        x = x + quant_feat_low
        for i, block in enumerate(self.generator.blocks_3):
            x = block(x) 
        
        return x