'''
VQGAN code, adapted from the original created by the Unleashing Transformers authors:
https://github.com/samb-t/unleashing-transformers/blob/master/models/vqgan.py

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY
import functools
from basicsr.archs.network_swinir import RSTB,RSTB_AR

def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


#  Define VQVAE classes
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta 
       

        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)


    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1)  - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t() )  
        
        min_encoding_indices = torch.argmin(d, dim=1)
        
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        dict_indices = {"min_encoding_indices" : min_encoding_indices}


        return z_q, loss, dict_indices

    def get_codebook_feat(self, indices,shape,soft_idx=None):

        if self.training:
            soft_idx = soft_idx.view(-1,soft_idx.shape[-1])
            logits_BLV_hard = soft_idx.argmax(dim=1)
            hard_one_hot = torch.nn.functional.one_hot(logits_BLV_hard, num_classes=soft_idx.size(1)).float()
            hard_one_hot = soft_idx + (hard_one_hot - soft_idx).detach()
            # get quantized latent vectors
            z_q = torch.matmul(hard_one_hot, self.embedding.weight.detach())
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        else:
            z_q = self.embedding(indices.view(-1))
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
            
        return  z_q
   


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class SwinAttnLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(1):
            layer = RSTB_AR(embed_dim, input_resolution, blk_depth, num_heads, window_size, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x):
      
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
          
        x = x.transpose(1, 2).reshape(b, c, h, w) 
      
        return x

    
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.ds = nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.ds(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class Generator(nn.Module):
    def __init__(self, ch_mult, res_blocks):
        super().__init__()
      
        self.ch_mult = ch_mult 
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        
        block_in_ch = 64

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(block_in_ch, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = 64 * self.ch_mult[i]

            for j in range(self.num_res_blocks+1):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if i == self.num_resolutions - 3 and j <2:
                        blocks.append(SwinAttnLayers(input_resolution=(32, 32),blk_depth=2, embed_dim=256))
                if i == self.num_resolutions - 2 and j <2:
                    blocks.append(SwinAttnLayers(input_resolution=(16, 16),blk_depth=2, embed_dim=128))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
               

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, 3, kernel_size=3, stride=1, padding=1))




        blocks_1 = blocks[:13]
        blocks_1.append(ResBlock(128,256))
        blocks_1.append(ResBlock(256,256))


        blocks_2 = []
        blocks_2.append(ResBlock(512,256))
        blocks_2.append(ResBlock(256,256))
        blocks_2.append(normalize(256))
        blocks_2.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

        blocks_3 = []
        blocks_3.append(ResBlock(256,256))
        for b in blocks[14:]:
            blocks_3.append(b)


        self.blocks_1 = nn.ModuleList(blocks_1)
        self.blocks_2 = nn.ModuleList(blocks_2)
        self.blocks_3 = nn.ModuleList(blocks_3)


    def forward(self, x,x_low,x_low_q):
        
        if x is not None:
            for block in self.blocks_1:
                x = block(x)
               

            x_fuse = torch.cat([x.clone(),x_low],dim=1)
            for block in self.blocks_2:
                x_fuse =  block(x_fuse)
               

            return x,x_fuse
            
        if x_low_q is not None:
            for block in self.blocks_3:
                x_low_q = block(x_low_q)
                
               
            return x_low_q


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        blocks = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1))
        blocks.append(ResBlock(64, 64))    #256  
        blocks.append(ResBlock(64, 64)) 
        blocks.append(Downsample(64))

        blocks.append(ResBlock(128, 128))     #128
        blocks.append(ResBlock(128, 128))
        blocks.append(Downsample(128))

        blocks.append(ResBlock(256, 128))    #64
        blocks.append(ResBlock(128, 128))
        blocks.append(Downsample(128))

        blocks.append(ResBlock(256, 256))    #32
        blocks.append(SwinAttnLayers(input_resolution=(32, 32),blk_depth=4, embed_dim=256))
        blocks.append(ResBlock(256, 256))
        blocks.append(Downsample(256))

        blocks.append(ResBlock(512, 256))     #16
        blocks.append(SwinAttnLayers(input_resolution=(16, 16),blk_depth=2, embed_dim=256))
        blocks.append(ResBlock(256, 256))
        blocks.append(Downsample(256))

        blocks.append(ResBlock(512, 256))     #8
        blocks.append(ResBlock(256, 256))
        blocks.append(ResBlock(256, 128))
        blocks.append(ResBlock(128, 128))
        blocks.append(ResBlock(128, 64))
        blocks.append(ResBlock(64, 64))
        blocks.append(ResBlock(64, 64))
        blocks.append(normalize(64))
        blocks.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

     
    def forward(self, x):
        i=0
        for block in self.blocks:
            x = block(x)
            if i == 12:
                x_low = x
            i = i + 1
        
        return x,x_low



@ARCH_REGISTRY.register()
class VQAutoEncoder(nn.Module):
    def __init__(self,  ch_mult, res_blocks=2, codebook_size=1024, emb_dim=256,
                beta=0.25,  model_path=None):
        super().__init__()
        logger = get_root_logger()

        self.n_blocks = res_blocks 
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult

        self.encoder = Encoder()

       
        self.beta = beta #0.25
        self.quantize = VectorQuantizer(self.codebook_size,self.embed_dim, self.beta)
        
        self.generator = Generator(
            self.ch_mult, 
            self.n_blocks, 
        )
        self.lq_encoder = LqEncoder()
        self.lq_generator = LQGenerator()

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'])
                logger.info(f'vqgan is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
                logger.info(f'vqgan is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')

    def forward(self, x,lr):
       
        x , x_low = self.encoder(x)

        x_hr = x.clone()
       
        x,x_fuse = self.generator(x,x_low,None)
        
        quant_low, codebook_loss, quant_stats = self.quantize(x_fuse)

        x = x + quant_low
      

        x = self.generator(None,None,x)
        if lr is not None:
            x_lr = self.lq_encoder(lr)
            img_lr = self.lq_generator(x_lr)
            return x, codebook_loss, quant_stats,x_hr,x_lr,img_lr
        else:
            return x



# patch based discriminator
@ARCH_REGISTRY.register()
class VQGANDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, model_path=None):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(VQGANDiscriminator, self).__init__()
       
        norm_layer = nn.BatchNorm2d
       
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_d' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
            else:
                raise ValueError(f'Wrong params!')

    def forward(self, input0, input1=None):
        """Standard forward."""
        
        return self.main(input0), self.main(input1) if input1 is not None else None



class Downsample2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.ds = nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.ds(x)
        return x

class LqEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
    
        blocks = []
        # initial convultion  #32x32 3
        blocks.append(nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)) #32x32
        blocks.append(ResBlock(32, 32))
        blocks.append(ResBlock(32, 32))  
        blocks.append(Downsample2(32))  

        blocks.append(ResBlock(64, 64))  #16x16
        blocks.append(ResBlock(64, 64))
        blocks.append(Downsample2(64)) 
       
        blocks.append(ResBlock(128, 64))   #8x8
        blocks.append(ResBlock(64, 64))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x


class LQGenerator(nn.Module):
    def __init__(self,out_channels=3):
        super().__init__()
        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        blocks.append(ResBlock(64, 64))
        blocks.append(ResBlock(64, 64))

        blocks.append(Upsample(64))
        blocks.append(ResBlock(64, 64))
        blocks.append(ResBlock(64, 64))

        blocks.append(Upsample(64))
        blocks.append(ResBlock(64, 32))
        blocks.append(ResBlock(32, 32))

        blocks.append(normalize(32))
        blocks.append(nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)


    def forward(self, x):    

        for block in self.blocks:
            x = block(x)

        return x