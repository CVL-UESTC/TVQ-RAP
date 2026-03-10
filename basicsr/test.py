import sys
from pathlib import Path
# 获取 train.py 的上上级目录
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import datetime
import logging
import math
import copy
import random
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info, get_root_logger, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import warnings
# ignore UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
warnings.filterwarnings("ignore", category=UserWarning)

def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project') is not None):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger'):
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_val_dataloader(opt, logger):
    # create val dataloaders
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: ' f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return  val_loader


def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(root_path, is_train=True)

    torch.backends.cudnn.benchmark = True
   
    make_exp_dirs(opt)
    if opt['logger'].get('use_tb_logger') and opt['rank'] == 0:
        mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)
    
    # create train and validation dataloaders
    val_loader = create_val_dataloader(opt, logger)

    model = build_model(opt)
    model.validation(val_loader, 0, tb_logger, True)

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
