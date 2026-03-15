import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLoggerINR, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse
import copy

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

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
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # tb_logger = init_tb_logger(log_dir=f'./logs/{opt['name']}') #mkdir logs @CLY
        tb_logger = init_tb_logger(log_dir=osp.join('logs', opt['name']))
    return logger, tb_logger


def create_INR_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of train images/folders in {dataset_opt["name"]}: '
                f'{len(train_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader

def main():
    
    # parse options, set distributed setting, set ramdom seed
    opt_og = parse_options(is_train=True)
    if "Windows_batch_size" in opt_og["datasets"]["train"]:
        opt_og["datasets"]["train"]["samples_batch_size"] = opt_og["datasets"]["train"]["Windows_batch_size"] * opt_og["datasets"]["train"]["window_size"]**2

    opt = copy.deepcopy(opt_og)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir for experiments and logger
    make_exp_dirs(opt)
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
            'name'] and opt['rank'] == 0:
        mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    total_metrics = {}

    # create train and validation dataloaders
    train_loader = create_INR_dataloader(opt, logger)

    # create message logger (formatted outputs)
    msg_logger = MessageLoggerINR(opt, start_iter=1, tb_logger=tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")


    prefetcher.reset()
    inr_data = prefetcher.next()

    total_iters = opt["train"]["total_iter"]
    start_time = time.time()
    

    model = create_model(opt)
    current_iter = 0


    rgb2bgr = opt['val'].get('rgb2bgr', True)
    # wheather use uint8 image to compute metrics
    use_image = opt['val'].get('use_image', True)
    
    while inr_data is not None:

        natural_name = osp.splitext(osp.basename(inr_data["natural_path"][0]))[0]
        style_name = osp.splitext(osp.basename(inr_data["style_path"][0]))[0]
        style = inr_data["style_path"][0].split('/')[-2]


        # training
        logger.info(
            f'Start training for style {style}:  Reference Image: {style_name}, and Input Image : {natural_name} for iter number: {total_iters}')
        start_time = time.time()


        model.feed_data(inr_data, is_val=False)
        
        while current_iter <= total_iters:
                        
            current_iter += 1
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            result_code = model.optimize_parameters(current_iter)
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'img_cnt': 1, 'total_imgs': 1, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

        current_iter = 0

        print(f"Train Time: { time.time() - start_time}")
        opt = copy.deepcopy(opt_og)

        # save models and training states
        if  opt['logger']['save_checkpoint']:
            logger.info(f'Saving model for for style: Before Edit: {natural_name}, and After Edit : {style_name} for iter number: {total_iters}')
            model.save(style)


        model.feed_data(inr_data, is_val=True)
        model.validation(current_iter, tb_logger, opt['val']['save_img'], rgb2bgr, use_image)

        if "metrics" in opt['val']:
            for k, v in model.collected_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

        inr_data = prefetcher.next()


    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')

    if "metrics" in opt['val']: 
        log_str = f'Final Results, \t'
        for metric, value in total_metrics.items():
            log_str += f'\t # {metric}: {value/(len(train_loader)):.4f}'
        logger = get_root_logger()
        logger.info(log_str)

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    import os
    os.environ['GRPC_POLL_STRATEGY']='epoll1'
    main()
