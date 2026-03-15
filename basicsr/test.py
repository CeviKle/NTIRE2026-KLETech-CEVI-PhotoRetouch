import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train_INR import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    test_names = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if 'test' in phase:
            dataset_opt['phase'] = 'test'
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)
        test_names.append(dataset_opt['name'])

    # create model
    model = create_model(opt)
    total_metrics = {}

    for i, test_loader in enumerate(test_loaders):
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # wheather use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)
        
        for data in test_loader:
            model.feed_data(data, is_val=True)
            model.validation(
                current_iter=opt['name'],
                tb_logger=None,
                save_img=opt['val']['save_img'],
                rgb2bgr=rgb2bgr, use_image=use_image)
            if "metrics" in opt['val']:
                for k, v in model.collected_metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v

        if "metrics" in opt['val']: 
            log_str = f'Final Results for {test_names[i]}, \t'
            for metric, value in total_metrics.items():
                log_str += f'\t # {metric}: {value/(len(test_loader)):.4f}'
            logger = get_root_logger()
            logger.info(log_str)

            



if __name__ == '__main__':
    main()
