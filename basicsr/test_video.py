import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train_INR import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
import statistics
import cv2
from basicsr.utils import img2tensor
from basicsr.utils import get_root_logger
import numpy as np

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
    # create model
    model = create_model(opt)

    video = opt["video_inference"]
    cap = cv2.VideoCapture(video)
    retaining = True

    cnt = 0
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        if cnt == 0:
            video_out = cv2.VideoWriter('video.avi', fourcc=cv2.VideoWriter_fourcc(*'MJPG') ,fps=cap.get(cv2.CAP_PROP_FPS),frameSize=(frame.shape[1],frame.shape[0]))

        frame = img2tensor(frame.astype(np.float32) / 255., bgr2rgb=True, float32=True,)
        data = {'natural':frame.unsqueeze(0),"natural_path":video}
        rgb2bgr = True
        # wheather use uint8 image to compute metrics
        use_image = True
        model.feed_data(data, is_val=True)
        model.validation(
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr, use_image=use_image)
        video_out.write(model.out_img)
        cnt += 1
    
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
