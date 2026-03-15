from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import random
import cv2
import numpy as np
from basicsr.utils.INR_utils import get_wh_mgrid
import torch
import os

class RTD_Benchmark(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RTD_Benchmark, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.sample_rate = opt['sample_rate'] if 'sample_rate' in opt else None
        
        self.ref_matrix = {}
        with open(opt['refs_file'],'r') as fi:
            for row in fi.readlines():
                refs = row.split(',')
                self.ref_matrix[refs[0]] = refs[1].rstrip()

        self.window_size = opt['window_size'] if 'window_size' in opt else None
        

        self.inp_natural, self.input_gt, self.style_natural, self.style_output = opt['inp_natural'], opt['input_gt'], opt['style_natural'], opt['style_output']

        self.files, self.nat_files, self.files_style = self.INR_retouch_pathes(self.inp_natural, self.input_gt, self.style_natural)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        style, file_name = self.files[index].split('/')

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        natural_path = self.inp_natural + '/' + file_name

        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(natural_path)
        try:
            img_natural = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception(" natural path {} not working".format(natural_path))

        gt_path = f"{self.input_gt}/{style}/{file_name}"

        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(gt_path)
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(img_gt))


        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_natural = img2tensor([img_gt, img_natural],
                                    bgr2rgb=True,
                                    float32=True,)


        ####################################### Style Reference ##########################################################
        

        style_ref = self.ref_matrix[file_name]
        
            
        style_path = f"{self.style_output}/{style}/{style_ref}"
        
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(style_path)
        try:
            img_style = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("style path {} not working".format(style_path))

        style_natural_path = self.style_natural + '/' + style_ref
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(style_natural_path)
        try:
            img_style_natural = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("style path {} not working".format(style_path))
        

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_style, img_style_natural = img2tensor([img_style, img_style_natural],
                                    bgr2rgb=True,
                                    float32=True,)
        
        if self.window_size:
            # C x Kernal^2 x L
            img_style_winds = torch.nn.functional.unfold(img_style.unsqueeze(0), self.window_size, dilation=1, padding=0, stride=self.sample_rate).reshape(img_style.shape[0],self.window_size**2,-1)
            img_style_winds = img_style_winds.permute(1, 2, 0)
            img_style_natural_winds = torch.nn.functional.unfold(img_style_natural.unsqueeze(0), self.window_size, dilation=1, padding=0, stride=self.sample_rate).reshape(img_style_natural.shape[0],self.window_size**2,-1)
            img_style_natural_winds = img_style_natural_winds.permute(1, 2, 0)

            img_style = img_style.permute(1, 2, 0)
            img_style_natural = img_style_natural.permute(1, 2, 0)


            corrds_style = get_wh_mgrid(img_style_natural.shape[0], img_style_natural.shape[1], flatten=False)
            corrds_style = corrds_style.float()
            corrds_style_window = torch.nn.functional.unfold(corrds_style.permute(2, 0, 1).unsqueeze(0), self.window_size, dilation=1, padding=0, stride=self.sample_rate).reshape(corrds_style.shape[-1],self.window_size**2,-1)
            corrds_style_window = corrds_style_window.permute(1, 2, 0)

            # Encoding input
            # wind_size*wind_size x samples x C
            style_input = torch.cat(
                [
                    corrds_style_window,
                    img_style_natural_winds,
                ],
                dim=2,
            )
            style_out = img_style_winds
    
        return {
            'natural': img_natural,
            'style_natural': img_style_natural,
            'gt': img_gt,
            'style': img_style,
            'samples_style_nat': style_input,
            'samples_style': style_out,
            'natural_path': natural_path,
            'gt_path': gt_path,
            'style_path': style_path,
        }
    
    def INR_retouch_pathes(self, inp_natural, input_gt, style_natural):
        """Generate paired paths from folders.

        Args:
            nature_folder (str): folder of the natural unprocessed images.
            style_folders (str): base folder of the folders that contain the style adjusted images.
        Returns:
            list[str]: Returned path list.
        """


        files_input = os.listdir(inp_natural)
        files_input.sort()
        files_style = os.listdir(style_natural)
        files_style.sort()
        styles = os.listdir(input_gt)
        styles.sort()
        files = []
        for preset in styles:
            for fi in files_input:
                fi_name  = f"{preset}/{fi}"
                files.append(fi_name)

        return files, files_input, files_style

    def __len__(self):
        return len(self.files)
