from torch.utils import data as data
from torchvision.transforms.functional import normalize


from basicsr.utils import FileClient, imfrombytes, img2tensor
import random
import cv2
import os


class INRInferenceDataset(data.Dataset):
    """Paired image dataset for Learning Edits.

    Read Natural (Image Before Edit) and Styled Image (Image After Edit)

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            inp_natural (str): Folder of the Inference Images.
            style_output (str): Folder of the GT Images.

    """

    def __init__(self, opt):
        super(INRInferenceDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.input_size = opt['resize_input'] if 'resize_input' in opt else None        

        self.inp_natural = opt['inp_natural']
        # self.files = os.listdir(self.inp_natural)
        if os.path.isdir(self.inp_natural):
            self.files = os.listdir(self.inp_natural)
        else:
            self.files = [os.path.basename(self.inp_natural)]
            self.inp_natural = os.path.dirname(self.inp_natural)
            self.input_gt = opt['inp_gt'] if 'inp_gt' in opt else None

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        img_fi = self.files[index]

        # image range: [0, 1], float32.
        natural_path = self.inp_natural + '/' + img_fi

        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(natural_path)
        try:
            img_natural = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception(" natural path {} not working".format(natural_path))
        
        if self.input_size:
            img_natural = cv2.resize(img_natural, (self.input_size, self.input_size), interpolation= cv2.INTER_LINEAR)
        

        # gt_path = self.input_gt
        gt_path = self.input_gt if hasattr(self, 'input_gt') and self.input_gt is not None else None
        if gt_path is not None:
            gt_path = gt_path + '/' + img_fi
        
            img_bytes = self.file_client.get(gt_path)
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(img_gt))
            if self.input_size:
                img_gt = cv2.resize(img_gt, (self.input_size, self.input_size), interpolation= cv2.INTER_LINEAR)
        else:
            img_gt =None


        if gt_path is not None:
            img_gt, img_natural = img2tensor([img_gt, img_natural],
                                        bgr2rgb=True,
                                        float32=True,)
        else: 
            img_natural = img2tensor(img_natural, bgr2rgb=True, float32=True,)


        if img_gt is not None:
            return {
                'natural': img_natural,
                'gt': img_gt,
                'natural_path': natural_path,
                'gt_path': gt_path,
            }
        else :
            return {
                'natural': img_natural,
                'natural_path': natural_path,
            }

    def __len__(self):
        return len(self.files)
