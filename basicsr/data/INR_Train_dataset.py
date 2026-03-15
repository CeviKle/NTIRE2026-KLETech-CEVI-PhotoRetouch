from torch.utils import data as data

from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.INR_utils import get_wh_mgrid
import torch


class INRTrainDataset(data.Dataset):
    """Paired image dataset for Learning Edits.

    Read Natural (Image Before Edit) and Styled Image (Image After Edit)

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            style_natural List[(str)]: List of Pathes of Images Before Edit.
            style_output List[(str)]: List of Pathes of Images After Edit.
            window_size (int): Size of the window samples for training.
            sample_rate: The stride size of Window sampling

    """

    def __init__(self, opt):
        super(INRTrainDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.sample_rate = opt['sample_rate'] if 'sample_rate' in opt else None
        
        self.window_size = opt['window_size'] if 'window_size' in opt else None

        self.style_natural, self.style_output = opt['style_natural'], opt['style_output']
        

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        ####################################### Style Reference ##########################################################
        

        for i, (style_nat, style_ref) in enumerate(zip(self.style_natural, self.style_output)):
            
            style_path = style_ref
            
            # print(', lq path', lq_path)
            img_bytes = self.file_client.get(style_path)
            try:
                img_style = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("style path {} not working".format(style_path))
           

            style_natural_path = style_nat
            # print(', lq path', lq_path)
            img_bytes = self.file_client.get(style_natural_path)
            try:
                img_style_natural = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("style path {} not working".format(style_path))
           

            img_style, img_style_natural = img2tensor([img_style, img_style_natural],
                                        bgr2rgb=True,
                                        float32=True,)


            # Create Train Window Samples
            # C x Kernal^2 x L
            img_style_winds = torch.nn.functional.unfold(img_style.unsqueeze(0), self.window_size, dilation=1, padding=0, stride=self.sample_rate).reshape(img_style.shape[0],self.window_size**2,-1)
            img_style_winds = img_style_winds.permute(1, 2, 0)
            img_style_natural_winds = torch.nn.functional.unfold(img_style_natural.unsqueeze(0), self.window_size, dilation=1, padding=0, stride=self.sample_rate).reshape(img_style_natural.shape[0],self.window_size**2,-1)
            img_style_natural_winds = img_style_natural_winds.permute(1, 2, 0)

            img_style = img_style.permute(1, 2, 0)
            img_style_natural = img_style_natural.permute(1, 2, 0)

            # Postion Encoding
            corrds_style = get_wh_mgrid(img_style_natural.shape[0], img_style_natural.shape[1], flatten=False)
            corrds_style = corrds_style.float()
            corrds_style_window = torch.nn.functional.unfold(corrds_style.permute(2, 0, 1).unsqueeze(0), self.window_size, dilation=1, padding=0, stride=self.sample_rate).reshape(corrds_style.shape[-1],self.window_size**2,-1)
            corrds_style_window = corrds_style_window.permute(1, 2, 0)

            # wind_size*wind_size x samples x C
            style_input = torch.cat(
                [
                    corrds_style_window,
                    img_style_natural_winds,
                ],
                dim=2,
            )
            style_out = img_style_winds

            if i == 0:
                style_input_samples = style_input
                style_out_samples = style_out
            else:
                style_input_samples = torch.cat([style_input_samples, style_input], dim=1 if self.window_size else 0)
                style_out_samples = torch.cat([style_out_samples, style_out], dim=1 if self.window_size else 0)
        
        return {
            'style_natural': img_style_natural,
            'style': img_style,
            'samples_style_nat': style_input_samples,
            'samples_style': style_out_samples,
            "natural_path": style_natural_path,
            'style_path': style_path,
            
        }
    

    def __len__(self):
        return 1
