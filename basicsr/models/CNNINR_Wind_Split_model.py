import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import time

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

from basicsr.utils.INR_utils import get_wh_mgrid, InputWindTensor
from basicsr.models.losses.losses import LabColorLoss
from basicsr.models.losses.losses import VGGPerceptualLoss, LabColorLoss

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class CNNINRSplitWindModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(CNNINRSplitWindModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])
        

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if self.cri_pix is None:
            raise ValueError('Both pixel loss is None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        # -------------------------
        # VGG + Color Loss
        # -------------------------

        if self.opt['train'].get('vgg_weight', 0) > 0:
            self.cri_vgg = VGGPerceptualLoss().to(self.device)
        else:
            self.cri_vgg = None

        if self.opt['train'].get('color_weight', 0) > 0:
            self.cri_color = LabColorLoss().to(self.device)
        else:
            self.cri_color = None

    def setup_optimizers(self):
        train_opt = self.opt['train']
        p_params = []
        reg_weights = train_opt['optim_g']["weight_decay"]
        del train_opt['optim_g']["weight_decay"]

        for k, v in self.net_g.model_p.named_parameters():
            if v.requires_grad:
                p_params.append(v)
                 
        s_params = []

        for k, v in self.net_g.model_s.named_parameters():
            if v.requires_grad:
                s_params.append(v)

        m_params = []

        for k, v in self.net_g.model_m.named_parameters():
            if v.requires_grad:
                m_params.append(v)
        

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_p = torch.optim.Adam([{'params': p_params}],
                                                **train_opt['optim_g'], weight_decay=reg_weights[0])
            self.optimizer_s = torch.optim.Adam([{'params': s_params}],
                                                **train_opt['optim_g'], weight_decay=reg_weights[1])
            self.optimizer_m = torch.optim.Adam([{'params': m_params}],
                                                **train_opt['optim_g'], weight_decay=reg_weights[2])
        elif optim_type == 'SGD':
            self.optimizer_p = torch.optim.SGD(p_params,
                                               **train_opt['optim_g'], weight_decay=reg_weights[0])
            self.optimizer_s = torch.optim.SGD(s_params,
                                               **train_opt['optim_g'], weight_decay=reg_weights[1])
            self.optimizer_m = torch.optim.SGD(m_params,
                                               **train_opt['optim_g'], weight_decay=reg_weights[2])
        elif optim_type == 'AdamW':
            self.optimizer_p = torch.optim.AdamW([{'params': p_params}],
                                                **train_opt['optim_g'], weight_decay=reg_weights[0])
            self.optimizer_s = torch.optim.AdamW([{'params': s_params}],
                                                **train_opt['optim_g'], weight_decay=reg_weights[1])
            self.optimizer_m = torch.optim.AdamW([{'params': m_params}],
                                                **train_opt['optim_g'], weight_decay=reg_weights[2])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_p)
        self.optimizers.append(self.optimizer_s)
        self.optimizers.append(self.optimizer_m)

    def feed_data(self, data, is_val=False):
        
        if not is_val:
            # expected patch_size 1 so remove the batch dim
            self.style = data['style'][0].to(self.device)
            self.style_natural = data['style_natural'][0].to(self.device)

            self.style_tensor =  InputWindTensor(data["samples_style_nat"][0].to(self.device), data["samples_style"][0].to(self.device),flatten=False)
            self.style_path = data["style_path"][0]
            self.path = data["natural_path"][0]

        else:
            
            self.natural =  data['natural'].to(self.device)
            self.natural_path = data['natural_path'][0]
            if "gt" in data: 
                self.gt = data['gt'].to(self.device)
                self.gt_path = data['gt_path'][0]

            corrds_inf = get_wh_mgrid(self.natural.shape[2], self.natural.shape[3], flatten=False).permute(2,0,1).unsqueeze(0).to(self.device)
            self.corrds_inf = corrds_inf.float()


    def optimize_parameters(self, current_iter):
        self.optimizer_p.zero_grad()
        self.optimizer_s.zero_grad()
        self.optimizer_m.zero_grad()

        self.wind_size = self.opt["datasets"]["train"]["window_size"]
        self.Windows_batch_size = self.opt["datasets"]["train"]["Windows_batch_size"]

        batch = torch.rand([self.Windows_batch_size], device=self.device, dtype=torch.float32)

        style_inp, style_out = self.style_tensor(batch)

        style_inp = style_inp.reshape(
            (self.wind_size, self.wind_size, self.Windows_batch_size, -1)
        ).permute((2, 3, 0, 1))

        style_out = style_out.reshape(
            (self.wind_size, self.wind_size, self.Windows_batch_size, -1)
        ).permute((2, 3, 0, 1))

        # Forward
        pred = self.net_g(style_inp)
        # pred = torch.sigmoid(self.net_g(style_inp))
        self.output = pred

        l_total = 0
        loss_dict = OrderedDict()

        # -------------------------
        # Pixel Loss
        # -------------------------
        if self.cri_pix:
            l_pix = self.cri_pix(pred, style_out)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix


        # -------------------------
        # VGG Perceptual Loss
        # -------------------------
        if self.cri_vgg and pred.shape[1] == 3:
            pred_vgg = torch.clamp(pred, 0, 1)
            gt_vgg = torch.clamp(style_out, 0, 1)

            l_vgg = self.cri_vgg(pred_vgg, gt_vgg) * self.opt['train']['vgg_weight']
            l_total += l_vgg
            loss_dict['l_vgg'] = l_vgg

        # -------------------------
        # Lab Color Loss
        # -------------------------
        if self.cri_color and pred.shape[1] == 3:
            pred_lab = torch.clamp(pred, 0, 1)
            gt_lab = torch.clamp(style_out, 0, 1)

            l_color = self.cri_color(pred_lab, gt_lab) * self.opt['train']['color_weight']
            l_total += l_color
            loss_dict['l_color'] = l_color

        # Safety trick (keeps graph stable like original code)
        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        if not torch.isfinite(l_total):
            print("NaN detected. Skipping step.")
            self.optimizer_p.zero_grad()
            self.optimizer_s.zero_grad()
            self.optimizer_m.zero_grad()
            return
        # Backprop
        l_total.backward()

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 1.0)

        self.optimizer_p.step()
        self.optimizer_s.step()
        self.optimizer_m.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            pred = self.net_g(torch.cat([self.corrds_inf,self.natural], dim=1))
            self.output = pred.detach().cpu().reshape(self.natural.shape)
        self.net_g.train()


    def validation(self, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        img_name = osp.splitext(osp.basename(self.natural_path))[0]

        self.test()

        if hasattr(self, 'gt'): sub_folder = self.gt_path.split('/')[-2]
        else: sub_folder = self.natural_path.split('/')[-2]
        
        visuals = self.get_current_visuals()
        out_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr, reorder=True)

        self.out_img=out_img

        if 'gt' in visuals:
            gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr, reorder=True)
            del self.gt

        # tentative for out of GPU memory
        del self.output
        torch.cuda.empty_cache()

        if save_img:
            save_img_path = osp.join(self.opt['path']['visualization'], sub_folder,
                                        f'{img_name}_out.jpg')


            save_gt_img_path = osp.join(self.opt['path']['visualization'], sub_folder,
                                        f'{img_name}_gt.jpg')

            imwrite(out_img, save_img_path)

            if 'gt' in visuals: imwrite(gt_img, save_gt_img_path)

        if with_metrics:
            # calculate metrics
            assert gt_img is not None, "GT Images are not provided"
            
            opt_metric = deepcopy(self.opt['val']['metrics'])
            if use_image:
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(out_img, gt_img, **opt_)
            else:
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)

            self.collected_metrics = collected_metrics
    

            self._log_validation_metric_values(current_iter, f"{img_name}",
                                               tb_logger, self.collected_metrics)
        return 0.


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['inp'] = self.natural.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, current_iter):
            
        self.save_network(self.net_g, f'INR_Preset', current_iter)