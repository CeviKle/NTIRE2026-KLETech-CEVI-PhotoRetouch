
import torch
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


def CNN_DW(chan_inp, chan_out, sin_w):
    module = [nn.Conv2d(chan_inp, chan_out, kernel_size=1),
              Sine(sin_w),
              nn.Conv2d(chan_out, chan_out, kernel_size=3, padding=1, groups=chan_out),
              Sine(sin_w),
              nn.Conv2d(chan_out, chan_out, kernel_size=1)
              ]
    return module


class CNNDWSplitSiren(nn.Module):

    def __init__(self, n_input_p=2, n_input_s=3, n_output_dims=3, sin_w=1, n_neurons: int = 64, 
                 n_hidden_p: int = 1, n_hidden_s: int = 1, n_hidden_m: int = 1, use_skip: bool = True):
        super().__init__()
        layers_p = [nn.Conv2d(n_input_p, n_neurons//2, kernel_size=1), Sine(sin_w)]
        for i in range(n_hidden_p):
           layers_p.append(nn.Conv2d(n_neurons//2, n_neurons//2, kernel_size=1))
           layers_p.append(Sine(sin_w))
        self.model_p = nn.Sequential(*layers_p)

        layers_s = [nn.Conv2d(n_input_s, n_neurons//2, kernel_size=1), Sine(sin_w)]
        for i in range(n_hidden_s):
           layers_s.append(nn.Conv2d(n_neurons//2, n_neurons//2, kernel_size=1))
           layers_s.append(Sine(sin_w))
        self.model_s = nn.Sequential(*layers_s)

        layers_m = []
        for i in range(n_hidden_m):
           layers_m.extend(CNN_DW(n_neurons, n_neurons, sin_w=sin_w))
           layers_m.append(Sine(sin_w))

        layers_m.append(nn.Conv2d(n_neurons, n_output_dims, kernel_size=1))

        self.model_m = nn.Sequential(*layers_m)

        self.use_skip = use_skip
       

    def forward(self, inp):
        p, s = inp[:, :2], inp[:, 2:]     
        out_p = self.model_p(p)
        out_s = self.model_s(s)
        output = self.model_m(torch.cat([out_p, out_s], dim=1))

        if self.use_skip :
            update = output
            output = 1 * update + inp[:, -3:] # global skip connectionn

        return output




if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = CNNDWSplitSiren(n_input_p=2, n_input_s=3, n_output_dims=3, sin_w=1, n_neurons = 64, 
                 n_hidden_p = 1, n_hidden_s = 1, n_hidden_m = 1, use_skip = True, return_update = False)


    inp_shape = (5, 1280, 720)
    # inp_shape = (1, 455, 5, 12, 12)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)
