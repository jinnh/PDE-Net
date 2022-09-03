import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
from option import opt

class BasicConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0), use_relu=True):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=False)
        self.use_relu = use_relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Dense3Dblock(nn.Module):
    def __init__(self, cin, cout, use_relu, fea_num):
        super(Dense3Dblock, self).__init__()

        self.spatiallayer = BasicConv3d(cout, cout, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),use_relu=use_relu)
        self.spectralayer = BasicConv3d(cout, cout, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), use_relu=use_relu)
        
        self.Conv_mixdence = nn.Conv3d(cin, cout, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.fea_num = fea_num

        self.component_num = cout
        self.feat_weight = nn.Parameter(torch.rand(fea_num*64), requires_grad=True)  # [N,*,c,u,v,h,w]
        self.component_weight1 = nn.Parameter(torch.rand(self.component_num), requires_grad=True)  # [N,*,c,,,]
        self.component_weight2 = nn.Parameter(torch.rand(self.component_num), requires_grad=True)  # [N,*,c,,,]
        self.temperature_1 = 0.2
        self.temperature_2 = 0.2


    def forward(self, x, epoch):

        t1 = 1.0
        t2 = 1.0
        if epoch <= 30:  # T  1 ==> 0.1
            self.temperature_1 = t1 * (1 - epoch / 35)
            self.temperature_2 = t2 * (1 - epoch / 35)
        else:
            self.temperature_1 = 0.05
            self.temperature_2 = 0.05

        if(self.fea_num>1):
            x = torch.cat(x, dim=1) # [fea_num B C H W]
            
            [B, L, C, H, W] = x.shape

            feat_weight = self.feat_weight.clamp(0.02, 0.98)
            feat_weight = feat_weight[None, :, None, None, None]
            # p shape[fea_num 1 1 1 1]
            # noise r1 r2
            noise_feat_r1 = torch.rand((B, self.fea_num*64))[:, :, None, None, None].cuda()  ##[dence_num,N,1,1,1,1]
            noise_feat_r2 = torch.rand((B, self.fea_num*64))[:, :, None, None, None].cuda()
            noise_feat_logits = torch.log(torch.log(noise_feat_r1) / torch.log(noise_feat_r2))
            feat_weight_soft = torch.sigmoid(
                (torch.log(feat_weight / (1 - feat_weight)) + noise_feat_logits) / self.temperature_1)
            feat_logits = feat_weight_soft

            x = x * feat_logits
        else:
            x = torch.cat(x, 1)

        # # SELECT NETWOKR
        component_weight1 = self.component_weight1.clamp(0.02, 0.98)
        component_weight1 = component_weight1[None, :, None, None, None]
        component_weight2 = self.component_weight2.clamp(0.02, 0.98)
        component_weight2 = component_weight2[None, :, None, None, None]

        [B, L, C, H, W] = x.shape

        #s2
        noise_component_r1 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()  ##[dence_num,N,1,1,1,1]
        noise_component_r2 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()
        noise_component_logits1 = torch.log(torch.log(noise_component_r1) / torch.log(noise_component_r2))
        component_weight_gumbel1 = torch.sigmoid(
            (torch.log(component_weight1 / (1 - component_weight1)) + noise_component_logits1) / self.temperature_2)
        logits2 = component_weight_gumbel1

        #s3
        noise_component_r3 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()  ##[dence_num,N,1,1,1,1]
        noise_component_r4 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()
        noise_component_logits2 = torch.log(torch.log(noise_component_r3) / torch.log(noise_component_r4))
        component_weight_gumbel2 = torch.sigmoid(
            (torch.log(component_weight2 / (1 - component_weight2)) + noise_component_logits2) / self.temperature_2)
        logits3 = component_weight_gumbel2

        output = self.relu(self.Conv_mixdence(x))
       
        output = self.spectralayer(output) * logits2 + output
        output = self.spatiallayer(output) * logits3 + output

        return output

class stage(nn.Module):
    def __init__(self):
        super(stage, self).__init__()

        kernel_size=3
        n_feats=32
        self.head = nn.Conv3d(1, n_feats, kernel_size, padding=kernel_size // 2, bias=False)
        self.denselayers = nn.ModuleList([
            Dense3Dblock(cin=1 * n_feats, cout=n_feats * 2, use_relu=True, fea_num=0),
            Dense3Dblock(cin=2 * n_feats, cout=n_feats * 2, use_relu=True, fea_num=1),
            Dense3Dblock(cin=4 * n_feats, cout=n_feats * 2, use_relu=True, fea_num=2),
            Dense3Dblock(cin=6 * n_feats, cout=n_feats * 2, use_relu=True, fea_num=3),
            Dense3Dblock(cin=8 * n_feats, cout=n_feats * 2, use_relu=True, fea_num=4),
            Dense3Dblock(cin=10 * n_feats, cout=n_feats * 1, use_relu=False, fea_num=5)
        ])

        scale = opt.upscale_factor
        tail = []
        tail.append(nn.ConvTranspose3d(n_feats, 1, kernel_size=(3,2+scale,2+scale), stride=(1,scale,scale), padding=(1,1,1), bias=False))
        tail.append(nn.Conv3d(1, 1, kernel_size, padding=kernel_size//2, bias=False))
        self.tail = nn.Sequential(*tail)

    def forward(self, LHSI, epoch):

        LHSI = LHSI.unsqueeze(1)
        T = self.head(LHSI)

        x = [T]
        x1 = []
        for layer in self.denselayers:
            x_ = layer(x, epoch)
            x1.append(x_)
            x=x1
        x = x[-1] + T
        x = self.tail(x)
        x = x.squeeze(1)

        return x

class reconnetHRHSI(nn.Module):
    def __init__(self):
        super(reconnetHRHSI, self).__init__()

        if (opt.upscale_factor == 4):
            random_degradation = torch.randn((25), requires_grad=True)
        elif(opt.upscale_factor == 8):
            random_degradation = torch.randn((81), requires_grad=True)
        self.degradation_conv_weight = torch.nn.Parameter(random_degradation)
        self.degradation_conv_weight = torch.nn.Parameter(random_degradation)
        self.register_parameter("degradation_conv_weight", self.degradation_conv_weight)
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

        self.upsamper = nn.Upsample(scale_factor=opt.upscale_factor, mode='bicubic', align_corners=False)
        self.stages = nn.ModuleList([
            stage(),
            stage(),
            stage(),
            stage()])

    def degradation(self, HSI):

        channels = HSI.size()[1]
        out_channel = channels
        kernel = self.softmax(self.degradation_conv_weight)
        weight = kernel.reshape([channels, out_channel, opt.upscale_factor+1, opt.upscale_factor+1])
        LHSI = torch.nn.functional.conv2d(HSI, weight, stride=opt.upscale_factor, padding=opt.upscale_factor//2)

        return LHSI

    def forward(self, LHSI, epoch):

        LHSI = [LHSI]
        recon_out = self.upsamper(LHSI[-1])


        for index, stage in enumerate(self.stages):

            recon = stage(LHSI[-1], epoch=epoch)
            recon_out = recon_out + recon

            [B, C, H, W] = recon_out.shape
            recon_out_degradation = torch.reshape(recon_out, [B * C, 1, H, W])
            recon_out_degradation = self.degradation(recon_out_degradation)
            recon_out_degradation = recon_out_degradation.reshape(
                [B, C, recon_out_degradation.shape[2], recon_out_degradation.shape[3]])

            lhsi_ = LHSI[0] - recon_out_degradation
            LHSI.append(lhsi_)

        return recon_out, lhsi_
