
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .module_util import ResidualBlock_noBN, make_layer, CALayer
from utils import b_Bicubic

import ipdb

# --------------------------------
# --------------------------------
def get_gi(lsr, nsr):
    ###lsr 还是空间域，nsr是频域
    lsr_freq = torch.fft.rfft2(lsr)
    lsr_freq = torch.stack((lsr_freq.real, lsr_freq.imag), dim=-1)

    inv_denominator = lsr_freq[:, :, :, :, 0] * lsr_freq[:, :, :, :, 0] + \
                        lsr_freq[:, :, :, :, 1] * lsr_freq[:, :, :, :, 1] + \
                            nsr[:, :, :, :, 0] * nsr[:, :, :, :, 0] + \
                                nsr[:, :, :, :, 1] * nsr[:, :, :, :, 1]
                        ##这里有一次nsr维度的自动转换
                        # nsr  ##a^^2 - (bi)^^2

    ##
    gi = torch.zeros_like(lsr_freq)

    gi[:, :, :, :, 0] = lsr_freq[:, :, :, :, 0] / inv_denominator
    gi[:, :, :, :, 1] = -1 * lsr_freq[:, :, :, :, 1] / inv_denominator

    return gi

# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    deblur_f = torch.zeros_like(inv_ker_f).cuda()
    deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
                            - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
    deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
                            + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    deblur_f = torch.complex(deblur_f[..., 0], deblur_f[..., 1])
    deblur = torch.fft.irfft2(deblur_f)
    return deblur

class UnfoldKEst(nn.Module):
    def __init__(
        self,
        in_nc=3,
        nf=64,
        num_blocks=3,
        scale=4,
        out_nc=21
        ):
        super(UnfoldKEst, self).__init__()

        self.scale = scale
        self.in_nc = in_nc

        ### extract features
        self.conv_first = nn.Conv2d(in_nc, nf, 7, 1, 3)

        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.body = make_layer(basic_block, num_blocks)

        ### solve Kernel Estimation
        self.conv_i = nn.Conv2d(2*in_nc, 2*in_nc, 1, 1, 0)
        self.relu_i = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_i_o = nn.Conv2d(2*in_nc, 2*in_nc, 1, 1, 0)

        self.conv_out = nn.Conv2d(nf, out_nc**2, 1, 1, 0)
        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        ### a channel 
        self.conv_channel = nn.Conv2d(in_nc, 1, 1, 1, 0)

    def forward(self, lr, sr):
        ### 归一化
        # ipdb.set_trace()
        
        feature = self.conv_first(lr)
        
        ##downsample the estimated hr first
        if (self.scale > 1):
            sr_downsample = b_Bicubic(sr, self.scale)
        else:
            sr_downsample = sr
        
        pad_sz = self.scale * 4
        pads = [pad_sz, pad_sz, pad_sz, pad_sz] ###方便计算正反傅里叶变换，左右上下
        feature_pad = F.pad(feature, pads, 'replicate')

        sr_downsample_pad = F.pad(sr_downsample, pads, 'replicate')

        b, c, h, w = feature_pad.shape

        ###需要预估nsr 
        ##假设sn可以被估计，sk=1
        # nsr
        ##干脆直接假设这个参数是被学习出来的好了
        ##估计噪声和模糊核的权重参数是个全局概念，因此收到感受野限制，我们直接在频域估计
        lr_pad =  F.pad(lr, pads, 'replicate')
        lr_f = torch.fft.rfft2(lr_pad)
        lr_f_feature = torch.stack((lr_f.real, lr_f.imag), dim=-1)
        lr_f_feature = lr_f_feature.permute(0, 1, 4, 2, 3).contiguous()
        lr_f_feature_o = lr_f_feature.view((b, -1, ) + lr_f_feature.shape[3:])

        nsr = self.conv_i_o(self.relu_i(self.conv_i(lr_f_feature_o)))

        nsr = nsr.view((b, -1, 2) + (lr_f_feature_o.shape[2:])).permute(0, 1, 3, 4, 2).contiguous() # (b, c, h, w, 2)
        out = torch.zeros(feature.size()).to(lr.device)

        filter_gi = get_gi(sr_downsample_pad, nsr)

        # ipdb.set_trace()
        for i in range(c):
            img_feature_ch = feature_pad[:, i:i+1, :, :]  #b, 1, h+2p, w+2p

            ## do filter
            # ipdb.set_trace()
            img_feature_ch = img_feature_ch.repeat([1, self.in_nc, 1, 1])
            numerator = torch.fft.rfft2(img_feature_ch)
            numerator = torch.stack((numerator.real, numerator.imag), dim=-1)
            
            filtered_fea = deconv(numerator, filter_gi)
            filtered_fea = torch.mean(filtered_fea, dim=1, keepdim=True)

            out[:, i:i+1, :, :] = filtered_fea[:, :, pad_sz:-pad_sz, pad_sz:-pad_sz]

        out = self.conv_out(out)
        out = self.globalPooling(out)

        return out



def p2o_rfft(psf, shape):
    '''
    # psf: NxCxhxw
    # shape: [H,W]
    # otf: NxCxHx[W//2 + 1}, complex
    '''

    # otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf).to(psf.device)
    otf = torch.zeros(psf.shape[:-2] + shape, device=psf.device).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.rfft2(otf)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[torch.abs(otf)<n_ops*2.22e-16] = torch.tensor(0).type_as(otf)

    return otf

class KFFUnitV2(nn.Module):

    def __init__(self, in_nc):
        super(KFFUnitV2, self).__init__()

        self.conv_k = nn.Conv2d(in_nc*2, in_nc*2, 1, 1, 0, bias=False)
        self.relu_k = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_k2 = nn.Conv2d(in_nc*2, in_nc*2, 1, 1, 0, bias=False)

        self.conv_i = nn.Conv2d(in_nc*2, in_nc*2, 1, 1, 0, bias=False)
        self.relu_i = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_i2 = nn.Conv2d(in_nc*2, in_nc*2, 1, 1, 0, bias=False)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_o = nn.Conv2d(in_nc*2, in_nc*2, 1, 1, 0, bias=False)

    def forward(self, img, ker):
        """
        img: BxCxHxW
        ker: B x ksize x ksize
        """

        batch = img.shape[0]
        C = img.shape[1]

        input_ker = ker.repeat(1, C, 1, 1)   #c=3  (16x3x256x256)

        otf = p2o_rfft(input_ker, img.shape[-2:]) #BxCxHx(W//2 + 1)
        otf = torch.stack((otf.real, otf.imag), dim=-1) #BxCxHx(W//2 + 1)x2
        otf = otf.permute(0, 1, 4, 2, 3).contiguous()  #BxCx2xHx(W//2 + 1)
        otf = otf.view((batch, -1, ) + otf.shape[3:]) #Bx(C*2)xHx(W//2 + 1)   (16x6x256x129)

        img_fft = torch.fft.rfft2(img)
        img_fft = torch.stack((img_fft.real, img_fft.imag), dim=-1) #BxCxHx(W//2 + 1)x2
        img_fft = img_fft.permute(0, 1, 4, 2, 3).contiguous()  #BxCx2xHx(W//2 + 1)
        img_fft = img_fft.view((batch, -1, ) + img_fft.shape[3:]) #Bx(C*2)xHx(W//2 + 1)  (16x6x256x129)      

        otf_fea = self.conv_k2(self.relu_k(self.conv_k(otf)))
        img_fea = self.conv_i2(self.relu_i(self.conv_i(img_fft)))
        # print(f'test: {otf_fea.shape} {img_fea.shape}')  #test: torch.Size([16, 6, 256, 129]) torch.Size([16, 6, 256, 129])

        fuse_real = otf_fea[:, :C, :, :] * img_fea[:, :C, :, :] - otf_fea[:, C:, :, :] * img_fea[:, C:, :, :]
        fuse_imag = otf_fea[:, :C, :, :] * img_fea[:, C:, :, :] + otf_fea[:, C:, :, :] * img_fea[:, :C, :, :]
        fuse_fea = self.conv_o(self.relu(torch.cat((fuse_real, fuse_imag), dim=1)))

        fuse_fea = fuse_fea.view((batch, -1, 2, ) + img_fft.shape[2:]).permute(0, 1, 3, 4, 2).contiguous()  # BxCxHx(W//2 + 1)x2 [16, 3, 256, 129, 2]

        out = torch.complex(fuse_fea[..., 0], fuse_fea[..., 1])

        out = torch.fft.irfft2(out)
        
        return out


class DPCAB(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=3, reduction=4):
        super().__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
        )

        self.body2 = nn.Sequential(
            nn.Conv2d(nf2, nf2, ksize2, 1, ksize2 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf2, nf2, ksize2, 1, ksize2 // 2),
        )

        self.CA_body1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf1+nf2, nf1, ksize1, 1, ksize1 // 2),
            CALayer(nf1, reduction))

        self.CA_body2 = CALayer(nf2, reduction)

    def forward(self, x):
        f1 = self.body1(x[0])
        f2 = self.body2(x[1])

        ca_f1 = self.CA_body1(torch.cat([f1, f2], dim=1))
        ca_f2 = self.CA_body2(f2)

        x[0] = x[0] + ca_f1
        x[1] = x[1] + ca_f2
        return x


class DPCAG(nn.Module):
    def __init__(self, nf1, nf2, ksize1, ksize2, nb):
        super().__init__()

        self.body = nn.Sequential(*[DPCAB(nf1, nf2, ksize1, ksize2) for _ in range(nb)])

    def forward(self, x):
        y = self.body(x)
        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]
        return y


class UnfoldKExp(nn.Module):
    def __init__(
        self,
        in_nc=3,
        nf=64,
        num_blocks=3,
        scale=4,
        reduction=4,
        out_nc=3,
        nb=8,
        ng=1
        ):
        super(UnfoldKExp, self).__init__()

        self.min = 0.0
        self.max = 1.0

        self.conv_first = nn.Conv2d(in_nc, nf, 7, 1, 3)

        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.body = make_layer(basic_block, num_blocks)

        nf2 = nf // reduction
        self.body1 = nn.Conv2d(nf, nf2, 3, 1, 1)

        self.body2 = KFFUnitV2(in_nc=nf)

        self.merge_layer = nn.Conv2d(nf+nf2, nf, 3, 1, 1)
        self.CALayer = CALayer(nf, reduction=4)

        body = [DPCAG(nf, nf2, 3, 3, nb) for _ in range(ng)]
        self.body_res = nn.Sequential(*body)

        self.fusion = nn.Conv2d(nf+nf2, nf, 3, 1, 1)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(nf,nf * scale,3,1,1,bias=True),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf,nf * scale,3,1,1,bias=True),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )
        elif scale == 1:
            self.upscale = nn.Conv2d(nf, out_nc, 3, 1, 1)

        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(nf,nf * scale ** 2,3,1,1,bias=True),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )



    def forward(self, lr, ker):
        ### lr : (b, c, h, w); ker: (b, ksize, ksize)
        # ipdb.set_trace()
        
        lr_fea_first = self.conv_first(lr)
        lr_fea = self.body(lr_fea_first)

        lr_fea_body1 = self.body1(lr_fea)

        lr_fea_body2 = self.body2(lr_fea, ker)

        fea_out = self.merge_layer(torch.cat((lr_fea_body1, lr_fea_body2), dim=1))
        fea_out = self.CALayer(fea_out)

        # ipdb.set_trace()
        inputs = [fea_out, lr_fea_body1]
        f2, f1 = self.body_res(inputs)

        f = self.fusion(torch.cat([f1, f2], dim=1)) + lr_fea_first
        out = self.upscale(f)

        return torch.clamp(out, min=self.min, max=self.max)


class UnfoldKE(nn.Module):
    def __init__(
        self,
        in_nc=3,
        nf=64,
        upscale=4,
        ksize=21,
        out_nc=3,
        nb=3,
        ng=1,
        loop=4,
        pca_matrix_path=None,
    ):
        super(UnfoldKE, self).__init__()

        self.ksize = ksize
        self.loop = loop
        self.scale = upscale

        self.Estimator = UnfoldKEst(in_nc=in_nc, nf=nf, scale=self.scale, out_nc=ksize)

        self.Restorer = UnfoldKExp(in_nc=in_nc, nf=nf, scale=self.scale, out_nc=out_nc, nb=nb, ng=ng)

        ##initialize kernel, delta function
        kernel = torch.zeros(1, self.ksize, self.ksize)
        kernel[:, self.ksize // 2, self.ksize // 2] = 1
        self.init_kernel = kernel

    def forward(self, lr):

        srs = []
        kernels = []

        # ipdb.set_trace()

        B, C, H, W = lr.shape
        # kernel = self.init_kernel.to(lr.device).repeat([B, 1, 1]).unsqueeze(1)  #(b, 1, ksize, ksize)
        # sr = self.Restorer(lr, kernel.detach())
        sr = b_Bicubic(lr, 1 / self.scale)

        for i in range(self.loop):
            # ipdb.set_trace()

            kernel = self.Estimator(lr, sr.detach())  #(b, 1, kszie, ksize)
            kernel = kernel.view(B, 1, self.ksize, self.ksize)

            # sr = self.Restorer(lr, kernel.detach())

            srs.append(sr)
            kernels.append(kernel)
        
        # ipdb.set_trace()
        return [srs, kernels]
