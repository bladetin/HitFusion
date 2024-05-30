from torchsummary import summary
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import cv2
from new_vit import ViT, NaViT
import os

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        self.reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)  
        out = self.conv2d(out)  
        if self.is_last is False: 
            out = F.leaky_relu(out, inplace=True)
        return out

class Laplace(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super(Laplace, self).__init__()
        lap_filter = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(lap_filter))

    def forward(self, x):
        laplacex = self.convx(x)
        x=torch.abs(laplacex)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, kernel_size=3, stride=1, is_last=True)
        self.bn1 = nn.BatchNorm2d(in_channels, affine=True)
        self.conv1_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, is_last=True)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)
        
        self.lapconv1 = Laplace(in_channels)
        self.lapconv2 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1, is_last=True)
        self.out_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        residual = self.lapconv1(x)
        residual = self.lapconv2(residual)

        out = out + residual
        out = self.out_relu(out)
        return out

class NoLapResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NoLapResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, kernel_size=3, stride=1, is_last=True)
        self.bn1 = nn.BatchNorm2d(in_channels, affine=True)
        self.conv1_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, is_last=True)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)
        
        self.resconv = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1, is_last=True)

        self.out_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        residual = self.resconv(x)

        out = out + residual
        out = self.out_relu(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.residual_1 = ResidualBlock(in_channels=16, out_channels=16)
        self.residual_2 = ResidualBlock(in_channels=32, out_channels=16)
        self.residual_3 = ResidualBlock(in_channels=48, out_channels=16)
    
    def forward(self, x):
        out1 = self.residual_1(x)
        out2 = self.residual_2(torch.cat([x, out1], dim=1))
        out3 = self.residual_3(torch.cat([x, out1, out2], dim=1))
        return torch.cat([x, out1, out2, out3], dim=1)

class NoLapDenseBlock(nn.Module):
    def __init__(self):
        super(NoLapDenseBlock, self).__init__()
        self.residual_1 = NoLapResidualBlock(in_channels=16, out_channels=16)
        self.residual_2 = NoLapResidualBlock(in_channels=32, out_channels=16)
        self.residual_3 = NoLapResidualBlock(in_channels=48, out_channels=16)
    
    def forward(self, x):
        out1 = self.residual_1(x)
        out2 = self.residual_2(torch.cat([x, out1], dim=1))
        out3 = self.residual_3(torch.cat([x, out1, out2], dim=1))
        return torch.cat([x, out1, out2, out3], dim=1)



class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.res = ResidualBlock(out_channels,out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res(x)
        x = self.conv2(x)
        return x


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        self.conv_in_ir = ConvLayer(1, 32, 1, 1)  
        self.conv_in_vis = ConvLayer(1, 32, 1, 1)   

        self.conv_in_down_ir = ConvLayer(32, 16, 1, 1)   
        self.conv_in_down_vis = ConvLayer(32, 16, 1, 1)   

        self.dense_ir = NoLapDenseBlock()
        self.dense_vis = DenseBlock()

        self.compress_ir_0 = ConvLayer(64, 32, 1, 1)
        self.compress_ir_1 = ConvLayer(32, 16, 1, 1)
        self.compress_ir_2 = ConvLayer(16, 8, 1, 1)
        self.compress_ir_3 = ConvLayer(8, 1, 1, 1)

        self.compress_vis_0 = ConvLayer(64, 32, 1, 1)
        self.compress_vis_1 = ConvLayer(32, 16, 1, 1)
        self.compress_vis_2 = ConvLayer(16, 8, 1, 1)
        self.compress_vis_3 = ConvLayer(8, 1, 1, 1)

        self.trans_ir = ViT()
        self.trans_vis = NaViT()

        self.conv_out_0 = ConvLayer(128, 64, 3, 1)
        self.conv_out_1 = ConvLayer(64, 32, 3, 1)    
        self.conv_out_2 = ConvLayer(32, 16, 3, 1)
        self.conv_out_3 = ConvLayer(16, 8, 3, 1)
        self.conv_out_4 = ConvLayer(8, 1, 1, 1, is_last=True)

    def forward(self, vi, ir):

        vi = vi[:,:1]

        H_img = vi.shape[2]
        W_img = vi.shape[3]

        H_patch = 16
        W_patch = 16

        H_num = H_img // H_patch
        W_num = W_img // W_patch

        x_in_ir = self.conv_in_ir(ir)     
        x_in_vis = self.conv_in_vis(vi)   

        x_in_ir = self.conv_in_down_ir(x_in_ir) 
        x_in_vis = self.conv_in_down_vis(x_in_vis)

        x_ir = self.dense_ir(x_in_ir)     
        x_vis = self.dense_vis(x_in_vis)  

        x_ir_m = self.compress_ir_0(x_ir)  
        x_ir_m = self.compress_ir_1(x_ir_m)     
        x_ir_m = self.compress_ir_2(x_ir_m)    
        x_ir_m = self.compress_ir_3(x_ir_m)      

        x_vis_m = self.compress_vis_0(x_vis)     
        x_vis_m = self.compress_vis_1(x_vis_m)   
        x_vis_m = self.compress_vis_2(x_vis_m)   
        x_vis_m = self.compress_vis_3(x_vis_m)  

        dict1 = {'q': x_ir_m, 'k': x_vis_m, 'v': x_vis_m}
        x_ir_m = self.trans_ir(x_ir_m)       
        x_vis_m = self.trans_vis(dict1)    
        x_vis_m = x_vis_m["v"]

        x_ir_reshape = rearrange(x_ir_m, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=H_num, w=W_num, p1=H_patch, p2=W_patch)  
        x_vis_reshape = rearrange(x_vis_m, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=H_num, w=W_num, p1=H_patch, p2=W_patch) 

        x_ir = x_ir_reshape * x_ir   
        x_vis = x_vis_reshape * x_vis  

        x_out = torch.cat([x_ir, x_vis], dim=1)

        x_out = self.conv_out_0(x_out)           
        x_out = self.conv_out_1(x_out)            
        x_out = self.conv_out_2(x_out)            
        x_out = self.conv_out_3(x_out)            
        x_out = self.conv_out_4(x_out)           

        return x_out


if __name__ == "__main__":
    mynet = FusionNet()
    print(mynet)

    x0_ir = torch.randn(32, 1, 256, 256)
    x0_vis = torch.randn(32, 1, 256, 256)
    
    x1_ir = torch.randn(32, 1, 640, 480)
    x1_vis = torch.randn(32, 1, 640, 480)

    y0 = mynet(x0_ir, x0_vis)
    y1 = mynet(x1_ir, x1_vis)

    print(y0.shape)
    print(y1.shape)

