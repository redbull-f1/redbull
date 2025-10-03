import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterSpeedModular(nn.Module):
    '''
    Main model for CenterSpeed
    '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, size_linear_layer = 64, p_dropout=0.3, image_size=256):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.head = nn.Sequential(nn.Conv2d(in_channels=channel_two, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  #nn.LeakyReLU(),
                                  nn.Flatten(),
                                  #nn.BatchNorm1d(*2),
                                  #nn.Dropout1d(p=p_dropout),
                                    nn.Linear((image_size//4)**2, size_linear_layer),
                                   nn.LeakyReLU(),
                                   nn.Linear(size_linear_layer, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=p_dropout)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        y = self.head(x)
        x = F.leaky_relu(self.deconv1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        x = self.sigmoid(x)
        return x,y


class CenterSpeedDense(nn.Module):
    '''
    Main model for CenterSpeed
    '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, image_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.bn4 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(channel_one)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.deconv1(x)))
        x = self.bn4(self.deconv2(x))
        return x

class CenterSpeedDensesigmoid(nn.Module):
    '''
    Main model for CenterSpeed
    '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, image_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.bn4 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(channel_one)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.deconv1(x)))
        x = self.bn4(self.deconv2(x))
        channels = []
        for i in range(4):
            if i == 0:
                channels.append(self.sigmoid(x[:,i:i+1,:,:]))
            else:
                channels.append(x[:,i:i+1,:,:])

        output = torch.cat(channels, dim=1)

        return output

class CenterSpeedDensev2(nn.Module):
    '''
    Main model for CenterSpeed
    '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, image_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.bn4 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(channel_one)
        self.sigmoid = nn.Sigmoid()
        self.velocity_cap = 10.0
        self.theta_cap = 3.14159 # 180 degrees


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.deconv1(x)))
        x = self.deconv2(x)

        heatmap, vx, vy, theta = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :]

        # Apply Sigmoid to heatmap channels, Tanh to velocity channels
        heatmap = torch.sigmoid(heatmap)  # Constrain heatmap values between 0 and 1
        vx = torch.tanh(vx) * self.velocity_cap   # Allow vx to take on both positive and negative values
        vy = torch.tanh(vy) * self.velocity_cap   # Allow vy to take on both positive and negative values
        theta = torch.tanh(theta) * self.theta_cap # Allow theta to take on both positive and negative values

        # Recombine outputs
        output = torch.stack((heatmap, vx, vy, theta), dim=1)

        return output

class CenterSpeedDenseResidual(nn.Module):
    '''
        Main model for CenterSpeed
        '''
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, image_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.bn4 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(channel_one)


    def forward(self, x):
        res = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(res)))
        x = F.leaky_relu(self.bn3(self.deconv1(x))) + res
        x = self.bn4(self.deconv2(x))
        return x



############Old models for testing purposes ################

class HourglassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.conv3(x)
        return x

class HourglassModelDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.conv3(x)
        return x

class HourglassModelD(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.conv3(x)

class HourglassModelDeepExp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.dropout(x)
        x = self.conv3(x)
        self.sigmoid(x)
        return x

class HourglassModelDeepExp2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  nn.Flatten(),
                                   nn.Linear(129*129, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        y = self.head(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x,y


class CenterSpeed(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.head = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  nn.Flatten(),
                                   nn.Linear(64*64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        y = self.head(x)
        x = F.leaky_relu(self.deconv1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        x = self.sigmoid(x)
        #TODO: maybe it is better for onnx/trt to concat this in a vector!!
        return x,y
class CenterSpeed_NOPAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=0, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=0, output_padding=0) #padding 0 for to reduce border effects
        self.head = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  nn.Flatten(),
                                   nn.Linear(63*63, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        print(x.shape)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        print(x.shape)
        y = self.head(x)
        x = F.leaky_relu(self.deconv1(x))
        print(x.shape)
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        print(x.shape)
        x = self.sigmoid(x)
        #TODO: maybe it is better for onnx/trt to concat this in a vector!!
        return x,y

class CenterSpeed_NODECONV(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.head = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
                                  nn.Flatten(),
                                   nn.Linear(64*64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        y = self.head(x)
        x = F.leaky_relu(self.conv3(self.upsample1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv4(self.upsample2(x)))
        x = self.sigmoid(x)
        #TODO: maybe it is better for onnx/trt to concat this in a vector!!
        return x,y

class CenterSpeed_hm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        fm = x
        x = F.leaky_relu(self.deconv1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        x = self.sigmoid(x)
        return x , fm


class CenterSpeed_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
                                    nn.Flatten(),
                                    nn.Linear(64*64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        y = self.head(x)
        return y


class CenterSpeedModular2(nn.Module):
    def __init__(self,input_channels=6, channel_one=64, channel_two=128, size_linear_layer = 64, p_dropout=0.3, image_size=256):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)
        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one, kernel_size=4, stride=1, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=1, kernel_size=4, stride=1, padding=1, output_padding=0) #padding 0 for to reduce border effects
        self.head = nn.Sequential(nn.Conv2d(in_channels=channel_two, out_channels=1, kernel_size=3, stride=1, padding=1),
                                #nn.LeakyReLU(),
                                nn.Flatten(),
                                #nn.BatchNorm1d(*2),
                                #nn.Dropout1d(p=p_dropout),
                                    nn.Linear((image_size//4)**2, size_linear_layer),
                                nn.LeakyReLU(),
                                nn.Linear(size_linear_layer, 3))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=p_dropout)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        y = self.head(x)
        x = F.leaky_relu(self.deconv1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.deconv2(x))
        x = self.sigmoid(x)
        #TODO: maybe it is better for onnx/trt to concat this in a vector!!
        return x,y



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out




# CBAM 클래스들은 이미 위에 정의돼 있다고 가정:
# from your_module import CBAM

class CenterSpeedDenseCBAM(nn.Module):
    '''
    Main model for CenterSpeed + CBAM
    - conv1 -> CBAM(ch=channel_one)
    - conv2 -> CBAM(ch=channel_two)
    - deconv1 (원래대로)
    - deconv2 (원래대로)
    '''
    def __init__(self, input_channels=4, channel_one=64, channel_two=128, image_size=128,
                 cbam_reduction=16, cbam_pool_types=['avg','max'], cbam_no_spatial=False):
        super().__init__()
        self.input_channels = input_channels

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=channel_one,
                               kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_one)

        self.conv2 = nn.Conv2d(in_channels=channel_one, out_channels=channel_two,
                               kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_two)

        # ----- CBAM blocks (채널 수에 맞춰 게이트 채널 설정) -----
        self.cbam1 = CBAM(gate_channels=channel_one,
                          reduction_ratio=cbam_reduction,
                          pool_types=cbam_pool_types,
                          no_spatial=cbam_no_spatial)

        self.cbam2 = CBAM(gate_channels=channel_two,
                          reduction_ratio=cbam_reduction,
                          pool_types=cbam_pool_types,
                          no_spatial=cbam_no_spatial)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_two, out_channels=channel_one,
                                          kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(channel_one)

        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_one, out_channels=4,
                                          kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(4)

    def forward(self, x):
        # Encoder stage 1
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1, inplace=True)
        x = self.cbam1(x)   # ← 채널/공간 어텐션 적용

        # Encoder stage 2
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1, inplace=True)
        x = self.cbam2(x)   # ← 채널/공간 어텐션 적용

        # Decoder
        x = F.leaky_relu(self.bn3(self.deconv1(x)), negative_slope=0.1, inplace=True)
        x = self.bn4(self.deconv2(x))  # 최종 로짓(회귀/분류용 출력)
        return x






# ----- Bottleneck: 1x1 -> 3x3 -> 1x1 (+ residual add) -----
class Bottleneck(nn.Module):
    def __init__(self, in_channels, expansion=4, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        # out_channels를 in_channels와 같게 유지하려면 mid = in_channels // expansion
        mid = max(1, in_channels // expansion)  # 32
        out_channels = mid * expansion  # 보통 in_channels와 동일

       
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)

        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)

        self.conv3 = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.act = act

        # in/out 채널 동일·stride=1 → identity 사용, 아니면 projection
        self.use_identity = (in_channels == out_channels)
        if not self.use_identity:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.act(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))

        if self.use_identity:
            y = y + x
        else:
            y = y + self.proj(x)
        return self.act(y)


import torch
import torch.nn as nn
import torch.nn.functional as F

# ↙️ 이미 위에서 정의된 CBAM/BasicConv/ChannelGate/SpatialGate/Bottleneck 등을 사용한다는 가정
# from your_module import CBAM, Bottleneck

class CenterSpeedBottleneckCBAM(nn.Module):
    """
    CenterSpeed (Encoder-Decoder) + Bottleneck + CBAM
    Input [B,4,128,128]
      -> Conv1 [B,64,64,64]           -> CBAM(64)
      -> Conv2 [B,128,32,32]          -> CBAM(128)
      -> Bottleneck [128->128]        -> (옵션) CBAM(128)
      -> Deconv1 [B,64,64,64]
      -> Deconv2 [B,4,128,128]  (logits: [heat, vx, vy, yaw])

    Args
    ----
    input_channels: int
        입력 채널 수 (기본 4: [occ_t-1, dens_t-1, occ_t, dens_t])
    channel_one: int
        첫 conv 출력 채널 (기본 64)
    channel_two: int
        두 번째 conv 출력 채널 (기본 128)
    image_size: int
        입력 해상도 (기본 128, 구조상 직접 사용하진 않음)
    cbam_reduction: int
        CBAM 채널 게이트 MLP 축소 비율
    cbam_pool_types: list[str]
        CBAM 채널 풀 종류 (예: ['avg','max'])
    cbam_no_spatial: bool
        True면 SpatialGate 생략(채널 주의만)
    cbam_after_bottleneck: bool
        Bottleneck 출력에도 CBAM 적용할지 여부
    """

    def __init__(
        self,
        input_channels: int = 4,
        channel_one: int = 64,
        channel_two: int = 128,
        image_size: int = 128,
        cbam_reduction: int = 16,
        cbam_pool_types = ['avg', 'max'],
        # cbam_no_spatial: bool = False,
        cbam_no_spatial: bool = True,
        cbam_after_bottleneck: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.cbam_after_bottleneck = cbam_after_bottleneck

        # ----- Encoder -----
        self.conv1 = nn.Conv2d(self.input_channels, channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(channel_one)

        self.conv2 = nn.Conv2d(channel_one, channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(channel_two)

        # ----- CBAM blocks -----
        self.cbam1 = CBAM(
            gate_channels=channel_one,
            reduction_ratio=cbam_reduction,
            pool_types=cbam_pool_types,
            no_spatial=cbam_no_spatial,
        )

        self.cbam2 = CBAM(
            gate_channels=channel_two,
            reduction_ratio=cbam_reduction,
            pool_types=cbam_pool_types,
            no_spatial=cbam_no_spatial,
        )

        # Bottleneck (128 -> 128)
        self.bottleneck = Bottleneck(in_channels=channel_two, expansion=4)

        if self.cbam_after_bottleneck:
            self.cbam_bottleneck = CBAM(
                gate_channels=channel_two,
                reduction_ratio=cbam_reduction,
                pool_types=cbam_pool_types,
                no_spatial=cbam_no_spatial,
            )

        # ----- Decoder -----
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=channel_two, out_channels=channel_one,
            kernel_size=4, stride=2, padding=1, output_padding=0
        )
        self.bn3     = nn.BatchNorm2d(channel_one)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=channel_one, out_channels=4,
            kernel_size=4, stride=2, padding=1, output_padding=0
        )
        self.bn4     = nn.BatchNorm2d(4)

    def forward(self, x):
        # ----- Encoder stage 1 -----
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1, inplace=True)  # [B,64,64,64]
        x = self.cbam1(x)

        # ----- Encoder stage 2 -----
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1, inplace=True)  # [B,128,32,32]
        x = self.cbam2(x)

        # ----- Bottleneck -----
        x = self.bottleneck(x)                                                        # [B,128,32,32]
        if self.cbam_after_bottleneck:
            x = self.cbam_bottleneck(x)

        # ----- Decoder -----
        x = F.leaky_relu(self.bn3(self.deconv1(x)), negative_slope=0.1, inplace=True) # [B,64,64,64]
        x = self.bn4(self.deconv2(x))                                                  # [B,4,128,128] (logits)
        return x



# ----- CenterSpeedDenseBottleneck (pipeline 고정형) -----
class CenterSpeedDenseBottleneck(nn.Module):
    '''
    Main model for CenterSpeed
    Input [B,4,128,128]
      -> Conv1 [B,64,64,64]
      -> Conv2 [B,128,32,32]
      -> Bottleneck [128→128]
      -> Deconv1 [B,64,64,64]
      -> Deconv2 [B,4,128,128] -> Output Heatmap (logits)
    '''
    def __init__(self, input_channels=4, channel_one=64, channel_two=128, image_size=128):
        super().__init__()
        self.input_channels = input_channels

        # Encoder
        self.conv1 = nn.Conv2d(self.input_channels, channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(channel_one)

        self.conv2 = nn.Conv2d(channel_one, channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(channel_two)

        # Bottleneck (128 -> 128)
        self.bottleneck = Bottleneck(in_channels=channel_two, expansion=4)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(channel_two, channel_one, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn3     = nn.BatchNorm2d(channel_one)

        self.deconv2 = nn.ConvTranspose2d(channel_one, 4, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn4     = nn.BatchNorm2d(4)

    def forward(self, x):
        # Input -> Conv1
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1, inplace=True)   # [B,64,64,64]
        # Conv1 -> Conv2
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1, inplace=True)   # [B,128,32,32]
        # Conv2 -> Bottleneck (128->128)
        x = self.bottleneck(x)                                                        # [B,128,32,32]
        # Bottleneck -> Deconv1
        x = F.leaky_relu(self.bn3(self.deconv1(x)), negative_slope=0.1, inplace=True) # [B,64,64,64]
        # Deconv1 -> Deconv2 -> Output (logits)
        x = self.bn4(self.deconv2(x))                                                 # [B,4,128,128]
        return x
    



import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 기본 Bottleneck (네가 이미 가진 클래스와 동일 가정) ----
class Bottleneck(nn.Module):
    def __init__(self, in_channels, expansion=4, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        mid = max(1, in_channels // expansion)
        out_channels = mid * expansion

        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)

        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)

        self.conv3 = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.act = act
        self.use_identity = (in_channels == out_channels)
        if not self.use_identity:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.act(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.use_identity:
            y = y + x
        else:
            y = y + self.proj(x)
        return self.act(y)

# ---- Squeeze-and-Excitation Block ----
class SEBlock(nn.Module):
    """
    SE: GlobalAvgPool -> FC(C->C/r) -> ReLU -> FC(C/r->C) -> Sigmoid -> scale
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        assert reduction >= 1
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))        # [B,C,1,1]
        return x * w                    # channel-wise reweight

# ---- CenterSpeed + Bottleneck + Squeeze-Excitation ----
class CenterSpeedBottleneckSENet(nn.Module):
    """
    Input [B,4,128,128]
      -> Conv1 [B,64,64,64] -> BN -> LeakyReLU -> SE(64)
      -> Conv2 [B,128,32,32] -> BN -> LeakyReLU -> SE(128)
      -> Bottleneck(128->128)          -> (옵션) SE(128)
      -> Deconv1 [B,64,64,64] -> BN -> LeakyReLU
      -> Deconv2 [B,4,128,128] -> BN
    Output logits: [heat, vx, vy, yaw]
    """
    def __init__(
        self,
        input_channels: int = 4,
        channel_one: int = 64,
        channel_two: int = 128,
        image_size: int = 128,            # 인터페이스 유지용
        se_reduction: int = 16,
        use_se_after_bottleneck: bool = True,
    ):
        super().__init__()
        self.use_se_after_bottleneck = use_se_after_bottleneck

        # Encoder
        self.conv1 = nn.Conv2d(input_channels, channel_one, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(channel_one)
        self.se1   = SEBlock(channel_one, reduction=se_reduction)

        self.conv2 = nn.Conv2d(channel_one, channel_two, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(channel_two)
        self.se2   = SEBlock(channel_two, reduction=se_reduction)

        # Bottleneck
        self.bottleneck = Bottleneck(in_channels=channel_two, expansion=4)
        if self.use_se_after_bottleneck:
            self.se_bottleneck = SEBlock(channel_two, reduction=se_reduction)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(channel_two, channel_one, kernel_size=4, stride=2, padding=1)
        self.bn3     = nn.BatchNorm2d(channel_one)

        self.deconv2 = nn.ConvTranspose2d(channel_one, 4, kernel_size=4, stride=2, padding=1)
        self.bn4     = nn.BatchNorm2d(4)

    def forward(self, x):
        # Encoder stage 1
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1, inplace=True)
        x = self.se1(x)

        # Encoder stage 2
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1, inplace=True)
        x = self.se2(x)

        # Bottleneck (+ optional SE)
        x = self.bottleneck(x)
        if self.use_se_after_bottleneck:
            x = self.se_bottleneck(x)

        # Decoder
        x = F.leaky_relu(self.bn3(self.deconv1(x)), negative_slope=0.1, inplace=True)
        x = self.bn4(self.deconv2(x))  # logits [B,4,H,W]
        return x
