
import torch
from torch import nn
#from torch.autograd import Variable
import torch.nn.functional as F
import models.resnet1d
import models.model_base as mbase

class ion_base(mbase.base):
    def __init__(self):
        super(ion_base, self).__init__()

        self.loss_func = None
        

    def forward(self, x):
        return NotImplementedError()

    def set_device(self, dev):
        self.device = dev

    def _initialize_weights(self):
        print('_initialize_weights')
        for m in self.modules():
            
            print('initialize', m)
            #m.has
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                print(m.weight.grad)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def loss(self, data, target, epoch, phase):

        output = self.forward(data.to(self.device))
        ret = self.loss_func(output, target.to(self.device))

        return ret

    def loss_output(self, data, target, output_list):
        
        output = self.forward(data.to(self.device))
        ret = self.loss_func(output, target.to(self.device))

        output = output.cpu().numpy()
        
        if output_list is None:
            output_list = output
        else:
            output_list = np.concatenate((output_list, output), 0)

        return ret, output_list

class loss1(nn.Module):
    def __init__(self, length):
        super(loss1, self).__init__()

    def forward(self, x):
        pass


class conv1d_1(ion_base):
    name = 'conv1d_1'
    def __init__(self, t_len):
        super(conv1d_1, self).__init__()
        init = 256
        self.c11 = nn.Conv1d(1, init, 3, stride=1, padding=1)
        self.c12 = nn.Conv1d(init, init, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool1d(2)
        self.c21 = nn.Conv1d(init, init*2, 3, stride=1, padding=1)
        self.c22 = nn.Conv1d(init*2, init*2, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool1d(2)

        self.c31 = nn.Conv1d(init*2, init*4, 3, stride=1, padding=1)
        self.c32 = nn.Conv1d(init*4, init*4, 3, stride=1, padding=1)

        self.avepool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(init*4, init)
        self.linear2 = nn.Linear(init, t_len)
        self.mse = nn.MSELoss()

        self.loss_func = self.loss_definition

        self._initialize_weights()

    def loss_definition(self, output, target):
        diff = torch.clamp(torch.abs(output - target), 0, 3.0)
        loss = self.mse(diff, 0)
        #loss = torch.clamp(loss, 0, 5.) 
        return loss

    def forward(self, x):
        h = F.relu(self.c11(x))
        h = F.relu(self.c12(h))
        h1 = self.maxpool1(h)
        h = F.relu(self.c21(h1))
        h = F.relu(self.c22(h))
        h2 = self.maxpool1(h)

        h = F.relu(self.c31(h2))
        h = F.relu(self.c32(h))
        h = self.avepool(h)
        h = h.squeeze(2)
        h = F.relu(self.linear1(h))
        h = self.linear2(h)
        return h


class conv1d_1(ion_base):
    name = 'conv1d_1'
    def __init__(self, t_len):
        super(conv1d_1, self).__init__()
        init = 64
        self.c11 = nn.Conv1d(1, init, 3, stride=1, padding=1)
        self.c12 = nn.Conv1d(init, init, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool1d(2)
        self.c21 = nn.Conv1d(init, init*2, 3, stride=1, padding=1)
        self.c22 = nn.Conv1d(init*2, init*2, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool1d(2)

        self.c31 = nn.Conv1d(init*2, init*4, 3, stride=1, padding=1)
        self.c32 = nn.Conv1d(init*4, init*4, 3, stride=1, padding=1)

        self.avepool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(init*4, init)
        self.linear2 = nn.Linear(init, 1)
        self.mse = nn.MSELoss()

        self.loss_func = self.loss_definition

        self._initialize_weights()

    def loss_definition(self, output, target):
        return self.mse(output, target)

    def forward(self, x):
        h = F.relu(self.c11(x))
        h = F.relu(self.c12(h))
        h1 = self.maxpool1(h)
        h = F.relu(self.c21(h1))
        h = F.relu(self.c22(h))
        h2 = self.maxpool1(h)

        h = F.relu(self.c31(h2))
        h = F.relu(self.c32(h))
        h = self.avepool(h)
        h = h.squeeze(2)
        h = F.relu(self.linear1(h))
        h = self.linear2(h)
        return h


class resnet1d_1(ion_base):
    name = 'resnet1d_1'
    def __init__(self, t_len):
        super(resnet1d_1, self).__init__()
        

        output = 2048
        self.resnet = resnet1d.ResNet1D(256, resnet1d.Bottleneck, [3, 4, 6, 3], output)
        
        self.LeaklyReLu = nn.LeakyReLU(inplace=True)
        #self.c51 = nn.Conv1d(output, output//2, 1, stride=1, padding=0)
        #self.c52 = nn.Conv1d(output//2, output//2, 1, stride=1, padding=0)
        #self.c53 = nn.Conv1d(output//2, 1, 1, stride=1, padding=0)
        self.avepool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(output, output // 2)
        self.linear2 = nn.Linear(output // 2, 1)

        #self._initialize_weights()
        self.mse = nn.MSELoss()

        self.loss_func = self.loss_definition

        self._initialize_weights()

    def loss_definition(self, output, target):
        
        return self.mse(output, target)
        #return self.mse(target, output)

    def forward(self, x):

        h = self.resnet(x)

        h = h.unsqueeze(3)
        h = self.up1(h)
        h = h.squeeze(3)
        h = h + self.c1_skip(h1)

        
        h = F.relu(self.c41(h))
        h = F.relu(self.c42(h))
        h = h.unsqueeze(3)
        h = self.up2(h)
        h = h.squeeze(3)

        h = self.LeaklyReLu(self.c51(h))
        h = self.LeaklyReLu(self.c52(h))
        h = self.c53(h)

        return h

class conv_deconv_1(ion_base):
    name = 'conv_deconv_1'
    def __init__(self, t_len):
        super(conv_deconv_1, self).__init__()
        init = 64
        self.c11 = nn.Conv1d(1, init, 3, stride=1, padding=1)
        self.c12 = nn.Conv1d(init, init, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool1d(2)
        self.c21 = nn.Conv1d(init, init*2, 3, stride=1, padding=1)
        self.c22 = nn.Conv1d(init*2, init*2, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool1d(2)
        self.c31 = nn.Conv1d(init*2, init*4, 3, stride=1, padding=1)
        self.c32 = nn.Conv1d(init*4, init*4, 3, stride=1, padding=1)

        self.c_skip1 = nn.Conv1d(init*2, init*4, 1, stride=1, padding=0)
        self.c_skip2 = nn.Conv1d(init, init*4, 1, stride=1, padding=0)

        self.c41 = nn.Conv1d(init*8, init*4, 3, stride=1, padding=1)
        self.c42 = nn.Conv1d(init*4, init*4, 3, stride=1, padding=1)

        self.c51 = nn.Conv1d(init*8, init*4, 1, stride=1, padding=0)
        self.c52 = nn.Conv1d(init*4, init*2, 1, stride=1, padding=0)
        self.c53 = nn.Conv1d(init*2, init*1, 1, stride=1, padding=0)
        self.c54 = nn.Conv1d(init*1, 1, 1, stride=1, padding=0)

        self.up1 = nn.Upsample(scale_factor=(2, 1), mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=(2, 1), mode='bilinear')
        #self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.up1 = nn.Upsample(size=128, mode='bilinear')
        #self.up2 = nn.Upsample(size=256, mode='bilinear')

        self.relu = nn.ReLU(inplace=False)
        self.Lrelu = nn.LeakyReLU(inplace=False)

        self.dout = nn.Dropout2d(p=0.3, inplace=False)
        

        #self._initialize_weights()
        self.mse = nn.MSELoss()

        self.loss_func = self.loss_definition

        self._initialize_weights()

    def loss_definition(self, output, target):
        
        return self.mse(output, target)
        #return self.mse(target, output)

    def forward(self, x):

        identity = x
        h = self.Lrelu(self.c11(x))
        h1 = self.Lrelu(self.c12(h))#(N, 1L, I/1)
        h = self.maxpool1(h1)

        h = self.Lrelu(self.c21(h))
        h2 = self.Lrelu(self.c22(h))#(N, 2L, I/2)
        h = self.maxpool1(h2)

        h = self.Lrelu(self.c31(h))
        h = self.Lrelu(self.c32(h))#(N, 4L, I/4)

        h = h.unsqueeze(3)
        h = self.dout(h)

        h = self.up1(h)
        h = h.squeeze(3)#(N, 4L, I/2)
        
        h = torch.cat((h, self.c_skip1(h2)), dim=1)#(N, 4L+4L, I/2)

        h = self.Lrelu(h)
        h = self.Lrelu(self.c41(h))
        h = self.Lrelu(self.c42(h))#(N, 4L, I/2)
        h = h.unsqueeze(3)
        h = self.up2(h)
        h = h.squeeze(3)#(N, 4L, I/1)

        h = torch.cat((h, self.c_skip2(h1)), dim=1)#(N, 8L, I/1)
        h = self.Lrelu(h)

        h = self.Lrelu(self.c51(h))
        h = self.relu(self.c52(h))
        h = self.Lrelu(self.c53(h))
        h = self.c54(h)


        return h







class resnet_deconv_1(ion_base):
    name = 'resnet_deconv_1'
    
    def __init__(self, t_len):
        super(resnet_deconv_1, self).__init__()
        output = 512
        #output = 1024
        self.resnet = resnet1d.ResNet1D(256, resnet1d.Bottleneck, [2, 2, 2, 2], output)
        
        self.LeaklyReLu = nn.LeakyReLU(inplace=True)
        self.c51 = nn.Conv1d(output, output//2, 1, stride=1, padding=0)
        self.c52 = nn.Conv1d(output//2, output//2, 1, stride=1, padding=0)
        self.c53 = nn.Conv1d(output//2, 1, 1, stride=1, padding=0)
        
        #self._initialize_weights()
        self.mse = nn.MSELoss()

        self.loss_func = self.loss_definition

        self._initialize_weights()

    def loss_definition(self, output, target):
        
        return self.mse(output, target)

    def forward(self, x):

        h = self.resnet(x)
        h = self.LeaklyReLu(self.c51(h))
        h = self.LeaklyReLu(self.c52(h))
        h = self.c53(h)

        return h
