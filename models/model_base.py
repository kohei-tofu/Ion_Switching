import torch
from torch import nn


class base(nn.Module):
    def __init__(self):
        super(base, self).__init__()

    def forward(self, x):
        return NotImplementedError()

    def set_device(self, dev):
        self.device = dev

    def _initialize_weights(self):
        print('_initialize_weights')
        for m in self.modules():
            print('initialize', m)
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
