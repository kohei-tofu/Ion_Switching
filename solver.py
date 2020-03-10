
import torch.optim as optim
from torch.optim import lr_scheduler

import os, getpass
import os.path as osp
import argparse
from easydict import EasyDict as edict
#from cvpack.utils.pyt_utils import ensure_dir




#def get_optimizer(ptn, save_path, model, lst_epoch = -1, device='cpu', last = True):
def get_optimizer(cfg, model, lst_epoch = -1):

    #ptn = 'Adam'
    #ptn = 'SGD'
    print("last_ep", lst_epoch)
    save_path = cfg.MODEL.SAVE_PATH
    fname = None
    if cfg.SOLVER.LASTEPOCH <= 0:
        lst_epoch = -1
    elif cfg.SOLVER.FROM_CHECKPOINT == True:
        fname = save_path + "/optimizer_last.pth"
    else:
        fname = save_path + "/optimizer_" + str(lst_epoch) + ".pth"
        
    lr = cfg.SOLVER.BASE_LR
    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg.SOLVER.MOMENTUM)
        
    elif cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    elif cfg.SOLVER.OPTIMIZER == 'lbfg':
        lr = 1e-1
        optimizer = optim.LBFGS(model.parameters(), lr=lr)

    if fname != None:
        optimizer.load_state_dict(torch.load(fname, map_location=device))

    if cfg.SOLVER.SCHEDULER == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, 
                                        step_size=cfg.SOLVER.STEPSIZE, 
                                        gamma=cfg.SOLVER.GAMMA, 
                                        last_epoch=lst_epoch)

    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1, last_epoch=lst_epoch)

    return optimizer, scheduler
