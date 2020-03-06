
import os, getpass
import os.path as osp
import argparse
import models.models
import dataset

from easydict import EasyDict as edict

class Config:

    #
    #  DATASET
    #
    #DATALOADER = edict()

    DATASET = edict()
    #DATASET.NAME = 'COCO'
    DATASET.NAME = 'Ion_switching'
    DATASET.TRAIN_VAL = dataset.split_loader
    DATASET.TEST = dataset.split_loader
    DATASET.KEYWORDS = {'time_lentgh' : 256, 'num_train_rate' : 0.8}
    #DATASET.KEYWORDS = [256]
    #DATASET.GET_FUNC(**DATASET.KEYWORDS)


    #
    #  MODEL
    #
    MODEL = edict()
    MODEL.TYPE = models.conv1d_1
    #MODEL.TYPE = models.conv_deconv_1
    #MODEL.TYPE = models.resnet_deconv_1
    #MODEL.TYPE = models.densenet1d_1
    
    MODEL.KEYWORDS = {'t_len' : DATASET.KEYWORDS['time_lentgh']}
    MODEL.DEVICE = 'cuda'
    MODEL.SAVE_PATH = './result/' + MODEL.TYPE.name

    #
    # SOLVER
    #
    SOLVER = edict()
    SOLVER.FROM_CHECKPOINT = False
    SOLVER.MAX_ITER = 40 
    #SOLVER.BATCHSIZE = 256
    SOLVER.BATCHSIZE = 64
    SOLVER.SEED = 1234
    SOLVER.LASTEPOCH = -1
    SOLVER.N_GPU = 0
    #SOLVER.IMS_PER_GPU = 32
    
    SOLVER.CHECKPOINT_PERIOD = 5 

    SOLVER.SCHEDULER = 'StepLR'
    SOLVER.GAMMA = 0.5
    SOLVER.STEPSIZE = 10
    
    
    
    #SOLVER.OPTIMIZER = 'Adam'
    SOLVER.OPTIMIZER = 'SGD'
    SOLVER.BASE_LR = 1e-9
    SOLVER.MOMENTUM = 0.9


    SOLVER.WARMUP_FACTOR = 0.1
    SOLVER.WARMUP_ITERS = 2400 
    SOLVER.WARMUP_METHOD = 'linear'
    SOLVER.WEIGHT_DECAY = 1e-5 
    SOLVER.WEIGHT_DECAY_BIAS = 0
