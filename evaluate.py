from __future__ import print_function
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import util
import evaluate
import solver

def val(cfg, model, device, data_val, epoch):

    model.eval()
    test_loss = 0
    correct = 0

    #num_data = data_test.num_data
    loss_sum = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_val):

            loss = model.loss(data, target, epoch, 'val')
            if loss is None:
                print('None')
                continue

            d = batch_idx/len(data_val) * 100
            #print('val {}'.format(loss.item()))
            print('Val Epoch: [{} / {}({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx, len(data_val), d, loss.item()))
            loss_sum += loss.item()

    loss_sum /= len(data_val)

    print()
    print('val_loss : ' , loss_sum)
    return loss_sum



def output(cfg, model, device, data_val, epoch):

    model.eval()
    test_loss = 0
    correct = 0

    #num_data = data_test.num_data
    loss_sum = 0

    output = None
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_val):
            
            
            loss, output = model.loss_output(data, target, epoch, 'val')
            if loss is None:
                print('None')
                continue

            d = batch_idx/len(data_val) * 100
            #print('val {}'.format(loss.item()))
            print('Val Epoch: [{} / {}({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx, len(data_val), d, loss.item()))
            loss_sum += loss.item()

    loss_sum /= len(data_val)

    print()
    print('val_loss : ' , loss_sum)
    return loss_sum, output


def make_labels():

    pass

def main(cfg):

    torch.manual_seed(cfg.SOLVER.SEED)
    save_path = cfg.MODEL.SAVE_PATH
    
    #kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    data_test = cfg.DATASET.TEST(**cfg.DATASET.KEYWORDS)
    
    batchsize = cfg.SOLVER.BATCHSIZE
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batchsize)

    device = torch.device(cfg.SOLVER.N_GPU if cfg.SOLVER.N_GPU >= 0 else "cpu")
    
    model = cfg.MODEL.TYPE(**cfg.MODEL.KEYWORDS)
    
    print('device', device)
    model.to(device)
    model.set_device(device)
    model.load_state_dict(torch.load(save_path + "/model_last.pt", map_location=device))

    save_path = cfg.MODEL.SAVE_PATH
    util.make_directory(save_path)

    print('phase : validation')
    loss, output = val(cfg, model, device, test_loader, 0)
    print('loss', loss)




