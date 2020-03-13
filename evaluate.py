from __future__ import print_function
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import util
import evaluate
import solver
import pylab as pl



def get_output(model, device, data_loader):

    model.eval()
    output_list = None
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            
            output_list = model.output(data, output_list)

            d = batch_idx/len(data_loader) * 100
            print('Epoch: [{} / {}({:.0f}%)] '.format(batch_idx+1, len(data_loader), d))

    return output_list

def get_output2(model, device, data_loader):

    model.eval()
    output_list = None
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            
            output_list = model.output(data, output_list)

            d = batch_idx/len(data_loader) * 100
            print('Epoch: [{} / {}({:.0f}%)] '.format(batch_idx+1, len(data_loader), d))

    return output_list


def make_labels():

    pass


def draw_loss(save_path):

    train_loss = np.load(save_path + 'loss_train_last.npy').tolist()
    test_loss = np.load(save_path + 'loss_test_last.npy').tolist()
    epoch = range(len(train_loss))
    pl.clf()

    pl.subplot(121)
    pl.plot(epoch, train_loss)
    pl.plot(epoch, test_loss)
    pl.grid()

    pl.subplot(122)
    pl.plot(epoch, train_loss)
    pl.plot(epoch, test_loss)
    pl.yscale('log')
    pl.grid()

    pl.savefig(save_path + 'loss_curve.png')


def evaluate(output, data_, save_path, phase):

    dshape = data_.data.shape
    dflatten = data_.data.reshape((dshape[0] * dshape[1], dshape[2]))[:, 2]

    pl.figure()
    pl.subplot(121)
    pl.plot(output)
    pl.grid()
    pl.subplot(122)
    pl.plot(dflatten)
    pl.grid()
    pl.savefig(save_path + phase + '_evaluation.png')



def inference(output_list, data_, bsize):

    dshape = data_.data.shape
    ans = np.zeros((dshape[0], dshape[1], 2))
    d_len = data_.d_len

    for i, id in enumerate(range(len(data_))):
        
        b_num, d_num = data_.get_indices(id)   
        
        b_o = i // bsize
        i_o = i % bsize 
        #print(b_o, i_o)
        output = output_list[b_o][i_o]

        ans[b_num, (d_num):(d_num+d_len), 0] += output
        ans[b_num, (d_num):(d_num+d_len), 1] += np.ones(d_len)
    
    #take average
    ans = ans.reshape((dshape[0] * dshape[1], 2))
    print(ans.shape, np.min(ans[:, 1]))
    ans[:, 0] = ans[:, 0] / (ans[:, 1] + 1e-3)
    ans[:, 0] = ans[:, 0].round()

    print(ans[:, 1])

    #pl.clf()
    #pl.plot(dflatten)
    #pl.grid()
    #pl.savefig(save_path + phase + '_evaluation.png')

    return ans[:, 0]



def evaluate_train(cfg, save_path, batchsize, model, device):
    
    print('phase : train')

    #data_train, data_val = cfg.DATASET.TRAIN_VAL(cfg.DATASET)
    data_train = cfg.DATASET.TRAIN(cfg.DATASET)
    data_train.transform = False

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batchsize, shuffle=False)
    #val_loader = torch.utils.data.DataLoader(data_val, batch_size=batchsize, shuffle=False)

    output_train = get_output2(model, device, data_loader)
    inference_train = inference(output_train, data_train, batchsize)
    evaluate(inference_train, data_train, save_path, 'train')

    #output_val = inference(model, device, val_loader, data_val)
    #evaluate(output_val, data_val, save_path, 'val')



def evaluate_test(cfg, save_path, batchsize, model, device):
    
    print('phase : validation')
    data_test = cfg.DATASET.TEST(cfg.DATASET)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batchsize, shuffle=False)

    output_test = get_output(model, device, test_loader)
    inference_test = inference(output_test, data_test, batchsize)

    dshape = data_test.data.shape
    dflatten = data_test.data.reshape((dshape[0] * dshape[1], dshape[2]))

    
    print(inference_test.shape, dflatten.shape)
    savearray = np.vstack((inference_test, dflatten[:, 0])).T

    print(savearray.shape)
    print(save_path + 'inference_test.csv')
    np.savetxt(fname=save_path + 'inference_test.csv', X = savearray, header="time,open_channels")


def main(cfg):

    pl.figure()
    torch.manual_seed(cfg.SOLVER.SEED)
    save_path = cfg.MODEL.SAVE_PATH
    batchsize = cfg.SOLVER.BATCHSIZE

    draw_loss(save_path)
    
    #data_train, data_val = cfg.DATASET.TRAIN_VAL(cfg.DATASET)
    #train_loader = torch.utils.data.DataLoader(data_train, batch_size=batchsize)
    #val_loader = torch.utils.data.DataLoader(data_val, batch_size=batchsize)

    

    device = torch.device(cfg.SOLVER.N_GPU if cfg.SOLVER.N_GPU >= 0 else "cpu")
    
    model = cfg.MODEL.TYPE(**cfg.MODEL.KEYWORDS)
    
    print('device', device)
    model.to(device)
    model.set_device(device)
    model.load_state_dict(torch.load(save_path + "model_best.pt", map_location=device))

    save_path = cfg.MODEL.SAVE_PATH
    util.make_directory(save_path)

    print('train val')
    #evaluate_test(cfg, save_path, batchsize, model, device)


    #evaluate_train(cfg, save_path, batchsize, model, device)
    evaluate_test(cfg, save_path, batchsize, model, device)





