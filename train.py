from __future__ import print_function
import argparse
import solver
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import util
import evaluate
import dataset

def train(cfg, model, device, data_train, optimizer, epoch, n_epoch_e):

    model.train()

    print('len(data_train)', len(data_train))
    length_sum = 0
    loss_sum = 0
    None_counter = 0
    #with torch.set_grad_enabled(phase == 'train'):
    with torch.set_grad_enabled(True):
        for batch_idx, (data, target) in enumerate(data_train):

            #output = model(data.to(device))
            optimizer.zero_grad()
            #loss = model.loss(output, target, epoch, 'train')
            loss = model.loss(data, target, epoch, 'train')
        
            #print(loss)
            if loss is None:
                print('None')
                None_counter += 1
                del data, output
                continue

            #model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            d = batch_idx/len(data_train) * 100
            print('Train Epoch: {} / {} [{} / {}({:.0f}%)] Loss: {:.6f}'.format(
                epoch, n_epoch_e, batch_idx, len(data_train), d, loss.item()))
            #loss.item() / len(batch_idx)

        print(len(data_train))
        #loss_sum /= (len(data_train) - None_counter)
        loss_sum /= (len(data_train) - None_counter)
        print('train loss : ', loss_sum)
        return loss_sum






def main(cfg):

    torch.manual_seed(cfg.SOLVER.SEED)
    
    #kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}

    data_train, data_val = cfg.DATASET.TRAIN_VAL(**cfg.DATASET.KEYWORDS)
    #sampler_train = torch.utils.data.RandomSampler(data_train)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(data_train)

    batchsize = cfg.SOLVER.BATCHSIZE
    #train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batchsize, batch_sampler=sampler_train)
    #train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batchsize, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batchsize)

    device = torch.device(cfg.SOLVER.N_GPU if cfg.SOLVER.N_GPU >= 0 else "cpu")
    
    model = cfg.MODEL.TYPE(**cfg.MODEL.KEYWORDS)
    print('device', device)
    model.to(device)
    model.set_device(device)

    train_loss = []
    test_loss = []
    best_loss = 5000.
    loss = 0
    save_path = cfg.MODEL.SAVE_PATH
    util.make_directory(save_path)


    if cfg.SOLVER.FROM_CHECKPOINT != True:
        n_epoch_s = 1
        optimizer, scheduler = solver.get_optimizer(cfg, model, -1)

    elif cfg.SOLVER.FROM_CHECKPOINT == 'sequence':
        print('start from checkpoint')
        model.load_state_dict(torch.load(save_path + "/model_last.pt", map_location=device))
        train_loss = np.load(save_path + '/loss_train_last.npy', train_loss).tolist()
        test_loss = np.load(save_path + '/loss_test_last.npy', test_loss).tolist()
        n_epoch_s = len(train_loss) + 1
        best_loss = np.min(test_loss)
        optimizer, scheduler = config.get_optimizer(cfg, model, -1)


    n_epoch_e = n_epoch_s + cfg.SOLVER.MAX_ITER

    print('start', n_epoch_s, n_epoch_e)
    for epoch in range(n_epoch_s, n_epoch_e):

        #scheduler.step()

        print('epoch', epoch, model.name)
        print("learning rate", scheduler.get_lr())

        print('phase : train')
        loss = train(cfg, model, device, train_loader, optimizer, epoch, n_epoch_e)
        train_loss.append(loss)
        scheduler.step()

        print('phase : validation')
        loss = evaluate.val(cfg, model, device, val_loader, n_epoch_e)
        print('loss', loss)
        if loss < best_loss:
            best_loss = loss
            mname = save_path + "model_best.pt"
            torch.save(model.state_dict(), mname)
            print('saved', mname) 

        mname = save_path + "/model_last.pt"
        torch.save(model.state_dict(), mname)
        np.save(save_path + '/loss_train_last.npy', train_loss)
        np.save(save_path + '/loss_test_last.npy', test_loss)
        torch.save(optimizer.state_dict(), save_path + "/optimizer_last.pth")
        print('saved', mname) 

        if (epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0):
            mname = save_path + "model_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), mname)
            torch.save(optimizer.state_dict(), save_path + "optimizer_" + str(epoch) + ".pth")
            print('saved', mname)





