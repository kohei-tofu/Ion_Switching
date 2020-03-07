import numpy as np
import pylab as pl
import torch

def get_data(name):
    if name == "train" or name == "val":
        fname ="data/train.csv"
    else:
        fname ="data/test.csv"

    data = np.loadtxt(fname, delimiter = ',', skiprows=1).astype(np.float32)
    print(data.shape)
    len_data = data.shape[0] // 500000
    ret = data.reshape((len_data, 500000, data.shape[1]))
    return ret

class IonDataset(torch.utils.data.Sampler):

    def __init__(self, phase, d_len = 256, transform=None):
        super(IonDataset, self).__init__(None)
        self.data = get_data(phase)
        self.d_len = d_len
        self.d_len2 = self.d_len // 2
        self.num_batches = self.data.shape[0]
        self.len_data = self.data.shape[1]
        self.data_num = self.num_batches * (self.len_data - self.d_len)
        

    def _get_data():
        pass

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        b_num = idx // int(self.len_data - self.d_len)
        d_num = idx // (b_num + 1)
        _time = self.data[b_num, (d_num):(d_num+self.d_len), 0]
        _input = self.data[b_num, (d_num):(d_num+self.d_len), 1]
        _target = self.data[b_num, (d_num):(d_num+self.d_len), 2]
        ret_in = _input[np.newaxis, :]
        ret_out = _target[np.newaxis, :]
        return ret_in, ret_out
        

import json

class Preprocess:
    
    def __init__(self):
        self.ave = 0.
        self.sig = 1.

    def __call__(self, data):
        return (data - self.ave) / self.sig
    
    def make_params(self, data):
        temp = data.reshape((-1, 3))[:, 1]
        self.ave = np.average(temp)
        self.sig = np.average(np.sqrt((temp - self.ave)**2))

    def save_parms(self, path):
        d = {'ave':float(self.ave), 'sig':float(self.sig)}
        f = open(path + "preprocess.json", "w")
        json.dump(d, f, ensure_ascii=False)

    def load_parms(self, path):
        f = open(path + "preprocess.json")
        d = json.load(f)
        self.ave = d['ave']
        self.sig = d['sig']

class IonDataset_one(torch.utils.data.Sampler):
    
    def __init__(self, phase, d_len = 256, transform=None, indecies=None):
        #super(IonDataset_one, self).__init__(phase, d_len, transform, indecies)
        super(IonDataset_one, self).__init__(None)

        self.data = get_data(phase)
        self.d_len = d_len

        if phase != 'test':
            self.num_batches = self.data.shape[0]
            self.len_data = self.data.shape[1]
            data_num = self.num_batches * (self.len_data - self.d_len)
            self.transform = True
            if indecies is None:
                self.indecies = range(data_num)
            else:
                self.indecies = indecies

            self.data_num = len(self.indecies)
        else:
            
            self.num_batches = self.data.shape[0]
            self.len_data = self.data.shape[1]

        self.preprocess = Preprocess()
    


    def __getitem__(self, id):

        idx = self.indecies[id]
        b_num = idx // int(self.len_data - self.d_len)
        d_num = idx // (b_num + 1)
        _input = self.data[b_num, (d_num):(d_num+self.d_len), 1]
        #_target = self.data[b_num, (d_num):(d_num+self.d_len), 2]
        _target = self.data[np.newaxis, :][:, b_num, d_num, 2]

        if self.transform == True:
            noise = np.random.normal(0., 0.002, _input.shape[0])
            _input = _input + noise.astype(np.float32)

        _input = self.preprocess(_input)
        _input = _input[np.newaxis, :]

        return _input, _target

    def __len__(self):
        return self.data_num

class IonDataset_seq(IonDataset_one):
    
    def __init__(self, phase, d_len = 256, transform=None, indecies=None):
        super(IonDataset_seq, self).__init__(phase, d_len, transform, indecies)


    def __getitem__(self, id):

        idx = self.indecies[id]
        b_num = idx // int(self.len_data - self.d_len)
        d_num = idx // (b_num + 1)
        _input = self.data[b_num, (d_num):(d_num+self.d_len), 1]
        _target = self.data[b_num, (d_num):(d_num+self.d_len), 2]

        if self.transform == True:
            noise = np.random.normal(0., 0.02, _input.shape[0])
            _input = _input + noise.astype(np.float32)

        _input = self.preprocess(_input)
        _input = _input[np.newaxis, :]

        return _input, _target



def testdata_loader(DATASET):

    time_lentgh = DATASET.KEYWORDS['time_lentgh']
    num_train_rate = DATASET.KEYWORDS['num_train_rate']
    data_test = IonDataset_seq('test', time_lentgh, None)
    data_test.preprocess = data_train.preprocess
    
    data_test.preprocess.save_parms(DATASET.PATH)
    
    return data_test

def split_loader(DATASET):

    idx_train = np.load(DATASET.PATH + 'idx_train.npy')
    idx_val = np.load(DATASET.PATH + 'idx_val.npy')

    time_lentgh = DATASET.KEYWORDS['time_lentgh']
    num_train_rate = DATASET.KEYWORDS['num_train_rate']

    data_train = IonDataset_seq('train', time_lentgh, True, idx_train)
    data_val = IonDataset_seq('val', time_lentgh, None, idx_val)

    print(data_train.data_num)
    print(data_val.data_num)
    data_train.preprocess.load_parms(DATASET.PATH)
    data_val.preprocess.load_parms(DATASET.PATH)

    return data_train, data_val


def get_idx(num_train_rate):

    idx_max = 4997440
    num_train = int(idx_max*num_train_rate)
    
    idx = np.array(range(idx_max))
    idx_train = np.random.choice(idx, num_train, replace=False)
    i_bool = np.ones(idx_max, dtype=bool)
    i_bool[idx_train] = False
    idx_val = idx[i_bool]

    return idx_train, idx_val

def main(cfg):
    
    make_preprocess1(cfg.DATASET)


def make_preprocess1(DATASET):
    
    time_lentgh = DATASET.KEYWORDS['time_lentgh']
    num_train_rate = DATASET.KEYWORDS['num_train_rate']

    idx_train, idx_val = get_idx(num_train_rate)
    data_train = IonDataset_seq('train', time_lentgh, True, idx_train)

    np.save(DATASET.PATH + 'idx_train.npy', idx_train)
    np.save(DATASET.PATH + 'idx_val.npy', idx_val)

    data_train.preprocess.make_params(data_train.data)
    data_train.preprocess.save_parms(DATASET.PATH)

    


def test1():

    data_train = get_data("train")
    data_test = get_data("test")
    print("train", data_train.shape)
    print("test", data_test.shape)

    pl.figure()
    pl.subplot(131)
    for _ in range(data_train.shape[0]):
        pl.plot(data_train[_, :, 0], data_train[_, :, 2])

    pl.subplot(132)
    for _ in range(data_train.shape[0]):
        pl.plot(data_train[_, :, 0], data_train[_, :, 1])
    pl.ylim(-6.5, 12.5)

    pl.subplot(133)
    for _ in range(data_test.shape[0]):
        
        pl.plot(data_test[_, :, 0], data_test[_, :, 1])

    pl.ylim(-6.5, 12.5)
    pl.show()


if __name__ == "__main__":
    
    test1()

    