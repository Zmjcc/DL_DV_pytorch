import numpy as np
from scipy import io
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
import os
from scipy import io
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
from tqdm import tqdm, trange
from sklearn import preprocessing

class MyDataset(Dataset):
    def __init__(self, training_dataset, training_labelset,mode):
        self.dataset = training_dataset
        self.labelset = training_labelset
        self.mode = mode

    def __getitem__(self, index):
        # if mode=='su':
        channel_bar_iter = self.dataset[index,:]
        label_iter = self.labelset[index,:]
        return torch.from_numpy(channel_bar_iter),torch.from_numpy(label_iter)
        # else:
        #     channel_bar_iter = self.dataset[index, :]
        #     H_noiseless_iter = self.labelset['H_noiseless'][index,:]
        #     H_iter = self.labelset['H'][index, :]
        #     return torch.from_numpy(channel_bar_iter), [torch.from_numpy(H_noiseless_iter),torch.from_numpy(H_iter)]

    def __len__(self):
        return int(self.dataset.shape[0])


class Dataset_load():
    def __init__(self, dataset_root,SNR_channel_dB,SNR_dB,test_length,Nt,Nr,dk,K,mode='gpu'):
        total_dataset = io.loadmat(dataset_root)
        if mode=='gpu':
            data_num = len(total_dataset['H'])
        else:
            data_num = 2*test_length
        H = total_dataset['H'][-data_num:,:]

        labelset_su = total_dataset['V'][-data_num:,:]
        SNR = 10**(SNR_dB/10)
        SNR_channel = 10**(SNR_channel_dB/10)
        noise_energy = 1 / SNR_channel
        channel_noise = np.sqrt(1 / 2 * noise_energy) * (
                    np.random.randn(data_num, Nt, Nr, K) + 1j * np.random.randn(data_num, Nt, Nr, K))
        H_noiseless = H
        H = H_noiseless + channel_noise



        test_labelset_su = labelset_su[-test_length:,:]
        labelset_su = labelset_su[:-test_length, :]
        H = np.concatenate([np.real(H), np.imag(H)], axis=-1)  #shape:Nt*dk*(2K)
        H = np.reshape(H,(data_num,-1))
        H_noiseless = np.concatenate([np.real(H_noiseless), np.imag(H_noiseless)], axis=-1)#shape:Nt*dk*(2K)
        H_noiseless = np.reshape(H_noiseless,(data_num,-1))
        test_H = H[-test_length:, :]
        H = H[:-test_length, :]

        test_H_noiseless = H_noiseless[-test_length:, :]
        H_noiseless = H_noiseless[:-test_length, :]


        #channel_merge = {'H_noiseless':H_noiseless,'H':H}
        #test_chanenl_merge = {'H_noiseless': test_H_noiseless, 'H': test_H}
        train_su_dataset = MyDataset(H,labelset_su,mode='su')
        self.train_su_dataset,self.valid_su_dataset = torch.utils.data.random_split(train_su_dataset,
                                                        [len(train_su_dataset)-len(train_su_dataset)//10,
                                                         len(train_su_dataset)//10])
        train_un_dataset = MyDataset(H,H_noiseless,mode = 'un')
        self.train_un_dataset, self.valid_un_dataset = torch.utils.data.random_split(train_un_dataset,
                                                                                     [len(train_un_dataset) -
                                                                                      len(train_un_dataset) // 10,
                                                                                      len(train_un_dataset) // 10])
        #self.train_su_dataset = MyDataset(dataset,labelset_su)
        self.test_su_dataset = MyDataset(test_H,test_labelset_su,mode = 'un')
        self.test_un_dataset = MyDataset(test_H,test_H_noiseless,mode = 'un')