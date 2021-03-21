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
        transmit_pa = total_dataset['transmit_power_allocation'][-data_num:,:]
        upload_pa = total_dataset['upload_power_allocation'][-data_num:,:]
        MRT_ratio = total_dataset['MRT_ratio'][-data_num:,:]
        labelset_su = np.concatenate((transmit_pa, upload_pa, MRT_ratio), axis=-1)
        SNR = 10**(SNR_dB/10)
        SNR_channel = 10**(SNR_channel_dB/10)
        noise_energy = 1 / SNR_channel
        channel_noise = np.sqrt(1 / 2 * noise_energy) * (
                    np.random.randn(data_num, Nt, Nr, K) + 1j * np.random.randn(data_num, Nt, Nr, K))
        H_noiseless = H
        H = H_noiseless + channel_noise
        H_ensemble = np.transpose(H, (0, 3, 2, 1))
        H_ensemble = np.reshape(H_ensemble, (data_num, K * Nr, Nt))
        H_bar = np.zeros((data_num, K * Nr, K * Nr)) + 1j * np.zeros((data_num, K * Nr, K * Nr))
        for i in range(len(H_ensemble)):
            H_bar[i] = H_ensemble[i].dot(np.transpose(np.conjugate(H_ensemble[i])))
        dataset = np.triu(np.real(H_bar)) + np.tril(np.imag(H_bar))


        # preprocessing
        dataset = np.reshape(dataset, (data_num, -1))
        scaler = preprocessing.StandardScaler()
        scaler.fit(dataset)
        dataset = scaler.transform(dataset)
        dataset = np.reshape(dataset, (data_num, K * Nr, K * Nr))

        dataset = np.expand_dims(dataset, axis=-1)
        test_dataset = dataset[-test_length:, :]
        dataset = dataset[:-test_length, :]

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

        channel_merge = np.concatenate((H_noiseless,H),axis = -1)
        test_channel_merge = np.concatenate((test_H_noiseless, test_H), axis=-1)
        #channel_merge = {'H_noiseless':H_noiseless,'H':H}
        #test_chanenl_merge = {'H_noiseless': test_H_noiseless, 'H': test_H}
        train_su_dataset = MyDataset(dataset,labelset_su,mode='su')
        self.train_su_dataset,self.valid_su_dataset = torch.utils.data.random_split(train_su_dataset,
                                                        [len(train_su_dataset)-len(train_su_dataset)//10,
                                                         len(train_su_dataset)//10])
        train_un_dataset = MyDataset(dataset,channel_merge,mode = 'un')
        self.train_un_dataset, self.valid_un_dataset = torch.utils.data.random_split(train_un_dataset,
                                                                                     [len(train_un_dataset) -
                                                                                      len(train_un_dataset) // 10,
                                                                                      len(train_un_dataset) // 10])
        #self.train_su_dataset = MyDataset(dataset,labelset_su)
        self.test_su_dataset = MyDataset(test_dataset,test_labelset_su,mode = 'un')
        self.test_un_dataset = MyDataset(test_dataset,test_channel_merge,mode = 'un')