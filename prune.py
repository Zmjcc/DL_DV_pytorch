import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import io

import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
import os

from scipy import io
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
from tqdm import tqdm,trange
import math
from dataset_load import Dataset_load

from option import parge_config
from model import BeamformNet,Loss_utils
args = parge_config()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
Nt = args.Nt
Nr = args.Nr
K = args.K
dk = args.dk
SNR_dB = args.SNR
SNR = 10**(SNR_dB/10)
p = 1
sigma_2 = 1/SNR
SNR_channel_dB = args.SNR_channel
data_mode = args.mode
batch_size = args.batch_size
epochs = args.epoch
test_length = args.test_length

base_root = '/home/zmj/Desktop/precode/data/'
data_root = base_root + 'DUU_EZF_dataset_%d_%d_%d_%d_%d.mat'%(Nt,Nr,K,dk,SNR_dB)
train_mode = 'train'
model_name = 'CNN2D'

from nni.compression.pytorch.utils.counter import count_flops_params
import nni
from nni.compression.pytorch import apply_compression_results, ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import (
    LevelPruner,
    SlimPruner,
    FPGMPruner,
    L1FilterPruner,
    L2FilterPruner,
    AGPPruner,
    ActivationMeanRankFilterPruner,
    ActivationAPoZRankFilterPruner
)
str2pruner = {
    'level': LevelPruner,
    'l1filter': L1FilterPruner,
    'l2filter': L2FilterPruner,
    'slim': SlimPruner,
    'agp': AGPPruner,
    'fpgm': FPGMPruner,
    'mean_activation': ActivationMeanRankFilterPruner,
    'apoz': ActivationAPoZRankFilterPruner
}
def get_dummy_input(args, device):
    dummy_input = torch.randn([200, 40, 40, 1]).to(device)
    return dummy_input

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 设置随机数种子
setup_seed(2021)
if data_mode=='debug':
    epochs = 10
def validate(data_loader,model,criterion):
    model.eval()
    loss_list = []
    print("Validation started...")
    with torch.no_grad():
        for H_bar_data,label_data in data_loader:
            H_bar_data = H_bar_data.clone().detach().type(torch.float32)
            H_bar_data = H_bar_data.to(device)

            label_data = label_data.clone().detach().type(torch.float32)
            label_data = label_data.to(device)
            # forward
            model_output = model(H_bar_data)
            loss = criterion(label_data,model_output)
            loss_list.append(loss.item())
    model.train()
    return np.mean(loss_list)
def test(data_loader,model,criterion):
    model.eval()
    loss_list = []
    print("test started...")
    with torch.no_grad():
        for H_bar_data,label_data in data_loader:
            H_bar_data = H_bar_data.clone().detach().type(torch.float32)
            H_bar_data = H_bar_data.to(device)

            label_data = label_data.clone().detach().type(torch.float32)
            label_data = label_data.to(device)
            # forward
            model_output = model(H_bar_data)
            loss = -criterion(label_data,model_output)
            loss_list.append(loss.item())
    model.train()
    return np.mean(loss_list)
def train(data_loader, valid_data_loader, model, criterion,save_path,lr):
    min_loss = 1e5
    optimizier = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizier,
                                  factor=0.3,
                                  patience=5,
                                  mode='min',
                                  min_lr=1e-5,
                                  eps=1e-5,
                                  verbose=True)
    loss = None
    for epoch in range(epochs):
        step = 0
        with tqdm(data_loader, desc="epoch:" + str(epoch),
                  postfix={"train_loss": 0} if loss is None else {"train_loss": loss.data}) as iteration:
            for H_bar_data,label_data in iteration:
                iteration.set_description("epoch:" + str(epoch))
                H_bar_data = H_bar_data.clone().detach().type(torch.float32)
                H_bar_data = H_bar_data.to(device)

                label_data = label_data.clone().detach().type(torch.float32)
                label_data = label_data.to(device)
                # forward
                model_output = model(H_bar_data)
                loss = criterion(label_data,model_output)

                iteration.set_postfix(loss=('%.4f' % loss.data.item()))
                # backward

                optimizier.zero_grad()
                loss.backward()
                optimizier.step()
                step = step + 1
        for param_group in optimizier.param_groups:
            print(param_group['lr'])
        if optimizier.param_groups[0]['lr'] == 1e-5:
            print('early stop!')
            break
        valid_loss = validate(valid_data_loader, model, criterion)
        print('valid_loss:' + '%.7f' % valid_loss)
        scheduler.step(valid_loss)
        if valid_loss < min_loss and abs(valid_loss - min_loss) > 1e-5:
            min_loss = valid_loss
            print("model has been saved")
            torch.save(model, save_path)
if __name__ == "__main__":

    total_dataset = Dataset_load(data_root,SNR_channel_dB=SNR_channel_dB,SNR_dB=SNR_dB,test_length = test_length,
                                 Nt=Nt,Nr=Nr,dk=dk,K=K, mode=data_mode)
    Loss_util = Loss_utils(Nt,Nr,dk,K,p,sigma_2)
    MSE_loss = Loss_util.MSE_loss
    SMR_loss = Loss_util.SMR_loss
    model = BeamformNet(model_name, Nt, Nr, dk, K)
    device = torch.device(0)
    if torch.cuda.is_available():
        model.to(device)

    '''supervised learning'''
    best_su_model_path = './model/DUU_models_%d_%d_%d_%d_%d_su.pth'%(Nt,Nr,K,dk,SNR_dB)
    train_su_dataloader = DataLoader(total_dataset.train_su_dataset, shuffle=False, batch_size=batch_size, drop_last=True)

    valid_su_dataloader = DataLoader(total_dataset.valid_su_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    #train(train_su_dataloader,valid_su_dataloader, model, criterion=MSE_loss,save_path = best_su_model_path,  lr = 1e-2)
    #print('supervised learning complete!')

    '''unsupervised learning'''
    best_un_model_path = './model/DUU_models_%d_%d_%d_%d_%d_un.pth' % (Nt, Nr, K, dk, SNR_dB)
    train_un_dataloader = DataLoader(total_dataset.train_un_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    valid_un_dataloader = DataLoader(total_dataset.valid_un_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    #train(train_un_dataloader,valid_un_dataloader, model, criterion=SMR_loss,save_path = best_un_model_path, lr = 1e-3)
    #print('unsupervised learning complete!')
    model_path = best_un_model_path
    mask_path = './model/DUU_models_%d_%d_%d_%d_%d_un_mask_model.pth' % (Nt, Nr, K, dk, SNR_dB)
    config_list = [{'sparsity': 0.8, 'op_types': ['Conv2d']}]
    pruner = L2FilterPruner(model, config_list)
    pruner.compress()
    model.eval()
    pruner.export_model(model_path=model_path, mask_path=mask_path)
    apply_compression_results(model, mask_path, device)
    '''test'''
    test_dataloader = DataLoader(total_dataset.test_un_dataset, shuffle=False, batch_size=batch_size,
                                     drop_last=True)
    model = torch.load(best_su_model_path)
    su_performance = test(test_dataloader,model,criterion=SMR_loss)
    print('supervised learning performance:' + str(su_performance))
    model = torch.load(best_un_model_path)
    un_performance = test(test_dataloader,model,criterion=SMR_loss)
    print('unsupervised learning performance:' + str(un_performance))

#python train.py --Nt 64 --Nr 4 --K 10 --dk 2 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1 --test_length 2000

