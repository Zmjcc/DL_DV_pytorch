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

import tensorflow as tf
from tensorflow.keras.layers import Activation,Dense,Conv1D,Conv2D,Flatten,Permute,Reshape,Input,BatchNormalization,Concatenate,Add,Lambda,GlobalAveragePooling1D,Concatenate,GlobalAvgPool1D,Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend as BK
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

def valid(data_loader,model,criterion,save_path):
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
            loss = criterion(H_bar_data, label_data)
            loss_list.append(loss.item())
    model.train()
    return np.mean(loss_list)
def train(data_loader, valid_data_loader, model, criterion,save_path):
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
    return model, best_model_path


def minus_sum_rate_loss(y_true, y_pred):
    '''
                y_true is the channels
                y_pred is the predicted beamformers
                notice that, y_true has to be the same shape as y_pred
    '''
    ## construct complex data  channel shape:Nt,Nr,2*K   y_pred shape:Nt,dk,K,2
    y_true = tf.cast(tf.reshape(y_true, [-1, Nt, Nr, 2 * K]), tf.complex128)
    H = y_true[:, :, :, :K] + 1j * y_true[:, :, :, K:]
    y_pred = tf.cast(y_pred, tf.complex128)
    V0 = y_pred[:, :, :, :, 0] + 1j * y_pred[:, :, :, :, 1]

    ## power normalization of the predicted beamformers
    # VV = tf.matmul(V0,tf.transpose(V0,perm=[0,2,1],conjugate = True))
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[:, :, :, user], tf.transpose(V0[:, :, :, user], perm=[0, 2, 1], conjugate=True)))

    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    energy_scale = tf.tile(tf.reshape(energy_scale, (-1, 1, 1, 1)), (1, Nt, dk, K))
    V = V0 * tf.cast(energy_scale, tf.complex128)
    sum_rate = 0.0
    # import ipdb;ipdb.set_trace()
    for k in range(K):
        H_k = tf.transpose(H[:, :, :, k], perm=[0, 2, 1])  # NrxNt
        V_k = V[:, :, :, k]  # Ntx1
        signal_k = tf.matmul(H_k, V_k)
        signal_k_energy = tf.matmul(signal_k, tf.transpose(signal_k, perm=[0, 2, 1], conjugate=True))
        interference_k_energy = 0.0
        for j in range(K):
            if j != k:
                V_j = V[:, :, :, j]
                interference_j = tf.matmul(H_k, V_j)
                interference_k_energy = interference_k_energy + tf.matmul(interference_j,
                                                                          tf.transpose(interference_j, perm=[0, 2, 1],
                                                                                       conjugate=True))
        SINR_k = tf.matmul(signal_k_energy,
                           tf.linalg.inv(interference_k_energy + sigma_2 * tf.eye(Nr, dtype=tf.complex128)))
        rate_k = tf.math.log(tf.linalg.det(tf.eye(Nr, dtype=tf.complex128) + SINR_k)) / tf.cast(tf.math.log(2.0),
                                                                                                dtype=tf.complex128)
        sum_rate = sum_rate + rate_k
    sum_rate = tf.cast(tf.math.real(sum_rate), tf.float32)
    # loss
    loss = -sum_rate
    return loss


# @tf.function
def DUU_EZF(channel, p_list, q_list, factor):
    channel = tf.cast(tf.reshape(channel, [-1, Nt, Nr, 2 * K]), tf.complex128)
    channel = channel[:, :, :, :K] + 1j * channel[:, :, :, K:]
    channel = tf.transpose(channel, [0, 2, 1, 3])
    P = list()
    factor = tf.cast(factor, tf.complex128)
    p_list = tf.cast(p_list, tf.complex128)
    q_list = tf.cast(q_list, tf.complex128)
    import ipdb;ipdb.set_trace()
    for user in range(K):
        H_this_user = channel[:, :, :, user]
        _, _, v = tf.linalg.svd(H_this_user,full_matrices=True)
        P.append(v[:, :, :dk])
    P = tf.stack(P, axis=3)
    P = tf.reshape(P, [-1, Nt, K * dk])
    # import ipdb;ipdb.set_trace()
    B = sigma_2 * tf.tile(tf.reshape(tf.eye(K * dk, dtype=tf.complex128), (1, K * dk, K * dk)), (factor.shape[0], 1, 1))
    B = tf.tile(tf.reshape(factor, [-1, 1, 1]), (1, K * dk, K * dk)) * B
    temp = tf.matmul(P, tf.linalg.diag(tf.sqrt(q_list)))
    B = B + tf.matmul(tf.transpose(temp, [0, 2, 1], conjugate=True), temp)
    P = tf.matmul(P, tf.linalg.inv(B))
    V = list()
    for user in range(K):
        for ds in range(dk):
            V_temp = P[:, :, user * dk + ds]
            V_temp = tf.tile(tf.reshape(tf.sqrt(p_list[:, user * dk + ds]), [-1, 1]), (1, Nt)) * V_temp / tf.tile(
                tf.reshape(tf.norm(V_temp, axis=1), [-1, 1]), (1, Nt))
            V.append(V_temp)
    V = tf.reshape(tf.stack(V, 2), [-1, Nt, dk, K])
    # import ipdb;ipdb.set_trace()
    V = tf.reshape(V, [-1, Nt, dk, K, 1])
    # import ipdb;ipdb.set_trace()
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V
def DUU_EZF_loss(y_true,y_pred):
    #import ipdb;ipdb.set_trace()
    channel = y_true[:,:2*Nr*Nt*K]
    channel_noise = y_true[:,2*Nr*Nt*K:]
    #channel = y_true
    p_list_pred = y_pred[:,:K*dk]
    q_list_pred = y_pred[:,K*dk:2*K*dk]
    mrt_list_pred = y_pred[:,-1]
    V_restore = DUU_EZF(channel_noise,p_list_pred,q_list_pred,mrt_list_pred)
    import ipdb;ipdb.set_trace()
    return minus_sum_rate_loss(channel,V_restore)

if __name__ == "__main__":
    lr = 1e-2

    total_dataset = Dataset_load(data_root,SNR_channel_dB=SNR_channel_dB,SNR_dB=SNR_dB,test_length = test_length,
                                 Nt=Nt,Nr=Nr,dk=dk,K=K, mode=data_mode)
    Loss_util = Loss_utils(Nt,Nr,dk,K,p,sigma_2)
    MSE_loss = Loss_util.MSE_loss
    SMR_loss = Loss_util.SMR_loss
    model = BeamformNet(model_name, Nt, Nr, dk, K)
    device = torch.device(0)
    if torch.cuda.is_available():
        model.to(device)

    su_iter = iter(DataLoader(total_dataset.test_su_dataset, shuffle=False, batch_size=batch_size, drop_last=True))
    un_iter = iter(DataLoader(total_dataset.test_un_dataset, shuffle=False, batch_size=batch_size, drop_last=True))
    H_bar_su,pqm = next(su_iter)
    H_bar_un,hh = next(un_iter)
    #import ipdb;ipdb.set_trace()
    performance = DUU_EZF_loss(hh, pqm)
    '''supervised learning'''
    best_su_model_path = './model/DUU_models_%d_%d_%d_%d_%d_su.pth'%(Nt,Nr,K,dk,SNR_dB)
    train_su_dataloader = DataLoader(total_dataset.train_su_dataset, shuffle=False, batch_size=batch_size, drop_last=True)

    valid_su_dataloader = DataLoader(total_dataset.valid_su_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    train(train_su_dataloader,valid_su_dataloader, model, criterion=MSE_loss,save_path = best_su_model_path)
    print('supervised learning complete!')

    '''unsupervised learning'''
    best_un_model_path = './model/DUU_models_%d_%d_%d_%d_%d_un.pth' % (Nt, Nr, K, dk, SNR_dB)
    train_un_dataloader = DataLoader(total_dataset.train_un_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    valid_un_dataloader = DataLoader(total_dataset.valid_un_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    train(train_un_dataloader,valid_un_dataloader, model, criterion=SMR_loss,save_path = best_un_model_path)
    print('unsupervised learning complete!')

    '''test'''


