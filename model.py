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
def complex_det(A):
    A_real = A.real
    A_imag = A.imag
    upper_matrix = torch.cat((A_real,-A_imag),dim = 2)
    lower_matrix = torch.cat((A_imag,A_real),dim = 2)
    Matrix = torch.cat(((upper_matrix,lower_matrix)),dim = 1)
    det_result = torch.linalg.det(Matrix)
    return torch.sqrt(det_result)
class Loss_utils():
    def __init__(self,Nt,Nr,dk,K,p,sigma_2):
        super(Loss_utils).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.dk = dk
        self.K = K
        self.p = p
        self.sigma_2 = sigma_2
    def MSE_loss(self,y_true,y_pred):
        Nt = self.Nt
        Nr = self.Nr
        dk = self.dk
        K = self.K
        p = self.p
        sigma_2 = self.sigma_2

        loss = F.mse_loss(y_pred, y_true, reduction='mean')
        return loss

    def SMR_loss(self,y_true,y_pred):
        Nt = self.Nt
        Nr = self.Nr
        dk = self.dk
        K = self.K
        p = self.p
        sigma_2 = self.sigma_2
        batch_size = y_true.shape[0]
        #H_noiseless = torch.view_as_complex(y_true[:,:(2*Nt*Nr*K)].reshape((-1,Nt,Nr,2,K)).permute(0,1,2,4,3).contiguous())
        H = torch.view_as_complex(y_true.reshape((-1,Nt,Nr,2,K)).permute(0,1,2,4,3).contiguous())

        # p_list_pred = y_pred[:, :K * dk].type_as(H)
        # q_list_pred = y_pred[:, K * dk:2 * K * dk].type_as(H)
        # mrt_list_pred = y_pred[:, -1:].type_as(H)

        #restore V
        V = torch.view_as_complex(y_pred.reshape((-1,Nt,dk,K,2)).contiguous())
        #V = self.DUU_EZF(H,p_list_pred,q_list_pred,mrt_list_pred)
        '''need to change for normal runing'''
        sum_rate = torch.zeros(1).cuda()
        for user in range(K):
            H_k = H[:,:,:,user].permute(0,2,1)
            V_k = V[:,:,:,user]
            signal_k = torch.matmul(H_k, V_k)
            signal_k_energy = torch.matmul(signal_k,torch.conj(signal_k.permute(0,2,1)))
            interference_k_energy = sigma_2 * torch.eye(Nr).type_as(H).reshape((1,Nr,Nr)).repeat(batch_size,1,1)
            for j in range(K):
                if j!=user:
                    V_j = V[:, :, :, j]
                    interference_j = torch.matmul(H_k, V_j)
                    interference_k_energy = interference_k_energy + torch.matmul(interference_j,torch.conj(interference_j.permute(0,2,1)))
                SINR_k = torch.matmul(signal_k_energy, torch.linalg.inv(interference_k_energy))
                rate_k = torch.log2(complex_det(SINR_k + torch.eye(Nr).type_as(H).reshape((1,Nr,Nr)).repeat(batch_size,1,1)))
            sum_rate = sum_rate + rate_k
        sum_rate = - sum_rate
        return torch.mean(sum_rate)


class CNN_2D_net(nn.Module):
    def __init__(self, Nt, Nr, dk, K):
        super(CNN_2D_net, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.dk = dk
        self.K = K
        self.CNN1 = torch.nn.Conv1d(in_channels=Nr*2*K, out_channels=Nr*4*K, kernel_size=7, stride=1, padding=3)
        #padding=(kernel_size-1)/2 )
        self.bn1 = nn.BatchNorm1d(Nr*4*K)
        self.CNN2 = torch.nn.Conv1d(in_channels=Nr*4*K, out_channels=Nr*2*K, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(Nr*2*K)
        self.CNN3 = torch.nn.Conv1d(in_channels=Nr*2*K, out_channels=Nr*2*K, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(Nr*2*K)
        self.CNN4 = torch.nn.Conv1d(in_channels=Nr * 2 * K, out_channels=dk * 2 * K, kernel_size=3, stride=1, padding=1)
        # flatten_dim = (Nt-12)*2*K*dk
        # self.fcp = nn.Linear(flatten_dim,dk*K)
        # self.fcq = nn.Linear(flatten_dim, dk*K)
        # self.fcr = nn.Linear(flatten_dim, 1)
    def forward(self,x):
        #x = x.permute(0,3,1,2)
        x = x.reshape((-1,self.Nt,self.Nr*self.K*2)).permute(0,2,1)
        x = F.leaky_relu(self.bn1(self.CNN1(x)), negative_slope=0.3)
        x = F.leaky_relu(self.bn2(self.CNN2(x)), negative_slope=0.3)
        x = F.leaky_relu(self.bn3(self.CNN3(x)), negative_slope=0.3)
        x = self.CNN4(x)
        x = x.permute(0,2,1).reshape((-1,self.dk*self.K*2*self.Nt))

        # x_flatten = x.reshape((x.shape[0],-1))
        # p_pred = F.softmax(self.fcp(x_flatten),dim = 1)
        # q_pred = F.softmax(self.fcq(x_flatten),dim = 1)
        # mrt_pred = torch.sigmoid(self.fcr(x_flatten))
        # mrt_pred = 0.005 + 0.025 * mrt_pred
        return x

class BeamformNet(nn.Module):
    def __init__(self, model_name, Nt, Nr, dk, K):
        super(BeamformNet, self).__init__()
        self.bfnet = CNN_2D_net(Nt,Nr,dk,K)
    def forward(self,x):
        return self.bfnet(x)
