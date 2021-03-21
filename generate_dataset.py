import numpy as np

np.random.seed(2020)
import tensorflow as tf

from scipy import io
# import os
import os
import hdf5storage
import random
from sklearn import preprocessing
import logging
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
# tf.config.run_functions_eagerly(True)

# %% load and construct data
from option import parge_config

args = parge_config()
Nt = args.Nt
Nr = args.Nr
K = args.K
dk = args.dk
SNR_dB = args.SNR
SNR = 10 ** (SNR_dB / 10)
p = 1
sigma_2 = 1 / SNR
SNR_channel_dB = args.SNR_channel
data_mode = args.mode

mode = 'train'
data_mode = 'debug'
# if data_mode =='debug':
#     data_length = 2000
#     test_length = 2000
#     epochs = 10

dataset_root = '/home/zmj/Desktop/precode/data/DV_dataset'
data_root = '/home/zmj/Desktop/precode/data/channel_data_full_user.mat'
channels = hdf5storage.loadmat(data_root)['H_list'][:, :, :, :K]
model_root = './model/'
factor = 1
# LS estimation noise
# LS_noise = np.sqrt(1/2/SNR)*(np.random.randn(len(channels),Nt,Nr,K)+1j*np.random.randn(len(channels),Nt,Nr,K))
# channels_noisy = channels+LS_noise

# channels = np.concatenate([np.real(channels),np.imag(channels)],axis=-1)

# import ipdb;ipdb.set_trace()

H = channels
print(H.shape)
data_num = len(H)

H_ensemble = np.transpose(H, (0, 3, 2, 1))
H_ensemble = np.reshape(H_ensemble, (data_num, K * Nr, Nt))
print(H_ensemble.shape)

H_bar = np.zeros((data_num, K * Nr, K * Nr)) + 1j * np.zeros((data_num, K * Nr, K * Nr))
for i in range(len(H_ensemble)):
    H_bar[i] = H_ensemble[i].dot(np.transpose(np.conjugate(H_ensemble[i])))
print(H_bar.shape)

# %% construct the dataset
dataset = np.triu(np.real(H_bar)) + np.tril(np.imag(H_bar))

# preprocessing
dataset = np.reshape(dataset, (data_num, -1))
scaler = preprocessing.StandardScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)
dataset = np.reshape(dataset, (data_num, K * Nr, K * Nr))


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
    loss = sum_rate
    return loss


def EZF(channel):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, 2 * K]), tf.complex128)
    H = H[:, :, :, :K] + 1j * H[:, :, :, K:]
    H = tf.transpose(H, [0, 2, 1, 3])
    P = list()
    for user in range(K):
        H_this_user = H[:, :, :, user]
        _, _, v = tf.linalg.svd(H_this_user)
        P.append(v[:, :, :dk])
    P = tf.stack(P, axis=3)
    P = tf.reshape(P, [-1, Nt, K * dk])
    # import ipdb;ipdb.set_trace()
    V = tf.matmul(P, tf.linalg.inv(tf.matmul(tf.transpose(P, [0, 2, 1], conjugate=True), P)))  # B*Nt*Kdk
    V = tf.reshape(V, [-1, Nt, dk, K, 1])
    # import ipdb;ipdb.set_trace()
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V


def DUU_EZF_optimal_factor(channel, p_list, q_list):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, 2 * K]), tf.complex128)
    H = H[:, :, :, :K] + 1j * H[:, :, :, K:]
    H = tf.transpose(H, [0, 2, 1, 3])
    V_ensemble = list()
    p_list = tf.cast(p_list, tf.complex128)
    q_list = tf.cast(q_list, tf.complex128)
    for user in range(K):
        H_this_user = H[:, :, :, user]
        _, _, v = tf.linalg.svd(H_this_user)
        V_ensemble.append(v[:, :, :dk])
    V_ensemble = tf.stack(V_ensemble, axis=3)
    V_ensemble = tf.reshape(V_ensemble, [-1, Nt, K * dk])
    # import ipdb;ipdb.set_trace()
    factor_list = tf.cast(tf.linspace(0.01, 0.03, 21), tf.complex128)
    performance_list = list()
    temp1 = tf.matmul(V_ensemble, tf.linalg.diag(tf.sqrt(q_list)))
    temp1 = tf.matmul(tf.transpose(temp1, [0, 2, 1], conjugate=True), temp1)
    for factor in factor_list:
        B = factor * sigma_2 * tf.eye(K * dk, dtype=tf.complex128)
        B = B + temp1
        P = tf.matmul(V_ensemble, tf.linalg.inv(B))
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
        performance_temp = minus_sum_rate_loss(channel, V)
        performance_list.append(performance_temp)
    performance = tf.stack(performance_list, 1)
    max_index = tf.argmax(performance, axis=1)
    best_factor_list = 0.001 * tf.cast(max_index, tf.float32) + 0.01
    # import ipdb;ipdb.set_trace()
    return best_factor_list


def DUU_EZF(channel, p_list, q_list, factor):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, 2 * K]), tf.complex128)
    H = H[:, :, :, :K] + 1j * H[:, :, :, K:]
    H = tf.transpose(H, [0, 2, 1, 3])
    P = list()
    factor = tf.cast(factor, tf.complex128)
    p_list = tf.cast(p_list, tf.complex128)
    q_list = tf.cast(q_list, tf.complex128)
    for user in range(K):
        H_this_user = H[:, :, :, user]
        _, _, v = tf.linalg.svd(H_this_user)
        P.append(v[:, :, :dk])
    P = tf.stack(P, axis=3)
    P = tf.reshape(P, [-1, Nt, K * dk])
    # import ipdb;ipdb.set_trace()
    # factor1 = 0.016
    # B = sigma_2 * tf.eye(K*dk,dtype = tf.complex128) + 0*tf.matmul(tf.transpose(P,[0,2,1],conjugate=True),P)
    # B = factor1 * sigma_2 * tf.eye(K*dk,dtype = tf.complex128)
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


def WMMSE(channel):
    # EZF initial
    V = EZF(channel)
    V = tf.cast(V, tf.complex128)
    V = V[:, :, :, :, 0] + 1j * V[:, :, :, :, 1]
    UW = V2UW(channel, V) * 0
    # import ipdb;ipdb.set_trace()
    for i in range(10):
        print('WMMSE_iter:' + str(i))
        new_UW = V2UW(channel, V)
        new_V = UW_restore_V(channel, new_UW)
        if tf.norm(new_UW[:, 2 * Nr * dk * K:] - UW[:, 2 * Nr * dk * K:], 2) < 0.001:
            print(tf.norm(new_UW[:, 2 * Nr * dk * K:] - UW[:, 2 * Nr * dk * K:], 2))
            break
        # V = UW_restore_V(channel,V2UW(channel,V))
        UW = new_UW
        V = new_V
    V = tf.reshape(V, [-1, Nt, dk, K, 1])
    # import ipdb;ipdb.set_trace()
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V


def update_RMMWSE_U(H_bar, X):
    U_list = list()
    trace_Hbar_XX = 0
    for user_index in range(K):
        Xk = X[:, :, :, user_index]
        trace_Hbar_XX = trace_Hbar_XX + sigma_2 / p * tf.linalg.trace(
            tf.matmul(H_bar, tf.matmul(Xk, tf.transpose(Xk, [0, 2, 1], conjugate=True))))
    for user_index in range(K):
        HXXH = tf.zeros([Nr, Nr], dtype=tf.complex128)
        Xk = X[:, :, :, user_index]
        H_bark = H_bar[:, user_index * Nr:(user_index + 1) * Nr, :]
        for i in range(K):
            Xi = X[:, :, :, i]  #
            HX = tf.matmul(H_bark, Xi)
            HXXH = HXXH + tf.matmul(HX, tf.transpose(HX, perm=[0, 2, 1], conjugate=True))
        U_this_user = tf.matmul(tf.matmul(tf.linalg.inv(
            sigma_2 / p * tf.tile(tf.reshape(trace_Hbar_XX, (-1, 1, 1)), [1, Nr, Nr]) * tf.eye(Nr,
                                                                                               dtype=tf.complex128) + HXXH),
                                          H_bark), Xk)
        U_list.append(U_this_user)
    U = tf.stack(U_list, 3)  # B*Nr*K
    return U


def update_RMMWSE_W(H_bar, X, U):
    W_list = list()
    for user_index in range(K):
        Uk = U[:, :, :, user_index]
        H_bark = H_bar[:, user_index * Nr:(user_index + 1) * Nr, :]
        Xk = X[:, :, :, user_index]
        W_this_user = tf.linalg.inv(tf.eye(dk, dtype=tf.complex128) - tf.matmul(
            tf.matmul(tf.transpose(Uk, perm=[0, 2, 1], conjugate=True), H_bark), Xk))
        W_list.append(W_this_user)
    W = tf.stack(W_list, 3)
    return W


def update_RMMWSE_X(H_bar, U, W):
    X_list = list()
    B = H_bar * 0
    for user_index in range(K):
        Uk = U[:, :, :, user_index]
        Wk = W[:, :, :, user_index]
        Mk = tf.matmul(tf.matmul(Uk, Wk), tf.transpose(Uk, perm=[0, 2, 1], conjugate=True))
        H_bark = H_bar[:, user_index * Nr:(user_index + 1) * Nr, :]
        B = B + sigma_2 / p * tf.tile(tf.reshape(tf.linalg.trace(Mk), (-1, 1, 1)),
                                      [1, K * Nr, K * Nr]) * H_bar + tf.matmul(
            tf.matmul(tf.transpose(H_bark, perm=[0, 2, 1], conjugate=True), Mk), H_bark)
    B_inverse = tf.linalg.inv(B)
    for user_index in range(K):
        Uk = U[:, :, :, user_index]
        Wk = W[:, :, :, user_index]
        H_bark = H_bar[:, user_index * Nr:(user_index + 1) * Nr, :]
        HUWk = tf.matmul(tf.matmul(tf.transpose(H_bark, perm=[0, 2, 1], conjugate=True), Uk), Wk)
        X_this_user = tf.matmul(B_inverse, HUWk)
        X_list.append(X_this_user)
    X = tf.stack(X_list, 3)
    return X


def tf_pinv(a, rcond=None):
    """Taken from
    https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/ops/linalg/linalg_impl.py
    """
    dtype = a.dtype.as_numpy_dtype

    if rcond is None:
        def get_dim_size(dim):
            dim_val = a.shape[dim]
            if dim_val is not None:
                return dim_val
            return tf.shape(a)[dim]

        num_rows = get_dim_size(-2)
        num_cols = get_dim_size(-1)
        if isinstance(num_rows, int) and isinstance(num_cols, int):
            max_rows_cols = float(max(num_rows, num_cols))
        else:
            max_rows_cols = tf.cast(tf.maximum(num_rows, num_cols), dtype)
        rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = tf.convert_to_tensor(rcond, dtype=dtype, name='rcond')

    # Calculate pseudo inverse via SVD.
    # Note: if a is Hermitian then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,  # Sigma
        left_singular_vectors,  # U
        right_singular_vectors,  # V
    ] = tf.linalg.svd(
        a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = tf.cast(rcond, dtype=singular_values.dtype) * tf.reduce_max(singular_values, axis=-1)
    singular_values = tf.where(
        singular_values > cutoff[..., None], singular_values,
        np.array(np.inf, dtype))

    # By the definition of the SVD, `a == u @ s @ v^H`, and the pseudo-inverse
    # is defined as `pinv(a) == v @ inv(s) @ u^H`.
    a_pinv = tf.matmul(
        right_singular_vectors / tf.cast(singular_values[..., None, :], dtype=dtype),
        left_singular_vectors,
        adjoint_b=True)

    if a.shape is not None and a.shape.rank is not None:
        a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv


def RWMMSE(channel):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, 2 * K]), tf.complex128)
    H = H[:, :, :, :K] + 1j * H[:, :, :, K:]  # B*Nt*Nr*K
    H_ensemble = tf.reshape(tf.transpose(H, [0, 3, 2, 1]), [-1, K * Nr, Nt])  # B*KNr*Nt
    H_bar = tf.matmul(H_ensemble, tf.transpose(H_ensemble, [0, 2, 1], conjugate=True))
    V = EZF(channel)
    V = tf.cast(V, tf.complex128)
    V = V[:, :, :, :, 0] + 1j * V[:, :, :, :, 1]  # B*Nt*dk*K
    X_list = []
    # import ipdb;ipdb.set_trace()
    H_ensemble_pinv = tf_pinv(tf.transpose(H_ensemble, [0, 2, 1],
                                           conjugate=True))  # tf.matmul(tf.transpose(H_ensemble,[0,2,1],conjugate=True),tf.linalg.inv(tf.matmul(H_ensemble,tf.transpose(H_ensemble,[0,2,1],conjugate=True))))
    for user_index in range(K):
        Vk = V[:, :, :, user_index]
        X_list.append(tf.matmul(H_ensemble_pinv, Vk))
    X = tf.stack(X_list, 3)  # B*KNr*dk*K
    U = update_RMMWSE_U(H_bar, X)
    W = update_RMMWSE_W(H_bar, X, U)
    for i in range(50):
        # new_UW = V2UW(channel,V)
        # print('RWMMSE_iter:'+str(i))
        new_X = update_RMMWSE_X(H_bar, U, W)
        new_U = update_RMMWSE_U(H_bar, new_X)
        new_W = update_RMMWSE_W(H_bar, new_X, new_U)
        # import ipdb;ipdb.set_trace()
        if tf.norm(new_W - W, 2).numpy() < 0.0001:
            # print(tf.norm(new_UW[:,2*Nr*dk*K:]-UW[:,2*Nr*dk*K:],2))
            break
        U = new_U
        W = new_W
        X = new_X
    V_list = list()
    for user_index in range(K):
        Xk = X[:, :, :, user_index]
        Vk = tf.matmul(tf.transpose(H_ensemble, [0, 2, 1], conjugate=True), Xk)
        V_list.append(Vk)
    V = tf.stack(V_list, 3)
    #     V = tf.reshape(V,[-1,Nt,dk,K,1])
    #     V = tf.cast(tf.concat([tf.math.real(V),tf.math.imag(V)],axis=4),dtype=tf.float32)

    #     U = np.reshape(U,(-1,K*Nr*dk))
    #     X = np.reshape(X,(-1,K*Nr*dk*K))
    # #    print(W_list)

    #     W = np.triu(np.real(W))+np.tril(np.imag(W))
    #     W = np.reshape(W,(-1,K*dk*dk))
    return U, W, V, X


data_num = len(channels)
dataset = dataset[:data_num]
channels = channels[:data_num]
labelset = np.zeros((data_num, K * (2 * Nr * dk + dk * dk)))

beamform_dataset = np.zeros((data_num, K * (2 * dk * Nt )))
init_rate_list = []
final_rate_list = []

total_iter = len(channels) // 1000
# import ipdb;ipdb.set_trace()
EZF_performance = []
RWMMSE_performance = []
DUU_EZF_performance = []
for i in range(total_iter):
    print('iteration:' + str(i))
    channel_iter = channels[i * 1000:(i + 1) * 1000, :]
    channel_iter = np.concatenate([np.real(channel_iter), np.imag(channel_iter)], axis=-1)
    U_iter, W_iter, V_iter, X_iter = RWMMSE(channel_iter)
    q_list_iter = (tf.reshape(tf.norm(U_iter, axis=1), [-1, dk * K])) ** 2
    q_list_iter = tf.math.real(
        p * q_list_iter / tf.tile(tf.reshape((tf.reduce_sum(q_list_iter, axis=1)), [-1, 1]), (1, dk * K)))
    p_list_iter = tf.reshape(tf.norm(V_iter, axis=1), [-1, dk * K]) ** 2
    p_list_iter = tf.math.real(
        p * p_list_iter / tf.tile(tf.reshape((tf.reduce_sum(p_list_iter, axis=1)), [-1, 1]), (1, dk * K)))
    # import ipdb;ipdb.set_trace()
    best_factor_list = DUU_EZF_optimal_factor(channel_iter, p_list_iter, q_list_iter)
    DUU_EZF_output = DUU_EZF(channel_iter, p_list_iter, q_list_iter, best_factor_list)
    DUU_EZF_performance.append(np.mean(minus_sum_rate_loss(channel_iter, DUU_EZF_output)))
    # import ipdb;ipdb.set_trace()
    EZF_output = EZF(channel_iter)
    EZF_performance.append(np.mean(minus_sum_rate_loss(channel_iter, EZF_output)))
    V_iter = tf.reshape(V_iter, [-1, Nt, dk, K, 1])
    V_iter = tf.cast(tf.concat([tf.math.real(V_iter), tf.math.imag(V_iter)], axis=4), dtype=tf.float32)
    RWMMSE_performance.append(np.mean(minus_sum_rate_loss(channel_iter, V_iter)))
    V_iter = tf.reshape(V_iter,[1000,-1])
    # import ipdb;ipdb.set_trace()
    beamform_dataset[i * 1000:(i + 1) * 1000, :] = V_iter.numpy()

data_save_root = dataset_root + 'DUU_EZF_dataset_%d_%d_%d_%d_%d.mat' % (Nt, Nr, K, dk, SNR_dB)
io.savemat(data_save_root,
           {'V':beamform_dataset,
            'H': channels})

logger = logging.getLogger('mytest')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(dataset_root + 'DUU_EZF_dataset_%d_%d_%d_%d_%d.log' % (Nt, Nr, K, dk, SNR_dB))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('EZF sum rate:%.5f' % np.mean(EZF_performance))
logger.info('DUU EZF sum rate:%.5f' % np.mean(DUU_EZF_performance))
logger.info('RWMMSE sum rate:%.5f' % np.mean(RWMMSE_performance))

# python generate_DUU_EZF.py --Nt 64 --Nr 4 --dk 4 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 1000 --epoch 1000 --factor 2