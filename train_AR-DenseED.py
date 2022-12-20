"""
Autoregressive Convolutional Encoder-Decoder Networks for Image-to-Image Regression

"""

from dense_ed import DenseED
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
import h5py
import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from time import time
from data_utils import load_data_ar
from data_plot import plot_pred_ar, plot_r2_rmse

import pandas as pd
from scipy import stats, integrate
#import seaborn as sns
plt.switch_backend('agg')

# Reproducibility
np.random.seed(0)
th.manual_seed(0)

# default to use cuda
parser = argparse.ArgumentParser(description='Dense Encoder-Decoder Convolutional Network')
parser.add_argument('--exp-name', type=str, default='AR-DenseED', help='experiment name')
parser.add_argument('--blocks', type=list, default=(5, 10, 5), help='list of number of layers in each block in decoding net')
parser.add_argument('--growth-rate', type=int, default=40, help='output of each conv')
parser.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
parser.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
parser.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
parser.add_argument('--outsize_even', action='store_true', default=True, help='if the output size is even or odd (e.g. 65 x 65 is odd, 64 x 64 is even)')
parser.add_argument('--upsample', type=str, default=None, help='How to upsampling in the decoder layer, choices: [None = Conv2DTranspose, linear = UPsampling(lienar), nearest = UPsampling(nearest)))')
parser.add_argument('--init-features', type=int, default=48, help='# initial features after the first conv layer')

parser.add_argument('--data-dir', type=str, default=".", help='data directory')
parser.add_argument('--n-train', type=int, default=1000, help="number of training data")
parser.add_argument('--n-test', type=int, default=500, help="number of test data")

parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.005, help='learnign rate')
parser.add_argument('--weight-decay', type=float, default=5e-5, help="weight decay")
parser.add_argument('--batch-size', type=int, default=200, help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=50, help='input batch size for testing (default: 100)')
parser.add_argument('--log-interval', type=int, default=5, help='how many epochs to wait before logging training status')
parser.add_argument('--plot-interval', type=int, default=50, help='how many epochs to wait before plotting training status')

args = parser.parse_args()
# Check if cuda is available
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

print('------------ Arguments -------------')
print("Torch device:{}".format(device))
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


all_over_again = 'experiments'

exp_dir = all_over_again + "/{}/Ntrs{}__Bks{}_Bts{}_Eps{}_wd{}_lr{}_K{}".\
    format(args.exp_name, args.n_train,args.blocks,
           args.batch_size, args.n_epochs, args.weight_decay, args.lr, args.growth_rate)

output_dir = exp_dir + "/predictions"
model_dir = exp_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

ntimes = 10  # ntimes is the total number of time instances considered

''' 
 %% Kwargs for Dataloader 
    num_workers (int, optional): how many subprocesses to use for data loading. 
                                 0 means that the data will be loaded in the main process. (default: 0)
    
    pin_memory (bool, optional): If True, the data loader will copy Tensors 
                                into device/CUDA pinned memory before returning them. 
                                If your data elements are a custom type, or your collate_fn 
                                returns a batch that is a custom type, see the example below.
'''
kwargs = {'num_workers': 0,'pin_memory': True} if th.cuda.is_available() else {}

# load training data
# N x Nc x H x W, N: Number of samples, Nc: Number of input/output channels, H x W: image size
hdf5_dir = args.data_dir + "/data_pollutant"#"/afs/crc.nd.edu/user/s/smo/invers_mt3d/raw2/lsx4_lsy2_var0.5_smax8_D1r0.1/kle{}_lhs{}".format(args.kle_terms,args.n_train)
x_train, y_train, train_stats, train_loader = load_data_ar(hdf5_dir, args, kwargs, 'train')

# load test data
hdf5_dir = args.data_dir + "/data_pollutant"
x_test, y_test, test_stats, test_loader = load_data_ar(hdf5_dir, args, kwargs, 'test')


model = DenseED(x_train.shape[1], 
                y_train.shape[1], 
                blocks=args.blocks, 
                growth_rate=args.growth_rate,
                drop_rate=args.drop_rate, 
                outsize_even=args.outsize_even,
                bn_size=args.bn_size,
                num_init_features=args.init_features, 
                bottleneck=args.bottleneck,
                upsample=args.upsample,
                out_activation=None).to(device)

print(model)
print("number of parameters: {}\nnumber of layers: {}"
              .format(*model._num_parameters_convlayers()))

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)

scheduler = ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-08)

n_out_pixels_train = len(train_loader.dataset) * train_loader.dataset[0][1].numel()
n_out_pixels_test = len(test_loader.dataset) * test_loader.dataset[0][1].numel()


# compute the quality metrics based on the test data and plot predictions (at specified epochs)
def test(epoch, plot_intv):
    model.eval()
    loss = 0.
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)

        with th.no_grad():
            output = model(input)
        loss += F.mse_loss(output, target,size_average=False).item()

        # plot predictions
        if epoch % plot_intv == 0 and batch_idx == len(test_loader) - 1:
            n_samples = 4
            idx = th.LongTensor(np.random.choice(args.n_test, n_samples, replace=False))

            print("Index of data: {}".format(idx))
            print("X shape: {}".format(x_test.shape))

            for i in range(n_samples):
                model.eval()
                x = x_test[ idx[i] * ntimes: (idx[i]+1) * ntimes]
                y = y_test[ idx[i] * ntimes: (idx[i]+1) * ntimes]

                y_output = np.full( (ntimes,y_test.shape[1],y_train.shape[2],y_train.shape[3]), 0.0)
                x_ii = np.full((1,x_test.shape[1],y_train.shape[2],y_train.shape[3]), 0.0)
                y_ii_1 = x[0,0,:,:]     # initial concentration
                for ii in range(ntimes):
                    x_ii[0,0,:,:] = y_ii_1        # the ii_th predicted output
                    x_ii[0,1,:,:] = x[ii,1,:,:]   # exponent coefficient
                    x_ii[0,2,:,:] = x[ii,2,:,:]   # velocity of advection 
                    x_ii_tensor = (th.FloatTensor(x_ii)).to(device)
                    with th.no_grad():
                        y_hat = model(x_ii_tensor)
                    y_hat = y_hat.data.cpu().numpy()
                    y_output[ii] = y_hat
                    y_ii_1 = y_hat[0,0,:,:]  # treat the current output as input to predict the ouput at next time step

                y_target = np.full( (ntimes,1,y_train.shape[2],y_train.shape[3]), 0.0)
                y_target[:ntimes] = y[:,[0]]  # the concentration fields for one input conductivity fields at ntimes time steps
                y_pred = np.full( (ntimes,1,y_train.shape[2],y_train.shape[3]), 0.0)
                y_pred[:ntimes] = y_output[:,[0]]
                #y_pred[ntimes] = y_output[[0],[0]]
                
                plot_pred_ar(i, y_target, y_pred, ntimes, epoch, idx, output_dir)

    rmse_test = np.sqrt(loss / n_out_pixels_test)
    r2_score = 1 - loss / test_stats['y_var']
    print("epoch: {}, test r2-score:  {:.4f}".format(epoch, r2_score))
    return r2_score, rmse_test

def cal_R2():
    "compute the test R2 score"
    n_test = args.n_test
    y_sum = np.full( (ntimes,1,y_train.shape[2],y_train.shape[3]), 0.0)

    for i in range(n_test):
        y = np.full( (ntimes,1,y_train.shape[2],y_train.shape[3]), 0.0)
        y[:ntimes] = y_test[i * ntimes: (i+1) * ntimes,[0]] # concentration at n_t time instances
        #y[ntimes] = y_test[ i * ntimes,[1] ] # head
        y_sum = y_sum + y
    y_mean = y_sum / n_test

    nominator = 0.0
    denominator = 0.0
    for i in range(n_test):
        x = x_test[ i * ntimes: (i+1) * ntimes]
        y = np.full( (ntimes,1,y_train.shape[2],y_train.shape[3]), 0.0)
        y[:ntimes] = y_test[i * ntimes: (i+1) * ntimes,[0]] # concentration at n_t time instances
        #y[ntimes] = y_test[ i * ntimes,[1] ] # head

        y_output = np.full( (ntimes, 1,y_train.shape[2],y_train.shape[3]), 0.0)
        x_ii = np.full((1,x_test.shape[1],y_train.shape[2],y_train.shape[3]), 0.0)
        y_ii_1 = x[0,0,:,:]     # initial concentration
        for ii in range(ntimes):
            x_ii[0,0,:,:] = y_ii_1        # the ii_th predicted output
            x_ii[0,1,:,:] = x[ii,1,:,:]   # exponent coefficient
            x_ii[0,2,:,:] = x[ii,2,:,:]   # velocity of advection
            x_ii_tensor = (th.FloatTensor(x_ii)).to(device)
            model.eval()
            with th.no_grad():
                y_hat = model(x_ii_tensor)
            y_hat = y_hat.data.cpu().numpy()
            y_output[ii,0] = y_hat[0,0]
            # if ii == ntimes - 1:
            #     y_output[ii+1,0] = y_hat[0,1]
            y_ii_1 = y_hat[0,0,:,:]
        nominator = nominator + ((y - y_output)**2).sum()
        denominator = denominator + ((y - y_mean)**2).sum()

    R2 = 1 - nominator/denominator
    print("R2: {}".format(R2))
    return R2

#find the maximum absolute prediction error at Nt concentration fields in each test sample,
#i.e., the results shown in Figure 13 of the paper
def max_err():
    n_test = args.n_test
    ErrMax = np.zeros((n_test))
    for i in range(n_test):
        x = x_test[ i * ntimes: (i+1) * ntimes]
        y = np.full( (ntimes,1,y_train.shape[2],y_train.shape[3]), 0.0)
        y[:ntimes] = y_test[i * ntimes: (i+1) * ntimes,[0]] # concentration at n_t time instances

        y_output = np.full( (ntimes, 1,y_train.shape[2],y_train.shape[3]), 0.0)
        x_ii = np.full((1,x_test.shape[1],y_train.shape[2],y_train.shape[3]), 0.0)
        y_ii_1 = x[0,0,:,:]     # initial concentration
        for ii in range(ntimes):
            x_ii[0,0,:,:] = y_ii_1        # the ii_th predicted output
            x_ii[0,1,:,:] = x[ii,1,:,:]   # exponent coefficient
            x_ii[0,2,:,:] = x[ii,2,:,:]   # velocity of advection
            x_ii_tensor = (th.FloatTensor(x_ii)).to(device)
            model.eval()
            with th.no_grad():
                y_hat = model(x_ii_tensor)
            y_hat = y_hat.data.cpu().numpy()
            y_output[ii,0] = y_hat[0,0]
            y_ii_1 = y_hat[0,0,:,:]
        err = np.abs(y - y_output)

        ErrMax[i] = ( ( err.max(axis=1) ).max(axis=1) ).max(axis=1)

    np.savetxt(exp_dir +'/TestErrMax_ntrain{}.dat'.format(args.n_train), ErrMax, fmt='%10.4f')   # use exponential notation
    return None


# # * * * Uncomment the following lines to test using pretrained model * * * # #
# print('start predicting...')
# # load model
# model.load_state_dict(th.load(model_dir + '/model_epoch{}.pth'.format(args.n_epochs)))
# print('Loaded model')
# test(200, 25)
# sys.exit(0)

# MAIN ==============
tic = time()
R2_test_self = []
r2_train, r2_test = [], []
rmse_train, rmse_test = [], []
for epoch in range(1, args.n_epochs + 1):
    # train
    model.train()
    mse = 0.
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target= input.to(device), target.to(device)
        model.zero_grad()
        output = model(input)

        loss = F.l1_loss(output, target,size_average=False)

        # for computing the RMSE criterion solely
        loss_mse = F.mse_loss(output, target,size_average=False)

        loss.backward()
        optimizer.step()
        mse += loss_mse.item()

    rmse = np.sqrt(mse / n_out_pixels_train)
    if epoch % args.log_interval == 0:
        r2_score = 1 - mse / train_stats['y_var']
        print("epoch: {}, training r2-score: {:.6f}".format(epoch, r2_score))
        r2_train.append(r2_score)
        rmse_train.append(rmse)
        r2_t, rmse_t = test(epoch, plot_intv=args.plot_interval)
        r2_test.append(r2_t)
        rmse_test.append(rmse_t)
        print("loss: {}".format(loss))

    scheduler.step(rmse)

    # save model
    if epoch == args.n_epochs:
        th.save(model.state_dict(), model_dir + "/model_epoch{}.pth".format(epoch))
tic2 = time()
print("Done training {} epochs with {} data using {} seconds"
      .format(args.n_epochs, args.n_train, tic2 - tic))

# plot the convergence of R2 and RMSE
plot_r2_rmse(r2_train, r2_test, rmse_train, rmse_test, exp_dir, args)


# save args and time taken
args_dict = {}
for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
args_dict['time'] = tic2 - tic
n_params, n_layers = model._num_parameters_convlayers()
args_dict['num_layers'] = n_layers
args_dict['num_params'] = n_params
with open(exp_dir + "/args.txt", 'w') as file:
    file.write(json.dumps(args_dict))

R2_test_s = cal_R2()
R2_test_self.append(R2_test_s)
np.savetxt(exp_dir + "/R2_test_self.txt", R2_test_self)

#max_err()
