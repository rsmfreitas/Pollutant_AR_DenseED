#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:24:08 2022

@author: rodolfofreitas
"""

import h5py
import torch as th
import numpy as np


def load_data(hdf5_dir, args, kwargs, flag):
    if flag == 'train':
        n_data = args.n_train
        batch_size = args.batch_size
    elif flag == 'test':
        n_data = args.n_test
        batch_size = args.test_batch_size

    with h5py.File(hdf5_dir + "/input_mcs{}.hdf5".format(n_data), 'r') as f:
        x = f['dataset'][()]
    with h5py.File(hdf5_dir + "/output_mcs{}.hdf5".format(n_data), 'r') as f:
        y = f['dataset'][()]

    y = np.where(y>=0, y, 0.)
    

    y_var = np.sum((y - np.mean(y, 0)) ** 2)

    data = th.utils.data.TensorDataset(th.FloatTensor(x), th.FloatTensor(y))
    data_loader = th.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    n_out_pixels = len(data_loader.dataset) * data_loader.dataset[0][1].numel()

    print("total input data shape: {}".format(x.shape))
    print("total output data shape: {}".format(y.shape))

    return x, y, n_out_pixels, y_var, data_loader

def load_data_ar(hdf5_dir, args, kwargs, flag):
    if flag == 'train':
        n_data = args.n_train
        batch_size = args.batch_size
    elif flag == 'test':
        n_data = args.n_test
        batch_size = args.test_batch_size
        
    with h5py.File(hdf5_dir + "/input_mcs{}_ar.hdf5".format(n_data), 'r') as f:
        x = f['dataset'][()]
    with h5py.File(hdf5_dir + "/output_mcs{}_ar.hdf5".format(n_data), 'r') as f:
        y = f['dataset'][()]
        
    print("total input data shape: {}".format(x.shape))
    print("total output data shape: {}".format(y.shape))
    
    # Scaling the concentrations
    x[:,0,:,:] = (x[:,0,:,:] - x[:,0,:,:].min()) / (x[:,0,:,:].max() - x[:,0,:,:].min())
    y = (y - y.min()) / (y.max() - y.min())
   
    y_mean = np.mean(y, 0)
    y_var = np.sum((y - y_mean) ** 2)
    print('y_var: {}'.format(y_var))
    stats = {}
    stats['y_mean'] = y_mean
    stats['y_var'] = y_var
    
    data_ = th.utils.data.TensorDataset(th.FloatTensor(x),
                                                    th.FloatTensor(y))
    
    data_loader = th.utils.data.DataLoader(data_, 
                                           batch_size=batch_size,
                                           shuffle=True, **kwargs)
    

    return x, y, stats, data_loader