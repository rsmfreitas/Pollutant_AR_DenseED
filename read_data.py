#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:54:38 2022

@author: rodolfofreitas
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

def read_input(ndata, dx, ngx, ngy,load_dir,save_dir):
    x = np.full( (ndata,dx,ngx,ngy), 0.0)
    for idx in range(ndata):
        inp_ = np.load(load_dir+f'/sample{idx}/input{idx}.npy')
        x[idx,:, :, :] = inp_  # c is the first input channel, b is the second input channel, v_x is the third

    print("X: {}".format(x[0,]))
    hf = h5py.File(save_dir+f'/input_mcs{ndata}_T150.hdf5', 'w')
    hf.create_dataset('dataset', data = x, dtype ='f', compression = 'gzip')
    hf.close()
    
    return x

def read_input_output_ar(ndata, nt, q, ngx, ngy,load_dir,save_dir):
    inp_ = h5py.File(save_dir+f'/input_mcs{ndata}_T150.hdf5', 'r')['dataset'][()]
    out_ = h5py.File(save_dir+f'/output_mcs{ndata}_T150.hdf5', 'r')['dataset'][()]
    x = np.full( (ndata * (out_.shape[1]) , q, ngx, ngy), 0.0)
    y = np.full( (ndata * (out_.shape[1]) , 1, ngx, ngy), 0.0)
    
    for idx in range(ndata):
        for idt in range(out_.shape[1]):
            if idt == 0:
                x[idt + idx * (out_.shape[1]),:, :, :] = inp_[idx, :, :, :]  # c is the first input channel, b is the second input channel, v_x is the third
                y[idt + idx * (out_.shape[1]),-1, :, :] = out_[idx, idt, :, :]
            else:
                x[idt + idx * (out_.shape[1]),1:, :, :] = inp_[idx, 1:, :, :]
                x[idt + idx * (out_.shape[1]),0, :, :] = out_[idx, idt-1, :, :]
                y[idt + idx * (out_.shape[1]),:, :, :]  = out_[idx, idt, :, :]
            
    
    print("X: {}".format(x[0,]))
    hf = h5py.File(save_dir+f'/input_mcs{ndata}_ar_T150.hdf5', 'w')
    hf.create_dataset('dataset', data = x, dtype ='f', compression = 'gzip')
    hf.close()
    
    hf = h5py.File(save_dir+f'/output_mcs{ndata}_ar_T150.hdf5', 'w')
    hf.create_dataset('dataset', data = y, dtype ='f', compression = 'gzip')
    hf.close()
    
    return x, y
    
def read_output(ndata, dx, ngx, ngy,load_dir,save_dir):
    y = np.full( (ndata,dx,ngx,ngy), 0.0)
    for idx in range(ndata):
        k=0
        for idt in range(1,t_steps+1):
            out_ = np.load(load_dir+f'/sample{idx}/c{idt}0.npy')
            y[idx,k, :, :] = out_  # b is the first input channel, u_x is the second input channel
            k+=1

    print("Y: {}".format(y[0,]))
    hf = h5py.File(save_dir+f'/output_mcs{ndata}_T150_b_11_ux_1.hdf5', 'w')
    hf.create_dataset('dataset', data = y, dtype ='f', compression = 'gzip')
    hf.close()
    return y


samples = 10000
t_steps = 15
inp_    = 3
out_    = 15
q       = 3
nt      = 15
ngx     = 32
ngy     = 128
load_dir   = f'./fenics_data/{samples}_T150_b_11_ux_1' 
save_dir    = 'data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir) 
    



# read and write output data
#x = read_input(samples,inp_,ngx,ngy,load_dir,save_dir)
y = read_output(samples,out_,ngx,ngy,load_dir,save_dir)

y = (y - y.min()) / (y.max() - y.min()) 

c_25 = np.squeeze(y[:,:,16,32])
c_50 = np.squeeze(y[:,:,16,64])
c_75 = np.squeeze(y[:,:,16,96])

mu_25  = c_25.mean(axis=0)
std_25 = c_25.std(axis=0)
mu_50  = c_50.mean(axis=0)
std_50 = c_50.std(axis=0)
mu_75  = c_75.mean(axis=0)
std_75 = c_75.std(axis=0)

t =np.arange(10,151,10)

plt.figure()
plt.errorbar(t, mu_25, std_25, fmt='b^', capsize=2)
plt.errorbar(t, mu_50, std_50, fmt='rv', capsize=2)
plt.errorbar(t, mu_75, std_75, fmt='ks', capsize=2)

#x, y = read_input_output_ar(samples,nt,q,ngx,ngy,save_dir,save_dir)

# plt.figure(101,figsize=(10, 3), dpi=150)
# plt.imshow(y[0,0], origin='lower')
# cb=plt.colorbar(shrink=0.8, aspect=10, fraction=.2,pad=.025)
# cb.ax.tick_params(labelsize=16)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# plt.xlabel('x',fontsize=22)
# plt.ylabel('y',fontsize=22)
# plt.title(r'c [$\cdot$]',fontsize=22)