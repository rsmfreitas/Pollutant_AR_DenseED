#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 19:46:20 2022

@author: rodolfofreitas
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import dolfin as df 
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
import time
import scipy.io
import sys
import time
import os
import argparse

# Reproducibility
np.random.seed(0)

def freuddif(u,b,Kf,rho,phi):
    pw = u**(b-1)
    S0 = (Kf)*b*( pw )  
    Se = 1  + (rho/phi) * S0 
    return df.conditional(df.ge(u-1e-12, 0.0), Se ,1) 
   
def transport_sorb(u,v,u_n,uk,q,D,rho,phi,omega,dt,b,Kf):
    
    a = dt * omega * D * df.inner(df.grad(u), df.grad(v)) * df.dx \
        +  dt * omega * df.dot(q,df.grad(u)) * v * df.dx \
            + freuddif(uk,b,Kf,rho,phi) * u * v * df.dx
            
    L = freuddif(uk,b,Kf,rho,phi) * u_n * v * df.dx \
        + dt * (omega-1) * df.inner(D * df.grad(u_n), df.grad(v)) * df.dx \
        + dt * (omega-1) * df.dot(q, df.grad(u_n)) * v * df.dx 
    
    return a,L

def transport_1D(run, Lx, Ly, ngx_out, ngy_out, dt, vqx, vqy, T, alphaL, rho, phi , Dm, b, Kf, save_every, save_dir):
    """simulate 2D Transport' equation

    Args:
        run (int): # run
        b (float): exponent Freundlich 
        q (float): velocity field (u_x,u_y)
        D (float): longitudinal dispersivity 
        Kf (float): Freundlich constant
        Dm (float): molecular diffusion coefficient 
        rho(float): bulk density 
        alphaL(float): longitudinal dispersivity
        dt (float): time step for simulation
        vqx (float): velocity axis-x
        vqy (float): velocity axis-y
        T (float): simulation time from 0 to T
        ngx_out (int): output # grid in x axis
        ngy_out (int): output # grid in y axis
        save_dir (str): runs folder
        save_every (int): save frequency in terms of # dt
        
    """
    b   = float(b)
    vqx = float(vqx)
    # Folder to save the each sample
    save_dir = save_dir + f'/sample{run}'
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        
    mesh        = df.Mesh( "mesh/1D.xml")
    boundaries  = df.MeshFunction('size_t', mesh, 'mesh/1D_facet_region.xml')
    
    # mesh outputs
    mesh_out    = df.RectangleMesh(df.Point(0,0),df.Point(Lx,Ly),ngx_out-1,ngy_out-1)


    # this number is specify on geo mesh generation
    left        = 5

    V           = df.FunctionSpace ( mesh, "CG", 2 )
    Vout        = df.FunctionSpace ( mesh_out, "CG", 2 )
    ##################################### 
    # Boundary for concentration 
    bc_typec    = 0
    c_left      = 100
    c0          = 0.0
    
    if bc_typec==1: #flux# q = inputdat.v
        bc =[]   
    elif bc_typec==0: #when the boundary is specified used 
        c_leftc = df.DirichletBC(V, df.Constant(c_left), boundaries, left)
        bc = [c_leftc]
      
        
    u = df.TrialFunction(V)
    v = df.TestFunction(V)


    c_0 = df.Constant(c0)    
    u_n = df.interpolate(c_0 ,V)   
    uk  = df.interpolate(c_0 ,V)  


    omega = 0.5
    
    # convection velocity 
    vqy     = vqx  
    q       = df.Constant((vqx,vqy))
    
    D       = alphaL * vqx + Dm * phi 
    
    #call function
    [a,L] = transport_sorb(u,v,u_n,uk,q,D,rho,phi,omega,dt,b,Kf)
    
    u = df.Function(V) 
    
    # save the input data
    C0 = np.zeros((ngy_out, ngx_out))
    C0[:,0] = c_left 
    np.save(save_dir+f'/input{run}.npy',np.stack((C0, b*np.ones((ngy_out, ngx_out)),vqx*np.ones((ngy_out, ngx_out)))))
    
    t = 0
    i = 0
        
    # not much log
    df.set_log_level(30)
    tic = time.time()
    
    while t < T:
        
        t+=dt
        i+=1
        
        eps = 1.0           # error measure ||u-u_k||
        tol = 1.0E-6        # tolerance
        iter = 0            # iteration counter
        maxiter = 1000    
        while eps > tol and iter < maxiter:
            iter += 1 
            df.solve(a == L, u, bc)
            diff    = u.vector() - uk.vector()
            eps     = np.linalg.norm(diff)
            uk.assign(u) # update for next iteration
        
        u_n.assign(u)
        u_out = df.project(u, Vout)
        
        u_out.rename('u', 'u')
        if i % save_every == 0:
            u_out_ = u_out.compute_vertex_values(mesh_out).reshape(ngy_out, ngx_out)
            np.save(save_dir + f'/c{i}0.npy', u_out_)
            
        print(f'Run: solved {i} steps with total {time.time()-tic:.3f} seconds')
       
    
    return time.time() - tic


def pollutant_transport(samples, processes=12):
    
    ngx_out = 128
    ngy_out = 32
    #sorption parameters
    Kf      = 0.001
    b       = 0.6
    # dt should be small to ensure the stability
    dt      = 10
    T       = 150 # number of months
    alphaL  = 35 #longitudinal dispersivity
    phi     = 0.3 # porosity
    Dm      = 1e-9 # molecular diffusion coefficient 
    rho     = 1000 # bulk density
    Lx      = 100
    Ly      = 10
    vqy     = 1.
    # Generate data for uncertanty in b and vqx
    b       =  1.1 + 1.1 * 0.1 * (2. * np.random.rand(samples,1) - 1.)#0.6 + (1.2 - 0.6) * np.random.rand(samples,1)
    vqx     = 1.0#0.5 + (1.5 - 0.5) * np.random.rand(samples,1)
    
    # save every 10 mont
    save_dt    = dt
    save_every = int(save_dt / dt)
    save_dir = f'./fenics_data/{samples}_T150_b_11_ux_1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    
    pool = mp.Pool(processes=processes)
    print(f'Initialized pool with {processes} processes')
    results = [pool.apply_async(transport_1D, args=(run, Lx, Ly, ngx_out, ngy_out, dt, vqx, vqy, 
                T, alphaL,rho,phi,Dm, b[run], Kf, save_every, save_dir)) for run in range(samples)]
    time_taken = [p.get() for p in results]
    print(time_taken)
        
    
if __name__ == '__main__':

    print("Number of cpu : ", mp.cpu_count())

    parser = argparse.ArgumentParser(description='Sim 1D Transport equation')
    parser.add_argument('--samples', type=int, default=10000, help='number of samples (default: 1)')
    parser.add_argument('--processes', type=int, default=mp.cpu_count(), help='# processes (default: 12)')
    args = parser.parse_args()
    
    tic = time.time()
    pollutant_transport(args.samples, args.processes)
    print(f'Run: solved {args.samples} samples with total {(time.time()-tic)/60.:.3f} minutes')