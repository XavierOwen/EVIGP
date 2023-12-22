import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
import numpy as np

import matplotlib.pyplot as plt
import time
import scipy as scipy
from scipy.stats import qmc


torch.set_default_dtype(torch.float64)

def data_generator_xsinx(N,epsilon=0,LHSseed=0):
    '''
    x in 1 dim, seed fixed in generating training data
    '''
    BHsampler = qmc.LatinHypercube(d=1,seed=LHSseed)
    l_bounds = [0.,]
    u_bounds = [10.,]
    BHsample_zeroOne = BHsampler.random(n=N)
    BHsample = qmc.scale(BHsample_zeroOne, l_bounds, u_bounds)
    
    with torch.no_grad():
        x = torch.from_numpy(BHsample[:,[0]])
        y = x*torch.sin(x)
        y = y.view((N,1))
        y += epsilon * torch.randn(N, 1)
    return x, y

def data_generator_borehole(N,epsilon,LHSseed=None):
    '''
    x in 8 dim, here are r_w, r, T_u, H_u, T_l, H_l, L, K_w
    reference webpage: https://www.sfu.ca/~ssurjano/borehole.html
    '''
    BHsampler = qmc.LatinHypercube(d=8,seed=LHSseed)
    l_bounds = [0.05,100  ,63070 ,990 ,63.1,700,1120,9855]
    u_bounds = [0.15,50000,115600,1110,116 ,820,1680,12045]
    BHsample_zeroOne = BHsampler.random(n=N)
    BHsample = qmc.scale(BHsample_zeroOne, l_bounds, u_bounds)

    with torch.no_grad():
        r_w = torch.from_numpy(BHsample[:,0])
        r   = torch.from_numpy(BHsample[:,1])
        T_u = torch.from_numpy(BHsample[:,2])
        H_u = torch.from_numpy(BHsample[:,3])
        T_l = torch.from_numpy(BHsample[:,4])
        H_l = torch.from_numpy(BHsample[:,5])
        L   = torch.from_numpy(BHsample[:,6])
        K_w = torch.from_numpy(BHsample[:,7])

        numerator = 2*torch.pi*T_u*(H_u-H_l)
        denominator = torch.log(r/r_w)*(1+2*L*T_u/(torch.log(r/r_w)*r_w**2*K_w )+T_u/T_l)
        x = torch.from_numpy(BHsample_zeroOne); # Input data # N * dimx
        y = numerator/denominator
        y = y.view((N,1))
        y += epsilon * torch.randn(N, 1)
    return x, y

def data_generator_OTLcircuit(N,epsilon,LHSseed=None):
    '''
    x in 6 dim, here are Rb1, Rb2, Rf, Rc1, Rc2, beta
    reference webpage: https://www.sfu.ca/~ssurjano/otlcircuit.html
    '''
    l_bounds = [50, 25,0.5,1.2,0.25,50]
    u_bounds = [150,70,3,  2.5,1.2,300]
    dimx = len(l_bounds)
    BHsampler = qmc.LatinHypercube(d=dimx,seed=LHSseed)
    
    BHsample_zeroOne = BHsampler.random(n=N)
    BHsample = qmc.scale(BHsample_zeroOne, l_bounds, u_bounds)
    
    with torch.no_grad():
        Rb1 = torch.from_numpy(BHsample[:,0])
        Rb2 = torch.from_numpy(BHsample[:,1])
        Rf  = torch.from_numpy(BHsample[:,2])
        Rc1 = torch.from_numpy(BHsample[:,3])
        Rc2 = torch.from_numpy(BHsample[:,4])
        beta= torch.from_numpy(BHsample[:,5])
        
        Vb1 = 12*Rb2/(Rb1+Rb2)
        term1 = (Vb1+0.74)*beta*(Rc2+9)/(beta*(Rc2+9)+Rf)
        term2 = 11.35*Rf/(beta*(Rc2+9)+Rf)
        term3 = 0.74*Rf*beta*(Rc2+9)/(beta*(Rc2+9)+Rf)/Rc1
        x = torch.from_numpy(BHsample_zeroOne); # Input data # N * dimx
        y = term1 + term2 + term3
        y = y.view((N,1))
        y +=  torch.normal(mean=0,std=epsilon,size=(N, 1))
    return x, y
