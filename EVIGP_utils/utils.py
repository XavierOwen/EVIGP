import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
import numpy as np

import matplotlib.pyplot as plt
import time
from itertools import combinations

import scipy as scipy
from scipy.stats import qmc

torch.set_default_dtype(torch.float64)

def Cal_G_constMean(
    x: Tensor,  # N * d
) -> Tensor:
    
    g =  torch.ones(x.shape[0], 1) # intercept column, and the rest columns
    return g   # N * p

def Cal_G_linearMean(
    x: Tensor,  # N * d
) -> Tensor:
    g = torch.cat([torch.ones(x.shape[0], 1),x],dim=-1)

    return g   # N * p
def Cal_G_quadraticMean(
    x: Tensor,  # N * d
) -> Tensor:
    g = torch.cat([torch.ones(x.shape[0], 1),x],dim=-1)
    dimx = x.shape[1]
    for i,j in combinations(range(0,dimx),2):
        g = torch.cat([g,x[:,[i]]*x[:,[j]]],dim=-1)
    g = torch.cat([g,x**2],dim=-1)
    return g   # N * int(dimx*(dimx+3)/2)+1

def Calc_y_xsinx(x, resize = True):
    '''
    return xsinx function value at x 
    '''
    y = x * torch.sin(x)
    return y

def Calc_y_borehole(x, resize = True):
    '''
    return borehole function value at x (in 8 dim)
    '''
    if resize:

        l_bounds = torch.tensor([0.05,100  ,63070 ,990 ,63.1,700,1120,9855])
        u_bounds = torch.tensor([0.15,50000,115600,1110,116 ,820,1680,12045])
        x = x *(u_bounds-l_bounds)+l_bounds

    r_w = x[:,0]
    r   = x[:,1]
    T_u = x[:,2]
    H_u = x[:,3]
    T_l = x[:,4]
    H_l = x[:,5]
    L   = x[:,6]
    K_w = x[:,7]
    numerator = 2*torch.pi*T_u*(H_u-H_l)
    denominator = torch.log(r/r_w)*(1+2*L*T_u/(torch.log(r/r_w)*r_w**2*K_w )+T_u/T_l)
    y = numerator/denominator
    y = y.view(y.shape[0],1)
    return y

def Calc_y_OTLcircuit(x, resize = True):
    '''
    return OTL circuit function value at x (in 6 dim)
    '''
    if resize:
        
        l_bounds = torch.tensor([50, 25,0.5,1.2,0.25,50])
        u_bounds = torch.tensor([150,70,3,  2.5,1.2,300])
        x = x *(u_bounds-l_bounds)+l_bounds

    Rb1 = x[:,0]
    Rb2 = x[:,1]
    Rf  = x[:,2]
    Rc1 = x[:,3]
    Rc2 = x[:,4]
    beta= x[:,5]
    Vb1 = 12*Rb2/(Rb1+Rb2)
    term1 = (Vb1+0.74)*beta*(Rc2+9)/(beta*(Rc2+9)+Rf)
    term2 = 11.35*Rf/(beta*(Rc2+9)+Rf)
    term3 = 0.74*Rf*beta*(Rc2+9)/(beta*(Rc2+9)+Rf)/Rc1
    y = term1 + term2 + term3
    y = y.view(y.shape[0],1)
    return y

def createMesh(meshSize, meshMin, meshMax):
    eta   = torch.linspace(meshMin[0], meshMax[0], meshSize)
    omega = torch.linspace(meshMin[1], meshMax[1], meshSize)

    grid_x, grid_y = torch.meshgrid(eta,omega, indexing='ij')
    theta_all = torch.cat([grid_x.reshape(meshSize*meshSize, 1), grid_y.reshape(meshSize*meshSize, 1)], 1)
    return grid_x, grid_y, theta_all


def handle_beta_noninformative(
    y: Tensor,
    x: Tensor,
    G: Tensor,
    omega: Tensor,
    eta: Tensor
):
    df = 0
    p = G.shape[-1]
    N = x.shape[0]

    diff = x[:, None, :] - x[None, :, :] 
    Kn    = torch.exp(-torch.sum((diff ** 2) * omega, axis=-1)) # N * N
    In    = torch.eye(N)
    Sig   = Kn + eta * In
    L     = torch.linalg.cholesky(Sig)
    L     = L.t()
    Sig_invG = torch.linalg.solve(L,torch.linalg.solve(L.t(), eta*G))/eta
    Sig_invY = torch.linalg.solve(L,torch.linalg.solve(L.t(), eta*y))/eta

    inv_term1 = G.t() @ Sig_invG
    L = torch.linalg.cholesky(inv_term1)
    L = L.t()
    term1 = torch.linalg.solve(L,torch.linalg.solve(L.t(), torch.eye(p)*eta))/eta

    sn2_term1 = y.t() @ Sig_invG @ term1 @ G.t() @ Sig_invY
    sn2_term2 = y.t() @ Sig_invY
    sn2 = sn2_term1 + sn2_term2
    df = 0
    tau2 = (1 + sn2) / (df + N - p)

    Sig_beta_n = tau2 * term1
    hat_beta_n = Sig_beta_n @ (G.t() @ Sig_invY)/tau2
    return hat_beta_n, Sig_beta_n

def handle_beta_informative(
    y: Tensor,
    x: Tensor,
    G: Tensor,
    omega: Tensor,
    eta: Tensor,
    nu,
    R_inv
) -> Tensor:
    df = 7
    p = G.shape[-1]
    N = x.shape[0]

    diff = x[:, None, :] - x[None, :, :]
    Kn    = torch.exp(-torch.sum((diff ** 2) * omega, axis=-1)) # N * N
    In    = torch.eye(N)
    Sig   = Kn + eta * In
    L     = torch.linalg.cholesky(Sig)
    L     = L.t()
    Sig_invG = torch.linalg.solve(L,torch.linalg.solve(L.t(), G*eta))/eta
    Sig_invY = torch.linalg.solve(L,torch.linalg.solve(L.t(), y*eta))/eta
    inverse_cov = torch.linalg.solve(L,torch.linalg.solve(L.t(), In*eta))/eta

    inv_term1 = G.t() @ Sig_invG
    L = torch.linalg.cholesky(inv_term1)
    L = L.t()
    term1 = torch.linalg.solve(L,torch.linalg.solve(L.t(), torch.eye(p)*eta))/eta

    sn2_term1 = y.t() @ Sig_invG @ term1 @ G.t() @ Sig_invY
    sn2_term2 = y.t() @ Sig_invY
    sn2 = sn2_term1 + sn2_term2
    df = 0
    tau2 = (1 + sn2) / (df + N - p)

    inv_Sig_beta_n = 1/tau2 * G.t() @ Sig_invG + 1/nu**2 * R_inv
    L = torch.linalg.cholesky(inv_Sig_beta_n)
    L     = L.t()
    Sig_beta_n = torch.linalg.solve(L,torch.linalg.solve(L.t(), torch.eye(p)*eta))/eta
    hat_beta_n = Sig_beta_n @ (G.t() @ Sig_invY)/tau2
    return hat_beta_n,Sig_beta_n

from typing import List
def handle_prediction(
    x0: Tensor, # new position to predict
    G0: Tensor,
    Xn: Tensor, # position used in train
    y: Tensor,
    G: Tensor,
    omega: Tensor,
    eta: Tensor
)->List:
  
    p = G.shape[-1]
    N = y.shape[0]
    g_x_t = G0

    diff1 = Xn - x0
    Kx_Xn = torch.exp(-torch.sum((diff1 ** 2) * omega, axis=-1)) # N * N # this is same in .r
    diff  = Xn[:, None, :] - Xn[None, :, :]
    Kn    = torch.exp(-torch.sum((diff ** 2)  * omega, axis=-1)) # this is correct same in .r

    In    = torch.eye(N)
    Sig   = Kn + eta * In
    L     = torch.linalg.cholesky(Sig)
    L     = L.t()
    Sig_invG = torch.linalg.solve(L,torch.linalg.solve(L.t(), eta*G))/eta 
    Sig_invY = torch.linalg.solve(L,torch.linalg.solve(L.t(), eta*y))/eta
    Sig_invKxX_t = torch.linalg.solve(L,torch.linalg.solve(L.t(), eta*Kx_Xn.t()))/eta
    mu_hat = ((g_x_t- Kx_Xn @ Sig_invG) @  torch.inverse(G.t() @ Sig_invG) @ G.t()+Kx_Xn) @ Sig_invY
    
    sn2_term1 = y.t() @ Sig_invG @ torch.inverse(G.t() @ Sig_invG) @ G.t() @ Sig_invY
    sn2_term2 = y.t() @ Sig_invY
    sn2 = sn2_term1 + sn2_term2
    df = 0
    tau2 = (1 + sn2) / (df + N - p)

    c = g_x_t.t() - G.t() @ Sig_invKxX_t
    c = c.t()
    tempMatrix = G.t() @ Sig_invG
    tempL      = torch.linalg.cholesky(tempMatrix)
    tempL      = tempL.t()
    Sig_inv_c  = torch.linalg.solve(tempL,torch.linalg.solve(tempL.t(), eta*c))/eta
    var        = tau2*(1-Kx_Xn @ Sig_invKxX_t  + c.t() @ Sig_inv_c )
    
    return mu_hat, var


def rmspe_sd(pred_y,true_y):
    N_test = true_y.shape[0]
    pred_y = pred_y.view((N_test,1))
    SSPE = torch.sum( (pred_y-true_y)**2)
    MSPE = SSPE/N_test
    RMSPE = torch.sqrt(MSPE)
    STD = torch.std(true_y)
    return (RMSPE/STD).item()