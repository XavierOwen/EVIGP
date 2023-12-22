import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import time
import scipy as scipy
from scipy.stats import qmc
from scipy.stats import invgamma

def GP_lnp_noninformative( # return log of eqn15
    y: Tensor,
    x: Tensor,
    G: Tensor,
    theta: Tensor, # eta and omegas
    a_eta_1,
    a_eta_2,
    a_omega # gamma prior for omegas. 
) -> Tensor:
    
    dimx       = x.shape[-1]    # dim for omega only
    p          = dimx+1         # dim for beta
    N_Particle = theta.shape[0] # particle number used in EVI method
    N          = y.shape[0]     # training data number

    df = 0

    Log_Post = torch.ones(N_Particle, 1) # initialize particles

    for i in range(N_Particle):
        omega = theta[i, 1:]       # 1 * dimx
        eta =   theta[i, 0 ]       # 1 * 1

        if eta < 0 or torch.min(omega) < 0:
            Log_Post[i, :] = - 1e10*torch.tensor([1])    ## To prevent eta and omega become negative
        else:
            diff = x[:, None, :] - x[None, :, :]    # N * N * dimx
            Kn = torch.exp(-torch.sum((diff ** 2) * omega, axis=-1)) # N * N
            In = torch.eye(N)
            # inverse_cov = torch.inverse(Kn + eta * In)
            Sig   = Kn + eta * In
            L     = torch.linalg.cholesky(Sig)
            L     = L.t()
            Sig_invOne = torch.linalg.solve(L,torch.linalg.solve(L.t(), G*eta))/eta
            Sig_invY = torch.linalg.solve(L,torch.linalg.solve(L.t(), y*eta))/eta
            inverse_cov = torch.linalg.solve(L,torch.linalg.solve(L.t(), In*eta))/eta
            # might consider improving the performance by using np.multi_dot
            sn2_term1 = y.t() @ Sig_invOne @ torch.inverse(G.t() @ Sig_invOne) @ G.t() @ Sig_invY
            sn2_term2 = y.t() @ Sig_invY
            sn2 = sn2_term1 + sn2_term2
            tau2 = (1 + sn2) / (df + N - p)
            
            Post_term1 = torch.pow(tau2, -1/2*(df + N - p))
            #Post_term2 = torch.pow(torch.det(G.t() @ inverse_cov @ G), -1/2)
            Post_term2 = torch.pow(torch.det(torch.inverse(G.t() @ Sig_invOne)),1/2)
            #Post_term3 = torch.pow(torch.det(Kn + eta * In), -1/2)
            Post_term3 = torch.pow(torch.det(inverse_cov), 1/2)
            #print('theta',theta)
            if Post_term1*Post_term2*Post_term3 == 0:
                Log_Post[i, :] = - 1e10*torch.tensor([1])
                continue

            #Post = Post_term1*Post_term2*Post_term3
            #print(torch.det(G.t() @ inverse_cov @ G))
            #print(torch.det(Kn + eta * In))
            #Log_Post[i, :] = torch.log(Post_term1) + torch.log(Post_term2) + torch.log(Post_term3) 
            Log_Post[i, :] = -(df + N - p)*torch.log(tau2)/2 - torch.logdet(G.t() @ Sig_invOne)/2 - torch.logdet(Sig)/2
            #print('logPost',Log_Post[i, :])
            Log_Post[i, :] +=     torch.distributions.gamma.Gamma(a_eta_1, a_eta_2)          .log_prob(eta)
            for j, omega_j in enumerate(omega):
                Log_Post[i, :] += torch.distributions.gamma.Gamma(a_omega[j,0], a_omega[j,1]).log_prob(omega_j)
    return Log_Post

def GP_lnp_informative( # return log of eqn15
    y: Tensor,
    x: Tensor,
    G: Tensor,
    theta: Tensor, # eta and omegas
    a_eta_1,
    a_eta_2,
    a_omega, # gamma prior for omegas. 
    nu,
    R_inv
) -> Tensor:
    
    dimx       = x.shape[-1]              # dim for omega only
    p          = int(dimx*(dimx+3)/2)+1   # dim for beta: 1+dim+dim+(dim)(dim-1)/2 for 1+x_i+x_i^2+\sum_{ij} x_ix_j
    N_Particle = theta.shape[0]           # particle number used in EVI method
    N          = y.shape[0]               # training data number

    df = 7 # df for inv_gamma distribution and others

    Log_Post = torch.ones(N_Particle, 1) # initialize particles
    diff = x[:, None, :] - x[None, :, :]    # N * N * dimx


    for i in range(N_Particle):
        omega = theta[i, 1:]       # 1 * dimx
        eta =   theta[i, 0 ]       # 1 * 1

        if eta < 0 or torch.min(omega) < 0:
            Log_Post[i, :] = - 1e10*torch.tensor([1])    ## To prevent eta and omega become negative
        else:
            Kn = torch.exp(-torch.sum((diff ** 2) * omega, axis=-1)) # N * N
            In = torch.eye(N)
            # inverse_cov = torch.inverse(Kn + eta * In)
            Sig   = Kn + eta * In
            L     = torch.linalg.cholesky(Sig)
            L     = L.t()
            Sig_invOne = torch.linalg.solve(L,torch.linalg.solve(L.t(), G*eta))/eta
            Sig_invY = torch.linalg.solve(L,torch.linalg.solve(L.t(), y*eta))/eta
            inverse_cov = torch.linalg.solve(L,torch.linalg.solve(L.t(), In*eta))/eta
            # might consider improving the performance by using np.multi_dot
            sn2_term1 = y.t() @ Sig_invOne @ torch.inverse(G.t() @ Sig_invOne) @ G.t() @ Sig_invY
            sn2_term2 = y.t() @ Sig_invY
            sn2 = sn2_term1 + sn2_term2
            tau2 = (1 + sn2) / (df + N - p)
            
            Sig_beta_n = torch.inverse(1/tau2 * G.t() @ Sig_invOne + 1/nu**2 * R_inv)
            hat_beta_n = Sig_beta_n @ (G.t() @ Sig_invY)/tau2
            
            Post_term1 = torch.pow(torch.det(Sig_beta_n),1/2)
            Post_term2 = torch.exp(-1/2 * hat_beta_n.t() @ torch.inverse(Sig_beta_n) @ hat_beta_n + 1/(2*tau2) * y.t() @ Sig_invY )
            Post_term3 = torch.pow(torch.det(inverse_cov), 1/2)
            Post_term4 = torch.pow(tau2, -N/2)

            #print('theta',theta)
            if Post_term1*Post_term2*Post_term3*Post_term4 == 0:
                Log_Post[i, :] = - 1e10*torch.tensor([1])
                continue

            Log_Post[i, :] = torch.logdet(Sig_beta_n)/2 - 1/2 * hat_beta_n.t() @ torch.inverse(Sig_beta_n) @ hat_beta_n - 1/(2*tau2) * y.t() @ Sig_invY
            Log_Post[i, :] = Log_Post[i, :] - torch.logdet(Sig)/2 - N/2 * torch.log(tau2)
            
            tau2_rv = invgamma(df, scale=1/2)
            prob_tau2 = tau2_rv.pdf(tau2.item())
            Log_Post[i, :] +=     np.log(prob_tau2)
            Log_Post[i, :] +=     torch.distributions.gamma.Gamma(a_eta_1, a_eta_2)          .log_prob(eta)
            for j, omega_j in enumerate(omega):
                Log_Post[i, :] += torch.distributions.gamma.Gamma(a_omega[j,0], a_omega[j,1]).log_prob(omega_j)
    return Log_Post