import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'\..\..')

# import custom function
from EVIGP_utils.data_gen import data_generator_OTLcircuit as data_generator
from EVIGP_utils.gp_lnrho import GP_lnp_informative as GP_lnp
from EVIGP_utils.utils import rmspe_sd
from EVIGP_utils.utils import Cal_G_quadraticMean as Calc_G

# import measurement function
from EVIGP_utils.utils import handle_prediction

# basic imports from external library
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import time
import scipy as scipy
from scipy.stats import qmc
from scipy.stats import invgamma


# seed and plot setup
torch.set_default_dtype(torch.float64)
np.random.seed(0);
torch.manual_seed(0);

# args operation

import argparse

# GP params
parser = argparse.ArgumentParser(description='running examples for EVIGP, example_name arg is required and others are optional')
parser.add_argument('--N', type=int, help='training data size', default=200)
parser.add_argument('--N_test', type=int, help='testing data size', default=1000)
parser.add_argument('--epsilon',type=float, help='level of white noise', default=0.02)
# evi params
parser.add_argument('--N_particle', type=int, help='particle number used in EVI', default=100)
parser.add_argument('--TAU', type=float, help='step size used in EVI', default=0.1)
parser.add_argument('--h', type=float, help='kernel bandwidth used in EVI', default=0.001)
# cross-fold params
parser.add_argument('--fold_num', type=int, help='fold number used in cross validation', default=5)

args = parser.parse_args()

N = args.N
N_test = args.N_test
epsilon = args.epsilon
N_particle = args.N_particle
TAU = args.TAU
h = torch.tensor(1)*args.h
fold_num = args.fold_num

# parameters for priors
a_omega = np.ones((8,2))
a_omega[:,1] *=4.
a_1 = 1.
a_2 = 4

x,y = data_generator(N,epsilon)
G = Calc_G(x)
dimx = x.shape[-1]  # Dimension of x_i
Dim_particles = dimx + 1 

R_diag = torch.ones(G.shape[1])
R_diag[1:]      *=1/3
R_diag[dimx+1:] *=1/3
R_inv = torch.diag(1/R_diag)


def mainFunc(nu,foldnum):
    print('-----foldnum-'+str(foldnum+1)+'-----\n')

    theta = torch.rand(N_particle, Dim_particles)*0.1
    theta.requires_grad_()
    theta_0 = theta.detach().clone()

    def Loss(
        y: Tensor,
        x: Tensor,
        G: Tensor,
        theta: Tensor,
        theta_0: Tensor,
        a_1,
        a_2,
        TAU,
        h
    ):
        Dim_particle = theta.shape[-1]
        N_Particle   = theta.shape[0]
        
        Loss1 = torch.mean(torch.sum((theta - theta_0)**2, dim = 1, keepdim = True))/(2*TAU)

        diff = theta[:, None, :] - theta[None, :, :]
        kxy = torch.exp(-torch.sum(diff ** 2, axis=-1) / (2 * h **2)) / torch.pow(torch.pi * 2.0 * h * h, Dim_particle / 2)  # -1 last dimension
        sumkxy = torch.sum(kxy, axis=1)  # , keepdims=True)
        diffusion_control = 1
        Loss2 = torch.mean(torch.sum(diffusion_control*torch.log(sumkxy[:, None] / N_Particle)- GP_lnp(y, x, G, theta, a_1, a_2, a_omega, nu, R_inv)))

        return Loss1 + Loss2

    start = time.time()

    for epoch in range(101):
        optimizer = torch.optim.LBFGS( [theta],
                                history_size=50,
                                max_iter = 100,
                                line_search_fn= 'strong_wolfe')

        def closure():
            optimizer.zero_grad()
            loss = Loss(train_y, train_x, train_G, theta, theta_0, a_1, a_2, TAU, h)
            loss.backward()
            return loss

        optimizer.step(closure)
        parameterMovingDistance = torch.norm(theta-theta_0,p=2).item()
        if epoch % 10 == 0:
            print('theta_moved_distance '+str(parameterMovingDistance)+'\n')
            if parameterMovingDistance < 1e-5:
                break
        theta_0 = theta.detach().clone()

    end = time.time()

    sortedlnp, indices = torch.sort((GP_lnp(train_y, train_x, train_G, theta, a_1, a_2, a_omega, nu, R_inv)),dim=0)
    chosenTheta = theta[indices[-1],:]

    omega = chosenTheta[0, 1:]  ## 1 * dimx
    eta =   chosenTheta[0, 0 ]
    test_x, test_y = data_generator(N_test,epsilon=0,LHSseed=None)
    out1 = []
    for curr_x in test_x:
        curr_x = curr_x.view((1,dimx))
        m, _ = handle_prediction(curr_x, Calc_G(curr_x), Xn, train_y, train_G, omega, eta)
        out1.append(m[0][0].detach())
    pred_y = torch.tensor(out1).view((N_test,1))
    output_rmspe = rmspe_sd(pred_y,test_y)

    print('rmspe_sd:'+str(output_rmspe)+'\n')
    print('-----end of fold-'+str(foldnum+1)+'-----\n')
    return output_rmspe


fold_index_total = np.arange(N)
np.random.shuffle(fold_index_total)
fold_total = fold_num
fold_index_matrix = fold_index_total.reshape((int(N/fold_total),fold_total))

rmspe_result = np.zeros(fold_total)
nu_list = np.arange(0.05,5,0.05)
all_rmspe = np.zeros((fold_total,len(nu_list)))

for nu_index,nu in enumerate(nu_list):
    print('beginning of all fold for nu='+str(nu)+'\n')
    for fold_num in range(fold_total):
        test_index = fold_index_matrix[:,fold_num].ravel()
        train_index = np.array(list(set(fold_index_total)-set(test_index)))
        train_x,train_y,train_G = x[train_index,:], y[train_index], G[train_index,:]
        test_x,  test_y, test_G = x[ test_index,:], y[ test_index], G[ test_index,:]
        Xn = train_x
        train_N = len(train_index)
        test_N = len(test_index)

        this_rmspe = mainFunc(nu,fold_num)
        rmspe_result[fold_num] = this_rmspe
    print('this nu='+str(nu)+' has mean rmspe from all folds '+str(np.mean(rmspe_result))+'\n')

    print('end of all fold for nu='+str(nu)+'\n')
    all_rmspe[:,nu_index] = rmspe_result

np.save('RMSPE/Borehole/GPEVI-BH-QuadraticMean-informativePrior-mode-CVnu1.npy',all_rmspe)

all_rmspe = np.mean(all_rmspe,axis=0)

minimumIndex = np.argmin(all_rmspe)

print('the selected nu for borehole informative quadratic mean is', nu_list[minimumIndex])