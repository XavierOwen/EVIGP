import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'\..\..')

# import custom function
from EVIGP_utils.data_gen import data_generator_borehole as data_generator
from EVIGP_utils.gp_lnrho import GP_lnp_noninformative as GP_lnp
from EVIGP_utils.utils import rmspe_sd
from EVIGP_utils.utils import Cal_G_linearMean as Calc_G

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

args = parser.parse_args()

N = args.N
N_test = args.N_test
epsilon = args.epsilon
N_particle = args.N_particle
TAU = args.TAU
h = torch.tensor(1)*args.h

# parameters for priors
a_omega = np.ones((8,2))
a_omega[:,1] *=4.
a_1 = 1.
a_2 = 4

# generate training data
x,y = data_generator(N,epsilon);
G = Calc_G(x)
dimx = x.shape[-1]  # Dimension of x_i

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
    Loss2 = torch.mean(torch.sum(diffusion_control*torch.log(sumkxy[:, None] / N_Particle)- GP_lnp(y, x, G, theta, a_1, a_2, a_omega)))
    
    return Loss1 + Loss2

Dim_particles = dimx + 1 # dimx for omega and 1 for eta

theta = torch.rand(N_particle, Dim_particles)*0.1
theta.requires_grad_()
theta_0 = theta.detach().clone()

start = time.time()

for epoch in range(501):
    optimizer = torch.optim.LBFGS(
        [theta],
        history_size=50,
        max_iter = 100,
        line_search_fn= 'strong_wolfe'
    )
    def closure():
        optimizer.zero_grad()
        loss = Loss(y, x, G, theta, theta_0, a_1, a_2, TAU,h)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    parameterMovingDistance = torch.norm(theta-theta_0,p=2).item()

    if epoch % 5 == 0:
        if parameterMovingDistance < 1e-8:
            break
    theta_0 = theta.detach().clone()

end = time.time()
trainingResult_str = 'training for Borehole function, noninformative prior, linear mean model ends at epoch {epoch} using {time:d} seconds'.format(
    epoch=epoch,
    time = int(end-start)
)
print(trainingResult_str)

sortedlnp, indices = torch.sort((GP_lnp(y, x, G, theta, a_1, a_2, a_omega)),dim=0)
chosenTheta = theta[indices[-1],:]

omega = chosenTheta[0, 1:]  ## 1 * dimx
eta =   chosenTheta[0, 0 ] 

# testing
Xn = x

N_retest = 100
Retest_result = np.zeros(N_retest)

for i in range(N_retest):
    test_x, test_y = data_generator(N_test,epsilon=0,LHSseed=None)
    out1 = []
    for curr_x in test_x:
        curr_x = curr_x.view((1,dimx))
        m, _ = handle_prediction(curr_x, Calc_G(curr_x), Xn, y, G, omega, eta)
        out1.append(m[0][0].detach())
    pred_y = torch.tensor(out1).view((N_test,1))
    current_test_result = rmspe_sd(pred_y,test_y)
    Retest_result[i]=current_test_result

np.save('RMSPE/Borehole/GPEVI-BH-LinearMean-nonInformativePrior-mode.npy',Retest_result)
print('mean RMSPE',np.mean(Retest_result))