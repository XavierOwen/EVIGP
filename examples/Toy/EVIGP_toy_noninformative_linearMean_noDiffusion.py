import sys
sys.path.append('../..') # to get the models

# import custom function
from utils.data_gen import data_generator_xsinx
from utils.gp_lnrho import GP_lnp_noninformative
from utils.utils import createMesh, Calc_y_xsinx, rmspe_sd
from utils.utils import Cal_G_linearMean as Calc_G

# import measurement function
from utils.utils import handle_prediction

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

font = {'size'   : 30}
matplotlib.rc('font', **font)
cbformat = ticker.ScalarFormatter()   # create the formatter
cbformat.set_powerlimits((-2,2))                 # set the limits for sci. not.

# args operation

import argparse

# GP params
parser = argparse.ArgumentParser(description='running examples for EVIGP, example_name arg is required and others are optional')
parser.add_argument('--N', type=int, help='training data size', default=11)
parser.add_argument('--N_test', type=int, help='testing data size', default=100)
parser.add_argument('--epsilon',type=float, help='level of white noise', default=0.5)
# evi params
parser.add_argument('--N_particle', type=int, help='particle number used in EVI', default=100)
parser.add_argument('--TAU', type=float, help='step size used in EVI', default=1)
parser.add_argument('--h', type=float, help='kernel bandwidth used in EVI', default=0.02)

args = parser.parse_args()

N = args.N
N_test = args.N_test
epsilon = args.epsilon
N_particle = args.N_particle
TAU = args.TAU
h = torch.tensor(1)*args.h

# parameters for priors
a_omega = np.ones((8,2))
a_omega[:,1]= .5
a_1 = 1.
a_2 = .5

# generate training data
x,y = data_generator_xsinx(N,epsilon);
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
    diffusion_control = 0
    Loss2 = torch.mean(torch.sum(diffusion_control*torch.log(sumkxy[:, None] / N_Particle)- GP_lnp_noninformative(y, x, G, theta, a_1, a_2, a_omega)))
    
    return Loss1 + Loss2

Dim_particles = dimx + 1 # dimx for omega and 1 for eta

theta = torch.rand(N_particle, Dim_particles)*.1
theta[:,1]*=3
theta[:,1]+=.1
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
trainingResult_str = 'training for toy example, noninformative prior, constant mean model ends at epoch {epoch} using {time:d} seconds'.format(
    epoch=epoch,
    time = int(end-start)
)
print(trainingResult_str)

sortedlnp, indices = torch.sort((GP_lnp_noninformative(y, x, G, theta, a_1, a_2, a_omega)),dim=0)
chosenTheta = theta[indices[-1],:]


# plotting
meshSize= 121
meshMin = [1e-10,.1]
meshMax = [.5,.6]

grid_x, grid_y, theta_all = createMesh(meshSize, meshMin, meshMax)
logPost_mesh = GP_lnp_noninformative(y, x, G, theta_all, a_1, a_2, a_omega).reshape(meshSize, meshSize) # background

plt.figure(figsize=(13,10),facecolor='white')
cnt = plt.contourf(grid_x, grid_y, torch.exp(logPost_mesh.detach()), 64)

for c in cnt.collections:
    c.set_edgecolor("face")

cbar = plt.colorbar(cnt, format=cbformat)
cbar.ax.yaxis.get_offset_text().set_fontsize(26)
cbar.ax.yaxis.set_offset_position('left') 

plt.scatter(theta[:, 0].detach().numpy(), theta[:, 1].detach().numpy(), c='black', marker='o', s=30, alpha=1.0)
plt.scatter(chosenTheta[:, 0].detach().numpy(), chosenTheta[:, 1].detach().numpy(), c='red',   marker='o', s=50, alpha=1.0)
plt.xlabel(r'$\eta$')
plt.ylabel('$\omega$    ', rotation=0)
plt.xlim(meshMin[0],meshMax[0])
plt.ylim(meshMin[1],meshMax[1])
plt.savefig("figs/Toy/XSinX-EVIGP-LinearMean-noDif-lnrhoWithPars.pdf", format="pdf", bbox_inches="tight")


# testing
test_x, test_y = data_generator_xsinx(N_test)
test_G = Calc_G(test_x)
Xn = x

sorted_test_x, sortIdx_test_x = torch.sort(test_x,dim=0)
test_x = test_x[sortIdx_test_x.ravel(),:]
test_y = test_y[sortIdx_test_x.ravel(),:]
test_G = test_G[sortIdx_test_x.ravel(),:]

omega = chosenTheta[0, 1:]  ## 1 * dimx
eta =   chosenTheta[0, 0 ] 

out1 = []
out2 = []

out1 = []
out2 = []
for curr_x in test_x:
    curr_x = curr_x.view((1,dimx))
    m, v = handle_prediction(curr_x, Calc_G(curr_x), Xn, y, G, omega, eta)
    out1.append(m[0][0].detach())
    out2.append(v[0][0].detach().item())
out2 = np.array(out2)

fig, ax1 = plt.subplots(figsize=(25,10),facecolor='white')

ax1.plot(test_x, test_x*np.sin(test_x), 'red', label=r'true: $x \sin(x)$',linewidth=5)
ax1.scatter(Xn, y ,c='black',label='train, with noise level 0.5',s=50)
ax1.plot(test_x, out1, 'blue',label='params that maximize lnp (modes of the particle)',linewidth=5);

ax1.fill_between(
    test_x.ravel(),
    out1 - 1.96 * np.sqrt(out2),
    out1 + 1.96 * np.sqrt(out2),
    alpha=0.3,
    label="95% confidence interval",
)

ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$',rotation=0)
plt.savefig("figs/Toy/XSinX-EVIGP-LinearMean-noDif-withVar.pdf", format="pdf", bbox_inches="tight")

pred_y = torch.tensor(out1).view((N_test,1))


N_retest = 100
Retest_result = np.zeros(N_retest)

for i in range(N_retest):
    test_x, test_y = data_generator_xsinx(N_test,epsilon=0,LHSseed=None)
    test_G = Calc_G(test_x)
    sorted_test_x, sortIdx_test_x = torch.sort(test_x,dim=0)
    test_x = test_x[sortIdx_test_x.ravel(),:]
    test_y = test_y[sortIdx_test_x.ravel(),:]
    test_G = test_G[sortIdx_test_x.ravel(),:]
    
    out1 = []
    for curr_x in test_x:
        curr_x = curr_x.view((1,dimx))
        m, _ = handle_prediction(curr_x, Calc_G(curr_x), Xn, y, G, omega, eta)
        out1.append(m[0][0].detach())
    pred_y = torch.tensor(out1).view((N_test,1))
    current_test_result = rmspe_sd(pred_y,Calc_y_xsinx(test_x))
    Retest_result[i]=current_test_result

np.save('RMSPE/Toy/GPEVI-XSinX-LinearMean-nonInformativePrior-noDif.npy',Retest_result)
print('mean RMSPE',np.mean(Retest_result))