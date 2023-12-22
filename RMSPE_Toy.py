import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

import os

data_constMean_mode_str = './RMSPE/Toy/GPEVI-XSinX-ConstMean-nonInformativePrior-mode.npy'
data_constMean_noDif_str = './RMSPE/Toy/GPEVI-XSinX-ConstMean-nonInformativePrior-noDif.npy'
data_constMean_gpfit_str = './RMSPE/Toy/GPfit-XSinX-ConstMean.csv'
data_constMean_laGP_str = './RMSPE/Toy/laGP-XSinX-ConstMean.csv'
data_constMean_mlegp_str = './RMSPE/Toy/mleGP-XSinX-ConstMean.csv'
data_linearMean_mode_str = './RMSPE/Toy/GPEVI-XSinX-LinearMean-nonInformativePrior-mode.npy'
data_linearMean_noDif_str = './RMSPE/Toy/GPEVI-XSinX-LinearMean-nonInformativePrior-noDif.npy'
data_linearMean_mlegp_str = './RMSPE/Toy/mleGP-XSinX-LinearMean.csv'

if os.path.exists(data_constMean_mode_str):
    data_constMean_mode = np.load(data_constMean_mode_str)
else:
    os.system('python -m examples.Toy.EVIGP_toy_noninformative_constMean_mode')
    data_constMean_mode = np.load(data_constMean_mode_str)

if os.path.exists(data_constMean_noDif_str):
    data_constMean_noDif = np.load(data_constMean_noDif_str)
else:
    os.system('python -m examples.Toy.EVIGP_toy_noninformative_constMean_noDiffusion')
    data_constMean_noDif = np.load(data_constMean_noDif_str)

if os.path.exists(data_constMean_gpfit_str):
    data_constMean_gpfit = np.loadtxt(data_constMean_gpfit_str,skiprows=1)
else:
    os.system('rscript ./examples/Toy/GPfit_toy_constMean.r')
    data_constMean_gpfit = np.loadtxt(data_constMean_gpfit_str,skiprows=1)

if os.path.exists(data_constMean_laGP_str):
    data_constMean_laGP = np.loadtxt(data_constMean_laGP_str,skiprows=1)
else:
    os.system('rscript ./examples/Toy/laGP_toy_constMean.r')
    data_constMean_laGP = np.loadtxt(data_constMean_laGP_str,skiprows=1)

if os.path.exists(data_constMean_mlegp_str):
    data_constMean_mlegp = np.loadtxt(data_constMean_mlegp_str,skiprows=1)
else:
    os.system('rscript ./examples/Toy/mleGP_toy_constMean.r')
    data_constMean_mlegp = np.loadtxt(data_constMean_mlegp_str,skiprows=1)

if os.path.exists(data_linearMean_mode_str):
    data_linearMean_mode = np.load(data_linearMean_mode_str)
else:
    os.system('python -m examples.Toy.EVIGP_toy_noninformative_linearMean_mode')
    data_linearMean_mode = np.load(data_linearMean_mode_str)

if os.path.exists(data_linearMean_noDif_str):
    data_linearMean_noDif = np.load(data_linearMean_noDif_str)
else:
    os.system('python -m examples.Toy.EVIGP_toy_noninformative_linearMean_noDiffusion')
    data_linearMean_noDif = np.load(data_linearMean_noDif_str)

if os.path.exists(data_linearMean_mlegp_str):
    data_linearMean_mlegp = np.loadtxt(data_linearMean_mlegp_str,skiprows=1)
else:
    os.system('rscript ./examples/Toy/mleGP_toy_linearMean.r')
    data_linearMean_mlegp = np.loadtxt(data_linearMean_mlegp_str,skiprows=1)


data_combined = np.vstack([
    data_constMean_mode,
    data_constMean_noDif,
    data_constMean_gpfit,
    data_constMean_laGP,
    data_constMean_mlegp,
    data_linearMean_mode,
    data_linearMean_noDif,
    data_constMean_mlegp,
])

method_list = [
    r'\underline{constant, using mode}',
    r'\underline{constant, no diffusion}',
    r'constant, gpfit',
    r'constant, laGP',
    r'constant, mlegp',
    r'\underline{linear, using mode}',
    r'\underline{linear, no diffusion}',
    r'linear, mlegp'
]

our_method_index = [0,1,5,6]

font = {'size'   : 28,'family':"Times New Roman"}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)

fig, ax1 = plt.subplots(figsize=(25,7),facecolor='white')

ax1.boxplot(data_combined.transpose(),medianprops={"linewidth": 2},
                whiskerprops={"linewidth": 2},
                capprops={ "linewidth": 2});
ax1.set_ylabel(r'RMSPE')
ax1.set_xticklabels(method_list, rotation=-45, ha="left");
ax1.grid()

#for i in our_method_index:
#    ax1.xaxis.get_ticklabels()[i].set_color('red')
#for i in ax1.get_xticklabels():
#    plt.setp(i, fontfamily="Times New Roman")
plt.savefig("./figs/Toy/XSinX-RMSPE-compare.pdf", format="pdf", bbox_inches="tight")
#plt.show()