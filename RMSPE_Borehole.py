import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

import os

data_constMean_mode_str = './RMSPE/Borehole/GPEVI-BH-ConstMean-nonInformativePrior-mode.npy'
data_constMean_mlegp_str = './RMSPE/Borehole/mleGP-BH-ConstMean.csv'
data_linearMean_mode_str = './RMSPE/Borehole/GPEVI-BH-LinearMean-nonInformativePrior-mode.npy'
data_linearMean_mlegp_str = './RMSPE/Borehole/mleGP-BH-LinearMean.csv'
data_quadraticMean_mode_str = './RMSPE/Borehole/GPEVI-BH-QuadraticMean-nonInformativePrior-mode.npy'
data_quadraticMean_mlegp_str = './RMSPE/Borehole/mleGP-BH-QuadraticMean.csv'

data_QuadraticMeanQuadraticMean_mode_str1 = './RMSPE/Borehole/GPEVI-BH-QuadraticMean-informativePrior-mode-nuSelected1.npy'
data_QuadraticMeanQuadraticMean_mode_str2 = './RMSPE/Borehole/GPEVI-BH-QuadraticMean-informativePrior-mode-nuSelected2.npy'

# file check, and rerun
if os.path.exists(data_constMean_mode_str):
    data_constMean_mode = np.load(data_constMean_mode_str)
else:
    os.system('python -m examples.Borehole.EVIGP_BH_noninformative_constMean_mode')

if os.path.exists(data_constMean_mlegp_str):
    data_constMean_mlegp = np.loadtxt(data_constMean_mlegp_str,skiprows=1)
else:
    os.system('rscript ./examples/Borehole/mleGP_BH_constMean.r')

if os.path.exists(data_linearMean_mode_str):
    data_linearMean_mode = np.load(data_linearMean_mode_str)
else:
    os.system('python -m examples.Borehole.EVIGP_BH_noninformative_linearMean_mode')
    data_linearMean_mode = np.load(data_linearMean_mode_str)

if os.path.exists(data_linearMean_mlegp_str):
    data_linearMean_mlegp = np.loadtxt(data_linearMean_mlegp_str,skiprows=1)
else:
    os.system('rscript ./examples/Borehole/mleGP_BH_linearMean.r')
    data_linearMean_mlegp = np.loadtxt(data_linearMean_mlegp_str,skiprows=1)

if os.path.exists(data_quadraticMean_mode_str):
    data_quadraticMean_mode = np.load(data_quadraticMean_mode_str)
else:
    os.system('python -m examples.Borehole.EVIGP_BH_noninformative_quadraticMean_mode')
    data_quadraticMean_mode = np.load(data_quadraticMean_mode_str)

if os.path.exists(data_quadraticMean_mlegp_str):
    data_quadraticMean_mlegp = np.loadtxt(data_quadraticMean_mlegp_str,skiprows=1)
else:
    os.system('rscript ./examples/Borehole/mleGP_BH_quadraticMean.r')
    data_quadraticMean_mlegp = np.loadtxt(data_quadraticMean_mlegp_str,skiprows=1)

if os.path.exists(data_QuadraticMeanQuadraticMean_mode_str1):
    data_QuadraticMeanQuadraticMean_mode_1 = np.load(data_quadraticMean_mode_str)
else:
    os.system('python -m examples.Borehole.EVIGP_BH_noninformative_quadraticMean_nuSelected1')
    data_QuadraticMeanQuadraticMean_mode_1 = np.load(data_quadraticMean_mode_str)
if os.path.exists(data_QuadraticMeanQuadraticMean_mode_str2):
    data_QuadraticMeanQuadraticMean_mode_2 = np.load(data_quadraticMean_mode_str)
else:
    os.system('python -m examples.Borehole.EVIGP_BH_noninformative_quadraticMean_nuSelected2')
    data_QuadraticMeanQuadraticMean_mode_2 = np.load(data_quadraticMean_mode_str)


data_combined = np.vstack([
    data_constMean_mode,
    data_constMean_mlegp,
    data_linearMean_mode,
    data_linearMean_mlegp,
    data_quadraticMean_mode,
    data_quadraticMean_mlegp,
    data_QuadraticMeanQuadraticMean_mode_1,
    data_QuadraticMeanQuadraticMean_mode_2,
])

method_list = [
    r'\underline{constant, using mode}',
    r'constant, mlegp',
    r'\underline{linear, using mode}',
    r'linear, mlegp',
    r'\underline{quadratic, using mode}',
    r'quadratic, mlegp',
    r'\underline{quadratic, selection}',
    r'\underline{quadratic, selection twice}'
]

our_method_index = [0,2,4,6,7]

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

plt.savefig("./figs/Borehole/BH-RMSPE-compare.pdf", format="pdf", bbox_inches="tight")
plt.show()