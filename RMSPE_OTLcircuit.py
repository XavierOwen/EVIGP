import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

import os

data_constMean_mode_str = './RMSPE/OTLcircuit/GPEVI-OTL-ConstMean-nonInformativePrior-mode.npy'
data_constMean_mlegp_str = './RMSPE/OTLcircuit/mleGP-OTL-ConstMean.csv'
data_linearMean_mode_str = './RMSPE/OTLcircuit/GPEVI-OTL-LinearMean-nonInformativePrior-mode.npy'
data_linearMean_mlegp_str = './RMSPE/OTLcircuit/mleGP-OTL-LinearMean.csv'
data_quadraticMean_mlegp_str = './RMSPE/OTLcircuit/mleGP-OTL-QuadraticMean.csv'

data_QuadraticMeanQuadraticMean_mode_str1 = './RMSPE/OTLcircuit/GPEVI-OTL-QuadraticMean-informativePrior-mode-nuSelected1.npy'
data_QuadraticMeanQuadraticMean_mode_str2 = './RMSPE/OTLcircuit/GPEVI-OTL-QuadraticMean-informativePrior-mode-nuSelected2.npy'

# file check, and rerun
if os.path.exists(data_constMean_mode_str):
    data_constMean_mode = np.load(data_constMean_mode_str)
else:
    os.system('python -m examples.OTLcircuit.EVIGP_OTL_noninformative_constMean_mode')

if os.path.exists(data_constMean_mlegp_str):
    data_constMean_mlegp = np.loadtxt(data_constMean_mlegp_str,skiprows=1)
else:
    os.system('rscript ./examples/OTLcircuit/mleGP_OTL_constMean.r')

if os.path.exists(data_linearMean_mode_str):
    data_linearMean_mode = np.load(data_linearMean_mode_str)
else:
    os.system('python -m examples.OTLcircuit.EVIGP_OTL_noninformative_linearMean_mode')
    data_linearMean_mode = np.load(data_linearMean_mode_str)

if os.path.exists(data_linearMean_mlegp_str):
    data_linearMean_mlegp = np.loadtxt(data_linearMean_mlegp_str,skiprows=1)
else:
    os.system('rscript ./examples/OTLcircuit/mleGP_OTL_linearMean.r')
    data_linearMean_mlegp = np.loadtxt(data_linearMean_mlegp_str,skiprows=1)

if os.path.exists(data_quadraticMean_mlegp_str):
    data_quadraticMean_mlegp = np.loadtxt(data_quadraticMean_mlegp_str,skiprows=1)
else:
    os.system('rscript ./examples/OTLcircuit/mleGP_OTL_quadraticMean.r')
    data_quadraticMean_mlegp = np.loadtxt(data_quadraticMean_mlegp_str,skiprows=1)

if os.path.exists(data_QuadraticMeanQuadraticMean_mode_str1):
    data_QuadraticMeanQuadraticMean_mode_1 = np.load(data_QuadraticMeanQuadraticMean_mode_str1)
else:
    os.system('python -m examples.OTLcircuit.EVIGP_OTL_noninformative_quadraticMean_nuSelected1')
    data_QuadraticMeanQuadraticMean_mode_1 = np.load(data_QuadraticMeanQuadraticMean_mode_str1)
if os.path.exists(data_QuadraticMeanQuadraticMean_mode_str2):
    data_QuadraticMeanQuadraticMean_mode_2 = np.load(data_QuadraticMeanQuadraticMean_mode_str2)
else:
    os.system('python -m examples.OTLcircuit.EVIGP_OTL_noninformative_quadraticMean_nuSelected2')
    data_QuadraticMeanQuadraticMean_mode_2 = np.load(data_QuadraticMeanQuadraticMean_mode_str2)


data_combined = np.vstack([
    data_constMean_mode,
    data_constMean_mlegp,
    data_linearMean_mode,
    data_linearMean_mlegp,    
    data_QuadraticMeanQuadraticMean_mode_1,
    data_QuadraticMeanQuadraticMean_mode_2,
    data_quadraticMean_mlegp
])

method_list = [
    r'\underline{constant, EVI-MAP}',
    r'constant, mlegp',
    r'\underline{linear, EVI-MAP}',
    r'linear, mlegp',
    r'\underline{quadratic, informative}',
    r'\underline{quadratic, after selection}',
    r'quadratic, mlegp'
]

for i,j in enumerate(data_combined):
    print('method ',method_list[i],' has mean ', np.mean(j),' and standard deviation ', np.std(j))


our_method_index = [0,2,4,5]

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

plt.savefig("./figs/OTLcircuit/OTL-RMSPE-compare.pdf", format="pdf", bbox_inches="tight")