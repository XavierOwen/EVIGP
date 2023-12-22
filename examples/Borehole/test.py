import numpy as np

import sys
import os

a = np.array([1,2,3])
#np.save('RMSPE/Borehole/test.npy',a)
currentFolder = os.path.dirname(os.path.abspath(__file__))
print(currentFolder)
np.save(currentFolder+'\..\..'+'\RMSPE\Borehole\test2.npy',a)
