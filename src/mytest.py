from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import random
import time
from progress.bar import Bar
import torch


hm = np.zeros((2,3, 2, 2), dtype=np.float32)
hm = torch.from_numpy(hm)

hm[0][0] = 1
hm[0][1] = 2
hm[0][2] = 3
hm[1][0] = 6
hm[1][1] = 5
hm[1][2] = 4

print(hm)
hm_nokeep = (hm[:,0,:,:] != torch.max(hm,1)[0]).float()
print(hm_nokeep.size())
print(len(hm))
for i in range(len(hm)):
    hm[i] = hm[i] * hm_nokeep[i]
print(hm)
hm_nozero = hm[:,1:]
print(hm_nozero)

