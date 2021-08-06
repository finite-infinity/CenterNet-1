import torch
import numpy as np
import cv2
import os

path = './images_gray'
img_list  =os.listdir(path)
for img in img_list:
    img_path = os.path.join(path, img)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.dstack([img]*3)
    img = cv2.imwrite(img_path, img)