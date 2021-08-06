from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import glob

from opts_multibee_withbackground import opts
from detectors.detector_factory_withbackground import detector_factory
import numpy as np

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'h264', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam':
    vid = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False

    while True:
        _, img = vid.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
                
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

    beecount_gt = get_beecount_gt()
    beecount_difflist = []
    beecount_gtlist = []
    beecount_prlist = []
    for (image_name) in image_names:
      with open('./output/'+ os.path.basename(image_name)[:-4]+'.txt', 'w') as f:   
        ret = detector.run(image_name)
        if os.path.basename(image_name)[:-4] in beecount_gt:
          beecount_diff = abs(ret['beecount'] - beecount_gt[os.path.basename(image_name)[:-4]])
          beecount_difflist.append(beecount_diff)
          beecount_gtlist.append(beecount_gt[os.path.basename(image_name)[:-4]])
          beecount_prlist.append(beecount_diff/beecount_gt[os.path.basename(image_name)[:-4]])

        title = ['*'*10,'bee count:%d'%(ret['beecount']),'*'*10]
        np.savetxt(f,title, fmt='%s', newline='\n')         
        for j in range(1, len(ret['results']) + 1):
          np.savetxt(f, ret['results'][j], fmt='%f', delimiter=',', newline='\n')
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    beecount_prarray = np.array(beecount_prlist)
    beecount_pr = beecount_prarray.mean()
    print('\n'+'*'*50+'the mean pr is :%f'%(beecount_pr))
    print('the beecount gt in each file is :')
    print(beecount_gtlist)
    print('the beecount diff is :')
    print(beecount_difflist)
    print('the beecount pr in each file is :')
    print(beecount_prlist)


def get_beecount_gt():
  txt_dir = '/home/alex/works/dataset/bee/multibee/test/txt'
  txt_list = glob.glob(txt_dir + "/*.txt")
  txt_list = np.sort(txt_list)
  beecount_gt = {}
  for txt in txt_list:
    beeinfo = np.loadtxt(txt,dtype=int,delimiter=',')
    beecount = 0
    for obj in beeinfo:       
      categoryid = int(obj[2])
      if categoryid == 0:
        beecount = beecount +1
    txtname = os.path.basename(txt)[:-4]
    beecount_gt[txtname] = beecount
  print(beecount_gt)
  return beecount_gt

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
