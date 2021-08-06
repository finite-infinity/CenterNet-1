from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts_gatebee import opts
from detectors.detector_factory import detector_factory
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
      vid_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in video_ext:
              vid_names.append(os.path.join(opt.demo, file_name))
    else:
      vid_names = [opt.demo]
    
    for (vid_name) in vid_names:
      print('*'*50+'now infer the %s video'%(vid_name))
      vid = cv2.VideoCapture(vid_name)
      detector.pause = False
      video_FourCC = cv2.VideoWriter_fourcc(*'MP42')
      #video_FourCC = cv2.VideoWriter_fourcc(*'x264')
      video_fps       = vid.get(cv2.CAP_PROP_FPS)
      video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

      output_path = './output/'+ os.path.basename(vid_name)[:-4]+'.avi'
      out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)


      with open('./output/'+ os.path.basename(vid_name)[:-4]+'.txt', 'w') as f:    
        frameno = 0
        while True:
          return_value, frame = vid.read()
          if return_value != True:
              break
          if return_value:
              frameno = frameno + 1
              #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                         
          else:
              raise ValueError("No image!")     

          ret = detector.run(frame)

          out.write(ret['img'])
          
          title = ['*'*10,'the %d frame,bee count:%d'%(frameno, ret['beecount']),'*'*10]
          np.savetxt(f,title, fmt='%s', newline='\n')
          for j in range(1, len(ret['results']) + 1):
            np.savetxt(f, ret['results'][j], fmt='%f', delimiter=',', newline='\n')
          
          time_str = ''
          for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
          #print(time_str)
          if cv2.waitKey(1) == 27:
              break  # esc to quit
      vid.release()
      out.release()

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
