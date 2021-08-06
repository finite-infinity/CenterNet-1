from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug, random_contrast
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
from scipy.stats import norm

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i       

  # # 画椭圆
  # def ellipse_around_point(self, xc, yc, a, d, r1, r2):  #d = FR_D（图片大小）
  #   ind = np.zeros((2, d, d), dtype=np.int)  #坐标
  #   m = np.zeros((d, d), dtype=np.float32)
  #   for i in range(d):
  #       ind[0,:,i] = range(-yc, d-yc)
  #   for i in range(d):
  #       ind[1,i,:] = range(-xc, d-xc)  # 边长为1的（y,x）网格
  #   rs1 = np.arange(r1, 0, -float(r1)/r2)  # r1用r2步减到0
  #   rs2 = np.arange(r2, 0, -1.0)
  #   s = math.sin(a)
  #   c = math.cos(a)

  #   pdf0 = norm.pdf(0)
  #   for i in range(len(rs1)):
  #       i1 = rs1[i]
  #       i2 = rs2[i]
  #       v = norm.pdf(float(len(rs1) - i) / len(rs1)) / pdf0
  #   # 经旋转的椭圆 m（((-yc,d-yc)*s+ic)/rs[i],(-xc,d-xc)*c+is）...)=v 
  #       m[((ind[0,:,:] * s + ind[1,:,:] * c)**2 / i1**2 + (ind[1,:,:] * s - ind[0,:,:] * c)**2 / i2**2) <= 1] = v

  #   return m   #shape（d,d） 是一个椭圆 


  # def generate_segm_labels(self, img, pos, FR_D, w=10, r1=7, r2=12):
  #   res = np.zeros((4, FR_D, FR_D), dtype=np.float32) # data,labels_segm, labels_angle, weight
  #   res[0] = img  #data
  #   res[2] = -1

  #   for i in range(pos.shape[0]):
  #       x, y, obj_class, a = tuple(pos[i,:])
  #       obj_class += 1

  #       if obj_class == 2:   
  #           a = 2 * math.pi     # 细胞蜂没有角度
  #       else:
  #           a = math.radians(float(a))

  #       if obj_class == 1:   # 整蜂
  #           m = self.ellipse_around_point(x, y, a, FR_D, r1, r2)  
  #       else:                # 细胞蜂
  #           m = self.ellipse_around_point(x, y, a, FR_D, r1, r1)

  #       mask = (m != 0)
  #       res[1][mask] = obj_class  # 类别号
  #       res[2][mask] = a / (2*math.pi)  
  #       res[3][mask] = m[mask]

  #   res[3] = res[3]*(w - 1) + 1
  #   return res

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)
    # 此处转灰度图
    alpha_bound = np.arange(0.9, 1.2, 0.05)
    beta_bound = np.arange(0, 10)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.dstack([img]*3)
    # 随机对比度
    img = random_contrast(img, alpha_bound, beta_bound)
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:   
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:   
        # 长边缩放，随机裁剪，随机变换中心点的位置，变换的范围是图像的四边向内移动border距离。
        # border的取值是如果图像尺寸超过1024，border为512，否则为128.
        s = s * np.random.choice(np.arange(0.8, 1.3, 0.1))
        w_border = self._get_border(512, img.shape[1])
        h_border = self._get_border(512, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border) # 新中心
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True       #翻转
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)  # 改变对比度
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)      # 变为(b,w,h,c)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    angle = np.zeros((self.max_objs, 1), dtype=np.float32)
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      angle_ = ann['angle']
      angle_ = math.radians(angle_)/(2*math.pi)
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        angle_ = 1 - angle_
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))  # 高斯半径 
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        angle[k] = angle_
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0], ct[1], 
                       w, h, 1, cls_id, angle_])
    
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'angle': angle}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret