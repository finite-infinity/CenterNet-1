from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.ctdet_gray import CTDetDataset as CTDetDatasetGRAY
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP

from .dataset.coco_gatebee import COCO as COCOGatebee
from .dataset.coco_multibee import COCO as COCOMultibee
from .dataset.coco_multibee_withangle import COCO as COCOMultibee_WithANGLE
from .sample.ctdet_withangle import CTDetDataset as CTDetDataset_WithANGLE

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'coco_gatebee': COCOGatebee,
  'coco_multibee_withangle': COCOMultibee_WithANGLE
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset_WithANGLE,
  #'ctdet': CTDetDataset,
  #'ctdet': CTDetDataset_WithBackGround,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
