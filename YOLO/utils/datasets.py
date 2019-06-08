import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision as tv


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def RandomHorizontalFilp(img, target, p = 0.5):
    if np.random.random() < p:
        img = tv.transforms.functional.hflip(img)
        target[:, 2] = 1 - target[:, 2]
    return img, target

class VOCDataset(torch.utils.data.Dataset):
  
  def __init__(self, root, image_set, img_size = 224, train = False, pad = True, multiscale = True):
    super(VOCDataset, self).__init__()

    download = not os.path.isdir(root + '/VOCdevkit/')
    print('download = ', download)
    if image_set not in ['train', 'val', 'trainval']:
        raise ValueError('image_set can only be defined as "train", "val", or "trainval" !')
    self.dataset = tv.datasets.VOCDetection(root, year = '2012', image_set = image_set, download = download)
    self.img_size = img_size
    self.batch_count = 0
    self.min_size = self.img_size - 2 * 32
    self.max_size = self.img_size + 2 * 32
    self.train = train
    self.pad = pad
    self.multiscale = multiscale
  
  def collate_fn(self, batch):
    imgs, targets = list(zip(*batch))
    # Remove empty placeholder targets
    targets = [boxes for boxes in targets if boxes is not None]
    # Add sample index to targets
    for i, boxes in enumerate(targets):
        boxes[:, 0] = i
    targets = torch.cat(targets, 0)
    
    if self.train and self.multiscale and self.batch_count % 10 == 0:
      self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
    self.batch_count += 1

    imgs = torch.stack([resize(img, self.img_size) for img in imgs])
    return imgs, targets
  
  
  def pad_to_square(self, img, pad_value):
    w, h = img.size
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    # pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0) #padding_left,padding_right, \text{padding\_top}, \text{padding\_bottom})padding_top,padding_bottom
    pad = (0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)   # left, top, right and bottom
    # Add padding
    img = tv.transforms.functional.pad(img, pad,fill = pad_value, padding_mode = 'constant' )
    
    return img, pad


  
  def transform(self, img):
    transform_list = [
    tv.transforms.Resize((self.img_size, self.img_size)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ]
    
    if self.train:
      transform_list.insert(0, tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
    
    transform = tv.transforms.Compose(transform_list)
    return transform(img)
  
  def target_transform (self, target, pad):
    
    object_to_idx = {'aeroplane': 0,'bicycle': 1,'bird': 2,'boat': 3,
              'bottle': 4,'bus': 5,'car': 6,'cat': 7,
              'chair': 8,'cow': 9,'diningtable': 10,'dog': 11,
              'horse': 12,'motorbike': 13,'person': 14,'pottedplant': 15,
              'sheep': 16,'sofa': 17,'train': 18,'tvmonitor': 19}

    if isinstance(target['annotation']['object'], list):
      boxes = [[object_to_idx[x['name']]] + [x['bndbox']['xmin'],x['bndbox']['ymin'],x['bndbox']['xmax'],x['bndbox']['ymax']] for x in target['annotation']['object']]
    elif isinstance(target['annotation']['object'], dict):
      x = target['annotation']['object']['bndbox']
      boxes = [object_to_idx[target['annotation']['object']['name']]] + [x['xmin'],x['ymin'],x['xmax'],x['ymax']]
    else:
        raise AssertionError('target is of unknonwn type !')

    boxes = np.array(boxes).astype(np.float)
    boxes = boxes.reshape(-1, 5)
    
    xmin = boxes[:, 1] + pad[0]
    ymin = boxes[:, 2] + pad[1]
    xmax = boxes[:, 3] + pad[2]
    ymax = boxes[:, 4] + pad[3]
        
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    box_w = (xmax - xmin) 
    box_h = (ymax - ymin) 
    boxes[:, 1] = x_center
    boxes[:, 2] = y_center
    boxes[:, 3] = box_w
    boxes[:, 4] = box_h
    targets = torch.zeros(len(boxes), 6)
    targets[:, 1:] = torch.Tensor(boxes)
    targets[targets != targets] = 0 # clear nan values

    return targets
  
  

  def __getitem__(self, idx):
    x, d = self.dataset[idx]
    x_numpy = np.array(x)
    w, h = x.size

    if self.pad:
        x, pad = self.pad_to_square(x, 0)
    else:
        pad = (0, 0, 0, 0)

    padded_w, padded_h = x.size
    d = self.target_transform(d, pad)
    d[:, 2] = d[:, 2] / padded_w
    d[:, 4] = d[:, 4] / padded_w
    d[:, 3] = d[:, 3] / padded_h
    d[:, 5] = d[:, 5] / padded_h
    
    if self.train:
      x, d = RandomHorizontalFilp(x, d)

    x = self.transform(x)
    
    return x, d
  
  def __len__(self):
    return len(self.dataset)

