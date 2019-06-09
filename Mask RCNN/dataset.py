import torch.utils.data as td
import torchvision as tv
import skimage
import transforms as T
import os
import numpy as np

class VOCDataset(td.Dataset):

  def __init__(self, root, image_set, img_size = 224 , transforms = None):
    download = not os.path.isdir(root + '/VOCdevkit/')
    print('download = ', download)
    if image_set not in ['train', 'val', 'trainval']:
        raise ValueError('image_set can only be defined as "train", "val", or "trainval" !')
    self.dataset = tv.datasets.VOCSegmentation(root, year = '2012', image_set = image_set, download = download)
    self.transforms = transforms
    self.img_size = img_size
    
  def __getitem__(self, idx):
    img, mask = self.dataset[idx]
    
    img = tv.transforms.functional.resize(img, (self.img_size, self.img_size))
    mask = tv.transforms.functional.resize(mask, (self.img_size, self.img_size))
    
    mask = np.array(mask)
    mask[mask == 255] = 0
    mask_labeled = skimage.measure.label(mask)
    obj_ids = np.unique(mask_labeled)
    masks = mask_labeled == obj_ids[:, None, None]
    num_objs = len(obj_ids)

    boxes = []
    labels = []
    area = []

    for region in skimage.measure.regionprops(mask_labeled):
      y1,x1,y2,x2 = region.bbox
      centroid = region.centroid
      centroid = np.array(centroid).astype(int)
      label = mask[tuple(centroid)]
      labels.append(label)
      boxes.append([x1, y1, x2, y2])
      area.append(region.area)
      
    boxes = torch.as_tensor(boxes, dtype = torch.float32)
    labels = torch.as_tensor(labels, dtype = torch.int64).view(-1)
    masks = torch.as_tensor(masks, dtype = torch.uint8)
    area = torch.as_tensor(area, dtype = torch.float32)
    
    image_id = torch.tensor([idx])
    #suppose all instances are not crowd
    iscrowd = torch.zeros((num_objs, ), dtype = torch.int64)
    
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["masks"] = masks
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd
    
    if self.transforms is not None:
      img, target = self.transforms(img, target)
      
    return img, target
  
  def __len__(self):
    return len(self.dataset)

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
