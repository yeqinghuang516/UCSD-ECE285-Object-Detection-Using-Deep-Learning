import torch
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches



def Detect(img, model):
  transform = tv.transforms.Compose([tv.transforms.Resize(224), tv.transforms.ToTensor()])
  img = transform(img)
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  thre = 0.5 # confidence threshold
  class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
  model.eval()
  with torch.no_grad():
      prediction = model([img.to(device)])
  prediction = prediction[0]
  img = np.array(img)
  img = np.moveaxis(img, [0, 1, 2], [2, 0 ,1])
  
  plt.figure()
  fig, ax = plt.subplots(1, figsize = (12, 12))
  ax.imshow(img)
  
  
  # prediction items
  boxes = prediction['boxes']
  scores = prediction['scores']
  labels = prediction['labels']

  # Bounding-box colors
  cmap = plt.get_cmap("tab20b")
  colors = [cmap(i) for i in np.linspace(0, 1, 20)]

  #
  unique_labels = labels.cpu().unique()
  n_cls_preds = len(unique_labels)
  bbox_colors = random.sample(colors, n_cls_preds)

  for row in range(len(scores [scores > thre])):
    box = boxes[row, :].reshape(1, -1)
    box = box[0]
    x1 = box[0]
    y1 = box[1]
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    cls_pred = labels[row]
    cls_conf = scores[row]

    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor= color, facecolor="none")
    ax.add_patch(bbox)
    plt.text(x1, y1, fontsize = 14, s = class_names[int(cls_pred) - 1] + ' ' + ('%.4f' % cls_conf), color="white", verticalalignment="top", bbox={"color": color, "pad": 0})

  plt.show()
