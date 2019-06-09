import torch
import torchvision as tv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN(torch.nn.Module):
  def __init__(self, num_classes, pretrained = True):
    super(MaskRCNN, self).__init__()
    self.model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained = pretrained)
    
    # get the number of input features for the classifier
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    
  def forward(self, images, targets = None):
    loss_dict = self.model(images, targets)
    return loss_dict