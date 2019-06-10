## Object Detection Using Deep Learning

This is a course project for UCSD ECE285: Machine learning for image processing

In this project, we implement a representative multi-object detection algorithm: YOLOv3, and conduct a series of optimization and exploration to improve its performance, including image padding, data augmentation, multi-scale training, modifying loss function, modifying activation function and comparison with another object detection algorithm: Mask R-CNN. After those procedures, We provide a detailed analysis of their performance.

<img src= "assets\WhatWeDid.png" width = '500px' >

### About the code

All notebooks are highly suggested to run on Google Colab, simply select "Run All" and the notebook should be run  smoothly.



### Some Results

#### Model performance comparison at different stages

|                        | Precision |  Recall   |    mAP    |    f1     |
| :--------------------: | :-------: | :-------: | :-------: | :-------: |
|        Baseline        | **48.52** |   28.88   |   19.03   |   34.40   |
|        +Padding        |   47.93   |   36.25   |   23.69   |   39.60   |
|       +Data Aug        |   44.93   |   46.79   |   31.94   |   45.48   |
| +Multi-scale Training  |   43.32   |   49.28   |   34.57   |   45.88   |
| Modified Loss Function |   46.68   | **49.46** | **35.82** | **47.63** |

#### Precision-Recall curves after each procedure

<img src = "assets\pre-recall.png" width = "400px">

For more detailed results, please see the report



#### Successful Cases

|                                                              |                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src = "assets\s2.png" width = '400px'> | <img src = "assets\s3.png" width = '400px'> | <img src = "assets\s4.png" width = '400px'> |



### Code References

https://github.com/eriklindernoren/PyTorch-YOLOv3.git

<https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb>
