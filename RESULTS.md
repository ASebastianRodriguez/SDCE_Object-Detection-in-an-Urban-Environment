# *Results of the project #1: Object Detection in an Urban Environmets*

## *1) Random images in that the trained algorithm detect "cars (red)", "pedestrian (blue)" and "cycles (green)":*

Sample pictures shows Exploratory Data Analysis (EDA) in that we can see the bounding boxes marking the differents detected objects.

<p>&nbsp;</p>

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/1.png "Ten pictures with bounding boxes")

<p>&nbsp;</p>

## *2) Histogram indicating Qty=f(classes):*

The graphics shows the quantity of different classes detected in the dataset.

<p>&nbsp;</p>

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/bar.png "Bar graphic indicating quantity of detectios per class")

<p>&nbsp;</p>

## *3) First training without augmentation:*
The curves shows the different courves included in TensorBoard without augmentation.

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/Scalars.jpg "Courves without augmentation")

<p>&nbsp;</p>

## *4) Second training with augmentation:*

### *4.1) Augmentation added to pipeline config file (pipeline_new.config):*
Inclusion of five (5) augmentation blocks with: conversion from rgb to gray scale, brightness adjusting (2 instances), contrast adjusting and color distortion.

<p>&nbsp;</p>

```
data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  data_augmentation_options {
    random_rgb_to_gray {
    probability: 0.2
    }
  }
  data_augmentation_options {
    random_adjust_brightness {
    max_delta: 0.2
    }
  }
  data_augmentation_options {
    random_adjust_contrast {
    min_delta: 0.7
    max_delta: 1.3
    }
  }
  data_augmentation_options {
    random_adjust_brightness {
    max_delta: 0.2
    }
  }
  data_augmentation_options {
    random_distort_color {
    color_ordering: 1
    }
  }
```

### *4.2) Train inputs image examples:*

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/Images.JPG "Train images examples")

<p>&nbsp;</p>

### *4.3) Execution of augmentation process:*

- First terminal: augmentation execution.<br />
- Second terminal: execution of chromium-browser.<br />
- Third terminal: execution of TensorBoard server.<br />

<p>&nbsp;</p>

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/Web%20Browser-Eval-TensorBoard.JPG "Terminals with augmentation process")

<p>&nbsp;</p>

### *4.4) Evaluation process losses:*

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/Eval_loss.JPG "Evaluation Losses")

<p>&nbsp;</p>

### *4.5) Comparison between Training Loss (without augmentation) and Evaluation Loss (with augmentation):*
In the first figure we can see that the value of Training Set Total Loss has an approximate value of 41.5 at 2.5k:

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/Train_Total%20Loss.JPG "Training Total Loss")

<p>&nbsp;</p>

In the first figure we can see that the value of Evaluation Set Total Loss has an approximate value of 40.0 at 2.5k:

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/Eval_Total_Loss.JPG "Evaluation Total Loss")

<p>&nbsp;</p>

*Conclusion about comparison:*
We can see that the Total Loss value, by adding a greater number of modified images using "augmentation" algorithms, decreases approximately 1.5 points. This is very important because adding images with modifications such as: changes in brightness, conversion of color to gray scale and others, a better training of the neural network is generated, resulting in an increase in its efficiency. when detecting "vehicles", "pedestrians" and "cycles". 

## *5) Animation GIF:*

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/Animation.png "Capture of animation GIF")
