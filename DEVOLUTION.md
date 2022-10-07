In this project you have developed an Object detection based on SSD. Remember that single-stage architectures are preferred for autonomous vehicles because they provide a higher frame rate than other architectures (e.g. Dual-stage encoders such as R-CNN).
However, SSD is not the only single-stage encoder network. YOLO is another great single-stage network that is very popular in the automotive industry. For instance, you can see a comparison of both in the following paper from a couple of years ago. A relevant part of this paper is the comparison chart between YOLO and SSD, as shown below:

<p>&nbsp;</p>

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/Capture.JPG "Comparison between different neural networks")

<p>&nbsp;</p>

Thus, in general, SSD is a good approach to balance between FPS and mAP performance. Whereas, YOLO focuses more on FPS performance providing a better real-time method. You can find a good introduction to YOLO in the following [blog](https://gilberttanner.com/blog/yolo-object-detection-introduction/) or you can read the actual [paper: You Only Look Once(Unified, Real-Time Object Detection)](https://arxiv.org/pdf/1506.02640.pdf) that introduced the idea for the first time.
