# *Udacity Self Driving Car Engineer Course*

## **Project Nº 1: Object Detection in an Urban Environment**

### [Go To Results Directly!!!](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/RESULTS.md)

## **1) Project Introduction:**

In this project, you will apply the skills you have gained in this course to create a convolutional neural network to detect and classify objects using data from Waymo. You will be provided with a dataset of images of urban environments containing annotated cyclists, pedestrians and vehicles. First, you will perform an extensive data analysis including the computation of label distributions, display of sample images, and checking for object occlusions.

<p>&nbsp;</p>

An example night image from the Waymo dataset, with annotations for vehicles and pedestrians:

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/01_Grafico.png "An example night image from the Waymo dataset, with annotations for vehicles and pedestrians.")

<p>&nbsp;</p>

<div style="text-align: justify"> You will use this analysis to decide what augmentations are meaningful for this project. Then, you will train a neural network to detect and classify objects. You will monitor the training with TensorBoard and decide when to end it. Finally, you will experiment with different hyperparameters to improve your model's performance. This project will include use of the TensorFlow Object Detection API, where you can deploy you model to get predictions on images sent to the API. You will also be provided with code to create a short video of their model predictions. </div>

<p>&nbsp;</p>

### **1.1) Setting Up the Project:**
There are two options for the project: using the classroom workspace, with the necessary libraries and data already available for you, or local setup. If you want to use a local setup, you can use the below instructions for a Docker container if using your own local GPU, or otherwise creating a similar environment on a cloud provider's GPU instance.

<p>&nbsp;</p>

### **1.2) GPU Workspace Note:**
While you can shut off your GPU while writing code, note that anytime you need to run code in the workspace, you will want to have the GPU activated, as only then will all related Python libraries be available (such as TensorFlow). If a library appears to not be available, that is the first thing to check. However, it's important to note that while most files auto-save within a few seconds of you completing edits in the workspace, you must manually save Jupyter notebooks before switching on or off the GPU, or else you may lose that work. If you are using the classroom environment, you can skip to the next page - the rest of the instructions here are for local work.

<p>&nbsp;</p>

### **1.3) Project Files:**
First, obtain the project files from the related [Github repository](https://github.com/udacity/nd013-c1-vision-starter).
If you are unfamiliar with GitHub, Udacity has a brief [GitHub tutorial](https://www.udacity.com/blog/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed [free course on git and GitHub](https://www.udacity.com/course/version-control-with-git--ud123). To learn about README files and Markdown, Udacity provides a free course on READMEs as well.

<p>&nbsp;</p>

### **1.4) Docker Setup:**
For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the build directory of the starter code. The instructions below are also contained within the build directory of the starter code.

<p>&nbsp;</p>

### **1.5) Requirements:**
- NVIDIA GPU with the latest driver installed
- docker / nvidia-docker

<p>&nbsp;</p>

### **1.6) Build:**
Build the image with: <br />
`docker build -t project-dev -f Dockerfile .`<br />
Create a container with:<br />
`docker run --gpus all -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ --network=host -ti project-dev bash` and any other flag you find useful to your system (eg, `--shm-size`).

<p>&nbsp;</p>

### **1.7) Set up:**
Once in container, you will need to install gsutil, which you can easily do by running:<br />
`curl https://sdk.cloud.google.com | bash`

Once gsutil is installed and added to your path, you can auth using:<br />
`gcloud auth login`

<p>&nbsp;</p>

### **1.8) Debug:**
Follow this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation) if you run into any issue with the installation of the TF Object Detection API.

<p>&nbsp;</p>

## *2) Steps to Implement the Project:*

### <ins>2.1) Data:</ins>
For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).
[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

<p>&nbsp;</p>

### <ins>2.2) Structure:</ins>
#### *2.2.1) Data:*
The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
	- training_and_validation - contains 97 files to train and validate your models
    	- train: contain the train data (empty to start)
    	- val: contain the val data (empty to start)
    	- test - contains 3 files to test your model and create inference videos
```
The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.
You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

<p>&nbsp;</p>

#### *2.2.2) Experiments:*
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

<p>&nbsp;</p>

### <ins>2.3) Prerequisites:</ins>
#### *2.3.1) Local Setup:*
For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

<p>&nbsp;</p>

#### *2.3.2) Download and process the data:*
**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

<p>&nbsp;</p>

#### *2.3.3) Classroom Workspace:*
In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

<p>&nbsp;</p>

### <ins>2.4) Instructions:</ins>
#### *2.4.1) Exploratory Data Analysis:*
You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.
Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).

<p>&nbsp;</p>

#### *2.4.2) Create the training - validation splits:*
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
- create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
- split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

<p>&nbsp;</p>

#### *2.4.3) Edit the config file:*
Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

<p>&nbsp;</p>

#### *2.4.4) Training:*
You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
A training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
An evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

<p>&nbsp;</p>

#### *2.4.5) Improve the performances:*
Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
- experiment with the optimizer: type of optimizer, learning rate, scheduler etc
- experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.

<p>&nbsp;</p>

### <ins>2.5) Creating an animation:</ins>
#### *2.5.1) Export the trained model:*
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

<p>&nbsp;</p>

### <ins>2.6) Submission Template:</ins>
#### *2.6.1) Project overview:*
This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

<p>&nbsp;</p>

#### *2.6.2) Set up:*
This section should contain a brief description of the steps to follow to run the code for this repository.

<p>&nbsp;</p>

#### *2.6.3) Dataset:*
#### 2.6.3.1) Dataset analysis:
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.

<p>&nbsp;</p>

#### 2.6.3.2) Cross validation:
This section should detail the cross validation strategy and justify your approach.

<p>&nbsp;</p>

#### *2.6.4) Training:*
#### 2.6.4.1) Reference experiment:
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

<p>&nbsp;</p>

#### 2.6.4.2) Improve on the reference:
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

<p>&nbsp;</p>

## *3) Used Links:*
In the [USED_LINKS.md](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/USED_LINKS.md) file you can find the different sites from which I have obtained information to complete my project.

<p>&nbsp;</p>

## *4) Results:*

### [Please click here to go to results](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/RESULTS.md)

<p>&nbsp;</p>
