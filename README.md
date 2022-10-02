# *Udacity Self Driving Car Engineer Course*

## **Project Nº 1: Object Detection in an Urban Environment**

### **Project Introduction:**

In this project, you will apply the skills you have gained in this course to create a convolutional neural network to detect and classify objects using data from Waymo. You will be provided with a dataset of images of urban environments containing annotated cyclists, pedestrians and vehicles. First, you will perform an extensive data analysis including the computation of label distributions, display of sample images, and checking for object occlusions.

<p>&nbsp;</p>

An example night image from the Waymo dataset, with annotations for vehicles and pedestrians:

![alt text](https://github.com/HomeBrain-ARG/SDCE_Object-Detection-in-an-Urban-Environment/blob/main/Graphics/01_Grafico.png "An example night image from the Waymo dataset, with annotations for vehicles and pedestrians.")

<p>&nbsp;</p>

<div style="text-align: justify"> You will use this analysis to decide what augmentations are meaningful for this project. Then, you will train a neural network to detect and classify objects. You will monitor the training with TensorBoard and decide when to end it. Finally, you will experiment with different hyperparameters to improve your model's performance. This project will include use of the TensorFlow Object Detection API, where you can deploy you model to get predictions on images sent to the API. You will also be provided with code to create a short video of their model predictions. </div>

<p>&nbsp;</p>

### **Setting Up the Project:**
There are two options for the project: using the classroom workspace, with the necessary libraries and data already available for you, or local setup. If you want to use a local setup, you can use the below instructions for a Docker container if using your own local GPU, or otherwise creating a similar environment on a cloud provider's GPU instance.

<p>&nbsp;</p>

### **GPU Workspace Note:**
While you can shut off your GPU while writing code, note that anytime you need to run code in the workspace, you will want to have the GPU activated, as only then will all related Python libraries be available (such as TensorFlow). If a library appears to not be available, that is the first thing to check. However, it's important to note that while most files auto-save within a few seconds of you completing edits in the workspace, you must manually save Jupyter notebooks before switching on or off the GPU, or else you may lose that work. If you are using the classroom environment, you can skip to the next page - the rest of the instructions here are for local work.

<p>&nbsp;</p>

### **Project Files:**
First, obtain the project files from the related [Github repository](https://github.com/udacity/nd013-c1-vision-starter).
If you are unfamiliar with GitHub, Udacity has a brief [GitHub tutorial](https://www.udacity.com/blog/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed [free course on git and GitHub](https://www.udacity.com/course/version-control-with-git--ud123). To learn about README files and Markdown, Udacity provides a free course on READMEs as well.

<p>&nbsp;</p>

### **Docker Setup:**
For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the build directory of the starter code. The instructions below are also contained within the build directory of the starter code.

<p>&nbsp;</p>

### **Requirements:**
- NVIDIA GPU with the latest driver installed
- docker / nvidia-docker

<p>&nbsp;</p>

### **Build:**
Build the image with: <br />

`docker build -t project-dev -f Dockerfile .`

<p>&nbsp;</p>

Create a container with:<br />
`docker run --gpus all -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ --network=host -ti project-dev bash` and any other flag you find useful to your system (eg, `--shm-size`).

<p>&nbsp;</p>

### **Set up:**
Once in container, you will need to install gsutil, which you can easily do by running:<br />
`curl https://sdk.cloud.google.com | bash`

<p>&nbsp;</p>

Once gsutil is installed and added to your path, you can auth using:<br />
`gcloud auth login`

### **Debug:**
Follow this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation) if you run into any issue with the installation of the TF Object Detection API.




<div style="text-align: justify">
</div>
