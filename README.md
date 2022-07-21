# Detecting the Multiple States of Oyster Activity and Orientation using Deep Learning Image Processing and Computer Vision Algorithms

## This research project was done at Salisbury University as part of the NSF REU Summer 2022 Program:
- [Salisbury University Website](https://www.salisbury.edu/)
- [NSF REU Salisbury Homepage](http://faculty.salisbury.edu/~ealu/REU/Schedule.html)

## Faculty Mentors:
### - Dr. Yuanwei Jin
### - Dr. Enyue (Annie) 

## Project Title:
*Image Processing and Computer Vision Algorithms for Sustainable Shellfish Farming*

## Research Participants:
### - Joshua Comfort
### - Ian Rudy

# References & Acknowledgements: 
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv5_OBB](https://github.com/hukaixuan19970627/yolov5_obb)
- [Arbitrary-Oriented Object Detection](https://arxiv.org/abs/2003.05597v2)
- [Arbitrary-Oriented Object Detection Transformer](https://arxiv.org/abs/2205.12785)
- [Detecting and Counting Oysters](https://arxiv.org/abs/2105.09758)
- [Fish Recognition Dataset](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/)
- [Turning any CNN image classifier into an object detector](https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/)
- [Underwater image processing](https://www.sciencedirect.com/science/article/pii/S0923596520302137)
- [Oyster detection system](https://github.com/bsadr/oyster-detection)
- [In Chesapeake Oysters’ Future: Underwater Drones, Shellfish Barges?](https://southernmarylandchronicle.com/2021/04/20/in-chesapeake-oysters-future-underwater-drones-shellfish-barges/)
- [PyTorch - YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/)
- [DenseDepth](https://github.com/ialhashim/DenseDepth)

## Methods:

### - Dataset
The oyster detection dataset can be found under the dataset directory, it was compiled using 200 initially provided images and 800 images collected from various internet sources. All of the annotation was done using the online annotation tool [Roboflow](https://roboflow.com/). 

The dataset contains images spanning various environments and camera angles to help increase the model's ability to generalize the features that make up an oyster. The oysters contained in the images are classified into one of three states oyster-closed, oyster-semi-open, and oyster-fully-open. Improvements to the dataset that would help improve model perfromance include a greater number of images, image processing to clean up the images, and a consistent metric to help label oyster states accurately (This project used metrics proposed in a project done by researchers at UMES).

The dataset was exported from Roboflow using the Oriented Bounding Box format, however, the exportation from roboflow had some issues and the exported txt files contained negative coordinate values. To remove the negative values from the dataset, a shell script was written and can be found under the tools directory. A guide on how to properly format your own dataset using roboflow and the script can be found under the getting started guide.

### - Training
The training for this research was done using [Google Colab](https://colab.research.google.com/), Colab provides remote access to high-performance computing on GPU instances. By leveraging Google's services we trained these models under various numbers of epochs, dataset iterations, and backbones. The achieved results are shown below in the section *Results*. 

### - Evaluation
The models were evaluated using these common metrics [precision](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall), [recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall), [average precision (AP)](https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_482), and [mean average precision (mAP)](https://www.v7labs.com/blog/mean-average-precision#:~:text=Average%20Precision%20is%20calculated%20as,mAP%20varies%20in%20different%20contexts.)

In addition, inference was run on sample videos and images which can be seen in the *Results* section.

### - Orientation
One complication that arose when try to detect oysters' activity, is that oysters are not always oriented ways that make classification feasible. To help remedy this, this project also sought to detect the orientation of oysters to allow for a more accurate classification of activity. Using the [YOLOv5_OBB](https://github.com/hukaixuan19970627/yolov5_obb) oysters were localized with rotated bounding boxes to more closely fit their contours. Additionally, color coded arrows were drawn parallel to the axis of orientation and the arrow length was coded to the ratio of length to width times the area of the box.

To achieve better results it might prove benefitial to use depth inference or a depth camera to retrieve depth information from the scene, allowing for three dimensional data to be used. 

Basic experimentation found good results obtaining depth from 2 dimensional images using [DenseDepth](https://github.com/ialhashim/DenseDepth), examples can be seen below.

<img src="./docs/DepthInference.jpg" width="550">

# Results

Below are some examples of inference run on various images, the classes are shown in difference colors, the first value is the predicted class, followed by the confidence value, the rotation value, and the number of degrees the oyster is rotated off axis.

## Run 1
<img src="./docs/InferenceTestImage9.jpg" width="450"> <img src="./docs/TestImg5.jpg" width="450">

## Run 2
<img src="./docs/InferenceTestImage1.jpg" width="450"> <img src="./docs/TestImg.jpg" width="450">

## Run 3
<img src="./docs/InferenceTestImage4.jpg" width="300"> <img src="./docs/Before1.jpg" width="300">

# Getting Started 

#  Acknowledgements
I have used utility functions from other wonderful open-source projects. Espeicially thank the authors of:

* [ultralytics/yolov5](https://github.com/ultralytics/yolov5).
* [Thinklab-SJTU/CSL_RetinaNet_Tensorflow](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow).
* [jbwang1997/OBBDetection](https://github.com/jbwang1997/OBBDetection)
* [CAPTAIN-WHU/DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
## More detailed explanation
想要了解相关实现的细节和原理可以看我的知乎文章:   
* [自己改建YOLOv5旋转目标的踩坑记录](https://www.zhihu.com/column/c_1358464959123390464).

## 有问题反馈
在使用中有任何问题，建议先按照[install.md](./docs/install.md)检查环境依赖项，再按照[GetStart.md](./docs/GetStart.md)检查使用流程是否正确，善用搜索引擎和github中的issue搜索框，可以极大程度上节省你的时间。

若遇到的是新问题，可以用以下联系方式跟我交流，为了提高沟通效率，请尽可能地提供相关信息以便我复现该问题。

* 知乎（@[略略略](https://www.zhihu.com/people/lue-lue-lue-3-92-86)）
* 代码问题提issues,其他问题请知乎上联系

## 关于作者

```javascript
  Name  : "胡凯旋"
  describe myself："咸鱼一枚"

