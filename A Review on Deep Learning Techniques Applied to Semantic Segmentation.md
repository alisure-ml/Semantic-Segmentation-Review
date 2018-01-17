### 综述：深度学习技术在语义分割中的应用


### Abstract
* This paper provides a review on deep learning methods for semantic segmentation applied to various application areas.
* Firstly, we describe the terminology of this field as well as mandatory background concepts.
    * 首先，描述领域的术语以及背景概念。
* Next, the main datasets and challenges are exposed to help researchers decide which are the ones that 
best suit their needs and their targets.
    * 接下来，介绍主要的数据集和挑战，帮助研究者决定哪一个能够最好的满足他们的需要和目标。
* Then, existing methods are reviewed, highlighting their contributions and their significance in the field.
    * 然后，概述已有的方法，强调他们的贡献和再领域中的意义。
* Finally, quantitative results are given for the described methods and the datasets in which they were evaluated, 
following up with a discussion of the results.
    * 最终，衡量各个方法的结果和再同一数据集上的评估，接着对结果进行讨论。
* At last, we point out a set of promising future works and draw our own conclusions about 
the state of the art of semantic segmentation using deep learning techniques.
    * 最后，指出了有前途的工作，得出关于最先进的使用深度学习技术的语义分割技术的结论。


### Introduction

* The key contributions of our work are as follows:
    * provide a broad survey of existing datasers.
        * 提供了对现有数据集得广泛调查。
    * an in-depth and organized review of the most significant methods.
        * 对最重要的方法进行深入且有组织得审查。
    * a thorough performance evaluation which gathers quantitative metrics such as accuracy, execution time, and memory footprint.
        * 进行全面的性能评估，收集正确性、执行时间和内存占用等量化指标。
    * a discussion about the aforementioned results, as well as a list of possible future works.
        * 对前述结果进行讨论，同时也列举了一些可能的未来的工作。
    
* this paper is organized as follows:
    1. 介绍什么是语义分割问题，也介绍在文献文献追踪常见的符号和惯例，同时也介绍了深度神经网络等其它的背景概念。
    2. 描述了已有的数据库、挑战和基线。
    3. 基于已有方法的贡献，按照自底向上的复杂度介绍。重点在对理论的介绍和方法的强调，而不是评估性能。
    4. 基于在前面介绍的数据集上的结果，对已有的方法进行一个简短的讨论。
    5. 总结论文，得出结论。
    
    
### terminology and background concepts  术语和背景概念

* from coarse to fine inference:
    * Classification
    * Localization or detection
    * Semantic segmentation
    * Instance segmentation
    * Part-based segmentation
    
    ![from coarse to fine](readme/from_coarse_to_fine.png)

* background concepts
    * common networks
    * approaches
    * design decisions (that are often used as the basis for deep semantic segmentation systems)
    * common techniques for training such as transfer learning
    * data pre-processing and augmentation approaches

* Common Deep Network Architectures
    AlexNet,VGG-16,GoogleNet, and ResNet are currently being used as building blocks for many segmentation architectures.

* Transfer Learning
    * Training a deep neural network from scratch is often not feasible because of various reasons:
    a dataset of sufficient size is required (and not usually available) and 
    reaching convergence can take to long for the experiments to be worth.
        * 从零开始训练深度神经网络通常是不可行的，因为各种原因：
            * 需要足够大小的数据集（通常是不可用的）。
            * 需要很长的时间才能达到可用的收敛。
            
    * it is often helpful to start with pre-trained weights instead of random initialized ones.
        * 用预训练的权重而不是随机初始化通常时有帮助的。
    
    * Fine-tuning the weights of a pre-trained network is one of the major transfer learning scenarios.
        * 微调预训练权重
    
    * 网络的体系结构必须满足预训练网络
        * it is not usual to come up with a whole new architecture, it is common to reuse already 
        existing network architectures (or components) thus enabling transfer learning.
    
    * 从零开始训练和微调有不同的地方：
        * it is important to choose properly which layers to fine-tune.(选择合适的网络层来微调)
            * 通常选择网络的高级(higher-level)部分，因为低级(lower-level)部分趋向于获得更加抽象的特征。
        * pick an appropriate policy for the learning rate.(选择合适的学习率策略)
            * 通常选用小的学习率。
    
    * Due to the inherent difficult of gathering and creating per-pixel labelled segmentation datasets,
    their scale is not as large as the size of classification datasets such as ImageNet.
    For that reason, `transfer learning` and in particular `fine-tuning from pre-trained classification networks` 
    is a common trend for segmentation networks.
        
* Data Preprocessing and Augmentation
    * Data augmentation could `avoiding overfitting` and `increasing generalization capabilities`.
        * 数据增强可以避免过拟合和增强泛化能力。

    * It typically consist of applying a set of transformations in either data or feature spaces, or even both.
    通常包括一组在数据空间或者特征空间或者两者的变换。最常见的数据增强在数据空间。
    
    * 在数据空间的数据增强通常产生新的样本：通过在已存在的数据上应用变换。
     
    * There are many transformations that can be applied:
        * translation
        * rotation
        * warping
        * scaling
        * color space shifts
        * crops
        * etc
    

### Datasets and challenges

* 2D or plain RGB datasets, 2.5D or RGB-Depth(RGB-D) ones, and pure volumetric or 3D databases.

* 2D Datasets
    * PASCAL Visual Object Classes(VOC)
        * a group-truth annotated dataset of images and five different competitions:
            * classification, detection, segmentation, action classification, and person layout.
        * the dataset have 21 classes categorized.
    
    * PASCAL Context
        * 对 PASCAL VOC 2010 的扩展，包含10103张图片，用于检测挑战，总用540类，只有59类常用。
    
    * PASCAL Part
        * 对 PASCAL VOC 2010 的扩展，对每一个对象的每一个部分提供像素级掩码。
    
    * Semantic Boundaries Dataset (SBD)
        * 对 PASCAL VOC 的扩展，包含11355张图片，提供了category-level和instance-level的信息。
    
    * Microsoft Common Objects in Context (COCO)
        * image recognition, segmentation, and captioning large-scale dataset。
        * more than 80 classes, provides more than 82783 images for training, 40504 for validation, 
        and its test set consist of more than 800000 images.
    
    * SYNTHIA
        * a large-scale collection of photo-realistic renderings of a virtual city, semantically segmented, 
        whose purpose is scene understanding in the context of driving or urban scenarios.
        * the dataset provides fine-grained pixel-level annotations for 11 classes.
        * 13407 training images from rendered video streams.
    
    * Cityscapes
        * focuses on semantic understanding of urban street scenes.
    
    * CamVid
    
    * KITTI
    
    * Youtube-Objects
        
    * `Adobe’s Portrait Segmentation`
        * 800x600 pixels portrait images, captured with mobile front-facing cameras.
        * the database consist of 1500 training images and 300 reserved for testing, 
        both sets are fully binary annotated: person or background.
        * it suitable for person in foreground segmentation applications.
        
    * Materials in Context (MINC)
    
    * Densely-Annotated VIdeo Segmentation (DAVIS)
    
    * Stanford background
    
    * SiftFlow

* 2.5D Datasets
    not only RGB information but also depth maps.
    * ................................
    
* 3D Datasets
    * CAD meshes
    * point clouds


### Methods
    
1. Decoder Variants
    * SegNet
    
2. Integrating Context Knowledge
    * Conditional Random Fields
    * Dilated Convolutions
    * Multi-scale Prediction
        * Multi-scale convolutional architecture for semantic segmentation
            * two path: original resolution / doubles
        * A multi-scale cnn for affordance segmentation in rgb images
        * Multiscale fully convolutional network with application to industrial inspection
            * training each network independently and then the networks are combined and the last layer is fine-tuned.
    * Feature Fusion
        * FCN: skip connection
        * ParseNet: early fusion
        * SharpMask: refinement module
    * Recurrent Neural Networks
        * ReSeg
        * Long Short-Term Memorized Context Fusion
        * ...

3. Instance Segmentation
    * Its main purpose is to represent objects of `the same class split into different instances`. 
    most of the methods rely on existing object detectors.
    * `Instance labeling` provides us extra information for reasoning about `occlusion situations`, 
    also `counting the number of elements` belonging to the same class and for `detecting a particular object`.
        * 遮挡情况、同类计数、检测指定的对象
    * SDS: Simultaneous Detection and Segmentation
    * DeepMask: an object proposal approach based on a single ConvNet.
        * This model predicts a `segmentation mask for an input patch` and
         `the likelihood of this patch for containing an object`.
    * SharpMask: a novel architecture for `object instance segmentation` implementing a `top-down refinement process`.
        * the refinement merges spatially rich information from lower-level features 
        with high-level semantic cues encoded in upper layers.
    * MultiPath classifier
    
4. RGB-D Data
    * use depth information and not only photometric data.
    * depth data needs to be encoded with three channels at each pixel as if it was an RGB images.    
    * methods:
        1. input `depth images` to models designed for RGB data and improve in this way the performance 
        by `learning new features from structural information`.
        2. leverage a `multi-view approach` to improve existing `single-view works`.
            * 利用多视的方法改进存在的单视。

5. 3D Data

6. Video Sequences
    * clockwork FCN
    * 3DCNN / Convolutional 3D(C3D)


### Discussion
* Evaluation metrics
    1. execution time
    
    2. memory footprint
    
    3. accuracy
        Metrics are usually variations on pixel accuracy and IoU:
        * PA: Pixel Accuracy
        * MPA: Mean Pixel Accuracy
        * MIoU: Mean Intersection over Union
        * FWIoU: Frequency Weighted Intersection over Union

* Summary
    1. DeepLab is the most solid method.
    2. The 2.5D or multi-modal datasets are dominated by recurrent networks such as LSTM-CF.
    3. video sequences
    4. 3D convolutions
    
* Future Research Directions
    1. 3D datasets
    2. Sequence datasets
    3. Point cloud segmentation using Graph Convolutional Networks(GCNs)
    4. Context knowledge
    5. Real-time segmentation
    6. Memory
    7. Temporal coherency on sequences(序列上的时间一致性)
    8. Multi-view integration


### Conclusion

