### BlitzNet: A Real-Time Deep Network for Scene Understanding

> BlitzNet：用于场景理解的实时深度网络


### Introduction

在计算机视觉中，对象检测和语义分割是场景理解的两个基本问题。对象检测的目标是检测图片中出现的预定义类别的所有对象和它们的定位边界框，语义分割的目标是解析图像并将类标签与每个像素相关联。

不难发现，良好的分割对目标的检测会有帮助，同时，目标的正确识别也对分割很有用。所以，有很强的动机来同时解决这两个任务，使两者互补。

论文的目标是利用在“全局对象级别”，“像素级别”或两者都有标注的图像数据，同时有效地解决目标检测和语义分割这两个问题。

由于论文将目标检测和语义分割结合来解决多任务场景理解问题，为此提出了称为BlitzNet的架构。BlitzNet可以产生实时且精确的分割和目标边界框。


### 架构的全局视图

![BlitzNet architecture](readme/BlitzNet_architecture.png)

首先，输入图像经过一个CNN模块来提取具有高级特征的特征图。该模块是一个特征提取器，可以是ResNet-50等模型。

然后，和SSD方法类似，为了在多尺度上高效地搜索边界框，特征图的分辨率依次减少（蓝色部分）。和FCN方法类似，为了预测精确的分割图，通过反卷积层将特征图放大（紫色部分）。这两部分由一系列与ResSkip Blocks交织的反卷积层组成。

最终，在输出的多个尺度的反卷积层上，通过简单的卷积层实现预测：一个用于目标检测（上面），一个用于语义分割（下面）。


### ResSkip Blocks

![BlitzNet ResSkip block](readme/BlitzNet_ResSkip_block.png)


### 非极大值抑制（Non-Maximum Suppression）


### 目标函数与训练


### 实验设置


### 结论


### 相关连接

[项目地址：http://thoth.inrialpes.fr/research/blitznet/](http://thoth.inrialpes.fr/research/blitznet/)

[论文地址：https://arxiv.org/abs/1708.02813](https://arxiv.org/abs/1708.02813)

[源码地址：https://github.com/dvornikita/blitznet](https://github.com/dvornikita/blitznet)
