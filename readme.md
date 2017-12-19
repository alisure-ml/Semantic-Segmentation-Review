## A Review on Deep Learning Techniques Applied to Semantic Segmentation

### 路线
> 图像语义分割是一个从粗略到精细的过程

* 图像分类(图像中一个或多个实例分类)
* 实例定位
* 图像语义分割
* 实例分割(同一类中不同的实例分割)   

![from coarse to fine](readme/from_coarse_to_fine.png)


### FCN问题
* 缺少对不同特征的感知，阻碍了再具体问题和场景中的应用
* 由于固有的空间不变性，不能将全局的上下文信息考虑进去
* 不能感知实例
* 不能适应无结构的数据
* 需要大量的标签数据

### FCN改进方向
- 解码变种：对低分辨率的特征图的处理不同   
  * 编码器：卷积网络
  * 解码器：反卷积网络

- 整合上下文信息：整合不同空间尺度的信息，对局部信息和全局信息进行平衡   
  * 条件随机场：做为后期处理，组合低层次的像素级别的信息
  * 膨胀卷积：增大卷积核的步伐获得更宽的感受野（带孔卷积）
  * 多尺度聚合：
  * 特征融合：提取不同层的特征进行融合，包含了不同的局部上下文信息
  * 递归神经网络：

### Reference
* [A Review on Deep Learning Techniques Applied to Semantic Segmentation](paper/A%20Review%20on%20Deep%20Learning%20Techniques%20Applied%20to%20Semantic%20Segmentation.pdf)
    * [A Review on Deep Learning Techniques Applied to Semantic Segmentation](https://arxiv.org/pdf/1704.06857.pdf)
    * [深度学习图像分割的常用方法](http://blog.csdn.net/gqixf/article/details/78030203)
