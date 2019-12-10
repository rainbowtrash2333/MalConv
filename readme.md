# 前言

本文通过参考论文[《Malware Detection by Eating a Whole EXE》](https://arxiv.org/abs/1710.09435)，更具使用论文提出的MalConv模型，使用Tensorflow实现了恶意软件的分类。

原模型是侦测是否为恶意软件，本文通过修改模型（其实几乎没有修改）进而实现恶意软件的分类。

# 版本

v1：实现MalConv模型。

v2：在v1的基础上使用tensorboard实现可视化，并增加precision，recall， F1指数和显示每次分类的混淆矩阵。

# 数据集

本小马使用的数据集为[kaggle Microsoft Malware Classification Challenge](https://www.kaggle.com/c/malware-classification?tdsourcetag=s_pctim_aiomsg)。这个数据集共包含了九类恶意如软件。

# MalConv模型介绍

## 论文中的模型

论文第四节 “Model Architecture” 对MalConv模型进行说明，下面是论文中关于部分模型的摘要（如果我有闲心可能会翻译一下）：

> To best capture such high level location invariance, we choose to use a convolution network architecture. Combining the  convolutional activations with a global max pooling before going to fully connected layers allows our model to produce its activation regardless of the location of the detected features. Rather than perform convolutions on the raw byte values (i.e., using a scaled version of a byte’s value from 0 to 255), we use an embedding layer to map each byte to a fixed length (but learned) feature vector. We avoid the raw byte value as it implies an interpretation that certain byte values are intrinsically “closer” to each-other than other byte values, which we know a priori to be false, as byte value meaning is dependent on context. Training the embedding jointly with the convolution allows even our shallow network to activate for a wider breadth of input patterns. This also gives it a degree of robustness in the face of minor alterations in byte values. Prior work using byte n-grams lack this quality, as they are dependent on exact byte matches (Kolter and Maloof 2006; Raff et al. 2016).
>
> We note a number of difficult design choices that had to be made in developing a neural network architecture for such long input sequences. One of the primary limitations in practice was GPU memory consumption in the first convolution layer. Regardless of convolution size, storing the activations after the first convolution for forward propagation can easily lead to out-of-memory errors during back-propagation. We chose to use large convolutional filters and strides to control the memory used by activations in these early layers. 
>
> Attempts to build deep architectures on such long sequences requires aggressive pooling between layers for our data, which results in lopsided memory use. This makes model parallelism in frameworks like Tensorflow difficult to achieve. Instead we chose to create a shallow architecture with a large filter width of 500 bytes combined with an aggressive stride of 500. This allowed us to better balance computational workload in a data-parallel manner using PyTorch (Paszke, Gross, and Chintala 2016). Our convolutional architecture uses the gated convolution approach following Dauphin et al. (2016), with 128 filters.

下图是论文中MalConv模型的完整架构图：

![1](https://i.loli.net/2019/10/06/UlVkM6hBw2gGfFd.png)

## 本不知天高地厚的小马的理解

1. 由于需要 “吃“ 下恶意软件的所有byte，所以这个模型必然会非常消耗显存，自然也不可能处理太大的文件，所以我们选取小于2M的文件进行训练

2. 然后读取文件的byte流，将其转换成整数数组，如果长度不到2M，这在尾部添加0

3. 把数组进行8维embedding，得到E

4. 将E分为两个四维的A和B

5. 将A，B分别进行以为的卷积，其中`kernel size = 500`,`stride = 500`, `filters = 128`

6. 将B带入Sigmoid函数，其结果再与A相乘，得到G

7. G进行最大池化得到P

8. 进入全连接层

9. 最后进入softmax分类器输出结果



# 参考

[Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435)

[MalConv: Lessons learned from Deep Learning on executables](http://www.jsylvest.com/blog/2017/12/malconv/)

[恶意软件分类模型——MalConv的实现](https://www.wuuuudle.cn/2018/10/21/%E6%81%B6%E6%84%8F%E8%BD%AF%E4%BB%B6%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E2%80%94%E2%80%94MalConv%E7%9A%84%E5%AE%9E%E7%8E%B0/)





# 碎碎念

这是我《信息安全实训2》这门课的课业，由于本马没有学习过机器学习，不会tensorflow，完全是赶鸭子上架，一点点copy出来的，许多地方做的都不好，如果浪费您时间的话，深感抱歉。如果对您有帮助，我会非常高兴的。

最后，如果您感兴趣的话，可以到我的[博客](https://www.twikura.com/)看看，虽然都是写垃圾文。

