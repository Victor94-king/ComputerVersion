# 3D视觉感知之双目深度估计PSMNet: Pyramid Stereo Matching Network

论文地址: [[1803.08669] Pyramid Stereo Matching Network (arxiv.org)](https://arxiv.org/abs/1803.08669)

代码地址: [JiaRenChang/PSMNet: Pyramid Stereo Matching Network (CVPR2018) (github.com)](https://github.com/JiaRenChang/PSMNet)

Github链接 :[GIthub链接](https://github.com/Victor94-king/ComputerVersion)


<br />

## 1. 背景

3D感知任务相比于2D感知任务的情况更为复杂，而相比于单目相机双目相机的感知能力拥有以下几个特点:

* **优点**
  * 双目感知无需依赖强烈的先验知识和几何约束
  * 能够解决透视变化带来的歧义性(通俗的讲就是照片是由3D真实世界投影到2D图像然后再转换成3D，由于深度信息的丢失本身就十分困难)
  * 无需依赖物体检测的结果，对任意障碍物均有效
* **缺点**
  * 硬件: 摄像头需要精确配准，车辆运行过程中同样需要保持
  * 软件: 算法需要同时处理来自两个摄像头的数据，计算复杂度较高

<br />

<br />

而双目相机是如何实现3D视觉感知的呢？如下图：

B : 两个相机之间的距离

f : 相机的焦距

d: 视差(左右两张图象上同一个3d点之间的距离)

z: 物体相对于相机的深度，也是我们需要求解的值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a0d6c04c742242bf8cb016e7eec02f32.png#pic_center)

<br />

根据几何的知识我们可以得到 视差d 与 深度z是成反比的，所以双目相机的3D感知其实就是基于视差的估计来的，那么接下来核心来了,我们应该怎么得到每个像素点的视差呢？PSMNet 横空出世，它是一个利用端到端的卷积神经网络学习如何从输入的pair图像中获取每个像素点的视差
![在这里插入图片描述](https://img-blog.csdnimg.cn/3ae24a73615a43c8b983955b7563e13c.png#pic_center)
`<br />`

<br />

PSMNet 在原文中提到了以下几个亮点:

> * 端到端无需后处理的双目深度估计方法
> * 利用空间金字塔池化模块（SPP）和空洞卷积有效地整合全局上下文信息，从而提高了深度估计的准确性。
> * 采用三维卷积神经网络（3D CNN）stacked  hourglass 对cost map进行正则化处理，进一步提高了深度估计的精度。
> * 使用堆叠多个hourglass网络，并结合中间监督，进一步优化了3D CNN模块的性能。

<br />

<br />

---

## 2. 网络结构

![在这里插入图片描述](https://img-blog.csdnimg.cn/638a528f049c4a4eb4a77fdd78f58878.png#pic_center)

整体的网络结构如上图所示,首先网络的输入是成对的双目相机拍摄出来的左右片，通过一系列权重共享的特征提取网络提取特征，然后叠加构建costmap，然后经过一个3D卷积最后通过上采样恢复到原始输入大小的特征图即可。此文将网络结构分成4个模块然后会分别进行介绍

<br />

<br />

### 2.1 特征提取

第一个**CNN模块**，比较简单就是一个带残差模块和[空洞卷积](https://zhuanlan.zhihu.com/p/50369448)的卷积神经网络，将原始的H * W * 3 (kitti数据集里是375 X 1242 * 3)的图像下采样至 H/4 * W/4 * 128。所以整体的分辨率降低了4倍。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f0936f77004e41cba30c4ec105cd1c5e.png#pic_center)

<br />

第二个是**SPP模块**，这里可以看到下面有4个branch这里用的就是4个不同大小尺度的averagepooling，去收集不同分辨率下的局部的信息，然后通过双线性插值恢复到原始图像的1/4大小，然后与输出进SPP网络的原始输入进行拼接，这样SPP网络最后的输出就整合了全局(CNN网络的输出)以及局部(4个branch的pooling)的信息。

![在这里插入图片描述](https://img-blog.csdnimg.cn/7def92f628564bb9835f2864e1fe7153.png#pic_center)
PS： 这里查了查相关资料有关于[AveragePooling和MaxPooling的区别](https://blog.csdn.net/ytusdc/article/details/104415261)，主要来说如下:

* AveragePooling: 更有利于保留图像背景信息，一般用在
  * 当你需要用整合局部的信息的时候就用针对于像素级别的任务比如说分割任务
  * 下采样倍数较大的深层网络
  * 需要保留细节特征，利用全局的信息
* MaxPooling:更有利于保留图像纹理信息，相当于做了特征选择，选出了辨识度更好的特征，一般用在
  * 需要边缘的信息的任务，比如说分割类的任务

<br />

<br />

### 2.2 构建Cost Volume

在介绍构建Cost Volume之前，这里还需要估计引入一个概念就是视差的范围:前文提到计算深度就是**匹配视差其关键在于计算匹配误差**，即对于对于左视图的同一个物体我们只要找到其右视图的水平方向偏移的像素点，我们就可以知道其深度。因此接下来的几点是一个重点:
`<br />`

* [ ] 由于需要感知的深度范围有限，所以我们需要感知的视差的范围也是有限的(eg, 相机的深度范围是1 - 100m,对应的视差范围可能是1-10个pixel)因此对于视差我们就在可能的视差范围内搜寻值就可以了
* [ ] 对于每一个可能的视差（范围有限上一点提到的1-10个pixel），计算匹配误差，因此得到的三维的误差数据称为Cost Volume。
* [ ] 计算匹配误差时考虑像素点附近的局部区域(提取邻域的信息)，比如对局部区域内所有对应像素值的差进行求和
* [ ] 通过Cost Volume可以得到每个像素处的视差（对应最小匹配误差的𝑑𝑑），从而得到深度值。

<br />
<br />
<br />

好的有了以上的观点我们就可以继续回到PSMNet的costVolume 的构建了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/507b32eabd4841e8be8039a15e4fc365.png#pic_center)

由2.2部分的输出的左右图像分别大小分别是H' * W' * C(其实已经下采样到了1/4) 然后对D(程序中是192)个可能的视差范围将左右的特征图重合的部分拼接，不足的部分padding0，从而得到一个新的4维的特征张量H' * W' *D * 2C 。

**这里的含义可以看成拼接后的特征图同一个物体的cost比较小，不同的物体差异较大。所以就是计算左右特征图的在left[i:]与right[:-i]的相似度。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/32927c8cfbd14cc19ca5ed215187c4a9.png#pic_center)

<br />

**对应的代码如下:**

```
        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp//4): # 0 - 47 
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:] # LEFT 
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i] #RIGHT 
            else: # i == 0
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()
```

<br />

<br />

### 2.3 三维卷积

对于一个4D(H' * W' *D * 2C)的输入，作者采用了两种三维卷积的机制分别是basic 和Stacked hourglass，**且其作用均为对比左右特征图的在同一个位置的视差差异。**

<br />

#### 2.3.1 Basic

![在这里插入图片描述](https://img-blog.csdnimg.cn/d4d1d26ccf36433095586c50efec430d.png#pic_center)

这个结构就是多层跨链接的3D卷积，且卷积核为(3 * 3 * 3),可以看出其利用到了每个像素点周围邻域的信息(空间信息)也利用到了多个视差的信息，所以相比于只对比一个视差更加鲁棒。最后同一个线性插补上采样恢复原始分辨率，从而计算每个像素的深度值。下列是basic模块的代码实现

<br />

#### 2.3.2 Stacked hourglass

<br />

![在这里插入图片描述](https://img-blog.csdnimg.cn/9a433486f99645e5885274af1808eccc.png#pic_center)

这是作者提出一个相比于basic更加复杂的结构，是一个堆叠3次的hourglass结构，同样的这种hourglass的结构有3个好处:

* 能获取不同感受野的信息
* 利用skip连接可以在不同以及自身的结构内传递信息，更加鲁棒
* 与basic只有一个输出不同，stacked hourglass 在每个hourglass 结构都接了一个单独的输出，且在训练阶段加权求和得到总loss(具体权重参考第3部分)

```
        if self.training: # training 需要加上所有的cost 
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        pred3 = disparityregression(self.maxdisp)(pred3)
```

<br />

<br />

### 2.4 视差匹配

这里有两种做法:

* 硬分类: 直接取cost最小的视差值作为输出，但是这样有个缺点就是如果实际中最小值与第二小的值差别特别小，那么真实的视差应该处于二者之间，所以作者采用了一种软分类的机制。
* 软分类: 对网络输出的d个cost的值进行加权求和，权重就是输出的cost值，cost值越大权重越小。

![在这里插入图片描述](https://img-blog.csdnimg.cn/af6e651e2376483dbbb8931ef5b9ac25.png#pic_center)

```
class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True) # 加权求和
        return out
```

<br />

---

## 3. 损失函数

这里用的就是回归模型里比较常用的smoothL1，其是一种结合了L1和L2 的结合体，不会像L2对离群点敏感且容易梯度爆炸也不会像L1一样在0处不可导。

这里还要highligh一点就是之前提到的stacked hourglass操作里有三个预测头，在训练的时候这三个输出也对应着不同的loss，作者对其进行了调参结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/faeedcfc08414961ac54df556f64f40f.png#pic_center)

在源代码里体现如下:

```
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            # 三个loss 是加权平均 分别是 0.5 / 0.7 / 1.0
            loss =  0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + \
                    0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + \
                    F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
```

<br />

---

## 4. 效果

下图是原论文中截至于2018年3月，在kitti2015数据集上的效果，其中All 和 Noc 分别代表了所有像素点和未遮挡像素点的误差。 D1-bg / D1-fg / D1-all 分别代表的是背景/前景/所有点的误差百分比。可以看出效果还是很不错的，效率上由于引入3D卷积的操作时间上可能有待提供高。

![在这里插入图片描述](https://img-blog.csdnimg.cn/bd8602f7f9c8482fb449b31e3e94d5b8.png#pic_center)

<br />
<br />

---

## 5. 失败case与提升

![在这里插入图片描述](https://img-blog.csdnimg.cn/eb733b42ba7d45b5927286b3d78e9717.png#pic_center)

**原因 & 改进:**

**虽然考虑了邻域的信息但没用考虑高层的语义信息，无法理解场景 -> 用物体检测和语义分割的结果进行后处理，或者多个任务同时进行训练。或者增加注意力机制增加网络对纹理信息的理解提高深度的一致性能。**

<br />
<br />
![在这里插入图片描述](https://img-blog.csdnimg.cn/a8b9d21b560c442ead6a1363b1dfc312.png#pic_center)

**原因 & 改进:**

**远距离的视差值较小，在离散的图像像素上难以区分 -> 提高图像的空间分辨率，使得远距离物体也有较多的 像素覆盖；增加基线长度，从而增加视差的范围**

<br />
<br />

![在这里插入图片描述](https://img-blog.csdnimg.cn/96e6d44b2e3544cfa48d91f80d59b53e.png#pic_center)

**原因 & 改进:**

**低纹理或者低光照的区域内无法有效提取特征，用于计算匹配误差 -> 提高摄像头的动态范围，或者采用可以测距的传感器**

`<br /><br />``<br />`

---

**改进方向总结:**

* 针对3D卷积的 stacked hourglass 和 深层次的SPP结构 ， 会影响整体的效率 -> [一种基于 PSMNet 改进的立体匹配算法](https://zrb.bjb.scut.edu.cn/CN/abstract/abstract12982.shtml#:~:text=%E5%9C%A8%20PSMNet,%E7%AB%8B%E4%BD%93%E5%8C%B9%E9%85%8D%E7%BD%91%E7%BB%9C%E7%9A%84%E5%9F%BA%E7%A1%80%E4%B8%8A%E8%BF%9B%E8%A1%8C%E6%94%B9%E8%BF%9B%EF%BC%8C%E6%8F%90%E5%87%BA%E4%BA%86%E4%B8%80%E7%A7%8D%E5%85%B7%E5%A4%87%E6%B5%85%E5%B1%82%E7%BB%93%E6%9E%84%E4%B8%8E%E5%AE%BD%E9%98%94%E8%A7%86%E9%87%8E%E7%9A%84%E7%AB%8B%E4%BD%93%E5%8C%B9%E9%85%8D%E7%AE%97%E6%B3%95%E2%80%94%E2%80%94SWNet%E3%80%82%20%E6%B5%85%E5%B1%82%E7%BB%93%E6%9E%84%E8%A1%A8%E7%A4%BA%E7%BD%91%E7%BB%9C%E5%B1%82%E6%95%B0%E6%9B%B4%E5%B0%91%E3%80%81%E5%8F%82%E6%95%B0%E6%9B%B4%E5%B0%91%E3%80%81%E5%A4%84%E7%90%86%E9%80%9F%E5%BA%A6%E6%9B%B4%E5%BF%AB%3B%20%E5%AE%BD%E9%98%94%E8%A7%86%E9%87%8E%E5%88%99%E8%A1%A8%E7%A4%BA%E7%BD%91%E7%BB%9C%E7%9A%84%E6%84%9F%E5%8F%97%E9%87%8E%E6%9B%B4%E5%AE%BD%E5%B9%BF%EF%BC%8C%E8%83%BD%E5%A4%9F%E8%8E%B7%E5%8F%96%E5%B9%B6%E4%BF%9D%E7%95%99%E6%9B%B4%E5%A4%9A%E7%9A%84%E7%A9%BA%E9%97%B4%E4%BF%A1%E6%81%AF%E3%80%82)，作者提出一种浅层的ASPP和替代stacked hourglass的3个级联的残差结构 从而提高效率。
* 针对cost volume 的建立，只是直接concat，并没有考虑到相关性 -> [Group-wise Correlation Stereo Network](https://arxiv.org/abs/1903.04025) , 利用了相互之间的关系，将左特征和右特征沿着通道维度分成多组，在视差水平上对每组之间计算相关图，然后打包所有相关图以形成4D cost，这样一来，便可为后续的3D聚合网络提供更好的相似性度量，

<br />
<br /><br />
原作者给的代码里生成的深度图是灰度图，不利于肉眼对比效果，需要将灰度图转换成自设定的彩色图，对应的代码可以参考。

```
def disp_map(disp):
    map = np.array([
        [0, 0, 0, 114],
        [0, 0, 1, 185],
        [1, 0, 0, 114],
        [1, 0, 1, 174],
        [0, 1, 0, 114],
        [0, 1, 1, 185],
        [1, 1, 0, 114],
        [1, 1, 1, 0]
    ])
    # grab the last element of each column and convert into float type, e.g. 114 -> 114.0
    # the final result: [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    bins = map[0:map.shape[0] - 1, map.shape[1] - 1].astype(float)

    # reshape the bins from [7] into [7,1]
    bins = bins.reshape((bins.shape[0], 1))

    # accumulate element in bins, and get [114.0, 299.0, 413.0, 587.0, 701.0, 886.0, 1000.0]
    cbins = np.cumsum(bins)

    # divide the last element in cbins, e.g. 1000.0
    bins = bins / cbins[cbins.shape[0] - 1]

    # divide the last element of cbins, e.g. 1000.0, and reshape it, final shape [6,1]
    cbins = cbins[0:cbins.shape[0] - 1] / cbins[cbins.shape[0] - 1]
    cbins = cbins.reshape((cbins.shape[0], 1))

    # transpose disp array, and repeat disp 6 times in axis-0, 1 times in axis-1, final shape=[6, Height*Width]
    ind = np.tile(disp.T, (6, 1))
    tmp = np.tile(cbins, (1, disp.size))

    # get the number of disp's elements bigger than  each value in cbins, and sum up the 6 numbers
    b = (ind > tmp).astype(int)
    s = np.sum(b, axis=0)

    bins = 1 / bins

    # add an element 0 ahead of cbins, [0, cbins]
    t = cbins
    cbins = np.zeros((cbins.size + 1, 1))
    cbins[1:] = t

    # get the ratio and interpolate it
    disp = (disp - cbins[s]) * bins[s]
    disp = map[s, 0:3] * np.tile(1 - disp, (1, 3)) + map[s + 1, 0:3] * np.tile(disp, (1, 3))

    return disp


def disp_to_color(disp, max_disp=None):

    # grab the disp shape(Height, Width)
    h, w = disp.shape

    # if max_disp not provided, set as the max value in disp
    if max_disp is None:
        max_disp = np.max(disp)

    # scale the disp to [0,1] by max_disp
    disp = disp / max_disp

    # reshape the disparity to [Height*Width, 1]
    disp = disp.reshape((h * w, 1))

    # convert to color map, with shape [Height*Width, 3]
    disp = disp_map(disp)

    # convert to RGB-mode
    disp = disp.reshape((h, w, 3))
    disp = disp * 255.0

    return disp


def tensor_to_color(disp_tensor, max_disp=192):
    """
    The main target is to convert the tensor to image format
      so that we can load it into tensor-board.add_image()
    Args:
        disp_tensor (Tensor): disparity map
            in (BatchSize, Channel, Height, Width) or (BatchSize, Height, Width) layout
        max_disp (int): the max disparity value
    Returns:
        tensor_color (numpy.array): the converted disparity color map
            in (3, Height, Width) layout, value range [0,1]
    """
    if disp_tensor.ndimension() == 4:
        disp_tensor = disp_tensor[0, 0, :, :].detach().cpu()
    elif disp_tensor.ndimension() == 3:
        disp_tensor = disp_tensor[0, :, :].detach().cpu()
    else:
        disp_tensor = disp_tensor.detach().cpu()

    disp = disp_tensor.numpy()

    disp_color = disp_to_color(disp, max_disp) / 255.0
    disp_color = disp_color.transpose((2, 0, 1))

    return disp_color

```

<br />
