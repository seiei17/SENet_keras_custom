# SENet_keras_custom
A practice after reading paper.
# SENet

## 前言
近期研究很多都在探讨通过特征之间的空间关系来增强模型对特征的表示。

而论文则提出了一种考虑**channel**之间关系的方法——SE Block，进行特征重校准，用来学习全局信息，抑制更少有用的特征。

## SQUEEZE-AND-EXCITATION BLOCKS
常用的卷积操作把通道间的依赖信息卷入了局部的空间关系中。通过卷积得到的通道关系是不清楚且局部的。

作者通过squeeze和excitation两部分来提供一种接触全局信息和特征重校准的方法。

### Squeeze：嵌入全局信息
对于卷积操作，每个学习到的filter都是对空间局部进行操作，这导致输出无法表征其他位置的信息。

论文提出方法将空间信息压缩成通道描述子，这可以通过**全局平均池化**来进行实现。用z表示经过squeeze后的输出，那么<img src="https://latex.codecogs.com/gif.latex?$z_c$" title="$z_c$" />表示第c个通道上的z：

<img src="https://latex.codecogs.com/gif.latex?$$z_c=\operatorname{F}_&space;{sq}(u_c)={1\over{H*&space;C}}\sum_{i=1}^H\sum_{j=1}^Wu_c(i,j)$$" title="$$z_c=\operatorname{F}_ {sq}(u_c)={1\over{H* C}}\sum_{i=1}^H\sum_{j=1}^Wu_c(i,j)$$" />

squeeze操作主要是求得通道间的相关性，如果能屏蔽掉空间上的分布会有更好的效果。所以论文只是选择了最简单的一种方法——Global Average Pooling，其实还有很多方法都可以适用。

### Excitation：适应性重校准
为了利用squeeze操作整合的结果，论文紧跟着加入第二种操作去找到通道间的关联性。提出的function必须满足两个条件：

* 必须是灵活的，特别是要能学习通道间的非线性联系；
* 必须要能学习通道间的非互斥关系，我们要让有用的通道得到强调，而不是得到 一个one-hot向量。

得到excitation公式：

<img src="https://latex.codecogs.com/gif.latex?$$s=\operatorname{F}_&space;{ex}(z,W)=\sigma(g(z,W))=\sigma(W_2\delta(W_1z))$$" title="$$s=\operatorname{F}_ {ex}(z,W)=\sigma(g(z,W))=\sigma(W_2\delta(W_1z))$$" />

* <img src="https://latex.codecogs.com/gif.latex?$\delta$" title="$\delta$" />是ReLU操作。
* 为了减少计算负担，引入了瓶颈操作。即$W_1,W_2$表示两次FC操作的权重，其中第一次进行降维，第二次进行升维。进行降维的大小取决去**r**。经过测试，作者得出，**当r=16的时候效果最好**。

于是得到了对U进行scale的算子s：

<img src="https://latex.codecogs.com/gif.latex?$$\widetilde{x_c}=\operatorname{F}_&space;{scale}(u_c,s_c)=s_cu_c$$" title="$$\widetilde{x_c}=\operatorname{F}_ {scale}(u_c,s_c)=s_cu_c$$" />

### Instantiations
SEBlock能被加入到很多的网络中，如在**VGG**中，可以在每个卷积激活函数后面加入SEBlock。

在**GoogLeNet**中，<img src="https://latex.codecogs.com/gif.latex?$\operatorname{F}_&space;{tr}$" title="$\operatorname{F}_ {tr}$" />操作就代表每个inception模块，得到SE-Inception network。

![ins](./depository/ins.png)

在ResNet中，<img src="https://latex.codecogs.com/gif.latex?$\operatorname{F}_&space;{tr}$" title="$\operatorname{F}_ {tr}$" />操作就代表每个residual模块，得到SE-ResNet。

![res](./depository/res.png)

## 额外讨论
通过ResNet-50进行试验讨论。

### Reduction ratio

![reduce](./depository/reduce.png)

通过比较得出当r=16时，网络有最好的效果。

在实际情况中应该随机应变。

### Squeeze operator

![sqn](./depository/sqo.png)

论文只比较了maxpool和avgpool的差别，两者效果都很好。

### excitation operator

![exo](./depository/exo.png)

通过与ReLU和tanh函数比较，发现ReLu的效果最差，甚至低于原网络。

excitation操作的影响比较大，应该注意选择合适的操作。

### integration strategy

文中提出了几种不同的整合策略，同时给出了相应的错误率。

![its](./depository/its.png)

并且也分出了是否在residual之内添加seblock的区别，称为**SE_3x3**。

## SE BLOCKS的作用
### squeeze的作用
论文通过构造一个NoSqueeze的结构来验证，即不使用全局平均池化的操作，用1x1 conv.来代替excitation操作中的fc层。

得到的效果远远差于标准的SE结果。

得到结论，**全局信息的利用（全局平均池化）对于模型的效果有很大的作用**。

### excitation的作用
论文通过在resnet的不同的stage构造SEBlock来观察数据的激活。

在低层各种特征的数据相差不大，这说明低级特征更多的是共通性；而在高层各种特征的数据相差较大，各自有了不同的针对性。而且在靠近分类器的最后一个stage，激活趋向饱和，即全部都接近为1。

所以作者提出，可以除去靠近分类器的SE block，这样可以减少大量的参数且只损失一点点的精确度。

