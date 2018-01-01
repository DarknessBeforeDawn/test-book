---
title: 聚类
layout: post
share: false
---
# 1.有监督学习与无监督学习

首先，看一下有监督学习(Supervised Learning)和无监督学(Unsupervised Learning)习的区别，给定一组数据(input，target)为 $Z=(X,Y)$ 。

有监督学习: 最常见的是回归(regression)和分类(classification)。

$\bullet$ Regression: $Y$ 是实数向量。回归问题，就是拟合 $(X,Y)$ 的一条曲线，使得下式损失函数 $L$ 最小。

$$L(f,(X,Y))=\|f(X)-Y\|^2$$

$\bullet$ Classification: $Y$ 是一个finite number，可以看做类标号。分类问题需要首先给定有label的数据训练分类器，故属于有监督学习过程。分类问题中，cost function $L(X,Y)$ 是 $X$ 属于类 $Y$ 的概率的负对数。

$$L(f,(X,Y))=-logf_Y(X),\ \ f_i(X)=P(Y=i|X);\ \ f_Y(X)\geq 0, \ \ \sum_if_i(X)=1$$

无监督学习：无监督学习的目的是学习一个function $f$ ，使它可以描述给定数据的位置分布 $P(Z)$ 。 包括两种：密度估计(density estimation)和聚类(clustering).

$\bullet$ density estimation就是密度估计，估计该数据在任意位置的分布密度

$\bullet$ clustering就是聚类，将 $Z$ 聚集几类（如K-Means），或者给出一个样本属于每一类的概率。由于不需要事先根据训练数据去训练聚类器，故属于无监督学习。

$\bullet$ PCA和很多deep learning算法都属于无监督学习。

#### 聚类

聚类试图将数据集中的样本划分为若干个通常不相交的子集，每个子集称为一个“簇”(cluster).形式化地说，假定样本集

$$D=\{\mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_m\}$$

包含 $m$ 个无标记样本，每个样本 $\mathbf{x}_i=(x_i1;x_i2;\cdots;x_im)$ 是一个 $n$ 维特征向量，则聚类算法将样本集 $D$ 划分为 $k$ 个不相交的簇 

$$\{C_l|l=1,2,\cdots,k\}, \ \ C_{l`}\bigcap_{l`\neq l}C_l=\emptyset, \ \ \ D=\bigcup_{l=1}^kC_l$$

相应地，用

$$\lambda_j\in\{1,2,\cdots,k\}$$

表示样本 $\mathbf{x}_j$ 的“簇标记”(cluster label),即 $\mathbf{x}_j\in C_{\lambda_j}$ .于是，聚类的结果可以用包含 $m$ 个元素的簇标记向量 $\mathbf{\lambda}=(\lambda_1;\lambda_2;\cdots;\lambda_m)$ 表示。

基于不同的学习策略有很多种类型的聚类学习算法，这里先讨论两个基本问题，性能度量和距离计算。

# 2.性能度量

聚类性能度量亦称聚类“有效性指标”(validity index).对于聚类结果，需要通过某种性能度量来评估其好坏。聚类结果应“簇内相似度”(intra-cluster similarity)高且“簇间相似度”(inter-cluster similarity)低。

聚类性能度量大致分为两类.一类是将聚类结果与某个参考模型(reference model)进行比较，称为外部指标(external index);另一类是直接考察聚类结果而不利用任何参考模型，称为内部指标(internal index).

对数据集 $D$ ，假定通过聚类给出的簇划分结果为

$$C=\{C_1,C_2,\cdots,C_k\}$$

参考模型给出的簇划分为

$$C^*=\{C_1^*,C_2^*,\cdots,C_k^*\}$$

相应地，令$$\mathbf{\lambda},\mathbf{\lambda}^*$$表示$$C,C^*$$对应的簇标记向量。定义

$$a=|SS|, \ \ \ SS=\{(\mathbf{x}_i,\mathbf{x}_j)|\lambda_i=lambda_j,\lambda_i^*=lambda_j^*,i<j\}$$

$$b=|SD|, \ \ \ SD=\{(\mathbf{x}_i,\mathbf{x}_j)|\lambda_i=lambda_j,\lambda_i^*neq lambda_j^*,i<j\}$$

$$c=|DS|, \ \ \ DS=\{(\mathbf{x}_i,\mathbf{x}_j)|\lambda_i\neq lambda_j,\lambda_i^*=lambda_j^*,i<j\}$$

$$d=|DD|, \ \ \ DD=\{(\mathbf{x}_i,\mathbf{x}_j)|\lambda_i\neq lambda_j,\lambda_i^*\neq lambda_j^*,i<j\}$$

其中集合 $SS$ 包含了在 $C$ 中隶属于相同簇且在$$C^*$$中也隶属于相同簇的样本对；集合 $SD$ 包含了在 $C$ 中隶属于相同簇但在$$C^*$$中隶属于不同簇的样本对；……由于每个样本对 $(\mathbf{x}_i,\mathbf{x}_j)(i<j)$ 仅能出现在一个集合中，因此有 $a+b+c+d=m(m-1)/2$ 成立。

基于上述式子可导出以下常用的聚类性能度量外部指标：

$\bullet$ Jaccard系数(Jaccard Coefficient,简称JC)

$$JC=\frac{a}{a+b+c}$$

$\bullet$ FM指数(Fowlkes and Makkows Index,简称FMI)

$$FMI=\sqrt{\frac{a}{a+b}\cdot \frac{a}{a+c}}$$

$\bullet$ Rand指数(Rand Index,简称RI)

$$RI=\frac{2(a+d)}{m(m-1)}$$

上述性能度量的结果值均在 $[0,1]$ 区间，值越大越好。

定义

$$avg(C)=\frac{2}{|C|(|C|-1)}\sum_{1\leq i<j \leq |C|}dist(\mathbf{x}_i,\mathbf{x}_j)$$

$$diam(C)=\max_{1\leq i<j \leq |C|}dist(\mathbf{x}_i,\mathbf{x}_j)$$

$$d_{min}(C_i,C_j)=\min_{\mathbf{x}_i\in C_i,\mathbf{x}_j\in C_j}dist(\mathbf{x}_i,\mathbf{x}_j)$$

$$d_{cen}(C_i,C_j)=dist(\mathbf{\mu}_i,\mathbf{\mu}_j)$$

其中, $dist(\cdot,\cdot)$ 用于计算两个样本之间的距离； $\mathbf{\mu}$ 代表簇 $C$ 的中心点

$$\mathbf{\mu}=\frac{1}{|C|}\sum_{1\leq i\leq |C|}\mathbf{x}_i$$

显然， $avg(C)$ 对应簇 $C$ 内样本间的平均距离； $diam(C)$ 对应于簇 $C$ 内样本的最远距离； $d_{min}(C_i,C_j)$ 对应于簇 $C_i$ , $C_j$ 最近样本间的距离； $d_{cen}(C_i,C_j)$ 对应于簇 $C_i$ , $C_j$ 中心点间的距离。

由上述各种距离的定义可以导出以下常用的聚类性能度量内部指标:

$\bullet$ DB指数(Davies_Bouldin Index,简称DBI)

$$DBI=\frac{1}{k}\sum_{i=1}^k\max_{j\neq i}\biggl(\frac{avg(C_i)+avg(C_j)}{d_{cen}(C_i,C_j)}\biggr)$$

$\bullet$ Dunn指数(Dunn Index,简称DI)

$$DI=\min_{1\leq i\leq k}\biggl\{\min_{j\neq i}\biggl(\frac{d_{min}(C_i,C_j)}{\max_{1\leq l\leq k}diam(C)}\biggr)\biggr\}$$

# 3.距离计算

$dist(\cdot,\cdot)$ 是距离度量，它有一些基本性质:

$\bullet$ 非负性： $dist(\mathbf{x_i},\mathbf{x_j})\geq 0$ ;

$\bullet$ 同一性： $dist(\mathbf{x_i},\mathbf{x_j})= 0$ 当且仅当 $\mathbf{x_i}=\mathbf{x_j}$ ;

$\bullet$ 对称性： $dist(\mathbf{x_i},\mathbf{x_j})=dist(\mathbf{x_i},\mathbf{x_j})$ ;

$\bullet$ 直递性： $dist(\mathbf{x_i},\mathbf{x_j})\leq dist(\mathbf{x_i},\mathbf{x_k})+dist(\mathbf{x_k},\mathbf{x_j})$ .

给定样本 $\mathbf{x_i}=(x_{i1};x_{i2};\cdots;x_{in})$ 与 $\mathbf{x_j}=(x_{j1};x_{j2};\cdots;x_{jn})$ ,则闵可夫斯基距离(Minkowski distance)为

$$dist_{mk}(\mathbf{x_i},\mathbf{x_j})=\biggl(\sum_{u=1}^n|x_{iu}-x_{ju}|^p\biggr)^{\frac{1}{p}}$$

对于 $p\geq 1$ 上式满足距离度量的基本性质。

$p=2$ 时该距离称为欧氏距离(Euclidean distance) 

$$dist_{ed}(\mathbf{x_i},\mathbf{x_j})=\|\mathbf{x_i}-\mathbf{x_j}\|_2=\sqrt{\sum_{u=1}^n|x_{iu}-x_{ju}|^2}$$

$p=2$ 时为曼哈顿距离(Manhattan distance)

$$dist_{ed}(\mathbf{x_i},\mathbf{x_j})=\|\mathbf{x_i}-\mathbf{x_j}\|_1=\sum_{u=1}^n|x_{iu}-x_{ju}|$$

对于1,2,3这类数字型之间的距离，可以直接算出1与2的距离比较近、与3的距离比较远，这样的属性称为有序属性(ordinal attribute);而对于与飞机，火车，轮船这样的离散属性则不能直接在属性值上计算距离，称为无序属性(non-ordinal attribute)。无属性距离不可用闵可夫斯基距离计算。

对无序属性可采用VDM(Value Difference Metric).令 $m_{u,a}$ 表示在属性 $u$ 上取值为 $a$ 的样本个数， $m_{u,a,i}$ 表示在第 $i$ 个样本簇中在属性 $u$ 上取值为 $a$ 的样本数， $k$ 为样本簇数，则属性 $u$ 上两个离散值 $a$ 与 $b$ 之间的VDM为

$$VDM_p(a,b)=\sum_{i=1}^k\biggl |\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}\biggr |$$

将闵可夫斯基距离和VDM结合即可处理混合属性。假定有 $n_c$ 个有序属性， $n-n_c$ 个无序属性，令有序属性排列在无序属性之前，则

$$MinkovDM_p(\mathbf{x_i},\mathbf{x_j})=\biggl(\sum_{u=1}^{n_c}|x_{iu}-x_{ju}|^p+\sum_{u=n_c+1}^nVDM_p(x_{iu},x_{ju})\biggr)^{\frac{1}{p}}$$

当样本空间中不同属性的重要性不同时，可使用加权距离(weihted distance),以加权闵可夫斯基距离为例

$$dist_{wmk}(\mathbf{x_i},\mathbf{x_j})=\biggl(\sum_{u=1}^nw_i|x_{iu}-x_{ju}|^p\biggr)^{\frac{1}{p}}$$

其中权重 $w_i\geq 0(i=1,2,\cdots,n)$ 表征不同属性的重要性，一般 $\sum\limits_{i=1}^n=1$ .

通常相似度度量是基于某种形式的距离来定义的，距离越大，相似度越小。用于相似度度量的距离未必一定要满足距离度量的所有基本性质，尤其是直递性。不满足直递性的距离称为非度量距离(non-metric-distance).

# 4.原型聚类

原型聚类亦称基于原型的聚类(protorype-based clustering),此类算法假设聚类结构能通过一组原型刻画，算法先对原型进行初始化，然后对原型进行迭代更新。

## 4.1 k-means算法

给定样本集

$$D=\{\mathbf{x_1},\mathbf{x_2},\cdots,\mathbf{x_m}\}$$

k-means算法针对聚类所得簇划分

$$C=\{C_1,C_2,\cdots,C_k\}$$

最小化平方误差

$$E=\sum_{i=1}^k\sum_{\mathbf{x} \in C_i}\|\mathbf{x}-\mathbf{\mu_i}\|_2^2, \ \ \mathbf{\mu_i}=\frac{1}{|C_i|}\sum_{\mathbf{x} \in C_i}\mathbf{x}$$

其中 $\mu_i$ 是簇 $C_i$ 的均值向量。在一定程度上上式刻画了簇内样本围绕均值向量的紧密程度, $E$ 值越小则簇内样本相似度越高。

最小化上述平方误差并不简单，找到最优解需要考察样本集 $D$ 所有可能的簇划分，这是一个NP-Hard问题。因此，k-means算法采用了贪心策略，通过迭代优化近似解。

k-means算法流程：

———————————————————————————————

输入:聚类簇数 $k$ ,样本集$$D=\{\mathbf{x_1},\mathbf{x_2},\cdots,\mathbf{x_m}\}$$

过程：

1:从 $D$ 中随机选择 $k$ 个样本作为初始均值向量$$\{\mathbf{\mu_1},\mathbf{\mu_2},\cdots,\mathbf{\mu_k}\}$$

2:repeat

3: $~~~~$ 令 $C_i=\emptyset(1\leq i \leq k)$

4: $~~~~~~~~$ for $j=1,2,\cdots,m$ do

5: $~~~~~~~~~~~~$ 计算样本 $\mathbf{x_j}$ 与各均值向量 $\mathbf{\mu_i}(1\leq i \leq k)$ 的距离： $d_{ji}=\|\mathbf{x_j}-\mathbf{\mu_i}\|_2$ ;

6: $~~~~~~~~~~~~$ 根据距离最近的均值向量确定 $\mathbf{x_j}$ 的簇标记: $\lambda_j=\arg\min_id_{ji}$ ;

7: $~~~~~~~~~~~~$ 将样本 $\mathbf{x_j}$ 划入相应的簇: $C_{\lambda_j} = C_{\lambda_j}\bigcup\{\mathbf{x_j}\}$ ;

8: $~~~~~~~~$ end for

9: $~~~~~~~~$ for $i=1,2,\cdots,k$ do

10: $~~~~~~~~~~~~$ 计算新均值向量 $\mathbf{\mu_i}'$ ;

11: $~~~~~~~~~~~~$ if $\mathbf{\mu_i}'\neq \mathbf{\mu_i}$  then

12: $~~~~~~~~~~~~~~~~$ 将当前均值向量 $\mathbf{\mu_i}$ 更新为 $\mathbf{\mu_i}'$

13: $~~~~~~~~~~~~$ else

14: $~~~~~~~~~~~~~~~~$ 保持当前均值向量不变

15: $~~~~~~~~~~~~$ end if

16: $~~~~~~~~$ end for

17: until 当前均值向量均未更新

输出：簇划分$$C=\{C_1,C_2,\cdots,C_k\}$$

———————————————————————————————
