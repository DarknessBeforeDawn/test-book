---
title: 条件随机场
layout: post
share: false
---

条件随机场(Conditional Random Field,CRF)是给定一组输入随机变量条件下另一组输出随机变量的条件概率分布模型，其特点是假设输出随机变量构成马尔科夫随机场。

# 1. 概率无向图模型

概率无向图模型(Prodbabilistic Undirected Graphical Model),又称为马尔科夫随机场(Markov Random Field)，是一个可以由无向图模型表示的联合概率分布。

## 1.1 模型定义

图(Graph)是由结点(Node)及连接结点的边(Edge)组成的集合，结点和边分别记作 $v$ 和 $e$ ，结点和边的集合分别记作 $V$ 和 $E$ ,图记作 $G=(V,E)$ .无向图是指边没有方向的凸。

概率图模型(Prodbabilistic Graphical Model)是由图表示的概率分布。设有联合概率分布 $P(Y)$ , $Y\in \mathcal{Y}$ 是一组随机变量。由无向图 $G=(V,E)$ 表示概率分布 $P(Y)$ ，即在图 $G$ 中，结点 $v\in V$ 表示一个随机变量 $Y_v,Y=(Y_v)_{v\in V}$ ；边 $e\in E$ 表示随机变量之间的概率依赖关系。

给定一个联合概率分布 $P(Y)$ 和表示它的无向图 $G$ 。首先定义无向图表示的随机变量之间存在的成对马尔科夫性(Pairwise Markov Property)、局部马尔可夫性(Local Markov Property)和全局马尔可夫性(Global Markov Property).

成对马尔可夫性：设 $u$ 和 $v$ 是无向图 $G$ 中任意两个没有边连接的结点，结点 $u$ 和 $v$ 分别对应随机变量 $Y_u$ 和 $Y_v$ 。其他所有结点为 $O$ ,对应的随机变量组是 $Y_O$ .成对马尔可夫性是指给定随机变量组 $Y_O$ 的条件下随机变量 $Y_u$ 和 $Y_v$ 是条件独立的，即

$$P(Y_u,Y_v|Y_O)=P(Y_u|Y_O)P(Y_v|P_O)$$

局部马尔可夫性：设 $v$ 是无向图 $G$ 中任意一个结点， $W$ 是与 $v$ 有边连接的所有结点， $O$ 是 $v,W$ 以外的其他所有结点。 $v$ 表示的随机变量是 $Y_v,W$ 表示的随机变量组是 $Y_W,O$ 表示的随机变量组是 $Y_O$ .局部马尔可夫性是指在给定随机变量组 $Y_W$ 的条件下随机变量 $Y_v$ 与随机变量组 $Y_O$ 是独立的，即

$$P(Y_v,Y_O|Y_W)=P(Y_v|Y_W)P(Y_O|P_W)$$

有

$$P(Y_O|P_W)>0\Rightarrow P(Y_v|Y_W) = P(Y_v|Y_O,Y_W)$$

局部马尔可夫性如下图所示：

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/CRF1.jpg"/>
</center>

全局马尔可夫性:设结点集合 $A,B$ 是在无向图 $G$ 中被结点集合 $C$ 分开的任意结点集合，如下图：

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/CRF2.jpg"/>
</center>

结点集合 $A,B$ 和 $C$ 对应的随机变量组分别是 $Y_A,Y_B,Y_C$ ，全局马尔可夫性是指给定随机变量组 $Y_C$ 条件下随机变量组 $Y_A$ 和 $Y_B$ 是条件独立的，即

$$P(Y_A,Y_B|Y_C)=P(Y_A|Y_C)P(Y_B|P_C)$$

**概率无向图模型：** 设有联合概率分布 $P(Y)$ ，由无向图 $G=(V,E)$ 表示，在图 $G$ 中，结点表示随机变量，边表示随机变量之间的依赖关系。如果联合概率分布 $P(Y)$ 满足成对、局部或全局马尔科夫性，就称此联合概率分布为概率无向图模型(Prodbabilistic Undirected Graphical Model),又称为马尔科夫随机场(Markov Random Field)。

概率无向图模型的最大特点就是易于因子分解，这样便于模型的学习与计算。

## 1.2 概率无向图模型的因子分解

**团与最大团：** 无向图 $G$ 中任何两个结点均有边连接的结点子集称为团(Clique).若 $C$ 是无向图 $G$ 的一个团，并且不能再加进任何一个 $G$ 的结点使其成为一个更大的团，则称此 $C$ 为最大团(Maximal Clique).

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/CRF3.jpg"/>
</center>

如上图表示由4个点组成的无向图。图中由2个结点组成的团由5个：

$$\{Y_1,Y_2\},\{Y_2,Y_3\},\{Y_3,Y_4\},\{Y_2,Y_4\},\{Y_1,Y_3\},$$

有两个最大团：

$$\{Y_1,Y_2,Y_3\},\{Y_2,Y_3,Y_4\}$$

而

$$\{Y_1,Y_2,Y_3,Y_4\}$$

不是一个团。因为 $Y_1$ 和 $Y_4$ 没有边连接。

将概率无向图模型的联合概率分布表示为其最大团上的随机变量的函数的乘积形式的操作，称为概率无向图模型的因子分解(Factorization).

给定概率无向图模型，设其无向图为 $G$ ， $C$ 为 $G$ 上的最大团， $Y_C$ 表示 $C$ 对应的随机变量。那么概率无向图模型的联合概率分布 $P(Y)$ 可写作图中所有最大团 $C$ 上的函数 $Psi_C(Y_C)$ 的乘积形式，即

$$P(Y)=\frac{1}{Z}\prod_C\Psi_C(Y_C)$$

其中， $Z$ 是规范化因子(Normalization Factor)：

$$Z=\sum_Y\prod_C\Psi_C(Y_C)$$

规范化因子保证 $P(Y)$ 构成一个概率分布，函数 $\Psi_C(Y_C)$ 称为势函数(Potential Function). 这里要求势函数 $\Psi_C(Y_C)$ 是严格正的，通常定义为指数函数：

$$\Psi_C(Y_C)=-\exp\{-E(Y_C)\}$$

**Hammersley-Clifford 定理**  概率无向图模型的联合概率分布 $P(Y)$ 可以表示为如下形式：

$$P(Y)=\frac{1}{Z}\prod_C\Psi_C(Y_C)$$

$$Z=\sum_Y\prod_C\Psi_C(Y_C)$$

其中， $C$ 是无向图的最大团， $Y_C$ 是 $C$ 的结点对应的随机变量， $\Psi_C(Y_C)$ 是 $C$ 上定义的严格正函数，乘积是在无向图所有的最大团上进行的。

# 2. 条件随机场的定义与形式

## 2.1 条件随机场的定义

条件随机场(CRF)是给定随机变量 $X$ 条件下，随机变量 $Y$ 的马尔科夫随机场，本文主要介绍定义在线性链上的特殊的条件随机场，称为线性链条件随机场(Linear Chain Conditional Random Field).线性链条件随机场可以用于标注等问题。这时，在条件概率模型 $P(Y$ \| $X)$ 中， $Y$ 是输出量，表示标记序列， $X$ 是输入变量，表示需要标注的观测序列。也把标记序列称为状态序列(参见 $HMM$ ).学习时，利用训练数据集通过极大似然估计或正则化的极大似然估计得到条件概率模型 $\hat{P}(Y$ \| $X)$ ;预测是，对于给定的输入序列 $x$ ，求出条件概率 $\hat{P}(y$ \| $x)$ 最大的输出序列 $\hat{y}$ .

**条件随机场：** 设 $X$ 与 $Y$ 是随机变量， $P(Y$ \| $X)$ 是在给定 $X$ 的条件下 $Y$ 的条件概率分布，若随机变量 $Y$ 构成一个由无向图 $G=(V,E)$ 表示的马尔科夫随机场，即

$$P(Y_v|X,Y_w,w\neq v)=P(Y_v|X,Y_w,w\sim v)$$

对任意结点 $v$ 成立，则称条件概率分布 $P(Y$ \| $X)$ 为条件随机场。式中 $w\sim v$ 表示在无向图 $G=(V,E)$ 中与结点 $v$ 有边连接的所有结点 $w,w\neq v$ 表示结点 $v$ 以外的所有结点， $Y_v,Y_w$ 为结点 $u,w$ 对应的随机变量。

在定义中并没有要求 $X$ 和 $Y$ 具有相同的结构。现实中，一般假设 $X$ 和 $Y$ 有相同的图结构。一般考虑无向图如下图所示的线性链情况。

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/CRF4.jpg" width="350"/>
    <img src="https://darknessbeforedawn.github.io/test-book/images/CRF5.jpg" width="350"/>
</center> 

即

$$G=(V=\{1,2,\cdots,n\},~~~~E=\{(i,i+1)\})$$

在此情况下， $X=(X_1,X_2,\cdots,X_n),~~~Y=(Y_1,Y_2,\cdots,Y_n)$ ，最大团是相邻两个结点的集合。

**线性链条件随机场：** 设 $X=(X_1,X_2,\cdots,X_n),~~~Y=(Y_1,Y_2,\cdots,Y_n)$ 均为线性链表示的随机变量序列，若在给定随机变量序列 $X$ 的条件下，随机变量序列 $Y$ 的条件概率分布 $P(Y$ \| $X)$ 构成条件随机场，即满足马尔可夫性

$$P(Y_i|X,Y_1,\cdots,Y_{i-1},Y_{i+1},\cdots,Y_n)=P(Y_i|X,Y_{i-1},Y_{i+1}),~~~i=1,2,\cdots,n$$

则称 $P(Y$ \| $X)$ 为线性链条件随机场。其中，在 $i=1,i=n$ 时只考虑单边。在标注问题中， $X$ 表示输入观测序列， $Y$ 表示对应的输出标记序列或状态序列。

## 2.2 条件随机场的参数化形式

根据Hammersley-Clifford 定理，可以给出线性链条件随机场 $P(Y$ \| $X)$ 的因子分级式，各因子是定义在相邻两个结点上的函数。

**线性链条件随机场的参数化形式(定理)：** 设 $P(Y$ \| $X)$ 为线性链条件随机场，则在随机变量 $X$ 的取值为 $x$ 的条件下，随机变量 $Y$ 取值为 $y$ 的条件概率具有如下形式:

$$P(y|x)=\frac{1}{Z(x)}\exp\biggl(\sum_{i,k}\lambda_kt_k(y_{i-1},y_i,x,i)+\sum_{i,l}\mu_ls_l(y_i,x,i)\biggr)$$

其中，

$$Z(x)=\sum_y\exp\biggl(\sum_{i,k}\lambda_kt_k(y_{i-1},y_i,x,i)+\sum_{i,l}\mu_ls_l(y_i,x,i)\biggr)$$

式中， $t_k$ 和 $s_l$ 是特征函数， $\lambda_k$ 和 $\mu_k$ 是对应的权值， $Z(x)$ 是规范化因子，求和是在所有可能的输出序列上进行的，

上述两个公式是线性链条件随机场模型的基本形式，表示给定输入序列 $x$ ，对输出序列 $y$ 预测的条件概率。式中 $t_k$ 是定义在边上的特征函数，称为转移特征，依赖于当前和前一个位置， $s_l$ 是定义在结点上的特征函数，称为状态特征，依赖于当前位置. $t_k$ 和 $s_l$ 都依赖于位置，是局部特征函数。通常，特征函数 $t_k$ 和 $s_l$ 取值为1或0；当满足特征条件时取值为 1，否则为0.条件随机场完全由特征函数 $t_k,s_l$ 和对应的权值 $\lambda_k,\mu_l$ 确定。

线性链条件随机场也是对数线性模型(Log Linear Model).

## 2.3条件随机场的简化形式