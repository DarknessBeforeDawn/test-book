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

条件随机场可以由简化形式表示。条件随机场的参数化形式中同一特征在各个位置都有定义，可以对同一个特征在各个位置求和，将局部特征函数转化为一个全局特征函数，这样就可以将条件随机场写成权值向量和特征向量的内积形式，即条件随机场的简化形式。

为简便起见，首先将转移特征和状态特征及其权值用统一的符号表示。设有 $K_1$ 个转移特征， $K_2$ 个状态特征， $K=K_1+K_2$ ,记

$$f_k(y_{i-1},y_i,x,i)=\left
\{
\begin{aligned}  
&t_k(y_{i-1},y_i,x,i),~~~~~k=1,2,\cdots,K_1  \\
&s_l(y_i,x,i),~~~~~k=K_1+l;~~l=1,2,\cdots,K_2
\end{aligned}
\right.$$

然后，对转移与状态特征在各个位置 $i$ 求和，记作

$$f_k(y,x)=\sum_{i=1}^nf_k(y_{i-1},y_i,x,i),~~~~~k=1,2,\cdots,K$$

用 $w_k$ 表示特征 $f_k(y,x)$ 的权值，即

 $$w_k=\left
\{
\begin{aligned}  
&lambda_k,~~~~~k=1,2,\cdots,K_1  \\
&\mu_l,~~~~~k=K_1+l;~~l=1,2,\cdots,K_2
\end{aligned}
\right.$$

于是有

$$P(y|x) = \frac{1}{Z(x)}\exp\sum_{k=1}^Kw_kf_k(y,x)$$

$$Z(x)=\sum_y\exp\sum_{k=1}^Kw_kf_k(y,x)$$

若以 $w$ 表示权值向量，即

$$w=(w_1,w_2,\cdots,W_K)^T$$

以 $F(y,x)$ 表示全局特征向量，即

$$F(y,x)=(f_1(y,x),f_2(y,x),\cdots,f_K(y,x))^T$$

则条件随机场可以写成向量 $w$ 与 $F(y,x)$ 的内积形式:

$$P_w(y|x)=\frac{\exp(w\cdot F(y,x))}{Z_w(x)}$$

其中，

$$Z_w(x)=\sum_y\exp(w\cdot F(y,x))$$

## 2.4 条件随机场的矩阵形式

条件随机场还可以由矩阵表示。假设

$$P_w(y|x) = \frac{1}{Z_w(x)}\exp\sum_{k=1}^Kw_kf_k(y,x)$$

$$Z_w(x)=\sum_y\exp\sum_{k=1}^Kw_kf_k(y,x)$$

表示对给定观测序列 $x$ ,相应的标记序列 $y$ 的条件概率，引进特殊的起点和终点状态标记 $y_0=start,,y_{n-1}=stop$ ,这时 $P_w(y$ \| $x)$ 可以通过矩阵形式表示。

对观测序列 $x$ 的每一个位置 $i=1,2,\cdots,n+1$ ，定义一个 $m$ 阶矩阵( $m$ 是标记 $y_i$ 的取值个数)


$$M_i(x)=[M_i(y_{i-1},y_i|x)]$$

$$M_i(y_{i-1},y_i|x)=\exp(W_i(y_{i-1},y_i|x))$$

$$W_i(y_{i-1},y_i|x)=\sum_{k=1}^Kw_kf_k(y_{i-1},y_i,x,i)$$

这样，给定观测序列 $x$ ,相应标记序列 $y$ 的非规范化概率可以通过该序列 $n+1$ 个矩阵适当元素的乘积 $\prod_{i=1}^{n+1}M_i(y_{i-1},y_i$ \| $x)$ 表示，条件概率是:

$$P_w(y|x)\frac{1}{Z_w(x)}\prod_{i=1}^{n+1}M_i(y_{i-1},y_i|x)$$

其中 $Z_w(x)$ 是规范化因子，是 $n+1$ 个矩阵的乘积的 $(start,stop)$ 元素：

$$Z_w(x)=(M_1(x)M_2(x)\cdots M_{n+1}(x))_{start,stop}$$

注意， $y_0=start,y_{n+1}=stop$ 表示开始状态与终止状态，规范化因子 $Z_w(x)$ 是以 $start$ 为起点 $stop$ 为终点通过状态的所有路径 $y_1y_2\cdots y_n$ 的非规范化概率 $\prod_{i=1}^{n+1}M_i(y_{i-1},y_i$ \| $x)$ 之和。

# 3. 条件随机场的概率计算问题

条件随机场的概率计算问题是给定条件随机场 $P(Y$ \| $X)$ ，输入序列 $x$ 和输出序列 $y$ ,计算条件概率 $P(Y_i=y_i$ \| $x),P(Y_{i-1}=y_{i-1},Y_i=y_i$ \| $x)$ 以及相应的数学期望的问题。为了方便，像 $HMM$ 那样，引进前向-后向向量，递归地计算以上概率记期望。这样的算法称为前向-后向算法。

## 3.1 前向-后向算法

对每个指标 $i=0,1,\cdots,n+1$ ,定义前向向量 $\alpha_i(x)$

$$\alpha_0(y|x)=\left
\{
\begin{aligned}  
&1,~~~~~y=start  \\
&0,~~~~~else
\end{aligned}
\right.$$

递推公式为

$$\alpha_i^T(y_i|x)=\alpha_{i-1}^T(y_{i-1}|x)[M_i(y_{i-1},y_i|x)],~~~~~~i=1,2,\cdots,n+1$$

又可表示为

$$\alpha_i^T(x)=\alpha_{i-1}^T(x)M_i(x)$$

 $\alpha_i(y_i$ \| $x)$ 表示在位置 $i$ 的标记是 $y_i$ 并且到位置 $i$ 的前部分标记序列的非规范化概率， $y_i$ 可取的值有 $m$ 个，所以 $\alpha_i(x)$ 是 $m$ 维列向量。

同样，对每个指标 $i=0,1,\cdots,n+1$ ,定义后向向量 $\beta_i(x)$ :

$$\beta_{n+1}(y_{n+1}|x)=\left
\{
\begin{aligned}  
&1,~~~~~y_{n+1}=stop  \\
&0,~~~~~else
\end{aligned}
\right.$$

$$\beta_i(y_i|x)=[M_i(y_{i},y_{i+1}|x)]\beta_{i+1}(y_{i+1}|x)$$

又可表示为

$$\beta_i(x)=M_{i+1}(x)\beta_{i+1}(x)$$

 $\beta_i(y_i$ \| $x)$ 表示在位置 $i$ 的标记是 $y_i$ 并且从 $i+1$ 到 $n$ 的后半部分标记序列的非规范化概率。

由前向-后向向量定义可得:

$$Z(x)=\alpha_n^T(x)\cdot \mathbf{1}=\mathbf{1}^T\cdot \beta_1(x)$$

这里， $\mathbf{1}$ 是元素均为 1 的 $m$ 维列向量。

## 3.2 概率计算

按照前向-后向向量的定义，很容易计算标记序列在位置 $i$ 是标记 $y_i$ 的条件概率和在位置 $i-1$ 与 $i$ 是标记 $y_{i-1}$ 和 $y_i$ 的条件概率:

$$P(Y_i=y_i|x)=\frac{\alpha_i^T(y_i|x)\beta_i(y_i|x)}{Z(x)}$$

$$P(Y_{i-1},Y_i=y_i|x)=\frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}$$

其中，

$$Z(x)=\alpha_n^T(x)\cdot \mathbf{1}$$

## 3.3 期望计算

利用前向-后向向量，可以计算特征函数关于联合分布 $P(X,Y)$ 和条件分布 $P(Y$ \| $X)$ 的数学期望。

特征函数 $f_k$ 关于条件分布 $P(Y$ \| $X)$ 的数学期望是

$$
\begin{aligned} 
E_{P(Y|X)}[f_k]&=\sum_yP(y|x)f_k(y,x) \\
&=\sum_{i=1}^{n+1}\sum_{y_{i-1}y_i}f_k(y_{i-1},y_i,x,i)\frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}  \\
&k =1,2,\cdots,K
\end{aligned}$$

其中，

$$Z(x)=\alpha_n^T(x)\cdot \mathbf{1}$$

假设经验分布为 $\tilde{P}(X)$ ,特征函数 $f_k$ 关于联合分布 $P(X,Y)$ 的数学期望是

$$
\begin{aligned} 
E_{P(X,Y)}[f_k]&=\sum_{x,y}P(x,y)\sum_{i=1}^{n+1}f_k(y_{i-1},y_i,x,i) \\
&=\sum_x\tilde{P}(X)\sum_yP(y|x)f_k(y,x) \\
&=\sum_x\tilde{P}(X)\sum_{i=1}^{n+1}\sum_{y_{i-1}y_i}f_k(y_{i-1},y_i,x,i)\frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}  \\
&k =1,2,\cdots,K
\end{aligned}$$

其中，

$$Z(x)=\alpha_n^T(x)\cdot \mathbf{1}$$

上述两式是特征函数数学期望的一般计算公式。对于转移特征 $t_k(y_{i-1},y_i,x,i),~~~k=1,2,\cdots,K_1$ ，可以将式中的 $f_k$ 换成 $t_k$ ;对于状态特征，可以将式中的 $f_k$ 换成 $s_i$ ，表示为 $s_l(y_i,x,i), ~~~k=K_1+l,~~~l=1,2,\cdots,K_2$ .

有了概率和期望计算的公式，对给定的观测序列 $x$ 与标记序列 $y$ ,可以通过一次前向扫描计算 $\alpha_i$ 及 $Z(x)$ ，通过一次后向扫描计算 $\beta_i$ ,从而计算所有的概率和期望特征。

# 4. 条件随机场的学习算法

条件随机场模型实际上是定义在时序数据上的对数线形模型，其学习方法包括极大似然估计和正则化的极大似然估计。具体地算法有改进的迭代尺度法 $IIS$ ，梯度下降法以及拟牛顿法。

## 4.1 改进的迭代尺度法

已知训练数据集，由此可知经验概率分布 $\tilde{P}(X,Y)$ 可以通过极大训练数据的对数似然函数来求模型参数。

训练书的对数似然函数为:

$$L(w)=L_{\tilde{P}}(P_w)=\log \prod_{x,y}P_w(y|x)^{\tilde{P}(X,Y)}=\sum_{x,y}\tilde{P}(X,Y)\log P_w(y|x)$$

当 $P_w$ 是

$$P_w(y|x) = \frac{1}{Z_w(x)}\exp\sum_{k=1}^Kw_kf_k(y,x)$$

时，对数似然函数为

$$
\begin{aligned} 
L(w)&=\sum_{x,y}\tilde{P}(x,y)\log P_w(y|x) \\
&=\sum_{x,y}\biggl[\tilde{P}(x,y)\sum_{k=1}^Kw_kf_k(y,x)-\tilde{P}(x,y)\log Z_w(x)\biggr] \\
&=\sum_{j=1}^N\sum_{k=1}^{k}w_kf_k(y_{j},x_j)-\sum_{j=1}^N \log Z_w(x_j) \\
\end{aligned}$$

改进的迭代尺度算法通过迭代的方法不断优化对数似然函数改变量的下界，达到极大化对数似然函数的目的。假设模型的当前参数向量为 $w=(w_1,w_2,\cdots,w_K)^T$ ，向量的增量为 $\delta = (\delta_1,\delta_2,\cdots,delta_K)^T$ ,更新参数向量为 $w+\delta=(w_1+\delta_1,w_2+\delta_2,\cdots,w_K+\delta_K)^T$ 。在每一步迭代过程中，改进的迭代尺度法依次通过求解 $E_{\tilde{P}}[t_k]$ 和 $E_{\tilde{P}}[s_l]$ ，得到 $\delta = (\delta_1,\delta_2,\cdots,delta_K)^T$。

关于转移特征 $t_k$ 的更新方程为 

$$
\begin{aligned} 
E_{\tilde{P}}[t_k]&=\sum_{x,y}\tilde{P}(x,y)\sum_{i=1}^{n+1}t_k(y_{i-1},y_i,x,i) \\
&=\sum_{x,y}\tilde{P}(x)P(y|x)\sum_{i=1}^{n+1}t_k(y_{i-1},y_i,x,i)\exp (\delta_kT(x,y))\\
&~~~~~~~~~~~~~~ k=1,2,\cdots,K_1\\
\end{aligned}$$

关于状态特征 $s_l$ 的更新方程为

$$
\begin{aligned} 
E_{\tilde{P}}[s_l]&=\sum_{x,y}\sum_{i=1}^{n+1}s_l(,y_i,x,i) \\
&=\sum_{x,y}\tilde{P}(x)P(y|x)\sum_{i=1}^{n}s_l(,y_i,x,i)\exp (\delta_{k_1+l}T(x,y))\\
&~~~~~~~~~~~~~~ l=1,2,\cdots,K_2\\
\end{aligned}$$

这里， $T(x,y)$ 是在数据 $(x,y)$ 中出现所有特征数的总和:

$$T(x,y)=\sum_kf_k(y,x)=\sum_{k=1}^K\sum_{i=1}^{n+1}f_k(y_{i-1},y_i,x,i)$$

**条件随机场模型学习的改进的迭代尺度法**

输入: 特征函数 $t_1,t_2,\cdots,t_{K_1},~~~s_1,s_2,\cdots,s_{K_2}$ ; 经验分布 $\tilde{P}(x,y)$ ;

输出：参数估计值 $\hat{w}$ ; 模型 $P_{\hat{w}}$ .

(1)对所有

$$k\in\{1,2,\cdots,K\}$$

取初值 $w_k=0$

(2)对每一个

$$k\in\{1,2,\cdots,K\}$$

有：

(a)当 $k=1,2,\cdots,K_1$ 时，令 $\delta_k$ 是方程

$$\sum_{x,y}\tilde{P}(x)P(y|x)\sum_{i=1}^{n+1}t_k(y_{i-1},y_i,x,i)\exp (\delta_kT(x,y))=E_{\tilde{P}}[t_k]$$

的解；当 $k=k_1+l,~~l=1,2,\cdotsK_2$ 时，令 $\delta_{K_1+l}$ 是方程

$$\sum_{x,y}\tilde{P}(x)P(y|x)\sum_{i=1}^{n}s_l(,y_i,x,i)\exp (\delta_{k_1+l}T(x,y))=E_{\tilde{P}}[s_l]$$

的解。

(b)更新 $w_k$ 值： $w_k\leftarrow w_k+\delta_k$

(3)如果不是所有 $w_k$ 都收敛，重复步骤(2).




