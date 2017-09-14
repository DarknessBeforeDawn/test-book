---
title: 决策树
layout: post
share: false
---

决策树学习，假设给定训练数据集：


$$\begin{equation}
D=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}
\end{equation}$$

其中$x_i=(x_i^{(1)},x_i^{(2)},\cdots,x_i^{(n)})^T$为输入实例，$n$为特征个数，$y\in\{1,2,\cdots,K\}$为类标记，$N$为样本容量。学习目标是根据训练数据构建一个决策树模型，使它能够对实例正确分类。

熵（entropy）是表示随机变量不确定性的度量。设$X$是一个取有限个值得离散随机变量，其概率分布为：

$$P(X=x_i)=p_i,  i=1,2,\cdots,n$$

则随机变量$X$的熵定义为：

$$\begin{equation}
H(X)=-\sum_{i=1}^np_i\log p_i
\end{equation}$$

熵只依赖于$X$的分布，而与$X$的取值无关，因此可以将$X$的熵记做$H(P)$,熵越大随机变量的不确定性就越大，且$0 \leqslant H(p) \leqslant \log n$.

设随机变量$(X,Y)$,其联合概率分布为：


$$\begin{equation}
P(X,Y)=P(X=x_i,Y=y_i)=p_{ij},  i=1,2,\cdots,n;j=1,2,\cdots,m
\end{equation}$$

条件熵$H(Y|X)$表示在已知随机变量$X$的条件下随机变量$Y$的不确定性。$X$给定条件下$Y$的条件概率分布的熵对$X$的数学期望：

$$\begin{equation}
H(Y|X)=\sum_{i=1}^np_iH(Y|X_i),p_i=P(X=x_i),i=1,2,\cdots,n
\end{equation}$$

# 信息增益

特征$A$对训练数据集$D$的信息增益$g(D,A)$，定义为集合$D$的经验熵$H(D)$与特征$A$给定条件下$D$的经验熵$H(D|A)$之差，即：

$$\begin{equation}
g(D,A)=H(D)-H(D|A)
\end{equation}$$

$H(D)$表示对数据集$D$进行分类的不确定性，$H(D|A)$表示按特征$A$分类过后的所有熵的总和；二者的差表示分类前后不确定性减少的程度，因此信息增益大的特征有更强的分类能力。


设数据集为$D$,$|D|$表示样本个数，设有$K$个类$C_k$,$|C_k|$为属于$C_k$的样本个数，$\sum_{k=1}^K|C_k|=|D|$。设特征$A$有$n$个不同取值$\{a_1,a_2,\cdots,a_n\}$，根据$A$的取值将$D$划分为$n$个子集$D_1,D_2,\cdots,D_n$,$|D_i|$为$D_i$的样本个数，$\sum_{i=1}^n|D_i|=|D|$,记子集$D_i$中属于类$C_k$的样本的集合为$D_{ik}$,即$D_{ik}=D_i\bigcap C_k,|D_{ik}|$为$D_{ik}$的样本个数，则


$$\begin{equation}
H(D)=-\sum_{k=1}^K\frac{|C_k|}{|D|}\log\frac{|C_k|}{|D|}
\end{equation}$$

$$\begin{equation}
H(D|A)=-\sum_{i=1}^n\frac{|D_i|}{|D|}H(D_i)=-\sum_{i=1}^n\frac{|D_i|}{|D|}\sum_{k=1}^K\frac{|D_{ik}|}{|D_i|}\log\frac{|D_{ik}|}{|D_i|}
\end{equation}$$


# 信息增益比
特征$A$对训练数据集$D$的信息增益比：

$$\begin{equation}
g_R(D,A)=\frac{g(D,A)}{H_A(D)}
\end{equation}$$

其中$H_A(D)$为数据集$D$关于特征$A$的值的熵，$H_A(D)=-\sum_{i=1}^n\frac{|D_i|}{|D|}\log\frac{|D_i|}{|D|}$,$n$是特征$A$取值的个数。

ID3算法是计算所有特征的信息增益，每次选择信息增益最大的特征，并对改特征的不同取值构建子节点；再对子节点递归上述操作，来构建决策树。ID3算法生成的树容易产生过拟合。


C4.5算法是将ID3算法生成树过程中选择特征的方法换为了信息增益比。

# 决策树剪枝

决策树递归生成算法，易出现过拟合现象，这是由于在学习时过多的考虑如何提高对训练数据的正确分类，从而导致构建的决策树过于复杂，我们可以通过对决策树进行剪枝来简化决策树。

决策树剪枝通过极小化损失函数来实现。假设树$T$的叶节点个数为$|T|$,$t$是树$|T|$的叶节点，该叶节点有$N_t$个样本点，其中$k$类的样本点有$N_{tk}$个，$H_t(T)$位叶节点$t$上的经验熵，$\alpha \geqslant 0$为参数，则决策树的损失函数可定义为：

$$\begin{equation}
C_\alpha(T)=\sum_{t=1}^{|T|}N_tH_t(T)+\alpha |T|
\end{equation}$$

其中经验熵为：

$$\begin{equation}
H_t(T)=-\sum_k\frac{N_{tk}}{N_t}\log\frac{N_{tk}}{N_t}
\end{equation}$$

令

$$\begin{equation}
C(T)=\sum_{t=1}^{|T|}N_tH_t(T)=-\sum_{t=1}^{|T|}\sum_{k=1}^KN_{tk}\log\frac{N_{tk}}{N_t}
\end{equation}$$

则$C_\alpha(T) =C(T)+\alpha |T| $，其中$C(T)$表示模型对训练数据的预测误差，即模型与数据的拟合程度，$|T|$表示模型的复杂程度，较大的$\alpha$促使选择简单的模型，较小的$\alpha$促使选择较复杂的模型。

剪枝就是当$\alpha$确定时，选择损失函数最小的模型，假设一组叶节点回缩到其父节点之前与之后的整体树分别为$T_B$和$T_A$，其对应的损失函数数值分别为$C_\alpha(T_B)$和$C_\alpha(T_A)$,如果$C_\alpha(T_B)\geqslant C_\alpha(T_A)$，则进行剪枝，即将父节点变为新的叶节点。

# CART算法

### 1.回归生成树

首先，选择最优切分变量$j$与切分点$s$，将每个区域切分成两个子区域

$$\begin{equation}
R_1(j,s)=\{x|x^{(j)}\leqslant s\} , R_2(j,s)=\{x|x^{(j)}> s\}
\end{equation}$$

然后选择最优切分变量$j$和切分点$s$，具体地，求解：

$$\begin{equation}
min_{j,s}\biggl [min_{c_1}\sum_{x_i\in R_1(j,s)} (y_i-c_1)^2+min_{c_2}\sum_{x_i\in R_2(j,s)} (y_i-c_2)^2 \biggr]
\end{equation}$$

然后用选定的对$(j,s)$划分区域并决定相应的输出值：

$$\begin{equation}
\hat{c}_m=\frac{1}{N_m}\sum_{x_i\in R_m(j,s)} y_i, x\in R_m,m=1,2
\end{equation}$$

递归对子区域做上述操作，直到满足停止条件。将输入空间划分为$M$个区域$R_1,R_2,\cdots,R_M$，生成决策树：

$$\begin{equation}
f(x) = \sum_{m=1}^M\hat{c}_mI(x\in R_m)
\end{equation}$$

#####CART回归生成树例子
训练数据见下表，$x$的取值范围$[0.5,10.5],y$的取值范围$[5.0,10.0]$，学习这个回归问题的最小二叉回归树。

$$\begin{array}{l|lcr} x_i&1&2&3&4&5&6&7&8&9&10\\ \hline y_i &5.56&5.70&5.91&6.40&6.80&7.05&8.90&8.70&9.00&9.05 \\  \end{array}$$

求训练数据的切分点，根据所给数据，考虑如下切分点：

$$1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5$$

对各个切分点，求出对应的$R_1,R_2,c_1,c_2$及

$$\begin{equation}
m(s)=min_{j,s}\biggl [min_{c_1}\sum_{x_i\in R_1(j,s)} (y_i-c_1)^2+min_{c_2}\sum_{x_i\in R_2(j,s)} (y_i-c_2)^2 \biggr]
\end{equation}$$

当$s=1.5$时，$R_1=\{1\},R_1=\{2,3,\cdots,10\},c_1=5.56,c_2=7.50,m(s)=0+15.72=15.72$

$s$及$m(s)$的计算结果如下表：

$$\begin{array}{l|lcr} s&1.5&2.5&3.5&4.5&5.5&6.5&7.5&8.5&9.5\\ \hline m(s) &15.72&12.07&8.36&5.78&3.91&1.93&8.01&11.73&15.74 \\  \end{array}$$

由上表可知，当$x=6.5$的时候达到最小值，此时$R_1=\{1,2,\cdots,6\}$,$R_1=\{7,8,9,10\}$,$c_1=6.24$,$c_2=8.9$,所以回归树$T_1(x)$为：

$$
\begin{equation}
T_1(x)=\begin{cases}6.24,&x<6.5\\8.91,&x \geq  6.5\end{cases}
\end{equation}
$$

$$
\begin{equation}
f_1(x)=T_1(x)
\end{equation}
$$

用$f_1(x)$拟合训练数据的残差如下表，表中$r_{2i}=y_i-f_1(x_i)$

$$\begin{array}{l|lcr} x_i&1&2&3&4&5&6&7&8&9&10\\ \hline y_i &-0.68&-0.54&-0.33&0.16&0.56&0.81&-0.01&-0.21&0.09&0.14 \\  \end{array}$$

用$f_1(x)$拟合训练数据的平方误差：

$$
\begin{equation}
L(y,f_1(x))=\sum_{i=1}^{10}(y_i-f_1(x_i))^2=1.93
\end{equation}
$$

第二步求$T_2(x)$

### 2.分类树生成

##### 尼基指数

分类问题中，假设有$K$个类，样本点属于第$k$类的概率为$p_k$,则概率分布的尼基指数定义为:

$$\begin{equation}
Gini(p) = \sum_{k=1}^Kp_k(1-p_k)=1-\sum_{k=1}^Kp_k^2
\end{equation}$$

对于给定样本集合$D$，其尼基指数为：

$$\begin{equation}
Gini(D) =1-\sum_{k=1}^K\biggl(\frac{|C_k|}{|D|}\biggr)^2
\end{equation}$$

这里$C_k$是$D$中属于第$k$类的样本子集，$K$是类的个数。

如果样本集合$D$根据特征$A$是否取某一可能值$a$被分割成$D_1$和$D_2$两部分，即$D_1=\{(x,y)\in D|A(x)=a\},D_2=D-D_1$，则在特征$A$的条件下，集合$D$的尼基指数定义为：

$$\begin{equation}
Gini(D,A) =\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)
\end{equation}$$

$Gini(D)$表示集合$D$的不确定性，$Gini(D,A)$表示经过$A=a$分割后集合$D$的不确定性。尼基指数越大，样本集合的不确定性越大同熵。

决策树生成：

1.计算现有特征对数据集的尼基指数，对每一个特征$A$，对其可能取得每个值$a$,根据是否$A=a$将$D$切割成$D_1,D_2$,并计算$A=a$时的尼基指数。

2.在所有可能的特征$A$以及它们所有可能的切分点$a$中，选择尼基指数最小的特征及其对应的切分点最为最优特征与最优切分点。一最优特征和最优切分点，从现有节点生成两个子节点，将训练数据集以特征分配到两个子节点中。

对子节点递归做1,2操作直至满足停止条件，生成CART决策树。

# 集成化(Ensemble)

### Bagging(bootstrap aggregating)
首先，从大小为$N$的数据集$D$中，分别独立进行多次随机抽取$n$个数据，并使用每次抽取的数据训练弱分类器$C_i$。

然后，对新的数据进行分类时，首先将新的数据放进每一个分类器$C_i$中进行分类，得到每一个分类器对新数据的分类结果，并进行投票后获得优胜的结果。

### 随机森林(Random forest)
首先，从原始数据集从大小为$N$的数据集$D$中，有放回的抽取$N$个数据，同一条数据可以重复被抽取，假设进行$k$次，然后根据$k$个数据集构建$k$个决策树。

其次，设有$n$个特征，在每一棵树的每个节点随机抽取$m$个特征，并计算特征蕴含的信息量，并选择一个最具分类能力的特征进行节点分裂。每棵树最大限度的生长，不做任何剪裁。

最后，将生成的多棵树组成随机森林，用随机森林进行分类，分类结果桉树分类器投票多少而定。

### Boosting

假设训练集上一个有$n$个样例，并对每个样例赋上一个权重$W_i$,在训练起初每个点的权重相同；训练过程中提高在上次训练中分类错误的样例的权重，并降低分类正确样例的权重；在全部训练完成后得到$M$个模型，对$M$个模型的分类结果通过加权投票决定：

$$C(x)=sign\biggl[\sum_m^M\alpha_mC_m(x)\biggr]$$


## AdaBoost
###1.算法流程

设训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\},x_i\in \mathcal{X}\subseteq \mathbf{R}^n,x_i\in \mathcal{Y}=\{-1,+1\}$

(1)初始化时训练数据的权重

$$\begin{equation}
D_1=(w_{11},\cdots,w_{1i},\cdots,w_{1N}),w_{1i}=\frac{1}{N},i=1,2,\cdots,N
\end{equation}$$

(2)对$m=1,2,\cdots,M$，使用具有权重分布的$D_m$进行训练，得到基本分类器


$$\begin{equation}
G_m(x):\mathcal{X}\rightarrow \{-1,+1\}
\end{equation}$$


计算$G_m(x)$在训练数据集上的分类误差率：


$$\begin{equation}
e_m(x)=P(G_m(x_i)\neq y_i)=\sum_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)
\end{equation}$$

将$\sum\limits_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)$用$\sum\limits_{i=1\atop G_m(x_i)\neq y_i}^Nw_{mi}$表示更好理解，

$w_{mi}$表示第$m$轮中第$i$个实例的权重，$\sum\limits_{i=1}^Nw_{mi}=1$。计算$G_m(x)$的系数：

$$\begin{equation}
\alpha_m=\frac{1}{2}\ln \frac{1-e_m}{e_m}
\end{equation}$$

当$e_m\leqslant \frac{1}{2}$时，$\alpha_m\geqslant 0$，并且$\alpha_m$随着$e_m$的减小而增大，因此分类误差越小的基本分类器在最终分类器中的作用越大，更新训练数据集的权重分布：

$$\begin{equation}
D_{m+1}=(w_{m+1,1},\cdots,w_{m+1,i},\cdots,w_{m+1,N})
\end{equation}$$

$$\begin{equation}
w_{m+1,i}=\frac{w_{mi}}{Z_m}exp(-\alpha_my_iG_m(x_i)),i=1,2,\cdots,N
\end{equation}$$

当$y_i=G_m(x_i)$时$y_iG_m(x_i)=1$，因此被分类正确的样本权重在减小，而误分类的样本权重在增大。$Z_m$是规范因子：

$$\begin{equation}
Z_m=\sum_{i=1}^Nw_{mi}exp(-\alpha_my_iG_m(x_i))
\end{equation}$$


(3)构建基本分类器的线性组合

$$\begin{equation}
f(x)=\sum_{m=1}^M\alpha_mG_m(x)
\end{equation}$$

得到最终分类器：

$$\begin{equation}
G(x)=sign(f(x))=sign\biggl(\sum_{m=1}^M\alpha_mG_m(x)\biggr)
\end{equation}$$

###2.示例
给定训练样本：

$$\begin{array}{l|lcr}  n&1&2&3&4&5&6&7&8&9&10\\
\hline
x&0&1&2&3&4&5&6&7&8&9\\
y&1&1&1&-1&-1&-1&1&1&1&-1
\end{array}$$

初始化数据权值分布：

$$\begin{equation}
D_1=(w_{11},w_{12},\cdots,w_{110}),w_{1i}=0.1,i=1,2,\cdots,10
\end{equation}$$

##### 迭代过程1，$m=1$,

(a)在权值分布为$D_1$的训练数据上，阈值$v$取2.5时分类误差率最低，基本分类器为：

$$
\begin{equation}
G_1(x)=\begin{cases}1,&x<2.5\\-1,&x >  2.5\end{cases}
\end{equation}
$$

(b)$G_1(x)$在训练数据集上误差率$e_1=P(G_1(x_i)\neq y_i)=0.3$.

(c)计算$G_1(x)$系数：$\alpha_1=\frac{1}{2}\log\frac{1-e_1}{e_1}=0.4236$

(d)更新训练数据的权值分布：

$$\begin{equation}
D_2=(w_{21},w_{22},\cdots,w_{210}),w_{2i}=\frac{w_{1i}}{Z_1}\exp (\alpha_iy_iG_i(x_i))
\end{equation}$$

$$
D_2=(0.07143,0.07143,0.07143,0.07143,0.07143,0.07143,0.16667,0.16667,0.16667,0.07143)
$$

$f_1(x)=0.4236G_1(x)$，分类器$sign(f_1(x))$在训练集上有3个误分类点。

##### 迭代过程2，$m=2$,

(a)在权值分布为$D_2$的训练数据上，阈值$v$取8.5时分类误差率最低，基本分类器为：

$$
\begin{equation}
G_2(x)=\begin{cases}1,&x<8.5\\-1,&x >  8.5\end{cases}
\end{equation}
$$

(b)$G_2(x)$在训练数据集上误差率$e_2=0.2143$.

(c)计算$\alpha_2=0.6496$.

(d)更新训练数据的权值分布：

$$
D_3=(0.0455,0.0455,0.0455,0.16667,0.16667,0.16667,0.1060,0.1060,0.1060,0.0455)
$$

$f_2(x)=0.4236G_1(x)+0.6496G_2(x)$，分类器$sign(f_2(x))$在训练集上有3个误分类点。

##### 迭代过程3，$m=3$,

(a)在权值分布为$D_3$的训练数据上，阈值$v$取5.5时分类误差率最低，基本分类器为：

$$
\begin{equation}
G_3(x)=\begin{cases}1,&x<5.5\\-1,&x >  5.5\end{cases}
\end{equation}
$$

(b)$G_3(x)$在训练数据集上误差率$e_3=0.1820$.

(c)计算$\alpha_3=0.7514$.

(d)更新训练数据的权值分布：

$$
D_4=(0.125,0.125,0.125,0.102,0.102,0.102,0.065,0.065,0.065,0.125)
$$

$f_3(x)=0.4236G_1(x)+0.6496G_2(x)+0.7514G_3(x)$，分类器$sign(f_3(x))$在训练集上有0个误分类点。分类器最终为：

$$
\begin{equation}
G(x)=sign[f_3(x)]=sign[0.4236G_1(x)+0.6496G_2(x)+0.7514G_3(x)]
\end{equation}
$$

###3.AdaBoost训练误差分析

AdaBoost误差上界为：

$$
\begin{equation}
\frac{1}{N}\sum_{i=1}^NI(G(x_i) \neq y_i) \leq \frac{1}{N}\sum_i\exp (-y_if(x_i))=\prod_mZ_m
\end{equation}
$$

当$G(x_i) \neq y_i$时，$y_if(x_i)<0$,因此$\exp (-y_if(x_i))\geq 1$,前半部分得证。

后半部分，别忘了有：

$$\begin{equation}
w_{m+1,i}=\frac{w_{mi}}{Z_m}exp(-\alpha_my_iG_m(x_i))\Longrightarrow Z_mw_{m+1,i}=w_{mi}\exp(-\alpha_my_iG_m(x_i))
\end{equation}$$

推导如下：

$$
\begin{align*}
\frac{1}{N}\sum_i\exp (-y_if(x_i)) &=\frac{1}{N}\sum_i\exp \biggl(-\sum_{m=1}^M\alpha_my_iG_m(x_i)\biggr) \\
&=\sum_iw_{1i}\prod_{m=1}^M\exp (-\alpha_my_iG_m(x_i)) \\
&=Z_1\sum_iw_{2i}\prod_{m=2}^M\exp (-\alpha_my_iG_m(x_i)) \\
&=Z_1Z_2\sum_iw_{3i}\prod_{m=3}^M\exp (-\alpha_my_iG_m(x_i)) \\
&=Z_1Z_2\cdots Z_{M-1}\sum_iw_{Mi}\exp (-\alpha_my_iG_m(x_i)) \\
&=\prod_{m=1}^MZ_m
\end{align*} 
$$


因此我们可以在每一轮选取适当的$G_m$使得$Z_m$最小，从而使训练误差下降的最快。对于二分类问题，有如下结果：


$$\begin{equation}
\prod_{m=1}^MZ_m=\prod_{m=1}^M[2\sqrt{e_m(1-e_m)}]=\prod_{m=1}^M\sqrt{1-4\gamma_m^2}\leqslant \exp\biggl(-2\sum_{m=1}^M\gamma_m^2\biggr)
\end{equation}$$

其中$\gamma_m=\frac{1}{2}-e_m$.

##### 证明：
当$y_i=G_m(x_i)$时$y_iG_m(x_i)=1$,当$y_i\neq G_m(x_i)$时$y_iG_m(x_i)=-1$,$e_m=\sum\limits_{ G_m(x_i)\neq y_i}w_{mi}$,$\alpha_m=\frac{1}{2}\log \frac{1-e_m}{e_m}$.

$$
\begin{align*}
Z_m &=-\sum_{i=1}^Nw_{mi}\exp (-\alpha_my_iG_m(x_i)) \\
&=\sum_{y_i=G_m(x_i)}w_{mi}e^{-\alpha_m}+\sum_{y_i\neq G_m(x_i)}w_{mi}e^{\alpha_m}\\
&=(1-e_m)e^{-\alpha_m}+e_me^{\alpha_m} \\
&=(1-e_m)e^{-\frac{1}{2}\log\frac{1-e_m}{e_m}}+e_me^{\frac{1}{2}\log\frac{1-e_m}{e_m}} \\
&=(1-e_m)(e^{\log\frac{1-e_m}{e_m}})^{-\frac{1}{2}}+e_m(e^{\log\frac{1-e_m}{e_m}})^{\frac{1}{2}} \\
&=(1-e_m)\sqrt{\frac{e_m}{1-e_m}}+e_m\sqrt{\frac{1-e_m}{e_m}} \\
&= 2\sqrt{e_m(1-e_m)}\\
&=\sqrt{1-4\gamma_m^2}
\end{align*} 
$$

至于不等式：

$$\begin{equation}
\prod_{m=1}^M\sqrt{1-4\gamma_m^2}\leqslant \exp\biggl(-2\sum_{m=1}^M\gamma_m^2\biggr)
\end{equation}$$

可由$e^x$和$\sqrt{1-x}$在$x=0$处的泰勒展开推出$\sqrt{1-4\gamma_m^2}\leqslant \exp(-2\gamma_m^2)$,进而得到。

另外，如果存在$\gamma>0$,对所有$m$有$\gamma_m\geqslant \gamma$,则：

$$
\begin{equation}
\frac{1}{N}\sum_{i=1}^NI(G(x_i) \neq y_i) \leq \exp(-2M\gamma^2)
\end{equation}
$$

这个结论表明，AdaBoost的训练误差是以指数速率下降的。另外，AdaBoost算法不需要事先知道下界$\gamma$，AdaBoost具有自适应性，它能适应弱分类器各自的训练误差率。

###4.AdaBoost算法另一种解释
AdaBoost算法可以看做是模型为加法模型、损失函数为指数函数、学习方法为前向分步算法时的二分类学习方法。

#####4.1.前向分步算法

加法模型(additive model)如下：

$$
\begin{equation}
f(x)=\sum_{m=1}^M\beta_mb(x;\gamma_m)
\end{equation}
$$

其中$b(x;\gamma_m)$为基函数，$\gamma_m$为基函数参数，$\beta_m$为基函数系数。

在给定训练数据和损失函数$L(y,f(x))$的条件下，学习加法模型$f(x)$成为经验风险极小化即损失函数极小化问题：

$$
\begin{equation}
\min_{\beta_m,
\gamma_m}\sum_{i=1}^NL\biggl(y_i,\sum_{m=1}^M\beta_mb(x_i;\gamma_m)\biggr)
\end{equation}
$$

该问题可以作如此简化：从前向后，每步只学习一个基函数及其系数，逐步逼近上式，即每步只优化如下损失函数：

$$
\begin{equation}
\min_{\beta,
\gamma}\sum_{i=1}^NL\biggl(y_i,\beta b(x_i;\gamma)\biggr)
\end{equation}
$$

这就是向前分步算法。

#####向前分步算法流程

输入：训练数据集$T=\{(x_1,y_1),(x_2,y_),\cdots,(x_N,y_N)\}$;损失函数$L(y,f(x))$;基函数$\{b(x;\gamma)\}$。

输出：加法模型$f(x)$

(1)初始化$f_0(x)=0$

(2)对$m=1,2,\cdots,M$

(a)极小化损失函数

$$
\begin{equation}
(\beta_m,\gamma_m)=\arg \min_{\beta,
\gamma}\sum_{i=1}^NL\biggl(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma)\biggr)
\end{equation}
$$

得到参数$\beta_m,\gamma_m$

(2)更新

$$
\begin{equation}
f_m(x)=f_{m-1}(x)+\beta_mb(x;\gamma_m)
\end{equation}
$$

(3)得到加法模型

$$
\begin{equation}
f(x)=f_M(x)=\sum_{m=1}^M\beta_mb(x;\gamma_m)
\end{equation}
$$

前向分步算法将同时求解从$m=1$到$M$所有参数$\beta_m,\gamma_m$的优化问题简化为逐次求解各个$\beta_m,\gamma_m$的优化问题。

#####4.2.前向分步算法与AdaBoost
AdaBoost算法是前向分步加法算法的特例。其模型是由基本分类器组成的加法模型，其损失函数是指数函数。

AdaBoost的基本分类器为$G_m(x)$,其系数为$\alpha_m$,$m=1,2,\cdots,M$,AdaBoost的最终模型即最终的加法模型为：

$$
\begin{equation}
f(x)=\sum_{m=1}^M\alpha_mG_m(x)
\end{equation}
$$

前向分步算法逐一学习基函数的过程，与Adaboost算法逐一学习各个基本分类器的过程一致。

下面证明前向分步算法的损失函数是指数损失函数$L(y,f(x))=\exp[-yf(x)]$时，其学习的具体操作等价于AdaBoost算法学习的具体操作。

假设经过$m-1$轮迭代前向分步算法已经得到$f_{m-1}(x):$

$$
\begin{equation}
f_{m-1}(x)=f_{m-2}(x)+\alpha_{m-1}G_{m-1}(x)=\alpha_1G_1(x)+\cdots+\alpha_{m-1}G_{m-1}(x)
\end{equation}
$$

在第$m$轮迭代得到$\alpha_m,G_m(x)$和$f_m(x)$:

$$
\begin{equation}
f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)
\end{equation}
$$

目标是使前向分步算法得到的$\alpha_m$和$G_m(x)$使$f_m(x)$在训练数据集$T$上的指数损失最小，即

$$
\begin{equation}
(\alpha_m,G_m(x))=\arg\min_{\alpha,G}\sum_{i=1}^N\exp[-y_i(f_{m-1}(x_i)+\alpha G(x_i))]
\end{equation}
$$

假定$G_1(x),\cdots,G_{m-1}(x)$和$\alpha_1,\cdots,\alpha_{m-1}(x)$为已知参数，现在求解$G_m(x),\alpha_m$,并令$\overline{w}_{mi}=\exp[-y_i(f_{m-1}(x_i)]$,$\overline{w}_{mi}$与$\alpha,G$都无关，所以与最小化无关，$\overline{w}_{mi}$只依赖于与$f_{m-1}(x)$，并随着每一轮迭代而发生改变，于是上式可以表示为

$$
\begin{equation}
(\alpha_m,G_m(x))=\arg\min_{\alpha,G}\sum_{i=1}^N\overline{w}_{mi}\exp[-y_i\alpha G(x_i))]
\end{equation}
$$

接下来，便是要证使得上式达到最小的$\alpha_m^*$和$G_m^*(x)$就是Adaboost算法所求解得到的$\alpha_m$和$G_m(x)$。

接下来先求$G_m^*(x)$再求$\alpha_m^*$，对任意$\alpha>0$,使上式$(\alpha_m,G_m(x))$最小的$G(x)$由下式得到：

$$
\begin{equation}
G_m^*(x)=\arg\min_G\sum_{i=1}^N\overline{w}_{mi}I(y_i\neq G(x_i))
\end{equation}
$$

其中$\overline{w}_{mi}=\exp[-y_i(f_{m-1}(x_i)]$。AdaBoost算法中的误差率$e_m$为：

$$\begin{equation}
e_m(x)=P(G_m(x_i)\neq y_i)=\sum_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)
\end{equation}$$

$G_m^*(x)$即为AdaBoost算法中所求的$G_m(x)$,它是在第$m$轮加权训练数据时，使分类误差率最小的基本分类器;在Adaboost算法的每一轮迭代中，都是选取让误差率最低的阈值来设计基本分类器。

之后求$\alpha_m^*$，式$(\alpha_m,G_m(x))$后半部分为：

$$
\begin{align*}
\sum_{i=1}^N\overline{w}_{mi}\exp[-y_i\alpha G(x_i))] &=\sum_{y_i=G_m(x_i)}\overline{w}_{mi}e^{-\alpha}+\sum_{y_i\neq G_m(x_i)}\overline{w}_{mi}e^{\alpha} \\
&=(e^{\alpha}-e^{-\alpha})\sum_{i=1}^N\overline{w}_{mi}I(y_i\neq G(x_i))+e^{-\alpha}\sum_{i=1}^N\overline{w}_{mi}
\end{align*} 
$$

将$G_m^*$代入，并对$\alpha$求导，使导数等于0：

$$
\begin{align*}
&\frac{\partial\biggl((e^{\alpha}-e^{-\alpha})\sum\limits_{i=1}^N\overline{w}_{mi}I(y_i\neq G_m(x_i))+e^{-\alpha}\sum\limits_{i=1}^N\overline{w}_{mi}\biggr)}{\partial\alpha}\\ &=(e^{\alpha}+e^{-\alpha})\sum\limits_{i=1}^N\overline{w}_{mi}I(y_i\neq G_m(x_i))-e^{-\alpha}\sum\limits_{i=1}^N\overline{w}_{mi} \\
&=0
\end{align*} 
$$

即，

$$\begin{equation}
e^{2\alpha}+1=\frac{\sum\limits_{i=1}^N\overline{w}_{mi}}{\sum\limits_{i=1}^N\overline{w}_{mi}I(y_i\neq G_m(x_i))}=\frac{1}{\sum\limits_{i=1}^Nw_{mi}I(y_i\neq G_m(x_i))}=\frac{1}{e_m}
\end{equation}$$

可得：

$$\begin{equation}
\alpha_m^*=\frac{1}{2}\log\frac{1-e_m}{e_m}
\end{equation}$$

这里的$\alpha_m^*$与AdaBoost算法的$\alpha_m$完全一致。

最后看每一轮样本的权值更新，由$f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)$以及$\overline{w}_{mi}=\exp[-y_i(f_{m-1}(x_i)]$，可得：

$$\begin{equation}
\overline{w}_{m+1,i}=\exp[-y_i(f_m(x_i)]=\exp[-y_i(f_{m-1}(x)+\alpha_mG_m(x))]=\exp(-y_if_{m-1}(x))\exp(-y_i\alpha_mG_m(x))
\end{equation}$$

可得，$\overline{w}_{m+1,i}=\overline{w}_{m,i}\exp(-y_i\alpha_mG_m(x))$,这与AdaBoost算法中$w_{m+1,i}$

$$\begin{equation}
w_{m+1,i}=\frac{w_{mi}}{Z_m}\exp(-\alpha_my_iG_m(x_i)),i=1,2,\cdots,N
\end{equation}$$

只差规范因子$Z_m$：

$$\begin{equation}
Z_m=\sum_{i=1}^Nw_{mi}\exp(-\alpha_my_iG_m(x_i))
\end{equation}$$

因而二者等价。

##GBDT(Gradient Boosting Decision Tree)

###1.提升树

提升树模型实际采用加法模型（即基函数的线性组合）与前向分步算法，以决策树为基函数的提升方法称为提升树（Boosting Tree）。提升树模型可以表示为决策树的加法模型：

$$\begin{equation}
f_M(x)=\sum_{m=1}^MT(x;\Theta_m)
\end{equation}$$

其中，$T(x;\Theta_m)$表示决策树；$\Theta_m$为决策树的参数；$M$为树的个数。

#####1.1提升树算法
提升树算法采用前向分步算法。首先确定初始提升树$f_0(x)=0$,第$m$步的模型是：

$$\begin{equation}
f_m(x)=f_{m-1}(x)+T(x;\Theta_m)
\end{equation}$$

其中，$f_{m-1}(x)$为当前模型，通过经验风险极小化确定下一棵决策树的参数$\Theta_m$

$$\begin{equation}
\hat{\Theta}_m=\arg\min_{\Theta_m}\sum_{i=1}^NL(y_i,f_{m-1}(x_i)+T(x_i;\Theta_m))
\end{equation}$$

由于树的线性组合可以很好的拟合训练数据，即使数据中的输入和输出之间的关系很复杂也是如此，所以提升树是一个高功能的学习算法。

对于二分类问题，提升树算法只需将AdaBoost算法中的基本分类器限定为二分类树即可 。这里不再讨论，下边主要讨论回归问题的提升树。

已知训练集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$,$x_i\in \mathcal{X}\subseteq \mathbf{R}^n$,$\mathcal{X}$为输入空间，$y\in \mathcal{Y}\subseteq \mathbf{R}$,$\mathbf{R}$为输入空间。将输入空间\mathcal{X}划分为$J$个互不相交的区域$R_1,R_2,\cdots,R_J$，并且每个区域上确定输出的常量$c_j$,那么树可表示为

$$\begin{equation}
T(x;\Theta)=\sum_{j=1}^Jc_jI(x\in R_j)
\end{equation}$$

其中，参数$\Theta=\{(R_1,c_1),(R_2,c_2),\cdots,(R_J,c_J)\}$表示树的区域划分和各区域上的常数。$J$是回归树的复杂度即叶节点个数。