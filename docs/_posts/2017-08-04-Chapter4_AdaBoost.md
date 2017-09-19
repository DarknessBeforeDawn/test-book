---
title: AdaBoost
layout: post
share: false
---

## 1.算法流程

设训练数据集$$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\},x_i\in \mathcal{X}\subseteq \mathbf{R}^n,x_i\in \mathcal{Y}=\{-1,+1\}$$

(1)初始化时训练数据的权重

$$\begin{equation}
D_1=(w_{11},\cdots,w_{1i},\cdots,w_{1N}),w_{1i}=\frac{1}{N},i=1,2,\cdots,N
\end{equation}$$

(2)对$$m=1,2,\cdots,M$$，使用具有权重分布的$$D_m$$进行训练，得到基本分类器


$$\begin{equation}
G_m(x):\mathcal{X}\rightarrow \{-1,+1\}
\end{equation}$$


计算$$G_m(x)$$在训练数据集上的分类误差率：


$$\begin{equation}
e_m(x)=P(G_m(x_i)\neq y_i)=\sum_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)
\end{equation}$$

将$$\sum\limits_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)$$用$$\sum\limits_{i=1\atop G_m(x_i)\neq y_i}^Nw_{mi}$$表示更好理解，

$$w_{mi}$$表示第$$m$$轮中第$$i$$个实例的权重，$$\sum\limits_{i=1}^Nw_{mi}=1$$。计算$$G_m(x)$$的系数：

$$\begin{equation}
\alpha_m=\frac{1}{2}\ln \frac{1-e_m}{e_m}
\end{equation}$$

当$$e_m\leqslant \frac{1}{2}$$时，$$\alpha_m\geqslant 0$$，并且$$\alpha_m$$随着$$e_m$$的减小而增大，因此分类误差越小的基本分类器在最终分类器中的作用越大，更新训练数据集的权重分布：

$$\begin{equation}
D_{m+1}=(w_{m+1,1},\cdots,w_{m+1,i},\cdots,w_{m+1,N})
\end{equation}$$

$$\begin{equation}
w_{m+1,i}=\frac{w_{mi}}{Z_m}exp(-\alpha_my_iG_m(x_i)),i=1,2,\cdots,N
\end{equation}$$

当$$y_i=G_m(x_i)$$时$$y_iG_m(x_i)=1$$，因此被分类正确的样本权重在减小，而误分类的样本权重在增大。$$Z_m$$是规范因子：

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

## 2.示例
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

#### 迭代过程1，$$m=1$$,

(a)在权值分布为$$D_1$$的训练数据上，阈值$$v$$取2.5时分类误差率最低，基本分类器为：

$$
\begin{equation}
G_1(x)=\begin{cases}1,&x<2.5\\-1,&x >  2.5\end{cases}
\end{equation}
$$

(b)$$G_1(x)$$在训练数据集上误差率$$e_1=P(G_1(x_i)\neq y_i)=0.3$$.

(c)计算$$G_1(x)$$系数：$$\alpha_1=\frac{1}{2}\log\frac{1-e_1}{e_1}=0.4236$$

(d)更新训练数据的权值分布：

$$\begin{equation}
D_2=(w_{21},w_{22},\cdots,w_{210}),w_{2i}=\frac{w_{1i}}{Z_1}\exp (\alpha_iy_iG_i(x_i))
\end{equation}$$

$$
D_2=(0.07143,0.07143,0.07143,0.07143,0.07143,0.07143,0.16667,0.16667,0.16667,0.07143)
$$

$$f_1(x)=0.4236G_1(x)$$，分类器$$sign(f_1(x))$$在训练集上有3个误分类点。

#### 迭代过程2，$$m=2$$,

(a)在权值分布为$$D_2$$的训练数据上，阈值$$v$$取8.5时分类误差率最低，基本分类器为：

$$
\begin{equation}
G_2(x)=\begin{cases}1,&x<8.5\\-1,&x >  8.5\end{cases}
\end{equation}
$$

(b)$$G_2(x)$$在训练数据集上误差率$$e_2=0.2143$$.

(c)计算$$\alpha_2=0.6496$$.

(d)更新训练数据的权值分布：

$$
D_3=(0.0455,0.0455,0.0455,0.16667,0.16667,0.16667,0.1060,0.1060,0.1060,0.0455)
$$

$$f_2(x)=0.4236G_1(x)+0.6496G_2(x)$$，分类器$$sign(f_2(x))$$在训练集上有3个误分类点。

#### 迭代过程3，$$m=3$$,

(a)在权值分布为$$D_3$$的训练数据上，阈值$$v$$取5.5时分类误差率最低，基本分类器为：

$$
\begin{equation}
G_3(x)=\begin{cases}1,&x<5.5\\-1,&x >  5.5\end{cases}
\end{equation}
$$

(b)$$G_3(x)$$在训练数据集上误差率$$e_3=0.1820$$.

(c)计算$$\alpha_3=0.7514$$.

(d)更新训练数据的权值分布：

$$
D_4=(0.125,0.125,0.125,0.102,0.102,0.102,0.065,0.065,0.065,0.125)
$$

$$f_3(x)=0.4236G_1(x)+0.6496G_2(x)+0.7514G_3(x)$$，分类器$$sign(f_3(x))$$在训练集上有0个误分类点。分类器最终为：

$$
\begin{equation}
G(x)=sign[f_3(x)]=sign[0.4236G_1(x)+0.6496G_2(x)+0.7514G_3(x)]
\end{equation}
$$

## 3.AdaBoost训练误差分析

AdaBoost误差上界为：

$$
\begin{equation}
\frac{1}{N}\sum_{i=1}^NI(G(x_i) \neq y_i) \leq \frac{1}{N}\sum_i\exp (-y_if(x_i))=\prod_mZ_m
\end{equation}
$$

当$$G(x_i) \neq y_i$$时，$$y_if(x_i)<0$$,因此$$\exp (-y_if(x_i))\geq 1$$,前半部分得证。

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


因此我们可以在每一轮选取适当的$$G_m$$使得$$Z_m$$最小，从而使训练误差下降的最快。对于二分类问题，有如下结果：


$$\begin{equation}
\prod_{m=1}^MZ_m=\prod_{m=1}^M[2\sqrt{e_m(1-e_m)}]=\prod_{m=1}^M\sqrt{1-4\gamma_m^2}\leqslant \exp\biggl(-2\sum_{m=1}^M\gamma_m^2\biggr)
\end{equation}$$

其中$$\gamma_m=\frac{1}{2}-e_m$$.

#### 证明：
当$$y_i=G_m(x_i)$$时$$y_iG_m(x_i)=1$$,当$$y_i\neq G_m(x_i)$$时$$y_iG_m(x_i)=-1$$,$$e_m=\sum\limits_{ G_m(x_i)\neq y_i}w_{mi}$$,$$\alpha_m=\frac{1}{2}\log \frac{1-e_m}{e_m}$$.

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

可由$$e^x$$和$$\sqrt{1-x}$$在$$x=0$$处的泰勒展开推出$$\sqrt{1-4\gamma_m^2}\leqslant \exp(-2\gamma_m^2)$$,进而得到。

另外，如果存在$$\gamma>0$$,对所有$$m$$有$$\gamma_m\geqslant \gamma$$,则：

$$
\begin{equation}
\frac{1}{N}\sum_{i=1}^NI(G(x_i) \neq y_i) \leq \exp(-2M\gamma^2)
\end{equation}
$$

这个结论表明，AdaBoost的训练误差是以指数速率下降的。另外，AdaBoost算法不需要事先知道下界$$\gamma$$，AdaBoost具有自适应性，它能适应弱分类器各自的训练误差率。

## 4.AdaBoost算法另一种解释
AdaBoost算法可以看做是模型为加法模型、损失函数为指数函数、学习方法为前向分步算法时的二分类学习方法。

### 4.1.前向分步算法

加法模型(additive model)如下：

$$
\begin{equation}
f(x)=\sum_{m=1}^M\beta_mb(x;\gamma_m)
\end{equation}
$$

其中$$b(x;\gamma_m)$$为基函数，$$\gamma_m$$为基函数参数，$$\beta_m$$为基函数系数。

在给定训练数据和损失函数$$L(y,f(x))$$的条件下，学习加法模型$$f(x)$$成为经验风险极小化即损失函数极小化问题：

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

##### 向前分步算法流程

输入：训练数据集$$T=\{(x_1,y_1),(x_2,y_),\cdots,(x_N,y_N)\}$$;损失函数$$L(y,f(x))$$;基函数$$\{b(x;\gamma)\}$$。

输出：加法模型$$f(x)$$

(1)初始化$$f_0(x)=0$$

(2)对$$m=1,2,\cdots,M$$

(a)极小化损失函数

$$
\begin{equation}
(\beta_m,\gamma_m)=\arg \min_{\beta,
\gamma}\sum_{i=1}^NL\biggl(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma)\biggr)
\end{equation}
$$

得到参数$$\beta_m,\gamma_m$$

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

前向分步算法将同时求解从$$m=1$$到$$M$$所有参数$$\beta_m,\gamma_m$$的优化问题简化为逐次求解各个$$\beta_m,\gamma_m$$的优化问题。

### 4.2.前向分步算法与AdaBoost
AdaBoost算法是前向分步加法算法的特例。其模型是由基本分类器组成的加法模型，其损失函数是指数函数。

AdaBoost的基本分类器为$$G_m(x)$$,其系数为$$\alpha_m$$,$$m=1,2,\cdots,M$$,AdaBoost的最终模型即最终的加法模型为：

$$
\begin{equation}
f(x)=\sum_{m=1}^M\alpha_mG_m(x)
\end{equation}
$$

前向分步算法逐一学习基函数的过程，与Adaboost算法逐一学习各个基本分类器的过程一致。

下面证明前向分步算法的损失函数是指数损失函数$$L(y,f(x))=\exp[-yf(x)]$$时，其学习的具体操作等价于AdaBoost算法学习的具体操作。

假设经过$$m-1$$轮迭代前向分步算法已经得到$$f_{m-1}(x):$$

$$
\begin{equation}
f_{m-1}(x)=f_{m-2}(x)+\alpha_{m-1}G_{m-1}(x)=\alpha_1G_1(x)+\cdots+\alpha_{m-1}G_{m-1}(x)
\end{equation}
$$

在第$$m$$轮迭代得到$$\alpha_m,G_m(x)$$和$$f_m(x)$$:

$$
\begin{equation}
f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)
\end{equation}
$$

目标是使前向分步算法得到的$$\alpha_m$$和$$G_m(x)$$使$$f_m(x)$$在训练数据集$$T$$上的指数损失最小，即

$$
\begin{equation}
(\alpha_m,G_m(x))=\arg\min_{\alpha,G}\sum_{i=1}^N\exp[-y_i(f_{m-1}(x_i)+\alpha G(x_i))]
\end{equation}
$$

假定$$G_1(x),\cdots,G_{m-1}(x)$$和$$\alpha_1,\cdots,\alpha_{m-1}(x)$$为已知参数，现在求解$$G_m(x),\alpha_m$$,并令$$\overline{w}_{mi}=\exp[-y_i(f_{m-1}(x_i)]$$,$$\overline{w}_{mi}$$与$$\alpha,G$$都无关，所以与最小化无关，$$\overline{w}_{mi}$$只依赖于与$$f_{m-1}(x)$$，并随着每一轮迭代而发生改变，于是上式可以表示为

$$
\begin{equation}
(\alpha_m,G_m(x))=\arg\min_{\alpha,G}\sum_{i=1}^N\overline{w}_{mi}\exp[-y_i\alpha G(x_i))]
\end{equation}
$$

接下来，便是要证使得上式达到最小的$$\alpha_m^*$$和$$G_m^*(x)$$就是Adaboost算法所求解得到的$$\alpha_m$$和$$G_m(x)$$。

接下来先求$$G_m^*(x)$$再求$$\alpha_m^*$$，对任意$$\alpha>0$$,使上式$$(\alpha_m,G_m(x))$$最小的$$G(x)$$由下式得到：

$$
\begin{equation}
G_m^*(x)=\arg\min_G\sum_{i=1}^N\overline{w}_{mi}I(y_i\neq G(x_i))
\end{equation}
$$

其中$$\overline{w}_{mi}=\exp[-y_i(f_{m-1}(x_i)]$$。AdaBoost算法中的误差率$$e_m$$为：

$$\begin{equation}
e_m(x)=P(G_m(x_i)\neq y_i)=\sum_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)
\end{equation}$$

$$G_m^*(x)$$即为AdaBoost算法中所求的$$G_m(x)$$,它是在第$$m$$轮加权训练数据时，使分类误差率最小的基本分类器;在Adaboost算法的每一轮迭代中，都是选取让误差率最低的阈值来设计基本分类器。

之后求$$\alpha_m^*$$，式$$(\alpha_m,G_m(x))$$后半部分为：

$$
\begin{align*}
\sum_{i=1}^N\overline{w}_{mi}\exp[-y_i\alpha G(x_i))] &=\sum_{y_i=G_m(x_i)}\overline{w}_{mi}e^{-\alpha}+\sum_{y_i\neq G_m(x_i)}\overline{w}_{mi}e^{\alpha} \\
&=(e^{\alpha}-e^{-\alpha})\sum_{i=1}^N\overline{w}_{mi}I(y_i\neq G(x_i))+e^{-\alpha}\sum_{i=1}^N\overline{w}_{mi}
\end{align*}
$$

将$$G_m^*$$代入，并对$$\alpha$$求导，使导数等于0：

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

这里的$$\alpha_m^*$$与AdaBoost算法的$$\alpha_m$$完全一致。

最后看每一轮样本的权值更新，由$$f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)$$以及$$\overline{w}_{mi}=\exp[-y_i(f_{m-1}(x_i)]$$，可得：

$$\begin{equation}
\overline{w}_{m+1,i}=\exp[-y_i(f_m(x_i)]=\exp[-y_i(f_{m-1}(x)+\alpha_mG_m(x))]=\exp(-y_if_{m-1}(x))\exp(-y_i\alpha_mG_m(x))
\end{equation}$$

可得，$$\overline{w}_{m+1,i}=\overline{w}_{m,i}\exp(-y_i\alpha_mG_m(x))$$,这与AdaBoost算法中$$w_{m+1,i}$$

$$\begin{equation}
w_{m+1,i}=\frac{w_{mi}}{Z_m}\exp(-\alpha_my_iG_m(x_i)),i=1,2,\cdots,N
\end{equation}$$

只差规范因子$$Z_m$$：

$$\begin{equation}
Z_m=\sum_{i=1}^Nw_{mi}\exp(-\alpha_my_iG_m(x_i))
\end{equation}$$

因而二者等价。