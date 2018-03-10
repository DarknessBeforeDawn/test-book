---
title: 贝叶斯分类器
layout: post
share: false
---
# 1.贝叶斯公式

#### 条件概率公式

设 $A$ , $B$ 是两个事件，且 $P(B)>0$ ,则在事件 $B$ 发生的条件下，事件 $A$ 发生的条件概率(conditional probability)为：

$$P(A|B) = \frac{P(AB)}{P(B)}$$

#### 全概率公式

如果事件 $B_1,B_2,\cdots,B_n$ 满足 $B_i\cap B_j = \emptyset,i\neq j~~~~~i,j=1,2,\cdots,n$ , 且 $P(B_i)>0,B_1\cup B_2\cup\cdots B_n=\Omega$ ，设 $A$ 为任意事件，则有:

$$P(A)=\sum_{i=1}^nP(B_i)P(A|B_i)$$

上式即为全概率公式(formula of total probability)

全概率公式的意义在于，当直接计算 $P(A)$ 较为困难,而 $P(B_i),P(A$ \| $Bi)$ 的计算较为简单时，可以利用全概率公式计算 $P(A)$。思想就是，将事件 $A$ 分解成几个小事件，通过求小事件的概率，然后相加从而求得事件A的概率，而将事件A进行分割的时候，不是直接对 $A$ 进行分割，而是先找到样本空间 $\Omega$ 的一个个划分 $B_1,B_2,\cdots,B_n$ ,这样事件 $A$ 就被事件 $AB_1,AB_2,\cdots,AB_n$ 分解成了 $n$ 部分，而每一个 $B_i$ 发生都可能导致 $A$ 发生相应的概率是 

$$P(A|B_i)$$

#### 贝叶斯公式

与全概率公式解决的问题相反，贝叶斯公式是建立在条件概率的基础上寻找事件发生的原因（事件 $A$ 已经发生的条件下，分割中的小事件 $B_i$ 的概率），设 $B_1,B_2,\cdots,B_n$ 是样本空间 $\Omega$ 的一个划分，则对任一事件 $A(P(A)>0)$ ,有

$$P(B_i|A)=\frac{P(B_i)P(A|B_i)}{\sum\limits_{j=1}^nP(B_j)P(A|B_j)}$$

上式即为贝叶斯公式(Bayes formula)， $B_i$ 常被视为导致试验结果 $A$ 发生的\原因， $P(B_i)$ 表示各种原因发生的可能性大小，故称先验概率； $P(Bi$ \| $A)$ 则反映当试验产生了结果 $A$ 之后，再对各种原因概率的新认识，故称后验概率。

# 2.朴素贝叶斯法

## 2.1 基本方法

设输入空间 $\mathcal{X}\subseteq \mathbf{R^n}$ 为 $n$ 维向量的集合，输出空间为类标记集合$$\mathcal{Y}=\{c_1,c_2,\cdots,c_K\}$$.输入为特征向量 $x\in\mathcal{X}$ ,输出为类标记(class label) $y\in\mathcal{Y}$ . $X$ 是定义在输入空间 $\mathcal{X}$ 上的随机向量， $Y$ 是定义在输出空间 $\mathcal{Y}$ 上的随机变量。 $P(X,Y)$ 是 $X$ 和 $Y$ 的联合概率分布。训练数据集

$$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$$

由 $P(X,Y)$ 独立同分布产生。

朴素贝叶斯法通过训练数据集学习联合概率分布 $P(X,Y)$ .具体地，学习以下先验概率分布及条件概率分布。先验概率分布

$$P(Y=c_k),~~~k=1,2,\cdots,K$$

条件概率分布

$$P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},\cdots,X^{(n)}=x^{(n)}|Y=c_k),~~~k=1,2,\cdots,K$$

于是学习到联合概率分布 $P(X,Y)$ .

条件概率分布 $P(X=x|Y=c_k)$ 有指数级数量的参数，其估计实际不可行的。事实上，假设 $x^{(j)}$ 可取值有 $S_j$ 个， $j=1,2,\cdots,n,~~Y$ 可取值有 $K$ 个，那么参数个数为 $K\prod\limits_{j=1}nS_j$ .

朴素贝叶斯法对条件概率分布做了条件独立性的假设。由于这是一个较强的假设，朴素贝叶斯也由此得名。条件独立的假设是

$$P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},\cdots,X^{(n)}=x^{(n)}|Y=c_k)=\prod_{j=1}^nP(X^{(j)}=x^{(j)}|Y=c_k)$$

朴素贝叶斯是生成模型，它学习到的是生成数据的机制。条件独立假设等价于用于分类的特征在类确定的条件下都是条件独立的。这个假设使朴素贝叶斯法变得简单，但有时会使分类的准确率降低。

朴素贝叶斯法分类时，对给定的输入 $x$ ,通过学习到的模型计算后验概率分布 

$$P(Y=c_k|X=x)$$ 

将后验概率最大的类作为 $x$ 的类输出，后验概率计算根据贝叶斯定理进行

$$P(Y=c_k|X=x)=\frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_kP(X=x|Y=c_k)P(Y=c_k)}$$

根据条件独立性假设可得

$$P(Y=c_k|X=x)=\frac{P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_kP(Y=c_k)P(X^{(j)}=x^{(j)}|Y=c_k)}$$

这便是朴素贝叶斯的基本公式，朴素贝叶斯分类器可表示为

$$y=f(x)\arg\max_{c_k}\frac{P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_kP(Y=c_k)P(X^{(j)}=x^{(j)}|Y=c_k)}$$

其中分母对所有 $c_k$ 都相同，所以上述公式等价于

$$y=f(x)\arg\max_{c_k}P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)$$

## 2.2 后验概率最大化含义

朴素贝叶斯法将实例分到后验概率最大的类中，这等价于期望风险最小化。假设选择0-1损失函数

$$L(Y,f(X))=\left
\{
\begin{aligned}  
1, \ \ \ Y\neq f(X)  \\
0, \ \ \ Y= f(X) 
\end{aligned}
\right.$$


其中 $f(X)$ 是分类决策函数，则期望风险函数为

$$R_{\exp}(f)=E[L(X,f(X))]$$

期望是对联合分布 $P(X,Y)$ 取。由此取条件期望

$$R_{\exp}(f)=E_X\sum_{k=1}^K[L(c_k,f(X))]P(c_k|X)$$

为了使期望风险最小化，只需对 $X=x$ 逐个极小化，由此得到:

$$\begin{align}
f(x) &= \arg\min_{y\in \mathcal{Y}}\sum_{k=1}^KL(c_k,y)P(c_k|X=x)   \\
 &= \arg\min_{y\in \mathcal{Y}}\sum_{k=1}^KP(y\neq c_k|X=x)\\
&= \arg\min_{y\in \mathcal{Y}}(1-P(y= c_k|X=x))  \\
 &= \arg\max_{y\in \mathcal{Y}}P(y= c_k|X=x) \\
\end{align} $$

由以上推导就得到了后验概率最大化准则

$$f(x)=\arg\max_{c_k}P(c_k|X=x)$$

## 2.3 朴素贝叶斯参数估计

### 2.3.1 极大似然估计

在朴素贝叶斯法中，学习意味着估计 

$$P(Y=c_k),P(X^{(j)}=x^{(j)}|Y=c_k)$$

可以应用极大似然估计法估计相应概率，则

$$P(Y=c_k)=\frac{\sum\limits_{i=1}^NI(y_i=c_k)}{N},k=1,2,\cdots,K$$

设第 $j$ 个特征 $x^{(j)}$ 可能取值的集合为 

$$\{a_{j1},a_{j2},\cdots,a_{jS_j}\}$$

条件概率的极大似然估计

$$P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum\limits_{i=1}^NI(x^{(j)}_i=a_{jl},y_i=c_k)}{\sum\limits_{i=1}^NI(y_i=c_k)}$$

$$j=1,2,\cdots,n; ~~~l=1,2,\cdots,S_j;~~~k=1,2,\cdots,K$$

式中， $x_i^{(j)}$ 是第 $i$ 个样本的第 $j$ 个特征； $a_{jl}$ 是第 $j$ 个特征可取的第 $l$ 个值； $I$ 为指示函数。

### 2.3.2 贝叶斯估计

极大似然估计可能会出现所要估计的概率为0的情况。这时会影响到后验概率的计算结果，使分类产生偏差。使用贝叶斯估计可解决这一问题,如下：

$$P_\lambda(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum\limits_{i=1}^NI(x^{(j)}_i=a_{jl},y_i=c_k)+\lambda}{\sum\limits_{i=1}^NI(y_i=c_k)+S_j\lambda}$$

其中 $\lambda \geq 0$ .等价于在随机变量各个取值的频数上赋予一个正数 $\lambda>0$ .当 $\lambda=0$ 时就是极大似然估计。常取 $\lambda=1$ ，这时称为拉普拉斯平滑(Laplace smoothing).显然，对任何 $l=1,2,\cdots,S_j;~~~k=1,2,\cdots,K$ 有

$$P_\lambda(X^{(j)}=a_{jl}|Y=c_k)>0$$

$$\sum_{l=1}^{S_j}P_\lambda(X^{(j)}=a_{jl}|Y=c_k)=1$$

先验概率的贝叶斯估计是

$$P_\lambda(Y=c_k)=\frac{\sum\limits_{i=1}^NI(y_i=c_k)+\lambda}{N+K\lambda},k=1,2,\cdots,K$$

# 3.贝叶斯网路

## 3.1 定义

贝叶斯网络(Bayesian network)，又称信念网络(Belief Network)，或有向无环图模型(directed acyclic graphical model)，是一种概率图模型，于1985年由Judea Pearl首先提出。它是一种模拟人类推理过程中因果关系的不确定性处理模型，其网络拓朴结构是一个有向无环图(DAG)。 

贝叶斯网络的有向无环图中的节点表示随机变量$$\{X_1,X_2,\cdots,X_n\}$$，它们可以是可观察到的变量，或隐变量、未知参数等。将有因果关系（或非条件独立）的变量或命题用箭头来连接。若两个节点间以一个单箭头连接在一起，表示其中一个节点是“因(parents)”，另一个是“果(children)”，两节点就会产生一个条件概率值。连接两个节点的箭头代表此两个随机变量是具有因果关系，或非条件独立。

假设节点 $E$ 直接影响到节点 $H$ ，即 $E\rightarrow H$ ，则用从 $E$ 指向 $H$ 的箭头建立结点 $E$ 到结点 $H$ 的有向弧 $(E,H)$ ，权值(即连接强度)用条件概率 $P(H$ \| $E)$ 来表示，如下图所示：

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/Bayes1.png"/>
</center>

把某个研究系统中涉及的随机变量，根据是否条件独立绘制在一个有向图中，就形成了贝叶斯网络。其主要用来描述随机变量之间的条件依赖，用圈表示随机变量(random variables)，用箭头表示条件依赖(conditional dependencies)。

令 $G = (I,E)$ 表示一个有向无环图(DAG)，其中 $I$ 代表图形中所有的节点的集合，而 $E$ 代表有向连接线段的集合，且令 $x = (x_i),i \in I$ 为其有向无环图中的某一节点 $i$ 所代表的随机变量，若节点 $x$ 的联合概率可以表示成：

$$P(x)=\prod_{i\in }P(x_i|\pi_i)$$

则称 $x$ 为相对于一有向无环图 $G$ 的贝叶斯网络，其中， $\pi_i$ 表示节点 $i$ 之“因”，或称 $\pi_i$ 是 $i$ 的父节点集。

于任意的随机变量，其联合概率可由各自的局部条件概率分布相乘而得出：

$$P(x_1,\cdots,x_K) = P(x_K|x_1,\cdots,x_{K-1})\cdots P(x_2|x_1)P(x_1)$$

## 3.2 结构

给定如下图所示的一个贝叶斯网络：

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/Bayes2.png"/>
</center>

则其联合分布为:

$$P(x_1)P(x_2)P(x_3)P(x_4|x_1,x_2,x_3)P(x_5|x_1,x_3)P(x_6|x_4)P(x_7|x_4,x_5)$$

贝叶斯网络中三个变量之间的典型依赖关系有：同父结构(tail-to-tail),V型结构(head-to-head),顺序结构(head-to-tail).

#### tail-to-tail

如下图为tail-to-tail类型:

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/Bayes3.png"/>
</center>

考虑 $c$ 未知，跟 $c$ 已知这两种情况：

在 $c$ 未知的时候，有：

$$P(a,b,c)=P(c)P(a|c)P(b|c)$$

此时，没法得出 $P(a,b) = P(a)P(b)$ ，即 $c$ 未知时，$a,b$ 不独立。

在 $c$ 已知的时候，有：

$$P(a,b|c)=\frac{P(a,b,c)}{P(c)}=\frac{P(c)P(a|c)P(b|c)}{P(c)}=P(a|c)P(b|c)$$

 $c$ 已知时，$a,b$ 独立。

所以，在 $c$ 给定的条件下，$a,b$ 被阻断(blocked)，是独立的，称之为tail-to-tail条件独立，对应本节中最开始那张图中的“ $x_6$ 和 $x_7$ 在 $x_4$ 给定的条件下独立”。

#### head-to-head

如下图为head-to-head类型:

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/Bayes4.png"/>
</center>

联合分布为:

$$P(a,b,c) = P(a)P(b)P(c|a,b)$$

有:

$$\sum_cP(a,b,c)=\sum_cP(a)P(b)P(c|a,b)\Rightarrow P(a,b) = P(a)P(b)$$

即在 $c$ 未知的条件下， $a,b$ 被阻断(blocked)，是独立的，称之为head-to-head条件独立，对应本节中最开始那张图中的“ $x_1,x_2$ 独立”。

#### head-to-tail

如下图为head-to-tail类型:

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/Bayes5.png"/>
</center>

 $c$ 未知时，有：

$$P(a,b,c)=P(a)P(c|a)P(b|c)\nRightarrow P(a,b) = P(a)P(b)$$

即 $c$ 未知时，$a,b$ 不独立。

 $c$ 已知时，有：

$$P(a,b|c)=\frac{P(a,b,c)}{P(c)}$$

$$P(a,c) = P(a)P(c|a) = P(c)P(a|c)$$

则有：

$$P(a,b|c) = \frac{P(a)P(c|a)P(b|c)}{P(c)}=\frac{P(a,c)P(b|c)}{P(c)}= P(a|c)P(b|c)$$

所以，在 $c$ 给定的条件下，$a,b$ 被阻断(blocked)，是独立的，称之为head-to-tail条件独立。

这个head-to-tail其实就是一个链式网络，如下图所示：

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/Bayes6.png"/>
</center>

在 $x_i $给定的条件下， $x_{i+1}$ 的分布和 $x_1,x_2,\cdots,x_{i-1}$ 条件独立.也就是说 $x_{i+1}$ 的分布状态只和 $x_i$ 有关，和其他变量条件独立。当前状态只跟上一状态有关，跟上一个状态之前的所有状态无关。这种顺次演变的随机过程，就叫做马尔科夫链（Markov chain）。即：

$$P(X_{n+1}|X_0,X_1,\cdots,X_n)=P(X_{n+1}|X_n)$$

广义的讲，对于任意的结点集 $A,B,C$ ，考察所有通过 $A$ 中任意结点到 $B$ 中任意结点的路径，若要求 $A,B$ 条件独立，则需要所有的路径都被阻断(blocked)，即满足下列两个前提之一：

1. $A$ 和 $B$ 的“head-to-tail型”和“tail-to-tail型”路径都通过 $C$ ；

2. $A$ 和 $B$ 的“head-to-head型”路径不通过 $C$ 以及 $C$ 的子孙；

# 3.3 因子图

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/Bayes7.png"/>
</center>

 对于上图，在一个人已经呼吸困难（dyspnoea）的情况下，其抽烟（smoking）的概率是:

$$P(Smoking|Dyspnoea=yes)=?$$

$$\begin{align}
P(s|d=1) &= \frac{P(s,d=1)}{P(d=1)}\propto P(s,d=1)=\sum_{d=1,b,x,c}P(s)P(c|s)P(b|s)P(x|c,s)P(d|c,b)   \\
&= P(s)\sum_{d=1}\sum_bP(b|s)\sum_x\underbrace{\sum_cP(c|s)P(x|c,s)P(d|c,b)} _{f(s,d,b,x)}\\
\end{align} $$

上式首先对联合概率关于 $b,x,c$ 求和（在 $d=1$ 的条件下），从而消去 $b,x,c$ ，得到 $s$ 和 $d=1$ 的联合概率。由于 $P(s)$ 和 $d=1,b,x,c$ 都没关系，所以，可以提到式子的最前面。而且 $P(b|s)$ 和 $x,c$ 没关系，所以，也可以把它提出来，放到 $\sum_b$ 的后面，从而式子的右边剩下 $\sum_x$ 和 $\sum_c$ 。