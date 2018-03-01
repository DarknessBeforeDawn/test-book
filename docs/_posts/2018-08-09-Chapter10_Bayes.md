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

