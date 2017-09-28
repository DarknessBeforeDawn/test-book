---
title: EM算法
layout: post
share: false
---

# 1.基础知识
## 1.1.高斯混合模型

高斯混合模型是有如下形式的概率分布模型：

$$
\begin{equation}
p(x|\mathbf{\theta})=\sum_{k=1}^K\pi_kp(x|\theta_k)
\end{equation}
$$

其中 $\pi_k>0$ 为混合系数, $\sum\limits_{k=1}^K\pi_k=1$ , $\mathbf{\theta}=(\theta_1,\cdots,\theta_K)^T$ , $\theta_k=(\pi_k,\mu_k,\sigma_k^2)$ ，多元变量 $\theta_k=(\pi_k,\sum_k)$ , $\sum_k$ 为第 $k$ 个高斯混合成分的参数。

$$
p(X|\theta_k)=\frac{1}{\sqrt{z\pi}\sigma_k}\exp\biggl(-\frac{(\mathbf{x}-\mu)^2}{z\sigma_k^2}\biggr)
$$

假设观测数据 $x_1,x_2,\cdots,x_n \in \mathbf{R}$ 由 $K$ 个组分的高斯混合模型生成。

高斯混合分布的对数最大化似然函数为:

$$
\begin{align*}
\mathcal{l}(\mathbf{\theta}|\mathbf{X})&=\ln(L(\mathbf{\theta}|\mathbf{X}))=\ln p(\mathbf{X}|\theta)\\
&=\ln\prod_{i=1}^n\sum_{k=1}^K\pi_kp(\mathbf{x}_i|\theta_k)\\
&=\sum_{i=1}^n\ln\biggl(\sum_{k=1}^K\pi_kp(\mathbf{x}_i|\theta_k)\biggr)
\end{align*}
$$

对数里面有加和，无法直接通过求偏导解方程的方法求取最大值。
可以用EM算法解决这种难以计算的问题，EM算法是一种近似逼
近的迭代计算方法。

## 1.2.全概率公式
对于任一事件 $A$ ,若有互不相容的事件 $B_k,k=1,\cdots,K$ 且 $\bigcup_{k=1}^K\supset A$ ,则事件 $A$ 的概率可用下式计算：

$$
\begin{equation}
P(A)=\sum_{k=1}^KP(B_k)P(A|B_k)
\end{equation}
$$

上式称为全概率公式。

![](https://darknessbeforedawn.github.io/test-book/images/EM1.png)

## 1.3.Jensen不等式

如果 $f$ 是凸函数, $X$是随机变量，那么 $E[f(X)]\geq f(EX)$ ,如果 $f$ 是凹函数, $X$是随机变量，那么 $E[f(X)]\leq f(EX)$ ,如果 $f$ 是严格凹或凸函数, 那么 $E[f(X)]=f(EX)$ 当且仅当 $p(x=E[X])=1$ ,也就是 $X$ 是常量。如下图：

![](https://darknessbeforedawn.github.io/test-book/images/EM2.png)


# 2.EM算法原理及推导

实际问题中可能有一些数据是我们无法观测到的，假设观测数据$$X=\{x_1,\cdots,x_n\}$$,缺失数据$$Z=\{z_1,\cdots,z_n\}$$,观测数据与缺失数据组合成完整数据$$Y=\{y_1,\cdots,y_n\}$$,模型的参数 $\mathbf{\theta}$ 有待估计。其中缺失数据 $Z$ 将观测数据分割成 $K$ 个区域，且这 $K$ 个区域的数据服从的分布参数可能不同。这就造成观测数据下的似然函数难以求得解析解，即 $L(\theta$ \| $X)=P(X$ \| $\theta)$ 难以求解。但是完全数据下的似然函数 $L(\theta$ \| $Y)=P(Y$ \| $Y)$ 容易求解，观测数据条件下的隐变量概率 $P(Z$ \| $X,\theta)$ 容易求得。

 $Z$ 将观测数据分成若干个区域，则EM推导如下：

$$
\begin{align*}
\mathcal{l}(\mathbf{\theta}|X)&=\ln P(X|\mathbf{\theta})=\ln\sum_zP(X,z|\mathbf{\theta})\\
&=\ln\sum_z\frac{P(X,z|\mathbf{\theta})}{p(z|X,\mathbf{\theta}^{(t)})}p(z|X,\mathbf{\theta}^{(t)})\\
Jensen &\geqslant \sum_zp(z|X,\mathbf{\theta}^{(t)})\ln\frac{P(X,z|\mathbf{\theta})}{p(z|X,\mathbf{\theta}^{(t)})}\\
&=\sum_zp(z|X,\mathbf{\theta}^{(t)})\ln P(X,z|\mathbf{\theta}) -\sum_zp(z|X,\mathbf{\theta}^{(t)})\ln p(z|X,\mathbf{\theta}^{(t)})
&=\sum_zp(z|X,\mathbf{\theta}^{(t)})\ln P(X,z|\mathbf{\theta}) -\sum_zp(z|X,\mathbf{\theta}^{(t)})\ln p(z|X,\mathbf{\theta}^{(t)})
\end{align*}
$$





