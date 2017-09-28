---
title: EM算法
layout: post
share: false
---

# 1.高斯混合模型

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


