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
p(X|\theta_k)=\frac{1}{\sqrt{2\pi}\sigma_k}\exp\biggl(-\frac{(\mathbf{x}-\mu)^2}{2\sigma_k^2}\biggr)
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

实际问题中可能有一些数据是我们无法观测到的，假设观测数据$$X=\{x_1,\cdots,x_n\}$$,缺失数据$$Z=\{z_1,\cdots,z_n\}$$,观测数据与缺失数据组合成完整数据$$Y=\{y_1,\cdots,y_n\}$$,模型的参数 $\mathbf{\theta}$ 有待估计。其中缺失数据 $Z$ 由 $K$ 个分布组成，这 $K$ 个分布的参数可能不同，每个观测数据 $x_i$ 来自于这 $K$ 个分布其中之一。这就造成观测数据下的似然函数难以求得解析解，即 $L(\theta$ \| $X)=P(X$ \| $\theta)$ 难以求解。但是完全数据下的似然函数 $L(\theta$ \| $Y)=P(Y$ \| $Y)$ 容易求解，观测数据条件下的隐变量概率 $P(Z$ \| $X,\theta)$ 容易求得。

#### EM算法

对于观测数据 $X$ 样例间独立,我们想找到每个样例隐含的类别(分布) $z$，能使得 $p(x,z)$ 最大.对于每个样例 $x_i$ 假设 $Q_i$ 是关于隐含变量 $z$ 的概率分布，则 $\sum\limits_zQ_i(z)=1,Q_i(z)\geq 0$ .比如要将班上学生按身高聚类，假设隐藏变量 $z$ 是性别（男女的身高分布，即连续的高斯分布）。

$$
\begin{align*}
\mathcal{l}(\mathbf{\theta}|X)&=\ln P(X|\mathbf{\theta})=\ln\sum_zP(X,z|\mathbf{\theta})=\sum_i\ln\sum_{z^{(i)}}p(x_i,z^{(i)}|\mathbf{\theta})\\
&=\sum_i\ln\sum_{z^{(i)}}\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{Q_i(z^{(i)})}Q_i(z^{(i)})=\sum_i\ln E(\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{Q_i(z^{(i)})})\\
Jensen &\geqslant \sum_i E(\ln\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{Q_i(z^{(i)})})=\sum_i\sum_{z^{(i)}}Q_i(z^{(i)})\ln\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{Q_i(z^{(i)})}
\end{align*}
$$

根据Jensen不等式，想要等式成立，有如下条件:

$$
\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{Q_i(z^{(i)})}=c
$$

可以推出：

$$
\begin{align*}
p(x_i,z^{(i)}|\mathbf{\theta})&=cQ_i(z^{(i)})\\
\sum_{z^{(i)}}p(x_i,z^{(i)}|\mathbf{\theta})&=\sum_z^{(i)}cQ_i(z^{(i)}),\sum_{z^{(i)}}Q_i(z^{(i)})=1\\
\sum_{z}p(x_i,z|\mathbf{\theta})&=c
\end{align*}
$$

则：

$$
\begin{align*}
Q_i(z^{(i)})&=\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{c}=\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{\sum_{z}p(x_i,z|\mathbf{\theta})}\\
&=\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{p(x_i|\mathbf{\theta})}=p(z^{(i)}|x_i,\theta)
\end{align*}
$$

即 $Q_i$ 为在给定 $x_i,\theta^{(t)}$ 情况下 $x_i$ 属于分类 $z$ 的概率。将 $Q_i$带入可得：

$$
\begin{align*}
\mathcal{l}(\mathbf{\theta}|X)&=\ln P(X|\mathbf{\theta})=\ln\sum_zP(X,z|\mathbf{\theta})=\sum_i\ln\sum_{z^{(i)}}p(x_i,z^{(i)}|\mathbf{\theta})\\
&=\sum_i\ln\sum_{z^{(i)}}\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{Q_i(z^{(i)})}Q_i(z^{(i)})=\sum_i\ln E(\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{Q_i(z^{(i)})})\\
Jensen &\geqslant \sum_i E(\ln\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{Q_i(z^{(i)})})=\sum_i\sum_{z^{(i)}}Q_i(z^{(i)})\ln\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{Q_i(z^{(i)})}\\
&=\sum_i\sum_{z^{(i)}}p(z^{(i)}|x_i,\theta^{(t)})\ln\frac{p(x_i,z^{(i)}|\mathbf{\theta})}{p(z^{(i)}|x_i,\theta^{(t)})}\\
&=\sum_i\sum_{z^{(i)}}p(z^{(i)}|x_i,\theta^{(t)})\ln p(x_i,z^{(i)}|\mathbf{\theta}) +\biggl(-\sum_i\sum_{z^{(i)}}p(z^{(i)}|x_i,\theta^{(t)})\ln p(z^{(i)}|x_i,\theta^{(t)})\biggr)\\
&=\sum_{z}p(z|X,\theta^{(t)})\ln P(X,z|\mathbf{\theta}) +\biggl(-\sum_{z}p(z|X,\theta^{(t)})\ln P(z|X,\theta^{(t)})\biggr)
\end{align*}
$$

上式中第二项与 $\theta$ 无关为常数项（熵），令：

$$
Q(\theta|\theta^{(t)}) = \sum_{z}p(z|X,\theta^{(t)})\ln P(X,z|\mathbf{\theta})=E_z(\ln P(X,z|\mathbf{\theta})|X,\theta^{(t)})
$$

可得最大化似然函数等同于：

$$
\arg\max_\theta\mathcal{l}(\mathbf{\theta}|X)\Leftrightarrow\arg\max_\theta Q(\theta|\theta^{(t)})
$$

#### EM算法流程
由以上推导可知EM算法通过迭代求 $\mathcal{l}(\mathbf{\theta}|X)=\ln P(X|\mathbf{\theta})$ 的最大似然估计，每次迭代步骤如下：

(1)选择参数的初始值 $\theta^{(0)}$ 开始迭代

(2)E步（求期望）:

$$
Q(\theta|\theta^{(t)}) = \sum_{z}p(z|X,\theta^{(t)})\ln P(X,z|\mathbf{\theta})=E_z(\ln P(X,z|\mathbf{\theta})|X,\theta^{(t)})
$$

(3)M步：最大化第二步的期望值，然后对参数求偏导等于0后求得下一步迭代的参数:

$$
\theta^{(t+1)}\leftarrow\arg\max_\theta Q(\theta|\theta^{(t)})
$$

(4)重复E步和M步直至：

$$\|\theta^{(t+1)}-\theta^{(t)}\|>\varepsilon_1~~~~or~~~~\|Q(\theta^{(t+1)}|\theta^{(t)})-Q(\theta^{(t)}|\theta^{(t)})\|>\varepsilon_2$$

停止迭代，返回 $\theta^*=\theta^{(t+1)}$ .

EM算法是利用下边界逼近真实值，如下图：


![](https://darknessbeforedawn.github.io/test-book/images/EM3.png)

$$
\mathcal{l}(\mathbf{\theta}|X)\geqslant \sum_{z}p(z|X,\theta^{(t)})\ln P(X,z|\mathbf{\theta}) +\biggl(-\sum_{z}p(z|X,\theta^{(t)})\ln P(z|X,\theta^{(t)})\biggr)\triangleq g(\theta|\theta^{(t)})
$$

# 3.EM算法收敛性

观测数据的似然函数:

$$
P(X|\theta)=\frac{P(X,Z|\theta)}{P(Z|X,\theta)}
$$

$$
Q(\theta|\theta^{(i)}) = \sum_{Z}P(Z|X,\theta^{(i)})\ln P(X,Z|\mathbf{\theta})
$$

令

$$
H(\theta|\theta^{(i)}) = \sum_{Z}P(Z|X,\theta^{(i)})\ln P(Z|X,\mathbf{\theta})
$$

则对数似然函数可以写成：

$$
\ln P(X|\theta) =Q(\theta|\theta^{(i)})-H(\theta|\theta^{(i)})
$$

$$
\ln P(X|\theta^{(i+1)})-\ln P(X|\theta^{(i)})=[Q(\theta|\theta^{(i+1)})-Q(\theta|\theta^{(i)})]-[H(\theta|\theta^{(i+1)})-H(\theta|\theta^{(i)})]
$$

由于 $\theta^{(i+1)}$ 使 $Q(\theta|\theta^{(i)})$ 达到极大，所以有

$$
Q(\theta|\theta^{(i+1)})-Q(\theta|\theta^{(i)}) \geqslant 0
$$

第二项：

$$
\begin{align*}
H(\theta|\theta^{(i+1)})-H(\theta|\theta^{(i)})&=\sum_Z\biggl(\ln\frac{P(Z|X,\theta^{(i+1)})}{P(Z|X,\theta^{(i)})}\biggr)P(Z|X,\theta^{(i)})\\
&\leqslant \ln\sum_Z\biggl(\frac{P(Z|X,\theta^{(i+1)})}{P(Z|X,\theta^{(i)})}P(Z|X,\theta^{(i)})\biggr)\\
&=\ln\sum_Z\biggl(P(Z|X,\theta^{(i+1)})\biggr)=0
\end{align*}
$$

因此 $P(X$ \| $\theta)$ 是单调递增的，且有上界，因此收敛。

# 4.EM算法在高斯混合模型中的应用

高斯混合模型在第一节中已经提过，下面以单变量为例。

#### 明确隐变量，写出完整的对数似然函数

$$
\gamma_{ik}=\begin{cases} 1 & x_i\in k \\ 0 &else\end{cases}
~~~~i=1,\cdots,n;k=1,\cdots,K
$$

完整数据可表示为 $(x_i,\gamma_{i1},\cdots,\gamma_{iK})$

完整数据的似然函数可表示为：

$$
\begin{align*}
P(X,\gamma|\theta)&=\prod_{i=1}^nP(x_i,\gamma_{i1},\cdots,\gamma_{iK}|\theta)=\prod_{i=1}^n\prod_{k=1}^K(\pi_kp(x_i|\theta_k))^{\gamma_{ik}}\\
&=\prod_{k=1}^K(\pi_k)^{n_k}\prod_{i=1}^n\biggl(\frac{1}{\sqrt{2\pi}\sigma_k}\exp\biggl(-\frac{(x_i-\mu_k)^2}{2\sigma_k^2}\biggr)\biggr)^{\gamma_{ik}}\\
\ln P(X,\gamma|\theta)&=\sum_{k=1}^Kn_k\ln \pi_k+\sum_{i=1}^n\gamma_{ik}\biggl(\ln\frac{1}{\sqrt{2\pi}}-\frac{1}{2}\sigma_k^2-\frac{(x_i-\mu_k)^2}{2\sigma_k^2}\biggr),n_k=\sum_{i=1}^n\gamma_{ik},\sum_{k=1}^Kn_k=n
\end{align*}
$$

E-步

$$
\begin{align*}
Q(\theta|\theta^{(t)})&=\sum_{\gamma_{ik}}p(\gamma_{ik}|X,\theta^{(t)})\ln P(X,\gamma|\theta)\\
&=E_{\gamma_{ik}}\biggl(\sum_{k=1}^Kn_k\ln \pi_k+\sum_{i=1}^n\gamma_{ik}\biggl(\ln\frac{1}{\sqrt{2\pi}}-\frac{1}{2}\sigma_k^2-\frac{(x_i-\mu_k)^2}{2\sigma_k^2}\biggr)\biggr)\\
&=\biggl(\sum_{k=1}^K\sum_{i=1}^nE(\gamma_{ik})\ln \pi_k+\sum_{i=1}^nE(\gamma_{ik})\biggl(\ln\frac{1}{\sqrt{2\pi}}-\frac{1}{2}\sigma_k^2-\frac{(x_i-\mu_k)^2}{2\sigma_k^2}\biggr)\biggr)\\
&=\biggl(\sum_{k=1}^K\hat{n}_k\ln \pi_k+\sum_{i=1}^n\hat{\gamma}_{ik}\biggl(\ln\frac{1}{\sqrt{2\pi}}-\frac{1}{2}\sigma_k^2-\frac{(x_i-\mu_k)^2}{2\sigma_k^2}\biggr)\biggr)\\
\end{align*}
$$

M-步

$$
\theta^{(t+1)}=\arg\max_theta Q(\theta|\theta^{(t)})
$$

$$
\frac{\partial Q(\theta|\theta^{(t)})}{\partial \mu_k}=0\Rightarrow \mu_k^{(t+1)}=\frac{\sum\limits_{i=1}^n\gamma_{ik}x_i}{\sum\limits_{i=1}^n\gamma_{ik}}
$$

$$
\frac{\partial Q(\theta|\theta^{(t)})}{\partial \sigma_k^2}=0\Rightarrow (\sigma_k^2)^{(t+1)}=\frac{\sum\limits_{i=1}^n\gamma_{ik}(x_i-\mu_k)^2}{\sum\limits_{i=1}^n\gamma_{ik}}
$$

$$
\pi_k^{(t+1)}=\arg\max_{\pi_k}\biggl(Q(\theta|\theta^{(t)})+\lambda \biggl(\sum\limits_{k=1}^K\pi_k-1\biggr)\biggr)
$$

$$
\frac{\partial \biggl(Q(\theta|\theta^{(t)})+\lambda \biggl(\sum\limits_{k=1}^K\pi_k-1\biggr)\biggr)}{\partial \pi_k}=0\Rightarrow \pi_k^{(t+1)}=\frac{\sum\limits_{i=1}^n\pi_{ik}}{n}
$$

#### 多元高斯混合模型
与单元求解类似（注意向量和矩阵求导的细节即可）

$$
\mathbf{\mu}_k^{(t+1)}=\frac{\sum\limits_{i=1}^n\gamma_{ik}\mathbf{x}_i}{\sum\limits_{i=1}^n\gamma_{ik}}
$$

$$
\Sigma_k^{(t+1)}=\frac{\sum\limits_{i=1}^n\gamma_{ik}(\mathbf{x}_i-\mathbf{\mu}_k)(\mathbf{x}_i-\mathbf{\mu}_k)^T}{\sum\limits_{i=1}^n\gamma_{ik}}
$$

$$
\pi_k^{(t+1)}=\frac{\sum\limits_{i=1}^n\pi_{ik}}{n}
$$