---
title: 隐马尔可夫模型
layout: post
share: false
---

隐马尔可夫模型(Hidden Markov Model, HMM)是可用于标注问题的统计学习模型，描述由隐藏的马尔客服链随机生成观测序列的过程，属于生成模型。

# 1. $HMM$ 的基本概念

## 1.1 $HMM$ 的定义

 $HMM$ 是关于时序的概率模型，描述一个隐藏的马尔科夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。隐藏的马尔科夫链随机生成的状态的序列，称为状态序列(State Sequence);每一个状态生成一个观测，而由此产生的观测的随机序列，称为观测序列(Observation Seqyence).序列的每一个位置又可以看作一个时刻。

 $HMM$ 由初始概率分布、状态转移概率分布以及观测概率分布确定。HMM的形式如下:

设 $Q$ 是所有可能的状态集合， $V$ 是所有可能的观测的集合。

$$Q=\{q_1,q_2,\cdots,q_N\},~~~V=\{v_1,v_2,\cdots,v_M\}$$

其中， $N$ 是可能的状态数， $M$ 是可能的观测数。

 $I$ 是长度为 $T$ 的状态序列， $O$ 是对应的观测序列:

$$I=(i_1,i_2,\cdots,i_T),~~~O(o_1,o_2,\cdots,o_T)$$

 $A$ 是状态转移矩阵:

$$A=[a_{ij}]_{N\times N}$$

其中，

$$a_{ij}=P(i_{t+1}=q_j|i_t=q_i),~~~i=1,2,\cdots,N;~~j=1,2,\cdots,N$$

是在时刻 $t$ 处于状态 $q_i$ 的条件下在时刻 $t+1$ 转移到状态 $q_j$ 的概率。

 $B$ 是观测概率矩阵:

$$B=[b_j(k)]_{N\times M}$$

其中，

$$b_j(k)=P(o_t=v_k|i_t=q_j),~~~i=1,2,\cdots,M;~~j=1,2,\cdots,N$$

是在时刻 $t$ 处于状态 $q_j$ 的条件下生成观测 $v_k$ 的概率。

 $\pi$ 是初始状态概率向量:

$$\pi = (\pi_i)$$

其中，

$$\pi_i=P(i_1=q_i),~~~~i=1,2,\cdots,N$$

是时刻 $t=1$ 处状态 $q_i$ 的概率。

 $HMM$ 由初始状态概率向量 $\pi$ 、状态转移概率矩阵 $A$ 和观测概率矩阵 $B$ 决定。 $\pi$ 和 $A$ 决定状态序列， $B$ 决定观测序列。因此， $HMM~~\lambda$ 可以用三元符号表示，即

$$\lambda=(A,B,\pi)$$

 $A,B,\pi$ 称为 $HMM$ 的三要素。

状态转移矩阵 $A$ 与初始状态概率向量 $\pi$ 确定了隐马尔可夫链，生成不可观测的状态序列。观测概率矩阵 $B$ 确定了如何从状态生成观测，与状态序列综合取得了如何产生观测序列。

$HMM$ 作了两个基本假设:

(1) 齐次马尔科夫性假设，即假设隐藏的马尔科夫链在任意时刻 $t$ 的状态值依赖于前一时刻的状态，与其他时刻的状态及观测无关，也与时刻 $t$ 无关。

$$P(i_t|i_{t-1},o_{t-1},\cdots,i_1,o_1)=P(i_t|i_{t-1}), ~~~t=1,2,\cdots,T$$ 

(2) 观测独立性假设，即假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态，与其他观测及状态无关。

$$P(o_t|i_T,o_T,i_{T-1},o_{T-1},\cdots,i_{t+1},o_{t+1},i_t,i_{t-1},o_{t-1},\cdots,i_1,o_1)=P(o_t|i_t)$$

 $HMM$ 可用于标注，这时状态对应着标记。标注问题是给定观测的序列预测去对应的标记序列。可以假设标注问题的数据是由 $HMM$ 生成的。这样就可以利用 $HMM$ 的学习与预测算法进行标注。

## 1.2 观测序列的生成过程

根据 $HMM$ 的定义，可以将一个长度为 $T$ 的观测序列 $O=(o_1,o_2,\cdots,o_T)$ 的生成过程描述如下:

输入: $HMM  ~~~\lambda =(A,B,\pi)$ ，观测序列长度 $T$ ;

输出：观测序列 $O=(o_1,o_2,\cdots,o_T)$ 。

(1) 按照初始状态分布 $\pi$ 产生状态 $i_1$

(2) 令 $t=1$ 

(3) 按照状态 $i_t$ 的观测概率分布 $b_{i_t}(k)$ 生成 $o_t$ 

(4) 按照状态 $i_t$ 的状态转移概率分布 

$$\{a_{i_ti_{t+1}}\}$$

$~~~~~~~~$产生状态 $i_{t+1}, ~~ i_{t+1}=1,2,\cdots,N$

(5) 令 $t=t+1$ ；如果 $t<T$ ，转(3);否则，终止

## 1.2 $HMM$ 的三个基本问题

(1)概率计算问题，给定模型 $\lambda =(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,\cdots,o_T)$ ，计算在模型 $\lambda$ 下观测序列 $O$ 出现的概率 $P(O$ \| $\lambda)$

(2)学习问题。已知观测序列 $O=(o_1,o_2,\cdots,o_T)$ ，估计模型 $\lambda =(A,B,\pi)$ 参数，使得在该模型下观测序列概率 $P(O$ \| $\lambda)$ 最大.即用极大似然估计参数。

(3)预测问题，也称为解码(decoding)问题。已知模型 $\lambda =(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,\cdots,o_T)$ ，求给定观测序列条件概率 $P(I$ \| $O)$ 最大的状态序列 $O=(i_1,i_2,\cdots,i_T)$ .即给定观测序列，求最有可能的对应的状态序列。

# 2. 概率计算算法

## 2.1 直接计算

给定模型 $\lambda =(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,\cdots,o_T)$ ，计算观测序列 $O$ 出现的概率 $P(O$ \| $\lambda)$ .最直接的方法是按概率公式直接计算。通过列举所有可能的长度为 $T$ 的状态序列 $I=(i_1,i_2,\cdots, i_T)$ ，求各个状态序列 $I$ 与观测序列 $O=(o_1,o_2,\cdots,o_T)$ 的联合概率 $P(O,I$ \| $\lambda)$ ,然后对所有可能的状态序列求和，得到 $P(O$ \| $\lambda)$ 。

状态序列 $I=(i_1,i_2,\cdots, i_T)$ 的概率是

$$P(I|\lambda)=\pi_{i_1}a_{i_1i_2}a_{i_2i_3}\cdots a_{i_{T-1}i_T}$$

对固定状态序列 $I=(i_1,i_2,\cdots, i_T)$ ，观测序列 $O=(o_1,o_2,\cdots,o_T)$ 的概率是

$$P(O|I,\lambda)=b_{i_1}(o_1)b_{i_2}(o_2)\cdots b_{i_T}(o_T)$$ 

 $O$ 和 $I$ 同时出现的联合概率为

$$\begin{align}
P(O,I|\lambda)&=\frac{P(O,I,\lambda)}{P(\lambda)} = \frac{P(O,I,\lambda)}{P(I,\lambda)}\frac{P(I,\lambda)}{P(\lambda)}=P(O|I,\lambda)P(I|\lambda) \\
&= \pi_{i_1}b_{i_1}(o_1)a_{i_1i_2}b_{i_2}(o_2)a_{i_2i_3}\cdots a_{i_{T-1}i_T}b_{i_T}(o_T) \\
\end{align}$$

然后对所有可能的状态序列 $I$ 求和，得到观测序列 $O$ 的概率：

$$P(O|\lambda)=\sum_IP(O|I,\lambda)P(I|\lambda)=\sum_{i_1,i_2,\cdots, i_T}\pi_{i_1}b_{i_1}(o_1)a_{i_1i_2}b_{i_2}(o_2)a_{i_2i_3}\cdots a_{i_{T-1}i_T}b_{i_T}(o_T)$$

上式的计算量是 $O(TN^T)$ 阶的，计算量过大，此方法不可行。

## 2.2 前向算法

**前向概率:** 给定 $HMM ~~~\lambda$ ，定义到时刻 $t$ 部分观测序列为 $o_1,o_2,\cdots,o_t$  且状态为 $q_i$ 的概率为前向概率，记作

$$\alpha_t(i)=P(o_1,o_2,\cdots,o_t,i_t=q_i|\lambda)$$

可以递推地求前向概率 $\alpha_t(i)$ 及观测序列概率 $P(O$ \| $\lambda)$ .

**观测序列概率的前向算法**

输入: $HMM ~~\lambda$ ,观测序列 $O$ ；

输出: 观测序列概率 $P(O$ \| $\lambda)$ .

(1)初值

$$\alpha_1(i)=\pi_ib_i(o_1),~~~i=1,2,\cdots,N$$

(2)递推  对 $t=1,2,\cdots,T-1$ 

$$\alpha_{t+1}(i)=\biggl[\sum_{j=1}^N\alpha_t(j)a_{ji}\biggr]b_i(o_{t+1}),~~~i=1,2,\cdots,N$$

(3)终止

$$P(O|\lambda)=\sum_{i=1}^N\alpha_T(i)$$

(1)初始化前向概率，是初始时刻的状态 $i_1=q_i$ 和观测 $o_1$ 的联合概率。(2)是前向概率的递推公式，计算到时刻 $t+1$ 部分观测序列为 $o_1,o_2,\cdots,o_t,o_{t+1}$ 且在时刻 $t+1$ 处于状态 $q_i$ 的前向概率，如图。

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/HMM1.png"/>
</center>

在递推公式的方括号里, $\alpha_t(j)$ 是时刻 $t$ 观测到 $o_1,o_2,\cdots,o_t$ 并在时刻 $t$ 处于状态 $q_j$ 的前向概率，那么乘积 $\alpha_t(j)a_{ji}$ 就是是时刻 $t$ 观测到 $o_1,o_2,\cdots,o_t$ 并在时刻 $t$ 处于状态 $q_j$ 而在时刻 $t+1$ 到达状态 $q_i$ 的联合概率。对于这个乘积在时刻 $t$ 的所有可能的 $N$ 个状态 $q_j$ 求和，其结果解释到时刻 $t$ 观测为 $o_1,o_2,\cdots,o_t$ 并在时刻 $t+1$ 处于状态 $q_i$ 的联合概率。方括号里的值与观测概率 $b_i(o_{t+1})$  的乘积恰好就是是到时刻 $t+1$ 观测到 $o_1,o_2,\cdots,o_t,o_{t+1}$ 并在时刻 $t+1$ 处于状态 $q_i$ 的前向概率 $\alpha_{t+1}(i)$. 

(3)因为

$$\alpha_T(i)=P(o_1,o_2,\cdots,o_t,i_T=q_i|\lambda)$$

所以

$$P(O|\lambda)=\sum_{i=1}^N\alpha_T(i)$$

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/HMM2.jpg"/>
</center>

如上图，前向算法实际是基于状态序列的路径结构递推计算 $P(O$ \| $\lambda)$ 的算法。前向算法高兴的关键是其具备计算前向概率，然后利用路径结构将前向概率递推到全局，得到 $P(O$ \| $\lambda)$ .具体地，在时刻 $t=1$ ，计算 $\alpha_1(i)$ 的 $N$ 个值 $(i=1,2,\cdots,N)$ ；在各个时刻 $i=1,2,\cdots,T-1$ ，计算 $\alpha_{t+1}(i)$ 的 $N$ 个值, 而且每个 $\alpha_{t+1}(i)$ 的计算利用前一时刻 $N$ 个 $\alpha_t(j)$ .减少计算量的原因在于每一次计算直接引用前一时刻的计算结果，避免重复计算。这样利用前向概率计算 $P(O$ \| $\lambda)$ 的计算量是 $O(N^2T)$ 阶的。

## 2.3 后向算法

**后向概率:** 给定 $HMM ~~~\lambda$ ，定义在时刻 $t$ 状态为 $q_i$ 的条件下，从 $t+1$ 到 $T$ 的部分观测序列为 $o_{t+1},o_{t+2},\cdots,o_T$ 的概率为后向概率，记作

$$\beta_t(i)=P(o_{t+1},o_{t+2},\cdots,o_T|i_t=q_i,\lambda)$$

可以用递推地方法求得后向概率 $\beta_t(i)$ 及观测序列概率 $P(O$ \| $\lambda)$ 。

**观测序列概率的后向算法**

输入: $HMM ~~\lambda$ ,观测序列 $O$ ；

输出: 观测序列概率 $P(O$ \| $\lambda)$ .

(1)初值

$$\beta_T(i)=1,~~~i=1,2,\cdots,N$$

(2)递推  对 $t=T-1,T-2,\cdots,1$ 

$$\beta_t(i)=\sum_{j=1}^Na_{ij}b_j(o_{t+1})\beta_{t+1}(j),~~~i=1,2,\cdots,N$$

(3)终止

$$P(O|\lambda)=\sum_{i=1}^N\pi_ib_i(o_1)\beta_1(i)$$

(1)初始化后向概率，对最终时刻的所有状态 $q_i$ 规定 $\beta_T(i)=1$ 。(2)是后向概率的递推公式。

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/HMM3.png"/>
</center>

如上图，为了计算在时刻 $t$ 状态为 $q_i$ 条件下时刻 $t+1$ 之后的观测序列为 $o_{t+1},o_{t+2},\cdots,o_T$ 的后向概率 $\beta_t(i)$ ，只需考虑在时刻 $t+1$ 所有可能的 $N$ 个状态 $q_j$ 的转移概率 ( $a_{ij}$ 项)，以及在此状态下的观测 $o_{t+1}$ 的观测概率( $b_j(o_{t+1})$ 项)，然后考虑状态 $q_j$ 之后的观测序列的后向概率( $\beta_{t+1}(j)$ 项)。(3)求 $P(O$ \| $\lambda)$ 的思路与步骤(2) 一致，只是初始概率 $\pi_i$ 代替转移概率。

利用前向概率和后向概率的定义可以将观测序列概率 $P(O$ \| $\lambda)$ 统一写成

$$P(O|\lambda)=\sum_{i=1}^N\sum_{j=1}^N\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j), ~~~t=1,2,\cdots,T-1$$

此式当 $t=1$ 和 $t=T-1$ 时分别代表前向概率和后向概率所求的观测序列概率。