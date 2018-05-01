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

**例(盒子和球模型)**

假设有4个盒子，每个盒子里都装有红白两种颜色的球，盒子里的红白球数如下表:

| 盒子 | 1号 | 2号 | 3号 | 4号 |
|:---:|:---:|:---:|:---:|:---:|
| 红球数| 5 | 3 | 6 | 8 |
| 白球数| 5 | 7 | 4 | 2 |

抽球方法:开始，以等概率随机从4个盒子里选取1个盒子，从盒子里随机抽出1个球，记录颜色后放回；然后从当前盒子转移到下一个盒子，规则:如果当前盒子是1号，那么 下一个盒子一定是2号，如果当前盒子是2号或3号，那么分别以概率0.4和0.6转移到左边或右边的盒子，如果当前盒子是4号，那么各以0.5的概率停留在盒子4或转移到盒子3；确定转移的盒子后，再从这个盒子里随机抽出1个求，记录其颜色并放回；如此下去重复5次，得到 一个球的颜色的观测序列:

$$O=\{Red,Red,White,White,Red\}$$

在这个过程中，只能观测到求颜色的序列，不能观测到球是从哪个盒子取出的，即观测不到盒子的序列。

这个例子中有两个随机序列，一个是盒子的序列(状态序列)，一个是球颜色的观测序列(观测序列)。前者是隐藏的，只有后者是可观测的。这是 $HMM$ 的例子:

盒子对应状态，状态的集合:

$$Q=\{Box1,Box2,Box3,Box4\}, ~~~~~~N=4$$

球的颜色对应观测。观测序列集合:

$$V=\{Red,White\},~~~~~M=2$$

状态序列和观测序列长度 $T=5$ .

初始概率分布:

$$\pi = (0.25,0.25,0.25,0.25)^T$$

状态转移概率分布

$$A=\begin{bmatrix}
0 & 1 & 0 & 0\\
0.4 & 0 & 0.6 & 0\\
0 & 0.4 & 0 & 0.6\\
0 & 0 & 0.5 & 0.5
\end{bmatrix}$$

观测概率分布

$$B=\begin{bmatrix}
0.5 & 0.5\\
0.3 &0.7\\
0.6 & 0.4\\
0.8 & 0.2
\end{bmatrix}$$

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

## 1.3 $HMM$ 的三个基本问题

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

如上图，前向算法实际是基于状态序列的路径结构递推计算 $P(O$ \| $\lambda)$ 的算法。前向算法高效的关键是其具备计算前向概率，然后利用路径结构将前向概率递推到全局，得到 $P(O$ \| $\lambda)$ .具体地，在时刻 $t=1$ ，计算 $\alpha_1(i)$ 的 $N$ 个值 $(i=1,2,\cdots,N)$ ；在各个时刻 $i=1,2,\cdots,T-1$ ，计算 $\alpha_{t+1}(i)$ 的 $N$ 个值, 而且每个 $\alpha_{t+1}(i)$ 的计算利用前一时刻 $N$ 个 $\alpha_t(j)$ .减少计算量的原因在于每一次计算直接引用前一时刻的计算结果，避免重复计算。这样利用前向概率计算 $P(O$ \| $\lambda)$ 的计算量是 $O(N^2T)$ 阶的。

**例**

考虑盒子和球模型 $\lambda=(A,B,\pi)$ ，状态集合 

$$Q=\{1,2,3\}$$

观测集合

$$V=\{Red,White\}$$

$$A=\begin{bmatrix}
0.5 & 0.2 & 0.3\\
0.3 & 0.5 & 0.2\\
0.2 & 0.3 & 0.5
\end{bmatrix},B=\begin{bmatrix}
0.5 & 0.5\\
0.6 & 0.4\\
0.7 & 0.3
\end{bmatrix},\pi = (0.2,0.4,0.4)^T$$

设 $T=3,O=(Red,White,Red)$ ,用前向算法计算 $P(O$ \| $\lambda)$ 

**解**  

(1)计算初值

$$\begin{align}
\alpha_1(1)&=\pi_1b_1(o_1)=0.10 \\
\alpha_1(2)&=\pi_2b_2(o_1)=0.16 \\
\alpha_1(3)&=\pi_3b_3(o_1)=0.28
\end{align}$$

(2)递推计算

$$\begin{align}
\alpha_2(1)&=\biggl[\sum_{i=1}^3\alpha_1(i)a_{i1}\biggr]b_1(o_2)=0.154\times 0.5=0.077 \\
\alpha_2(2)&=\biggl[\sum_{i=1}^3\alpha_1(i)a_{i2}\biggr]b_2(o_2)=0.184\times 0.6=0.1104 \\
\alpha_2(3)&=\biggl[\sum_{i=1}^3\alpha_1(i)a_{i3}\biggr]b_3(o_2)=0.202\times 0.3=0.0606 \\
\alpha_3(1)&=\biggl[\sum_{i=1}^3\alpha_2(i)a_{i1}\biggr]b_1(o_3)=0.04187 \\
\alpha_3(2)&=\biggl[\sum_{i=1}^3\alpha_2(i)a_{i2}\biggr]b_2(o_3)=0.03551 \\
\alpha_3(3)&=\biggl[\sum_{i=1}^3\alpha_2(i)a_{i3}\biggr]b_3(o_3)=0.05284 \\
\end{align}$$

(3)终止

$$P(O|\lambda)=\sum_{i=1}^3\alpha_3(i)=0.13022$$


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

此式当 $t=1$ 和 $t=T-1$ 时分别代表前向概率和后向概率所求的观测序列概率。从 $t=1$ 时刻不断向前递推，将得到前向算法的计算公式，从 $t=T-1$ 时刻不断向后递推，将得到后向算法的计算公式。

把 $t=T-1$ 代入，得

$$P(O|\lambda)=\sum_{i=1}^N\sum_{j=1}^N\alpha_{T-1}(i)a_{ij}b_j(o_{T})\beta_{T}(j)$$

由于 $\alpha$ 是对 $i$ 的累加，与 $j$ 无关，于是上式可变化为：

$$P(O|\lambda)=\sum_{i=1}^N\alpha_{T-1}(i)\sum_{j=1}^Na_{ij}b_j(o_{T})\beta_{T}(j)$$

由 $\beta_t(i)$ 的递推公式:

$$\beta_t(i)=\sum_{j=1}^Na_{ij}b_j(o_{t+1})\beta_{t+1}(j)$$

得

$$P(O|\lambda)=\sum_{i=1}^N\alpha_{T-1}(i)\beta_{T-1}(i)$$

由 $\alpha_{T-1}(i)$ 的递推公式得

$$\begin{align}
P(O|\lambda)&=\sum_{i=1}^N\beta_{T-1}(i)\sum_{j=1}^N\alpha_{T-2}(j)a_{ji}b_i(O_{t-1}) \\
&=\sum_{i=1}^N\sum_{j=1}^N\alpha_{T-2}(j)a_{ji}b_i(O_{t-1})\beta_{T-1}(i) \\
&=\sum_{i=1}^N\alpha_{T-2}(i)\beta_{T-2}(i) \\
&=\cdots \\
&=\sum_{i=1}^N\alpha_1(i)\beta_1(i)
\end{align}$$

在前向算法和后向算法中，给每一个 $t$ 时刻的隐含状态结点定义了实际的物理含义，即 $\alpha_t(i),\beta_t(i)$ 两个中间变量分别从两边进行有向加权和有向边汇聚，形成一种递归结构，并且不断传播至两端，对任意 $t=1,t=T-1$ 时刻，分别进行累加就能求得 $P(O$ \| $\lambda)$  

## 2.4 概率与期望的计算

利用前向概率和后向概率，可以得到关于单个状态和两个状态概率的计算公式。

1.给定模型 $\lambda$ 和观测 $O$ ，在时刻 $t$ 处于状态 $q_i$ 的概率， 记

$$\gamma_t(i) = P(i_t=q_i|O,\lambda)$$

可以通过前向后向概率计算，

$$\gamma_t(i)=P(i_t=q_i|O,\lambda)=\frac{P(i_t=q_i,O,\lambda)}{P(O,\lambda)}=\frac{P(i_t=q_i,O,\lambda)}{P(\lambda)P(O|\lambda)}=\frac{P(i_t=q_i,O|\lambda)}{P(O|\lambda)}$$

由前向概率 $\alpha_t(i)$ 和后向概率 $\beta_t(i)$ 定义可知, $\alpha_t(i)\beta_t(i)$ 为在 $HMM~~~\lambda$ 下，由前向和后向算法导出同一个中间节点 $S$ ， 且 $t$ 时刻状态为 $q_i$ 的概率。

$$\alpha_t(i)\beta_t(i)=P(i_t=q_i,O|\lambda)$$



于是有

$$\gamma_t(i)=\frac{P(i_t=q_i,O|\lambda)}{P(O|\lambda)}=\frac{\alpha_t(i)\beta_t(i)}{\sum_{j=1}^N\alpha_t(j)\beta_t(j)}$$

2.给定模型 $\lambda$ 和观测 $O$ ,在时刻 $t$ 处于状态 $q_i$ 且在时刻 $t+1$ 处于状态 $q_j$ 的概率：

$$\xi_t(i,j)=P(i_t=q_i,i_{t+1} = q_j|O,\lambda)$$

通过前向后向概率计算:

$$\xi_t(i,j)=\frac{P(i_t=q_i,i_{t+1} = q_j,O|\lambda)}{P(O|\lambda)}=\frac{P(i_t=q_i,i_{t+1} = q_j,O|\lambda)}{\sum\limits_{i=1}^N\sum\limits_{j=1}^NP(i_t=q_i,i_{t+1} = q_j,O|\lambda)}$$

而 $\xi_t(i,j)$ ,其物理含义：从 $t$ 时刻出发由前向算法导出的中间节点 $S_i$ 和从 $t+1$ 时刻出发，由后向算法导出的中间节点 $S_j$ ,且节点 $S_i,S_j$ 中间还有一条加权有向边的关系 $a_{ij}b_j(O_{t+1})$ ，如下图:

<center class="half">
    <img src="https://darknessbeforedawn.github.io/test-book/images/HMM4.png"/>
</center>

$$P(i_t=q_i,i_{t+1} = q_j,O|\lambda)=\alpha_t(i)a_{ij}b_j(O_{t+1})\beta_{t+1}(j)$$

所以 

$$\xi_t(i,j)=\frac{\alpha_t(i)a_{ij}b_j(O_{t+1})\beta_{t+1}(j)}{\sum\limits_{i=1}^N\sum\limits_{j=1}^N\alpha_t(i)a_{ij}b_j(O_{t+1})\beta_{t+1}(j)}$$

3.将 $\gamma_t(i)$ 和 $\xi_t(i,j)$ 对各个时刻 $t$ 求和,可以得到一些有用的期望值:

(1)在观测 $O$ 下状态 $i$ 出现的期望

$$\sum_{t=1}^T=\gamma_t(i)$$

(2)在观测 $O$ 下由状态 $i$ 转移的期望

$$\sum_{t=1}^{T-1}=\gamma_t(i)$$

(3)在观测 $O$ 下由状态 $i$ 转移到状态 $j$ 的期望值

$$\sum_{t=1}^{T-1}=\xi_t(i,j)$$

# 3. 学习算法

## 3.1 监督学习算法

假设已给训练数据包含 $S$ 个长度相同的观测序列和对应的状态序列

$$\{(O_1,I_1),(O_2,I_2),\cdots,(O_S,I_S)\}$$ 

那么可以利用极大似然估计法来估计 $HMM$ 的参数:

1.转移概率 $a_{ij}$ 的估计

设样本中时刻 $t$ 处于状态 $i$ 时刻 $t+1$ 转移到状态 $j$ 的频数为 $A_{ij}$ ,那么状态转移概率 $a_{ij}$ 的估计是

$$\hat{a}_{ij} =\frac{A_{ij}}{\sum\limits_{j=1}^NA_{ij}},~~~~i,j=1,2,\cdots,N$$ 

2.观测概率 $b_j(k)$ 的估计

设样本中状态为 $j$ 并观测为 $k$ 的频数是 $B_{jk}$ ,那么状态为 $j$ 观测为 $k$ 的概率 $b_j(k)$ 的估计是

$$\hat{b}_j(k)=\frac{B_{jk}}{\sum\limits_{k=1}^MB_{jk}},~~~~~j=1,2,\cdots,N;k=1,2,\cdots,M$$

3.初始状态概率 $\pi_i$ 的估计 $\hat{\pi}_i$ 为 $S$ 个样本中初始化状态为 $q_i$ 的频率

由于监督学习需要使用训练数据，而人工标注训练数据代价很高，有时就会用非监督学习的方法。

## 3.2 Baum-Welch算法

假设给定训练数据只包含 $S$ 个长度为 $T$ 的观测序列

$$\{O_1,O_2,\cdots,O_S\}$$ 

而没有对应的状态序列，目标是学习隐马尔科夫模型 $\lambda=(A,B,\pi)$ 的参数。将观测序列数据看作观测数据 $O$ ，状态序列数据看作不可观测的隐数据 $I$ ，那么 $HMM$ 事实上是一个含有隐变量的概率模型

$$P(O|\lambda)=\sum_IP(O|I,\lambda)P(I|\lambda)$$

它的参数学习可以由 $EM$ 算法实现。

1.确定完全数据的对数似然函数

所有观测数据写成 $O=(o_1,o_2,\cdots,o_T)$ ，所有隐数据写成 $I=(i_1,i_2,\cdots,i_T)$ ，完全数据是 $(O,I)=$(o_1,o_2,\cdots,o_T,i_1,i_2,\cdots,i_T)$ 。完全数据的对数似然函数是 

$$\log P(O,I|\lambda)$$

2. $EM$ 算法的 $E$ 步: 求 $Q$ 函数 $Q(\lambda,\overline{\lambda})$

$$Q(\lambda,\overline{\lambda}) = E_I[\log P(O,I|\lambda)|O,\overline{\lambda}]=\sum_IP(I|O,\overline{\lambda})\log P(O,I|\lambda)=\sum_I\frac{P(O,I|\overline{\lambda})}{P(O|\overline{\lambda})}\log P(O,I|\lambda)$$

其中， $\overline{\lambda}$ 是 $HMM$ 参数的当前估计值， $\lambda$ 是要极大化的 $HMM$ 参数，对于 $\lambda$ 来说 $frac{1}{P(O|\overline{\lambda})}$ 为常数因子，可省略。

$$P(O,I|\lambda)=\pi_{i_1}b_{i_1}(o_1)a_{i_1i_2}b_{i_2}(o_2)\cdots a_{i_{T-1}i_T}b_{i_T}(o_T)$$

于是有:

$$Q(\lambda,\overline{\lambda})=\sum_I\log \pi_{i_1} P(O,I|\overline{\lambda})+\sum_I\biggl(\sum_{t=1}^{T-1}\log a_{i_ti_{t+1}}\biggr)P(O,I|\overline{\lambda}) +\sum_I\biggl(\sum_{t=1}^{T}\log b_{i_t}(o_t)\biggr)P(O,I|\overline{\lambda})$$ 

式中求和都是对所有训练数据的序列总长度 $T$ 进行的。

3. $EM$ 算法的 $M$ 步：极大化 $Q$ 函数求模型参数 $A,B,\pi$

由于要极大化的参数在 $Q$ 函数中单独地出现在3个项中，所以只需对各项分别极大化

(1)推导出的 $Q$ 函数中的第一项可写成

$$\sum_I\log \pi_{i_1} P(O,I|\overline{\lambda})=\sum_{i=1}^N\log \pi_{i} P(O,i_1=i|\overline{\lambda})$$

注意到 $\pi_i$ 满足约束条件 $\sum\limits_{i=1}^N\pi_i=1$ ，利用拉格朗日乘子法，写出拉格朗日函数:

$$\sum_{i=1}^N\log \pi_{i} P(O,i_1=i|\overline{\lambda}) + \gamma \biggl(\sum_{i=1}^N\pi-1\biggr)$$

对其求偏导，并令结果为 0

$$\frac{\partial}{\partial\pi_i}\biggl[\sum_{i=1}^N\log \pi_{i} P(O,i_1=i|\overline{\lambda}) + \gamma \biggl(\sum_{i=1}^N\pi-1\biggr)\biggr]=0$$

得

$$P(O,i_1=i|\overline{\lambda})+\gamma\pi_i = 0$$

对 $i$ 求和得到 $\gamma$

$$\gamma = -P(O|\overline{\lambda})$$

则 $\pi_i$ 为

$$\pi_i=\frac{P(O,i_1=i|\overline{\lambda})}{P(O|\overline{\lambda})}$$

(2)推导出的 $Q$ 函数的第二项可以写成

$$\sum_I\biggl(\sum_{t=1}^{T-1}\log a_{i_ti_{t+1}}\biggr)P(O,I|\overline{\lambda}) = \sum_{i=1}^N\sum_{j=1}^N\sum_{t=1}^{T-1}\log a_{ij} P(O,i_t=i,i_{t+1}=j|\overline{\lambda})$$

类似第一项，应用具有约束条件 $\sum\limits_{i=1}^Na_{ij}=1$ 的拉格朗日乘子法可以求出

$$a_{ij}=\frac{\sum\limits_{t=1}^{T-1}P(O,i_t=i,i_{t+1}=j|\overline{\lambda})}{\sum\limits_{t=1}^{T-1}P(O,i_t=i|\overline{\lambda})}$$

(3)推导出的 $Q$ 函数的第三项为：

$$\sum_I\biggl(\sum_{t=1}^{T}\log b_{i_t}(o_t)\biggr)P(O,I|\overline{\lambda})=\sum_{j=1}^N\sum_{t=1}^{T}\log b_j(o_t)P(O,i_t=j|\overline{\lambda})$$

同样用拉格朗日乘子法，约束条件是 $\sum\limits_{k=1}^Mb_j(k)=1$ .注意只有在 $o_t=v_k$ 时 $b_j(o_t)$ 对 $b_j(k)$ 的偏导树才不为 0 ，以 $I(o_t=v_k)$ 表示，求得

$$b_j(k)=\frac{\sum\limits_{t=1}^TP(O,i_t=j|\overline{\lambda})I(o_t=v_k)}{\sum\limits_{t=1}^TP(O,i_t=j|\overline{\lambda})}$$

## 3.3 Baum-Welch模型参数估计公式

由

$$\gamma_t(i)=\frac{P(i_t=q_i,O|\lambda)}{P(O|\lambda)}$$

$$\xi_t(i,j)=P(i_t=q_i,i_{t+1}=q_j|O,\lambda)$$

可将上节估计出的参数写成:

$$a_{ij}=\frac{\sum\limits_{t=1}^{T-1}\xi_t(i,j)}{\sum\limits_{t=1}^{T-1}\gamma_t(i)}$$

$$b_j(k)=\frac{\sum\limits_{t=1,o_t=v_k}^T\gamma_t(j)}{\sum\limits_{t=1}^T\gamma_t(j)}$$

$$\pi_i=\gamma_1(i)$$

以上三个结果就是 Baum-Welch 算法，它是 $EM$ 算法在 $HMM$ 学习中的具体实现，由 Baum 和 Welch 提出。

**Baum-Welch 算法流程**

输入: 观测数据 $O=(o_1,o_2,\cdots,o_T)$ ;

输出: $HMM$ 参数

(1)初始化

对 $n=0$ ,选取 $a_{ij}^{(0)},b_j(k)^{(0)},\pi_i^{(0)}$ ，得到模型 $\lambda^{(0)}=(A^{(0)},B^{(0)},\pi^{(0)})$ .

(2)递推。 对 $n=1,2,\cdots,$

$$a_{ij}^{(n+1)}=\frac{\sum\limits_{t=1}^{T-1}\xi_t(i,j)}{\sum\limits_{t=1}^{T-1}\gamma_t(i)}$$

$$b_j(k)^{(n+1)}=\frac{\sum\limits_{t=1,o_t=v_k}^T\gamma_t(j)}{\sum\limits_{t=1}^T\gamma_t(j)}$$

$$\pi_i^{(n+1)}=\gamma_1(i)$$

右端各值按观测 $O=(o_1,o_2,\cdots,o_T)$ 和模型 $\lambda^{(n)}=(A^{(n)},B^{(n)},\pi^{(n)})$ 计算。

(3) 终止，得到模型参数 $\lambda^{(n+1)}=(A^{(n+1)},B^{(n+1)},\pi^{(n+1)})$

# 4. 预测算法

## 4.1 近似算法

近似算法思想是，在每个时刻 $t$ 选择在该时刻最优肯出现的状态$$i_t^*$$,从而得到一个状态序列$$I^*=(i_1^*,i_2^*,\cdots,i_T^*)$$,将它作为预测的结果。

给定 $HMM~~~\lambda$ 和观测序列 $O$ ，在时刻 $t$ 处于状态 $q_i$ 的概率 $\gamma_t(i)$ 是

$$\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)}=\frac{\alpha_t(i)\beta_t(i)}{\sum\limits_{j=1}^N\alpha_t(j)\beta_t(j)}$$

在每一时刻 $t$ 最有可能的状态$$i_t^*$$是

$$i_t^*=\arg\max_{1\leqslant i\leqslant N}[\gamma_t(i)],t=1,2,\cdots,T$$

而得到状态序列$$I^*=(i_1^*,i_2^*,\cdots,i_T^*)$$.

近似算法的有点是计算简单，其确定是不能保证预测的状态序列整体是最有可能的状态序列，因为预测的状态序列可能有实际不发生的部分。事实上，上述方法得到的状态序列中有可能存在转移概率为 0 的相邻状态，即对某些 $i,j,a_{ij}=0$ 时，尽管如此，近似算法仍然是有用的。

## 4.2 维特比算法

维特比算法是用动态规划解 $HMM$ 的预测问题，用动态规划(Dynamic Programming)求概率最大路径。这时一条路径对应一个状态序列。

根据动态规划原理，最优路径具有如下特性:如果最优路径在时刻 $t$ 通过结点$$i_t^*$$,那么这一路径从结点$$i_t^*$$到终点$$i_T^*$$的部分路径，对于从$$i_t^*$$到$$i_T^*$$的所有可能的部分路径来说，必须是最优的。因为如果从$$i_t^*$$到$$i_T^*$$有另一条更好的部分路径存在，那么把它和从$$i_1^*$$到$$i_t^*$$的部分路径连接起来，就会形成一条比原来更优的路径，这是矛盾的。依据这一原理，我们只需从时刻 $t=1$ 开始，递推地计算在时刻 $t$ 状态为 $i$ 的各条部分路径的最大概率，直至得到时刻 $t=T$ 状态为 $i$ 的各条路径的最大概率。时刻 $t=T$ 的最大概率即为最优路径的概率$$P^*$$,最优路径的终结点$$i_T^*$$也同时得到。之后，为了找出最优路径的各个结点，从终结点$$i_T^*$$开始，由后向前逐步求得结点$$i_{T-1}^*,\cdots,i_{1}^*$$，得到最优路径$$I^*=(i_1^*,i_2^*,\cdots,i_T^*)$$.以上即为维特比算法。

首先导入两个变量 $\delta$ 和 $\psi$ .定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $(i_1,i_2,\cdots,i_t)$ 中概率最大值为

$$\delta_t(i)=\max_{i_1,i_2,\cdots,i_{t-1}}P(i_t=i,i_{t-1},\cdots,i_1,o_t,\cdots,o_1|\lambda),~~~i=1,2,\cdots,N$$

由定义可得变量 $\delta$ 的递推公式:

$$\delta_{t+1}(i)=\max_{i_1,i_2,\cdots,i_{t}}P(i_{t+1}=i,i_{t},\cdots,i_1,o_{t+1},\cdots,o_1|\lambda)=\max_{1\leqslant j\leqslant N}[\delta_t(j)a_{ji}]b_i(o_{t+1}),~~~t=1,2,\cdots,T-1$$

定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $(i_1,i_2,\cdots,i_{t-1},i)$ 中概率最大的路径的第 $t-1$ 个结点为

$$\psi_t(i)=\arg \max_{1\leqslant j\leqslant N}[\delta_{t-1}(j)a_{ji}],~~~i=1,2,\cdots,N$$

**维特比算法流程**

输入：模型 $\lambda=(A,B,\pi)$ 和观测 $O=(o_1,o_2,\cdots,o_T)$ ;

输出：最优路径

$$I^*=(i_1^*,i_2^*,\cdots,i_T^*)$$

(1)初始化

$$\delta_1(i)=\pi_ib_i(o_1),~~~\psi_1(i)=0,~~~i=1,2,\cdots,N$$

(2)递推。对 $t=2,3,\cdots,T$ 

$$\delta_{t}(i)=\max_{1\leqslant j\leqslant N}[\delta_{t-1}(j)a_{ji}]b_i(o_{t}),~~~i=1,2,\cdots,N$$

$$\psi_t(i)=\arg \max_{1\leqslant j\leqslant N}[\delta_{t-1}(j)a_{ji}],~~~i=1,2,\cdots,N$$

(3)终止

$$P^*=\max_{1\leqslant i\leqslant N}\delta_T(i)$$

$$i^*_T=\arg \max_{1\leqslant i\leqslant N}[\delta_T(i)]$$

(4)最优路径回溯。对 $t=T-2,T-2,\cdots,1$ 

$$i_t^*\psi_{t+1}(i^*_{t+1})$$

求得最优路径

$$I^*=(i_1^*,i_2^*,\cdots,i_T^*)$$