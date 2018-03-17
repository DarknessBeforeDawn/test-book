---
title: 条件随机场
layout: post
share: false
---

条件随机场(Conditional Random Field,CRF)是给定一组输入随机变量条件下另一组输出随机变量的条件概率分布模型，其特点是假设输出随机变量构成马尔科夫随机场。

# 1. 概率无向图模型

概率无向图模型(Prodbabilistic Undirected Graphical Model),又称为马尔科夫随机场(Markov Random Field)，是一个可以由无向图模型表示的联合概率分布。

# 1.1 模型定义

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