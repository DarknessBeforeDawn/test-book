---
title: logistic回归
layout: post
share: false
---

对于二分类问题，假设$$y\in \{0,1\}$$,而线性回归预测值$$z=\theta^Tx$$是一个实值，对于这个问题，我们引入sigmoid函数:$$y=\frac{1}{1+e^{-z}}$$，sigmoid函数可以将$$z$$值转化为0到1之间的一个值,sigmoid函数特性$$y'=y(1-y)$$。即预测函数
\begin{equation}
h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}
\end{equation}
可推导出
\begin{equation}
\ln\frac{y}{1-y}=\theta^Tx
\end{equation}

若将$$y$$看做$$x$$为正样本的概率，则$$1-y$$即为$$x$$为负样本的概率，$$\frac{y}{1-y}$$称为几率（odds）,反映$$x$$作为正样本的相对可能性，$$\ln\frac{y}{1-y}$$为对数几率(log odds,亦称logit).

对于样本$$x$$，其为正样本和负样本的概率为：
\begin{equation}
P(y=1|x;\theta)=\frac{1}{1+e^{-\theta^Tx}}=\frac{e^{\theta^Tx}}{1+e^{\theta^Tx}}
\end{equation}
\begin{equation}
P(y=0|x;\theta)=1-P(y=1|x;\theta)=\frac{e^{-\theta^Tx}}{1+e^{-\theta^Tx}}=\frac{1}{1+e^{\theta^Tx}}
\end{equation}
上述两个式子可以合并成：
\begin{equation}
P(y|x;\theta)=(P(y=1|x;\theta))^y(P(y=0|x;\theta))^{1-y}\end{equation}
然后利用最大似然估计，写出似然函数：
\begin{equation}
L(\theta)=P(y^{(i)}|x^{(i)};\theta)=\prod_{i=1}^m(h_{\theta}(x^{(i)}))^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}
\end{equation}
取对数，得到对数似然函数：
\begin{equation}
l(\theta)=\ln L(\theta)=\sum_{i=1}^m\biggl(y^{(i)}\ln h_{\theta}(x^{(i)})+(1-y^{(i)})ln (1-h_{\theta}(x^{(i)}))\biggr)
\end{equation}
在logit回归中：
\begin{equation}Cost(h_\theta(x),y)=
\begin{cases}
-\ln(h_\theta(x))& \text{y=1}\\
-\ln(1-h_\theta(x))& \text{y=0}
\end{cases}
\end{equation}

\begin{equation}
J(\theta)=\frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}),y^{(i)})=-\frac{1}{m} l(\theta)
\end{equation}
加入正则化项的损失函数：
\begin{equation}
J(\theta)=-\frac{1}{m}\sum_{i=1}^m\biggl(y^{(i)}\ln h_{\theta}(x^{(i)})+(1-y^{(i)})ln (1-h_{\theta}(x^{(i)}))\biggr)+\frac{\lambda}{2m}\sum_{j=1}^n\theta _j^2
\end{equation}
最大似然是求$$l(\theta)$$的最大值，而损失函数$$J(\theta)$$在$$l(\theta)$$乘以$$-\frac{1}{m}$$变为求最小值。


依旧使用梯度下降求解最小值
\begin{equation}
\mathbf{\theta}_j:=\mathbf{\theta}_j-\alpha \frac{\partial }{\partial \mathbf{\theta}_j}J(\mathbf{\theta})
\end{equation}

\begin{equation}
\begin{align*}  
  \frac{\partial }{\partial \mathbf{\theta}_j}J(\mathbf{\theta}) &= -\frac{1}{m}\sum_{i=1}^m\biggl(\frac{y^{(i)}}{h_{\theta}(x^{(i)})}\frac{\partial }{\partial \mathbf{\theta}_j}h_{\theta}(x^{(i)})-\frac{(1-y^{(i)})}{1-h_{\theta}(x^{(i)})}\frac{\partial }{\partial \mathbf{\theta}_j}h_{\theta}(x^{(i)})\biggr) \\  
 &= -\frac{1}{m}\sum_{i=1}^m\biggl(\frac{y^{(i)}}{h_{\theta}(x^{(i)})}-\frac{(1-y^{(i)})}{1-h_{\theta}(x^{(i)})}\biggr) \frac{\partial }{\partial \mathbf{\theta}_j}h_{\theta}(x^{(i)})\\  
 &= -\frac{1}{m}\sum_{i=1}^m\biggl(\frac{y^{(i)}}{h_{\theta}(x^{(i)})}-\frac{(1-y^{(i)})}{1-h_{\theta}(x^{(i)})}\biggr) h_{\theta}(x^{(i)}) (1-h_{\theta}(x^{(i)})) \frac{\partial }{\partial \mathbf{\theta}_j}\theta ^Tx^{(i)} \\ 
&= -\frac{1}{m}\sum_{i=1}^m\biggl(y^{(i)}(1-h_{\theta}(x^{(i)}))-(1-y^{(i)})h_{\theta}(x^{(i)})\biggr)x^{(i)}_j \\
&= \frac{1}{m}\sum_{i=1}^m\biggl(h_{\theta}(x^{(i)})-y^{(i)}\biggr)x^{(i)}_j
\end{align*} 
\end{equation}
则$$\theta$$的更新过程：
\begin{equation}
\mathbf{\theta}_j:=\mathbf{\theta}_j-\frac{\alpha}{m}\sum_{i=1}^m\biggl(h_{\theta}(x^{(i)})-y^{(i)}\biggr)x^{(i)}_j
\end{equation}
加入正则化项的更新过程：
\begin{equation}
\mathbf{\theta}_j:=\mathbf{\theta}_j-\frac{\alpha}{m}\sum_{i=1}^m\biggl(h_{\theta}(x^{(i)})-y^{(i)}\biggr)x^{(i)}_j-\frac{\lambda}{m}\theta _j
\end{equation}

[logistic回归相关代码](https://github.com/DarknessBeforeDawn/test-book/blob/master/code/logistic_regression/logistic_regression.md)
