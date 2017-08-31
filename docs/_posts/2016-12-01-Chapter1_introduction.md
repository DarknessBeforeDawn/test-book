---
title: 线性回归
layout: post
share: false
---

线性回归最简单的形式:

$D=\{(x_i, y_i)\}_{i=1}^m$,$x_i \in\mathbb{R}$,线性回归试图学得合适的$w$和$b$,使得
$$f(x_i)=wx_i+b,f(x_i)\simeq y_i$$
即使得$f(x_i)$与$y_i$之间的差别尽量小,因此我们可以使其均方误差最小，即
$$min\sum_{i=1}^m(y_i-wx_i-b)^2$$
令$E(w,b)=\sum_{i=1}^m(y_i-wx_i-b)^2$
分别对$w,b$求偏导
$$\frac{\partial E(w,b)}{\partial w}=2\sum_{i=1}^m(y_i-wx_i-b)(-x_i)=2\biggl(w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-b)x_i\biggr)$$
$$\frac{\partial E(w,b)}{\partial b}=-2\sum_{i=1}^m(y_i-wx_i-b)=2\biggl(mb-\sum_{i=1}^m(y_i-wx_i)\biggr)$$
令偏导为零可得
$$b=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i);$$
$$w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-\frac{1}{m}\sum_{i=1}^m(y_i-wx_i))x_i=w\sum_{i=1}^mx_i^2-\sum_{i=1}^mx_iy_i+\frac{1}{m}\sum_{i=1}^m\sum_{i=1}^m(x_iy_i-wx_i^2)$$


$$=w\sum_{i=1}^mx_i^2-\sum_{i=1}^mx_iy_i+\overline{x}\sum_{i=1}^my_i-\frac{w}{m}\sum_{i=1}^m\sum_{i=1}^mx_i^2=0；$$求得
$$w=\frac{\sum_{i=1}^my_i(x_i-\overline{x})}{\sum_{i=1}^mx_i^2-\frac{1}{m}\sum_{i=1}^m\sum_{i=1}^mx_i^2}；$$
其中$\overline{x}=\frac{1}{m}\sum_{i=1}^mx_i$.

对于多参数情形：
$$f(\mathbf{x}_i)=w^Tx_i+b,f(\mathbf{x}_i)\simeq y_i$$

令$\mathbf{\widehat{x}}_i =(\mathbf{x}_i;1)$,$\mathbf{\widehat{w}} = (\mathbf{w};1)$,$\mathbf{X}$为所有$\mathbf{\widehat{x}}_i$组成的矩阵，则
$$E_{\mathbf{\widehat{w}}} = (\mathbf{y}-\mathbf{X\widehat{w}})^T(\mathbf{y}-\mathbf{X\widehat{w}})$$
对$\mathbf{\widehat{w}}$求导得（参考矩阵求导），
$$\frac{\partial E_{\mathbf{\widehat{w}}}}{\partial \mathbf{\widehat{w}}}=2\mathbf{X}^T(\mathbf{X}{\mathbf{\widehat{w}}}-\mathbf{y})$$
当$\mathbf{X}^T\mathbf{X}$为满秩矩阵或正定矩阵时，令导数为零，求得
$$\mathbf{\widehat{w}}^*=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
但在实际问题中$\mathbf{X}^T\mathbf{X}$往往不是满秩矩阵，并且当参数多并且数据较多时，求导的计算量是非常大的。在实际问题中,令$f(\mathbf{x})=\mathbf{\theta}^T\mathbf{x}$，并将$\mathbf{x}$到$y$的映射函数$f$记作$\mathbf{\theta}$的函数$h_{\mathbf{\theta}}(\mathbf{x})$,则线性回归的损失函数一般定义为：
$$J(\mathbf{\theta})=\frac{1}{2m}\sum_{i=1}^m(h_{\mathbf{\theta}}(\mathbf{x}^{(i)})-y^{(i)})$$并通过梯度下降法进行迭代逐步接近最小点，迭代过程中$\mathbf{\theta}$不断更新：
$$\mathbf{\theta}_j:=\mathbf{\theta}_j-\alpha \frac{\partial }{\partial \mathbf{\theta}_j}J(\theta _0,\theta _1,\cdots,\theta _n)$$
其中$\alpha$为步长，也称为学习率。线性回归相关代码。