---
title: 线性回归
layout: post
share: false
---

线性回归最简单的形式:$$D=\{(x_i, y_i)\}_{i=1}^m,x_i \in\mathbb{R}$$ ,线性回归试图学得合适的$$w$$和$$b$$,使得
\begin{equation}
f(x_i)=wx_i+b,f(x_i)\simeq y_i
\end{equation}
即使得$$f(x_i)$$与$$y_i$$之间的差别尽量小,因此我们可以使其均方误差最小，即
\begin{equation}
min\sum_{i=1}^m(y_i-wx_i-b)^2
\end{equation}
令$$E(w,b)=\sum_{i=1}^m(y_i-wx_i-b)^2$$
分别对$$w,b$$求偏导
\begin{equation}
\frac{\partial E(w,b)}{\partial w}=2\sum_{i=1}^m(y_i-wx_i-b)(-x_i)=2\biggl(w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-b)x_i\biggr)
\end{equation}
\begin{equation}
\frac{\partial E(w,b)}{\partial b}=-2\sum_{i=1}^m(y_i-wx_i-b)=2\biggl(mb-\sum_{i=1}^m(y_i-wx_i)\biggr)
\end{equation}
令偏导为零可得
\begin{equation}
b=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i);
\end{equation}
\begin{equation}
w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-\frac{1}{m}\sum_{i=1}^m(y_i-wx_i))x_i=w\sum_{i=1}^mx_i^2-\sum_{i=1}^mx_iy_i+\frac{1}{m}\sum_{i=1}^m\sum_{i=1}^m(x_iy_i-wx_i^2)
\end{equation}


\begin{equation}=w\sum_{i=1}^mx_i^2-\sum_{i=1}^mx_iy_i+\overline{x}\sum_{i=1}^my_i-\frac{w}{m}\sum_{i=1}^m\sum_{i=1}^mx_i^2=0；\end{equation}求得
\begin{equation}w=\frac{\sum_{i=1}^my_i(x_i-\overline{x})}{\sum_{i=1}^mx_i^2-\frac{1}{m}\sum_{i=1}^m\sum_{i=1}^mx_i^2}；
\end{equation}
其中$$\overline{x}=\frac{1}{m}\sum_{i=1}^mx_i$$.

对于多参数情形：
\begin{equation}f(\mathbf{x}_i)=w^Tx_i+b,f(\mathbf{x}_i)\simeq y_i
\end{equation}

令$$\mathbf{\widehat{x}}_i =(\mathbf{x}_i;1)$$,$$\mathbf{\widehat{w}} = (\mathbf{w};1)$$,$$\mathbf{X}$$为所有$$\mathbf{\widehat{x}}_i$$组成的矩阵，则
\begin{equation}E_{\mathbf{\widehat{w}}} = (\mathbf{y}-\mathbf{X\widehat{w}})^T(\mathbf{y}-\mathbf{X\widehat{w}})
\end{equation}
对$$\mathbf{\widehat{w}}$$求导得（参考矩阵求导），
\begin{equation}\frac{\partial E_{\mathbf{\widehat{w}}}}{\partial \mathbf{\widehat{w}}}=2\mathbf{X}^T(\mathbf{X}{\mathbf{\widehat{w}}}-\mathbf{y})
\end{equation}
当$$\mathbf{X}^T\mathbf{X}$$为满秩矩阵或正定矩阵时，令导数为零，求得
\begin{equation}\mathbf{\widehat{w}}^*=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
\end{equation}
但在实际问题中$$\mathbf{X}^T\mathbf{X}$$往往不是满秩矩阵，并且当参数多并且数据较多时，求导的计算量是非常大的。在实际问题中,令$$f(\mathbf{x})=\mathbf{\theta}^T\mathbf{x}$$，并将$$\mathbf{x}$$到$$y$$的映射函数$$f$$记作$$\mathbf{\theta}$$的函数$$h_{\mathbf{\theta}}(\mathbf{x})$$,则线性回归的损失函数一般定义为：
\begin{equation}J(\mathbf{\theta})=\frac{1}{2m}\sum_{i=1}^m(h_{\mathbf{\theta}}(\mathbf{x}^{(i)})-y^{(i)})^2
\end{equation}并通过梯度下降法进行迭代逐步接近最小点，迭代过程中$$\mathbf{\theta}$$不断更新：
\begin{equation}\mathbf{\theta}_j:=\mathbf{\theta}_j-\alpha \frac{\partial }{\partial \mathbf{\theta}_j}J(\theta _0,\theta _1,\cdots,\theta _n)
\end{equation}
其中$$\alpha$$为步长，也称为学习率。

当我们的模型比较复杂，学习能力比较强时，容易造成过拟合的情况，例如如下模型：
\begin{equation}\theta _0+\theta _1x+\theta _2x^2+\theta _3x^3+\theta _4x^4
\end{equation}

对于过拟合，我们可以在损失函数中加入相应的正则化项来控制参数幅度，添加正则化项后的损失函数：\begin{equation}
J(\mathbf{\theta})=\frac{1}{2m}\biggl[\sum_{i=1}^m(h_{\mathbf{\theta}}(\mathbf{x}^{(i)})-y^{(i)})^2+\lambda \sum_{j=1}^n \theta _j^2\biggr]
\end{equation}


[线性回归参考代码](https://github.com/DarknessBeforeDawn/test-book/blob/master/code/linear_regression/linear_regression.md)


