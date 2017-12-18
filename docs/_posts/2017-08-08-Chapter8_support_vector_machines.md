---
title: SVM(支持向量机)
layout: post
share: false
---

# 1.预备知识

### 1.1 $KKT$ 条件

#### 无约束优化
对于变量 $x\in\mathbb{R}^n$ 的函数 $f(x)$ ,无约束优化问题如下:

$$\min_xf(x)$$

直接找到使目标函数导数为 $0$ 的点即可，即 $f'(x)=0$ ,如果没有解析解可以使用梯度下降或牛顿法等通过迭代使 $x$ 沿负梯度方向逐渐逼近最小值点。

##### 等式约束

如下等式约束问题：

$$\begin{aligned}
&\min_xf(x)  \\  
&s.t.~~~~h_i(x) = 0 , i = 1,2,...,m \\ 
\end{aligned}$$

约束条件会将解的范围限定在一个可行区域，此时不一定能找到 $f'(x)$ 为 $0$ 的点，只需找到可行区域内使得 $f(x)$ 最小的点即可，一般使用拉格朗日乘子法来进行求解，引入拉格朗日乘子 $\alpha\in\mathbb{R}^m$ ,构建拉格朗日函数：

$$L(x,\alpha)=f(x)+\sum_{i=1}^m\alpha_ih_i(x)$$

并分别对 $\alpha$ 和 $x$ 求偏导：

$$\left 
\{ 
\begin{aligned}  
\frac{\partial L(x,\alpha)}{\partial x}= 0  \\ 
\frac{\partial L(x,\alpha)}{\partial \alpha}= 0 
\end{aligned} 
\right.$$

求得 $x$ 、 $\alpha$ 的值，将 $x$ 代入 $f(x)$ 即为在约束条件 $h_i(x)$ 下的可行解。下面用一个示例来进行说明，对于二维的目标函数 $f(x,y)$ ,在平面中画出   $f(x,y)$ 的等高线，如下图虚线，并给出一个约束等式 $h(x,y)=0$ ,如下图绿线，目标函数 $f(x,y)$ 与约束 $g(x,y)$ 只可能相交，相切或没交集，只有相交或相切时才有可能是解，而且只有相切才可能得到可行解，因为相交意味着肯定还存在其它的等高线在该条等高线的内部或者外部，使得新的等高线与目标函数的交点的值更大或者更小。

![](https://darknessbeforedawn.github.io/test-book/images/SVM1.png)

因此,拉格朗日乘子法取得极值的必要条件是目标函数与约束函数相切，这时两者的法向量是平行的，即

$$f'(x)-\alpha h'(x)=0$$

所以只要满足上述等式，且满足约束 $h_i(x) = 0 , i = 1,2,…,m$ 即可得到解，联立起来，正好得到就是拉格朗日乘子法。以上为拉格朗日乘子法的几何推导。

##### 不等式约束

给定如下不等式约束问题：

$$\begin{aligned}
&\min_xf(x)  \\  
&s.t.~~~~g(x) \leq 0 , i = 1,2,...,m \\ 
\end{aligned}$$

对应的拉格朗日函数：

$$
L(x,\lambda) = f(x)+\lambda g(x)
$$

这时的可行解必须落在约束区域 $g(x)$ 之内，稀土给出了目标函数的等高线与约束：

![](https://darknessbeforedawn.github.io/test-book/images/SVM2.png)

由图可知可行解 $x$ 只能在 $g(x) \leq 0$的区域里:

(1)当可行解 $x$ 落在 $g(x)<0$ 的区域内，此时直接极小化 $f(x)$ 即可；

(2)当可行解 $x$ 落在 $g(x)=0$ 即边界上，此时等价于等式约束问题。

当约束区域包含目标函数原有的可行解时，此时加上约束可行解仍然落在约束区域内部，对应 $g(x)<0$ 的情况，这时约束条件不起作用；当约束区域不包含目标函数原有的可行解时，此时加上约束后可行解落在边界 $g(x)=0$ 上。下图分别描述了两种情况，右图表示加上约束可行解会落在约束区域的边界上。

![](https://darknessbeforedawn.github.io/test-book/images/SVM3.png)

以上两种情况，要么可行解落在约束边界上即得 $g(x)=0$ ，要么可行解落在约束区域内部，此时约束不起作用，令 $\lambda = 0$ 消去约束即可，所以无论哪种情况都会得到：

$$\lambda g(x)=0$$

在等式约束优化中，约束函数与目标函数的梯度只要满足平行即可，而在不等式约束中则不然，若 $\lambda \neq 0$，则可行解 $x$ 是落在约束区域的边界上，这时可行解应尽量靠近无约束时的解，所以在约束边界上，目标函数的负梯度方向应该远离约束区域朝向无约束时的解，此时正好可得约束函数的梯度方向与目标函数的负梯度方向应相同：

$$-\nabla_xf(x)=\lambda\nabla_xg(x)$$

上式需要满足 $\lambda > 0$ ，这个问题可以举一个形象的例子，假设你去爬山，目标是山顶，但有一个障碍挡住了通向山顶的路，所以只能沿着障碍爬到尽可能靠近山顶的位置，然后望着山顶叹叹气，这里山顶便是目标函数的可行解，障碍便是约束函数的边界，此时的梯度方向一定是指向山顶的，与障碍的梯度同向，下图描述了这种情况:

![](https://darknessbeforedawn.github.io/test-book/images/SVM4.png)

对于不等式约束，只要满足一定的条件，依然可以使用拉格朗日乘子法解决，这里的条件便是 $KKT$ 条件。

对于以下约束问题:

$$\begin{aligned}
&\min_xf(x)  \\  
&s.t.~~~~h_i(x)=0,i = 1,2,...,m \\
&~~~~~~~~~~g(x) \leq 0 , j = 1,2,...,n \\ 
\end{aligned}$$

对应拉格朗日函数：

$$L(x,\alpha,\beta)=f(x)+\sum_{i=1}^m\alpha_ih_i(x)+\sum_{j=1}^n\beta_ig_i(x)$$

则不等式约束后的可行解 $x$ 需要满足的 $KKT$ 条件为：

$$\begin{align}
\nabla_x L(x,\alpha,\beta) &= 0   \\ 
\beta_jg_j(x) &= 0  , \ j=1,2,...,n\\ 
h_i(x)&= 0 , \ i=1,2,...,m  \\ 
g_j(x) &\le 0  , \  j=1,2,...,n  \\ 
\beta_j &\ge  0 , \ j=1,2,...,n  \\ 
\end{align} $$

满足 $KKT$ 条件后极小化拉格朗日函数即可得到在不等式约束条件下的可行解。 $KKT$ 条件:

(1)拉格朗日取得可行解的必要条件；

(2)这就是以上分析的一个比较有意思的约束，称作松弛互补条件；

(3)(4)初始的约束条件；

(5)不等式约束的 Lagrange Multiplier 需满足的条件。

主要的 $KKT$ 条件便是 (3) 和 (5) ，只要满足这俩个条件便可直接用拉格朗日乘子法， $SVM$ 中的支持向量便是来自于此，需要注意的是 KKT 条件与对偶问题也有很大的联系。

### 1.2 对偶问题

对于任意一个带约束的优化都可以写成如下形式：

$$\begin{aligned}  
&\min_x \  f(x)  \\  
&s.t.  \ \ \ h_i(x) = 0 , \  i = 1,2,...,m \ \\  
& \ \ \ \ \ \ \ \ \ \   g_j(x) \le 0, \  j = 1,2,...,n 
\end{aligned}$$

如果 $g_j(x)$ 全是凸函数，并且 $h_i(x)$ 全是仿射函数（ $Ax+b$ 的形式），上述优化就是一个凸优化问题，凸优化极值唯一。

定义如下 $Lagrangian$ ：

$$
L(x,\alpha,\beta) =f(x) + \sum_{i=1}^m \alpha_i h_i(x) + \sum_{j=1}^n\beta_jg_j(x)
$$

它通过一些系数把约束条件和目标函数结合起来，将带约束的优化问题转化为无约束问题。

现在我们针对参数 $\alpha,\beta$ 对 $L(x,\alpha,\beta)$ 取最大值，令:

$$
Z(x) =\max_{\alpha_i,\beta_j \geq 0}L(x,\alpha,\beta)
$$

满足约束条件的 $x$ 使得 $h_i(x)=0$ ,并且 $g_j(x) \leq 0,\beta_j \geq 0$ ,则有 $\beta_jg_j(x) \leq 0$ .因此对于满足约束条件的 $x$ 有 $f(x)=Z(x)$ .而对于那些不满足约束条件的 $x$ 有 $Z(x)=\infty$ ,这样将导致问题无解，因此必须满足约束条件。这样一来，原始带约束的优化问题等价于如下无约束优化问题：

$$
\min_x f(x) = \min_xZ(x) = \min_x \max_{\alpha_i,\beta_j \geq 0}L(x,\alpha,\beta)
$$

这个问题称作原问题(primal problem),与之相对应的为对偶问题(dual problem),器形式非常类似，只是把 $\min,\max$ 交换了一下：

$$
\max_{\alpha_i,\beta_j \geq 0}\min_xL(x,\alpha,\beta)
$$

原问题是在最大值中取最下的那个，对偶问题是在最小值取最大的那个。令：

$$
D(\alpha, \beta)= \min_xL(x,\alpha,\beta)
$$

如果原问题的最小值记为 $p^*$ ,那么对于所有 $\alpha_i,\beta_j \geq 0$ 有：

$$
D(\alpha, \beta) \leq p^*
$$

由于对于极值点(包括所有满足约束条件的点) $x^*$ , 并且 $\beta_j \geq 0$ ,总有 :

$$
\sum_{i=1}^m \alpha_i h_i(x^*) + \sum_{j=1}^n\beta_jg_j(x^*) \leq 0
$$ 

因此

$$
L(x^*,\alpha,\beta) = f(x^*) +\sum_{i=1}^m \alpha_i h_i(x^*) + \sum_{j=1}^n\beta_jg_j(x^*) \leq f(x^*)
$$

于是

$$
D(\alpha, \beta)= \min_xL(x,\alpha,\beta) \leq L(x^*,\alpha,\beta) \leq f(x^*) = p^*
$$

因此 

$$
\max_{\alpha_i,\beta_j \geq 0}D(\alpha, \beta)
$$

实际上就是原问题的最大下界，最大下界离我们要逼近的值最近。记对偶问题的最优值为 $d^*$ ，则有：

$$d^* \leq p^*$$

这个性质叫做弱对偶性（weak duality），对于所有优化问题都成立，即使原始问题非凸。这里还有两个概念： $f(x)–D(\alpha, \beta)$ 叫做对偶间隔（duality gap）， $p^*–d^*$ 叫做最优对偶间隔（optimal duality gap）。无论原始问题是什么形式，对偶问题总是一个凸优化的问题，这样对于那些难以求解的原始问题 （甚至是 NP 问题），均可以通过转化为偶问题，通过优化这个对偶问题来得到原始问题的一个下界， 与弱对偶性相对应的有一个强对偶性（strong duality） ，强对偶即满足：

$$d^* = p^*$$

强对偶是一个非常好的性质，因为在强对偶成立的情况下，可以通过求解对偶问题来得到原始问题的解，在 $SVM$ 中就是这样做的。当然并不是所有的对偶问题都满足强对偶性 ，在 $SVM$ 中是直接假定了强对偶性的成立，其实只要满足一些条件，强对偶性是成立的，比如说 $Slater$ 条件与 $KKT$ 条件。

#### $Slater$ 条件

若原始问题为凸优化问题，且存在严格满足约束条件的点 $x$ ，这里的“严格”是指 $g_j(x)\leq 0$ 中的“ $≤$ ”严格取到“ $<$ ”，即存在 $x$ 满足 $g_j(x)<0 ,i=1,2,…,n$ ，则存在 $x^*,α^*,β^*$ 使得 $x^*$ 是原始问题的解， $α^*,β^*$ 是对偶问题的解，且满足：

$$p^* = d^* = L(x^*,\alpha^* ,\beta^*)$$

也就是说如果原始问题是凸优化问题并且满足 $Slater$ 条件的话，那么强对偶性成立。需要注意的是，这里只是指出了强对偶成立的一种情况，并不是唯一的情况。例如，对于某些非凸优化的问题，强对偶也成立。$SVM$ 中的原始问题 是一个凸优化问题（二次规划也属于凸优化问题），$Slater$ 条件在 $SVM$ 中指的是存在一个超平面可将数据分隔开，即数据是线性可分的。当数据不可分时，强对偶是不成立的，这个时候寻找分隔平面这个问题本身也就是没有意义了，所以对于不可分的情况预先加个 $kernel$ 就可以了。

#### $KKT$ 条件

假设 $x^*$ 与 $α^*,β^*$ 分别是原始问题（并不一定是凸的）和对偶问题的最优解，且满足强对偶性，则相应的极值的关系满足：

$$
\begin{aligned}  f(x^*) &= d^* = p^* =D(\alpha^*,\beta^*)  \\  &=\min_x f(x)+ \sum_{i = 1}^m \alpha_i^*h_i(x) + \sum_{j=1}^n\beta_j^*g_j(x) \\  & \le f(x^*)+ \sum_{i = 1}^m \alpha_i^*h_i(x^*) + \sum_{j=1}^n\beta_j^*g_j(x^*) \\ &\le f(x^*)  \end{aligned}
$$

由于两头是相等的，所以这一系列的式子里的不等号全部都可以换成等号。根据第一个不等号我们可以得到 $x^*$ 是 $L(x,\alpha^*,\beta^*)$ 的一个极值点，因此 $L(x,\alpha^*,\beta^*)$ 在 $x^*$ 处的梯度为0，即: 

$$\nabla_{x^*} L(x,\alpha^*,\beta^*) = 0$$

由第二个不等式，并且 $\beta_j^*g_j(x^*)$ 都是非正的，因此有：

$$\beta_j^*g_j(x^*)=0, j=1,\cdots,m$$

显然，如果 $\beta_j^* > 0$ ,那么必定有 $g_j(x^*)=0$ ,如果 $g_j(x^*) <0$ ,那么可以得到 $\beta_j^* = 0$ 。再将其他一些显而易见的条件写到一起，就是传说中的 KKT (Karush-Kuhn-Tucker) 条件：

$$
\begin{align}
\nabla_x L(x,\alpha,\beta) &= 0   \\ 
\beta_jg_j(x) &= 0  , \ j=1,2,...,n\\ 
h_i(x)&= 0 , \ i=1,2,...,m  \\ 
g_j(x) &\le 0  , \  j=1,2,...,n  \\ 
\beta_j &\ge  0 , \ j=1,2,...,n  \\ 
\end{align}
$$

总结来说就是说任何满足强对偶性的优化问题，只要其目标函数与约束函数可微，任一对原始问题与对偶问题的解都是满足 $KKT$ 条件的。即满足强对偶性的优化问题中，若 $x^*$ 为原始问题的最优解，$α^*,β^*$ 为对偶问题的最优解，则可得 $x^*,α^*,β^*$ 满足 $KKT$ 条件。

上面只是说明了必要性，当满足原始问题为凸优化问题时，必要性也是满足的，也就是说当原始问题是凸优化问题,且存在 $x^*,α^*,β^*$ 满足 $KKT$ 条件，那么它们分别是原始问题和对偶问题的极值点并且强对偶性成立.

##### 证明
首先原始问题是凸优化问题，固定 $α^*,β^*$ 之后对偶问题 $D(α^*,β^*)$ 也是一个凸优化问题，$x^*$ 是 $L(x,α^*,β^*)$ 的极值点：

$$
\begin{aligned}  
D(\alpha^*,\beta^*)  &= \min_x L(x,\alpha^*,\beta^*) \\ 
&= L(x^*,\alpha^*,\beta^*) \\ 
& =f(x^*)+\sum_{i=1}^m\alpha_i^*h_i(x^*)+\sum_{j=1}^n\beta_j^*g_j(x^*) \\  
&= f(x^*)  
\end{aligned}
$$

最后一个式子是根据 $KKT$ 条件中的 $h_i(x)=0$ 与 $\beta_jg_j(x)=0$ 得到的。这样一来，就证明了对偶间隔为零，也就是说，强对偶成立。


对于一个约束优化问题，找到其对偶问题，当弱对偶成立时，可以得到原始问题的一个下界。而如果强对偶成立，则可以直接求解对偶问题来解决原始问题。 $SVM$ 就是这样的。对偶问题由于性质良好一般比原始问题更容易求解，在 $SVM$ 中通过引入对偶问题可以将问题表示成数据的内积形式从而使得 kernel trick 的应用更加自然。此外，还有一些情况会同时求解对偶问题与原始问题 ，比如在迭代求解的过程中，通过判断对偶间隔的大小，可以得出一个有效的迭代停止条件。

# 2.支持向量机

## 2.1 线性可分支持向量机与硬间隔最大化

### 2.1.1 线性可分支持向量机

SVM 一直被认为是效果最好的现成可用的分类算法之一，$SVM$ 内容比较多，学习 $SVM$ 首先要从线性分类器开始。考虑一个二分类问题，给定训练数据集

$$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\},~~x_i\in\mathcal{X}=\mathbf{R}^n,~~y_i\in\mathcal{Y}=\{+1,-1\},~~i=1,2,\cdots,n$$

其中 $x_i$ 为第 $i$ 个特征向量， $y_i$ 为 $x_i$ 的类标记（有些地方会选 0 和 1 ，当然其实分类问题选什么都无所谓，只要是两个不同的数字即可，不过这里选择 +1 和 -1 是为了方便 SVM 的推导），学习的目标是在特征空间中找到一个分离超平面(线性分类器),其方程可表示为：

$$w^Tx+b=0$$

其中 $w=(w_1;w_2;\cdots;w_d)$ 为法向量，决定了超平面的方向； $b$ 为位移项，决定了超平面与原点之间的距离。在二维空间中的例子就是一条直线。通过这个超平面可以把两类数据分隔开来，一部分是正类，一部分是负类，法向量指向的一侧为正类，另一侧为负类。这里我们首先来说明线性可分的情况，对于线性不可分的情况后边会进行分析。

### 2.1.2 函数间隔与几何间隔

![](https://darknessbeforedawn.github.io/test-book/images/SVM5.png)
 
如图所示，两种标记的点分别代表两个类别，直线表示一个可行的超平面。将数据点 $x$ 代入 $f(x)$ 中，如果得到的结果小于 0 ，则赋予其类别 -1 ，如果大于 0 则赋予类别 1 。对于 $f(x)$ 的绝对值很小(包括)的情况，是很难处理的，因为细微的变动（比如超平面稍微转一个小角度）就有可能导致结果类别的改变，也就是越接近超平面的点越“难”分隔。

在超平面 $w^Tx+b=0$ 确定的情况下 $$|w^Tx+b|$$ 能够相对的表示点 $x$ 与超平面的距离。 $w^Tx+b$ 的符号与类标记 $y$ 的符号是否一致可判断分类是否正确,因此 $y(w^Tx+b)$ 可以用来表示分类的正确性及确信度。

隔函数间隔（functional margin）：

$$\hat\gamma_i = y_i(w^Tx_i+b)=y_if(x_i)$$

而超平面关于 $T$ 中所有样本点 $(x_i,y_i)$ 的函数间隔最小值（ $i$ 表示第 $i$ 个样本），便为超平面关于训练数据集 $T$ 的函数间隔：

$$\hat\gamma=\min_{i=1,2,\cdots,n} \hat\gamma_i $$

这样定义的函数间隔有问题，如果成比例的改变 $w$ 和 $b$（如将它们改成 $2w$ 和 $2b$ ），则函数间隔的值 $f(x)$ 却变成了原来的2倍（虽然此时超平面没有改变）。但可以通过对超平面的法向量 $w$ 加一些约束，如规范化 $\|w\|=1$ ,使间隔确定。这时的函数间隔即为几何间隔(geometrical margin)。

![](https://darknessbeforedawn.github.io/test-book/images/SVM6.png)

如图所示，对于一个点 $x$ ，令其垂直投影到超平面上的对应的为 $x_0$ ，由于 $w$ 是垂直于超平面的一个向量， $\gamma$ 为样本 $x$ 到超平面的距离,则有：

$$x=x_0+\gamma \frac{w}{\|w\|}$$

其中 $\|w\|$ 为 $w$ 的二阶范数（范数是一个类似于模的表示长度的概念）， $\frac{w}{\|w\|}$ 是单位向量（一个向量除以它的模称之为单位向量）。又由于 $x_0$ 是超平面上的点，满足 $f(x0)=0$ ，代入超平面的方程 $w^Tx+b=0$ ，可得 $w^Tx_0=-b$。

$x=x_0+\gamma \frac{w}{\|w\|}$ 两边同时乘以 $w^T$ , 再有 $w^Tw = \|w\|^2$ ,可得：

$$\gamma = \frac{w^Tx+b}{\|w\|} = \frac{f(x)}{\|w\|}$$

为了得到 $\gamma$ 的绝对值，令 $\gamma$ 乘上对应的类别 $y$ ,并取最小值，即可得出几何间隔（用 $\tilde{\gamma}$ 表示）的定义：


$$\tilde{\gamma} = \min y\gamma =\frac{\hat\gamma}{\|w\|}$$

几何间隔就是函数间隔除以 $\|w\|$ ，而且函数间隔 $y(w^Tx+b) = yf(x)$ 实际上就是$$|f(x)|$$，只是人为定义的一个间隔度量，而几何间隔$$\frac{|f(x)|}{\|w\|}$$才是直观上的点到超平面的距离。

### 2.1.3 硬间隔最大化

 对一个数据点进行分类，当超平面离数据点的“间隔”越大，分类的确信度（confidence）也越大。所以，为了使得分类的确信度尽量高，需要让所选择的超平面能够最大化这个“间隔”值。通过由前面的分析可知：函数间隔不适合用来最大化间隔值，因为在超平面固定以后，可以等比例地缩放 $w$ 的长度和 $b$ 的值,可以使函数间隔的值任意大；而几何间隔的大小不会随着 $w$ 和 $b$ 的缩放而改变。因此，最大间隔分类超平面中的“间隔”指的是几何间隔。

于是最大间隔分类器（maximum margin classifier）的目标函数可以定义为：

$$\max_{w,b}\tilde\gamma$$

同时需满足一些条件，根据间隔的定义，有

$$y_i\gamma_i=y_i\frac{w^Tx_i+b}{\|w\|}=\frac{\hat\gamma_i}{\|w\|}\geq\frac{\hat\gamma}{\|w\|}=\tilde\gamma~~~\Longrightarrow ~~~y_i(w^Tx_i+b)\geq \hat\gamma, i=1,2,\cdots,n$$

于是这个问题可以改写为：

$$\begin{aligned}  
&\max_{w,b} \  \frac{\hat\gamma}{\|w\|}  \\  
&s.t.  \ \ \ y_i(w^Tx_i+b)\geq \hat\gamma, i=1,2,\cdots,n
\end{aligned}$$

函数间隔 $\hat\gamma$ 的取值是点到超平面的最小间隔，并不影响最优化问题的解， 我们固定 $\hat\gamma$ 的值也只会将 $w,b$ 成比例缩放并不会改变超平面，因此我们可以令 $\hat\gamma=1$ ，并且最大化 $\frac{1}{\|w\|}$ 等价于最小化 $\frac{1}{2}\|w\|^2$ ,于是线性可分支持向量机学习的最优化问题可以写为：

$$\begin{aligned}  
&\min_{w,b} \  \frac{1}{2} \|w\|^2 \\  
&s.t.  \ \ \ y_i(w^Tx_i+b)-1\geq 0, i=1,2,\cdots,n
\end{aligned}$$


(1)分离超平面的存在性

由于训练数据集线性可分，上述优化问题一定存在可行解。又由于目标函数有下界，并且数据集中既有正类又有负类，因而最优解 $(w^*,b^*)$ 必满足 $w^*\neq 0$ .由此得知分离超平面的存在性。

(2)超平面的唯一性

假设上述优化问题有两个最优解 $(w^*_1,b^*_1)$ 和 $(w^*_2,b^*_2)$ .显然 $\|w^*_1\|=\|w^*_2\|=c$ ,其中 $c$ 是一个常数。令 $w =\frac{w^*_1+w^*_2}{2},b =\frac{b^*_1+b^*_2}{2}$ ,易知 $(w,b)$ 也为原问题的可行解(非最优解)，从而有

$$c\leq \|w\|\leq \frac{1}{2}\|w^*_1\|+\frac{1}{2}\|w^*_2\|=c$$

从而有 $w^*_1=\lambda w^*_2$ ,易知 $\lambda =1$ 或 $\lambda=-1$ ,但 $\lambda=-1$ 时 $w=0$ 不是原问题的可行解，则有 $w^*_1=w^*_2$.

可以将两个最优解分别写为 $(w^*,b^*_1),(w^*,b^*_2)$ ,然后证 $b^*_1=b^*_2$ .设 $x_1',x_2'$ 是 $y_i=+1$ 集合中分别对应 $(w^*,b^*_1),(w^*,b^*_2)$ 使得问题的不等式等号成立的点， $x_1'',x_2''$ 是 $y_i=-1$ 集合中分别对应 $(w^*,b^*_1),(w^*,b^*_2)$ 使得问题的不等式等号成立的点，则有:

$$b^*_1=-\frac{1}{2}((w^*)^T x_1'+(w^*)^T x_1''),b^*_2=-\frac{1}{2}((w^*)^T x_2'+(w^*)^T x_2'')$$

使得

$$b^*_1-b^*_2 = -\frac{1}{2}[(w^*)^T (x_1'-x_2')+(w^*)^T (x_1''-x_2'')]$$

又因为 

$$(w^*)^T x_2'+b_1^*\geq 1= (w^*)^T x_1'+b_1^*$$

$$(w^*)^T x_1'+b_2^*\geq 1= (w^*)^T x_2'+b_2^*$$

所以， $(w^*)^T (x_1'-x_2')=0$ ,同理有 $(w^*)^T (x_1''-x_2'')=0$ ,因此

$$b_1^*-b_2^*=0$$

得两个最优解是相同的，即最优解唯一。

#### 支持向量和间隔边界

![](https://darknessbeforedawn.github.io/test-book/images/SVM7.png)

如上图所示，中间的实线便是寻找到的最优超平面（Optimal Hyper Plane），其到两条虚线边界的距离相等，这个距离便是几何间隔 $\tilde\gamma$ ,两条虚线间隔边界之间的距离等于 $2\tilde\gamma$ ,而虚线间隔边界上的点则是支持向量。由于这些支持向量刚好在虚线间隔边界上，所以它们满足 $y(w^Tx+b)=1$ ,而对于所有不是支持向量的点，则显然有 $y(w^Tx+b)>1$ 。

### 2.1.4 对偶

线性可分支持向量机最优化问题的原问题的拉格朗日函数为:

$$\begin{equation}
L(w,b,\alpha)= \frac{1}{2}\|w\|^2-\sum_{i=1}^N\alpha_iy_i(w^Tx_i+b)+\sum_{i=1}^N\alpha_i
\end{equation}$$

其中， $\alpha_i\geq 0$ .

则线性可分支持向量机最优化原问题等价于如下无约束问题:


$$
\min_{w,b} \max_{\alpha}L(w,b,\alpha)
$$

其对偶问题为:

$$\max_{\alpha}\min_{w,b} L(w,b,\alpha)$$

首先求 $L(w,b,\alpha)$ 对 $w,b$ 的极小，由

$$
\begin{aligned}  
\nabla_w L(w,b,\alpha) &= w- \sum_{i=1}^N\alpha_iy_ix_i=0\\ 
\nabla_b L(w,b,\alpha) &=  -\sum_{i=1}^N\alpha_iy_i=0
\end{aligned}
$$

得

$$\begin{equation}
w=\sum_{i=1}^N\alpha_iy_ix_i
\end{equation}$$

$$\begin{equation}
\sum_{i=1}^N\alpha_iy_i=0
\end{equation}$$


将上述结果代入拉格朗日函数得：

$$\min_{w,b} L(w,b,\alpha)=\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^Tx_j)$$

再对 $\min\limits_{w,b} L(w,b,\alpha)$ 求 $\alpha$ 的极大,即得对偶问题:

$$\begin{aligned}  
&\max_\alpha \  \sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^Tx_j)\\  
&s.t.  \ \ \ \sum_{i=1}^N\alpha_iy_i=0 \\
& \ \ \ \ \ \ \ \ \ \alpha_i\geq 0 , i=1,2,\cdots,N
\end{aligned}$$

上式可转化为:

$$\begin{aligned}  
&\min_\alpha \  \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^Tx_j)-\sum_{i=1}^N\alpha_i\\  
&s.t.  \ \ \ \sum_{i=1}^N\alpha_iy_i=0 \\
& \ \ \ \ \ \ \ \ \ \alpha_i\geq 0  , i=1,2,\cdots,N
\end{aligned}$$

可以通过求解上述对偶问题的解，进而确定分离超平面。即，设 $\alpha^*=(\alpha^*_1,\alpha^*_2,\cdots,\alpha^*_N)^T$ 为上述对偶问题的一个解，若存在一个 $\alpha_j^*$ 使得 $0<\alpha_j^*$ ，则原始问题的解 $w^*,b^*$ 为:

$$w^*=\sum_{i=1}^N\alpha_i^*y_ix_i$$

$$b^* = y_j-\sum_{i=1}^N\alpha_i^*y_i(x_i^Tx_j)$$

将 $w^*$ 代入分离超平面函数可得:


$$f(x)=\biggl(\sum_{i=1}^N\alpha_i^*y_ix_i\biggr)^Tx + b =\sum_{i=1}^N\alpha_i^*y_i\langle x_i,x\rangle +b$$

其中 $\langle \cdot,\cdot\rangle$ 代表向量内积，这里的形式的有趣之处在于，对于新点 $x$ 的预测，只需要计算它与训练数据点的内积即可.

由拉格朗日函数和 $KKT$ 互补条件可得：

$$\alpha_i(y_i(w^*)^Tx_i)-1)=0 , i=1,2,\cdots,N$$

而当 $\alpha_i>0$ 时，有

$$y_i(w^*)^Tx_i)-1=0 $$

即这些 $x_i$ 一定在间隔边界上， 而对于那些不在间隔边界上的实例 $x_i$ 所对应的 $\alpha_i=0$ .


## 2.2 线性不可分支持向量机与软间隔最大化

### 2.2.1 线性支持向量机

线性可分问题的支持向量机的学习方法，对线性不可分训练数据不适用。通常情况是，训练集中有一些特异点，将这些特异点除去后，剩下大部分样本点组成的集合是线性可分的。

线性不可分意味着某些样本点 $(x_i, y_i)$ 不能满足函数间隔大于等于1的约束条件。为了解决这个问题，可以对每个样本点 $(x_i, y_i)$ 引入一个松弛变量 $\xi_i\geq 0$ ,使函数间隔加上松弛变量大于等于1，即：

$$y_i(w^Tx_i+b)\geq 1-\xi_i$$

而对于每个松弛变量 $\xi_i$ ,需要做一些惩罚，则目标函数可写为:

$$\begin{equation}
\frac{1}{2}\|w\|^2 + C\sum_{i=1}^N\xi_i
\end{equation}$$

这里， $C>0$ 称为惩罚参数， $C$ 值大时对误分类的惩罚增大， $C$ 值小时对误分类的惩罚减小。最小化上述目标函数，即要使间隔最大化，同时使误分类的个数尽量小， $C$ 是调和二者的系数。相对于硬间隔最大化，这个称为软间隔最大化。


线性不可分的线性支持向量机的学习问题可描述成如下凸二次优化问题:

$$\begin{aligned}  
&\min_{w,b,\xi} \  \frac{1}{2} \|w\|^2 + C\sum_{i=1}^N\xi_i\\  
&s.t.  \ \ \ y_i(w^Tx_i+b)\geq 1-\xi_i, i=1,2,\cdots,N \\
& \ \ \ \ \ \ \ \ \ \xi_i\geq 0 ,\ \ i=1,2,\cdots,N
\end{aligned}$$

上述问题是一个凸二次规划问题，因而关于 $(w,b,\xi)$ 的解是存在的。并且 $w$ 唯一， $b$ 不唯一， $b$ 的解存在于一个区间。

对于线性不可分时的线性支持向量机简称为线性支持向量机。显然，线性支持向量机包含线性可分支持向量机。而由于现实中训练数据往往是不可分的，因此信息支持向量机具有更广的实用性。

### 2.2.2 对偶

线性不可分支持向量机最优化问题的原问题的拉格朗日函数为:

$$\begin{equation}
L(w,b,\xi,\alpha,\mu)\equiv \frac{1}{2}\|w\|^2+C\sum_{i=1}^N\xi_i-\sum_{i=1}^N\alpha_i(y_i(w^Tx_i+b)-1+\xi_i)-\sum_{i=1}^N\mu_i\xi_i
\end{equation}$$

其中， $\alpha_i\geq 0,\mu_i\geq 0$ ,在线性可分支持向量机可以认为 $\xi_i=0$.

则线性不可分支持向量机最优化原问题等价于如下无约束问题:


$$
\min_{w,b,\xi} \max_{\alpha,\mu}L(w,b,\xi,\alpha,\mu)
$$

其对偶问题为:

$$\max_{\alpha,\mu}\min_{w,b,\xi} L(w,b,\xi,\alpha,\mu)$$

首先求 $L(w,b,\xi,\alpha,\mu)$ 对 $w,b,\xi$ 的极小，由

$$
\begin{aligned}  
\nabla_w L(w,b,\xi,\alpha,\mu) &= w- \sum_{i=1}^N\alpha_iy_ix_i=0\\ 
\nabla_b L(w,b,\xi,\alpha,\mu) &=  -\sum_{i=1}^N\alpha_iy_i=0\\ 
\nabla_{\xi_i} L(w,b,\xi,\alpha,\mu) &=  C-\alpha_i-\mu_i =0
\end{aligned}
$$

得

$$\begin{equation}
w=\sum_{i=1}^N\alpha_iy_ix_i
\end{equation}$$

$$\begin{equation}
\sum_{i=1}^N\alpha_iy_i=0
\end{equation}$$

$$\begin{equation}
C-\alpha_i-\mu_i =0
\end{equation}$$

将上述结果代入拉格朗日函数得：

$$\min_{w,b,\xi} L(w,b,\xi,\alpha,\mu)=\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^Tx_j)$$

再对 $\min\limits_{w,b,\xi} L(w,b,\xi,\alpha,\mu)$ 求 $\alpha$ 的极大(上式中已不包含 $\mu$ ),即得对偶问题:

$$\begin{aligned}  
&\max_\alpha \  \sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^Tx_j)\\  
&s.t.  \ \ \ \sum_{i=1}^N\alpha_iy_i=0 \\
& \ \ \ \ \ \ \ \ \ C-\alpha_i-\mu_i =0\\
& \ \ \ \ \ \ \ \ \ \alpha_i\geq 0\\
& \ \ \ \ \ \ \ \ \ \mu_i\geq 0 , i=1,2,\cdots,N
\end{aligned}$$

上式可转化为:

$$\begin{aligned}  
&\min_\alpha \  \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i^Tx_j)-\sum_{i=1}^N\alpha_i\\  
&s.t.  \ \ \ \sum_{i=1}^N\alpha_iy_i=0 \\
& \ \ \ \ \ \ \ \ \ 0\leq\alpha_i\leq C  , i=1,2,\cdots,N
\end{aligned}$$

可以通过求解上述对偶问题的解，进而确定分离超平面。即，设 $\alpha^*=(\alpha^*_1,\alpha^*_2,\cdots,\alpha^*_N)^T$ 为上述对偶问题的一个解，若存在一个 $\alpha_j^*$ 使得 $0<\alpha_j^*<C$ ，则原始问题的解 $w^*,b^*$ 为:

$$w^*=\sum_{i=1}^N\alpha_i^*y_ix_i$$

$$b^* = y_j-\sum_{i=1}^N\alpha_i^*y_i(x_i^Tx_j)$$

### 2.2.3 支持向量

在线性不可分情况，对偶问题的解 $\alpha^*=(\alpha^*_1,\alpha^*_2,\cdots,\alpha^*_N)^T$ 中对应于 $\alpha^*_i>0$ 的样本点 $(x_i,y_i)$ 的实例 $x_i$ 称为软间隔的支持向量。如下图所示，图中分离超平面由实线表示，间隔边界由虚线表示，正离由圈表示，负例由叉表示。 $x_i$ 到间隔边界的距离 $\frac{\xi_i}{\|w\|}$ .

![](https://darknessbeforedawn.github.io/test-book/images/SVM8.png)

软间隔的支持向量 $x_i$ 或者在间隔边界上，或者在间隔边界与分离超平面之间，或者在分离超平面误分的一侧。若 $\alpha_i^*<C$, 则 $\xi_i=0$,支持向量 $x_i$ 恰好落在间隔边界上；若 $\alpha_i^*=C,0<$\xi_i<1$ ,则 $x_i$ 间隔边界与分离超平面之间；若 $\alpha_i^*=C,$\xi_i=1$ ,则 $x_i$ 在分离超平面上；若 $\alpha_i^*=C,$\xi_i>1$ ,则 $x_i$ 在分离超平面误分一侧。

### 2.2.4 合页损失函数

线性支持向量机原始最优化问题:

$$\begin{aligned}  
&\min_{w,b,\xi} \  \frac{1}{2} \|w\|^2 + C\sum_{i=1}^N\xi_i\\  
&s.t.  \ \ \ y_i(w^Tx_i+b)\geq 1-\xi_i, i=1,2,\cdots,N \\
& \ \ \ \ \ \ \ \ \ \xi_i\geq 0 ,\ \ i=1,2,\cdots,N
\end{aligned}$$

等价于最优化问题:

$$\begin{equation}
\min_{w,b}\sum_{i=1}^N[1-y_i(w^Tx_i+b)]_++\lambda\|w\|^2
\end{equation}$$


函数

$$\begin{equation}
L(y(w^Tx+b)) =[1-y(w^Tx+b)]
\end{equation}$$

称为合页损失函数(hinge loss function). 下标“+”表示以下取正直的函数:

$$[z]_+=\left 
\{ 
\begin{aligned}  
z, z> 0  \\ 
0, z\leq 0 
\end{aligned} 
\right.$$

以上说明，当样本点 $(x_i,y_i)$ 被正确分类且函数间隔(确信度) $y_i(w^Tx_i+b)$ 大于1时，损失是0，否则损失是 $1-y_i(w^Tx_i+b)$ .目标函数第二项是系数为 $\lambda$ 的 $w$ 的 $L_2$ 范数，是正则化项。

令:

$$[1-y_i(w^Tx_i+b)]_+ = \xi_i \Longrightarrow \xi_i\geq 0$$

$$1-y_i(w^Tx_i+b) >0 \Longrightarrow y_i(w^Tx_i+b)=1-\xi_i$$

$$1-y_i(w^Tx_i+b) \leq 0 \Longrightarrow xi_i=0, y_i(w^Tx_i+b)\geq 1-\xi_i$$

所以等价优化问题可写为:

$$\begin{equation}
\min_{w,b}\sum_{i=1}^N\xi_i+\lambda\|w\|^2
\end{equation}$$

若取 $\lambda=\frac{1}{2C}$ ,则

$$\min_{w,b}\frac{1}{2}\biggr(\frac{1}{2}\|w\|^2+C\sum_{i=1}^N\xi_i\biggl)$$

合页损失函数的图形如下图，横轴是函数间隔，纵轴是损失。由于函数形状像一个合页，故名合页函数。

![](https://darknessbeforedawn.github.io/test-book/images/SVM9.png)

图中还花出0-1损失函数，可以认为它是二分类问题的真正的损失函数，而合页损失函数是0-1损失函数的上界。由于0-1损失函数不是连续可导的，直接优化其构成的目标函数比较困难，可以认为线性2支持向量机是优化由0-1损失函数的上界(合页损失函数)构成的目标函数。这时的上界损失函数又成为代理损失函数(surrogate loss function)。

图中虚线显示的是感知机的损失函数 $[-y_i(w^Tx_i+b)]_+$ .这时，当样本点 $(x_i,y_i)$ 被正确分类时，损失是0，否则损失是 $-y_i(w^Tx_i+b)$ .相比之下，合页损失函数不仅要分类正确，而且确信度足够高时损失才是0.也就是合页损失函数对学习有更高要求。

## 2.3 非线性支持向量机与核函数

对于线性分类问题，线性分类支持向量机是一种十分有效的方法。但，有时分类问题是非线性的，这时就得使用非线性支持向量机，其主要特点是利用核技巧(kernel trick).

### 2.3.1 核技巧

![](https://darknessbeforedawn.github.io/test-book/images/SVM10.png)

如上图可见，对于非线性分类问题无法用直线将正负例正确分离，但可以用一条椭圆曲线将它们正确分开。同样也可以通过将数据映射到高维空间，使正负样例变的线性可分。具体来说，在线性不可分的情况下，支持向量机首先在低维空间中完成计算，然后通过核函数将输入空间映射到高维特征空间，最终在高维特征空间中构造出最优分离超平面，从而把平面上本身不好分的非线性数据分开。

设原空间为 $\mathcal{X}\subset\mathbf{R^2},x=(x^{(1)},x^{(2)})\in\mathcal{X} $ , 新空间为 $\mathcal{Z}\subset\mathbf{R^2},z=(z^{(1)},z^{(2)})\in\mathcal{Z} $ ，定义从原空间到新空间的变换(映射)：

$$z=\phi(x)=((x^{(1)})^2,(x^{(2)})^2)^T$$

经过变换 $z=\phi(x)$ ,原空间 $\mathcal{X}\subset\mathbf{R^2} $ ,变换为新空间 $\mathcal{Z}\subset\mathbf{R^2} $ ,原空间中的点相应地变换为新空间中的点，原空间中的椭圆 

$$w_1(x^{(1)})^2+w_2(x^{(2)})^2+b =0$$

变换为新空间中的直线

$$w_1z^{(1)}+w_2z^{(2)}+b =0$$

这样，原空间的非线性可分问题就变成了新空间的线性可分问题。

可知，用线性分类方法求解非线性分类问题可以分为两步（核技巧）：

1.用一个变换将原空间的数据映射到新空间；

2.在新空间里用线性分类学习方法从训练数据中学习分类模型。

核技巧应用到支持向量机，就是通过一个非线性变换将输入空间（欧氏空间 $\mathbf{R^n}$ 或离散集合）对应于一个特征空间（希伯特空间 $\mathcal{H}$ ）,使得在输入空间中的超取面模型对应于特征空间中的超平面模型。这样，分类问题的学习任务通过在特征空间中求解线性支持向量机就可以完成。

#### 核函数

通过映射函数可以将对偶问题的目标函数转化为：

$$W(\alpha) = \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\langle\phi(x_i),\phi(x_j)\rangle-\sum_{i=1}^N\alpha_i$$

这样拿到非线性数据，就找一个映射 $\phi(\cdot)$  ，然后把原来的数据映射到新空间中，再做线性 $SVM$ 即可.但是，假设我们对一个二维空间做映射，选择的新空间是原始空间的所有一阶和二阶的组合，得到了五个维度；如果原始空间是三维，那么我们会得到 19 维的新空间，这个数目是呈爆炸性增长的，这样 $\phi(\cdot)$ 的计算就非常困难。下面讨论如何解决这种问题。

设两个向量 $x_1=(\eta_1,\eta_2)^T,x_2(\xi_1,\xi_2)^T$ , 则通过映射后的内积为:

$$\langle \phi(x_1),\phi(x_2)\rangle =\langle(\eta_1,\eta_1^2,\eta_2,\eta_2^2,\eta_1\eta_2)^T,(\xi_1,\xi_1^2,\xi_2,\xi_2^2,\xi_1\xi_2)^T,\rangle = \eta_1\xi_1+\eta_1^2\xi_1^2 +\eta_2\xi_2+\eta_2^2\xi_2^2 + \eta_1\eta_2\xi_1\xi_2$$

另外:

$$(\langle x_1,x_2\rangle +1)^2 = 2\eta_1\xi_1+\eta_1^2\xi_1^2+\eta_2\xi_2+\eta_2^2\xi_2^2+2\eta_1\eta_2\xi_1\xi_2+1$$

我们可以通过把映射函数的某几个维度线性缩放，然后加上一个常数维度， 使得 $\langle \phi(x_1),\phi(x_2)\rangle$ 的结果和 $(\langle x_1,x_2\rangle +1)^2$ 相同，该映射为:

$$\phi(X_1,X_2) = (\sqrt{2}X_1,X_1^2,\sqrt{2}X_2,X_2^2,\sqrt{2}X_1X_2,1)^T$$

这样一来我们就可以将映射到高维空间中，然后计算内积，改为直接在原来的地位空间中进行计算，而不需要显示地写出映射后的结果。

##### 这里的计算两个向量在隐式映射过后的空间中的内积的函数叫做核函数 (Kernel Function)，记为 $k(x,z)$.

刚才的例子中，核函数为:

$$k(x_1,x_2) = (\langle x_1,x_2\rangle +1)^2$$

继而将对偶问题的目标函数转化为：

$$W(\alpha) = \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jk(x_i,x_j)-\sum_{i=1}^N\alpha_i$$

这样就避开了直接在高维空间中进行计算内积，简化了计算。在实际应用中，往往依赖领域知识直接选择核函数，核函数选择的有效性需要通过实验验证。

### 2.3.2 正定核

在已知映射函数 $\phi$ ，可以通过 $\phi(x)$ 和 $\phi(z)$ 的内积求得核函数 $K(x,z)$ .在不用构造映射 $\phi(x)$ 能否判断一个给定的函数是不是 $K(x,z)$ 是不是核函数？接下来就来说明这个问题。

给定 $m$ 个训练样本 

$$\{x^{(1)},x^{(2)},\cdots,x^{(m)}\}$$

每个 $x^{(i)}$ 对应一个特征向量，并且假设 $K(x,z)$ 为一个有效的核函数，我们可以计算 $K_{ij} =K(x^{(i)},x^{(j)}),i,j=1,2,\cdots,m$ ,这样我们可以计算出 $m*m$ 的核函数矩阵 $K$ (Kernel Matrix).

由以上可知，矩阵 $K$ 为对称矩阵。即

$$K_{ij}=K(x^{(i)},\phi(x^{(j)}) = \langle \phi(x^{(i)},\phi(x^{(j)})\rangle = \langle \phi(x^{(j)},\phi(x^{(i)})\rangle = k_{ji}$$

首先使用 $\phi_k(x)$ 表示映射函数 $\phi(x)$ 的第 $k$ 维属性值。那么对任意向量 $z$, 有

$$
\begin{align}
z^TKz &= \sum_i\sum_jz_iK_{ij}z_j   \\ 
&= \sum_i\sum_jz_i\langle \phi(x^{(i)}),\phi(x^{(j)})\rangle z_j\\ 
&= \sum_i\sum_jz_i\sum_k \phi_k(x^{(i)})\phi_k(x^{(j)})z_j \\ 
&=\sum_k\sum_i\sum_jz_i\phi_k(x^{(i)})\phi_k(x^{(j)})z_j \\ 
&=\sum_k\biggl(\sum_iz_i\phi_k(x^{(i)})\biggr)^2 \\ 
&\geq 0
\end{align}
$$

上述推导说明，核函数矩阵 $K$ 是半正定的 ( $K\geq 0$ ).即， $K(x,z)$ 是有效核函数，可推出核函数矩阵 $K$ 是对称半正定的。

##### 正定核充要条件:

设 $K : \mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$ 是对称函数.则 $K(x,z)$ 为正定核的充要条件是，对任意 $x_i\in\mathcal{X},i=1,2,\cdots,m$ ，$K(x,z)$ 对应的核函数矩阵 $K$ 是对称半正定的. ( $\mathcal{X}$ 为欧式空间 $\mathbf{R}^n$ 的子集或离散集合)

必要性已经证过，下面来说明充分性。

##### 充分性:


1.定义映射，构成向量空间 $S$

定义映射为: $\phi :x\rightarrow K(\cdot,x)$ 

根据映射，对任意 $x_i\in\mathcal{X}, \alpha_i\in \mathbf{R},i=1,2,\cdots,m$ ,定义线性组合

$$f(\cdot) = \sum_{i=1}^m\alpha_iK(\cdot,x_i)$$

由线性组合为元素的集合 $S$ 对加法和数乘运算是封闭的，所以 $S$ 构成一个向量空间。

2.在 $S$ 上定义内积，使其成为内积空间

在 $S$ 上定义一个运算 $*$ : 对任意 $f,g\in S$ 

$$f(\cdot) =\sum_{i=1}^m\alpha_iK(\cdot,x_i),g(\cdot) =\sum_{j=1}^l\beta_jK(\cdot,z_j)$$

定义运算 $*$ 

$$f*g=\sum_{i=1}^m\sum_{j=1}^l\alpha_i\beta_jK(x_i,z_j)$$

证明运算 $*$ 是空间 $S$ 的内积，需要证:

(1) $(cf)*g=c(f*g),c\in\mathbf{R}$

(2) $(f+g)*h = f*h+g*h, h\in S$

(3) $f*g=g*f$

(4) $f*f\geq 0, f*f=0\Leftrightarrow f=0$

其中1-3有上述假设和 $K(x,z)$ 的对称性容易得到，现在只需证4

$$f*f = \sum_{i,j=1}^m\alpha_i\alpha_jK(x_i,x_j)$$

由核矩阵 $K$ 的半正定性可知上式非负，即 $f*f\geq 0$ .

接着证 $f*f=0\Leftrightarrow f=0$ 

充分性：当 $f=0$ 时，显然有 $f*f=0$ .

必要性：首先证

$$|f*g|^2 \leq (f*f)(g*g)$$

设 $f,g\in S$ , $\lambda\in\mathbf{R}$ ,则 $f+\lambda g\in S$ ,则有

$$(f+\lambda g)*(f+\lambda g) \geq 0$$

$$f*f +2\lambda (f*g) + \lambda^2(g*g) \geq 0$$

上式为 $\lambda$ 的二次三项式，非负，则其判别式小于等于0，即

$$(f*g)^2-(f*f)(g*g) \leq 0$$

设 

$$f(\cdot)=\sum_{i=1}^m\alpha_iK(\cdot,x_i)$$

则有

$$K(\cdot,x)*f = \sum_{i=1}^m\alpha_iK(x,x_i)=f(x)$$

于是

$$|f(x)|^2=|K(\cdot, x)*f|^2$$

根据上述证明的不等式可得

$$|K(\cdot, x)*f|^2 \leq (K(\cdot,x)*K(\cdot,x))(f*f)=K(x,x)(f*f)$$

上式表明，当 $f*f=0$ 时，对任意 $x$ 都有 

$$|f(x)|=0$$

以上可说明 $*$ 运算就是向量空间 $S$ 的内积运算，仍然可用 $\bullet$ 表示。赋予内积的向量空间为内积空间。


3.将内积空间 $S$ 完备化为希尔伯特空间

由内积的定义可以得到范数

$$\|f\|=\sqrt{f\bullet f}$$

这样， $S$ 就是一个赋范向量空间。根据泛函分析理论，对于不完备的赋范向量空间 $S$ ,一定可以使之完备化，得到完备的赋范向量空间 $\mathcal{H}$ ,同时又是内积空间，这就是希尔伯特空间。

这一希尔伯特空间 $\mathcal{H}$ 称为再生核希伯特空间(Reproducing Kernel Hilbert Space, RKHS).这是由于核 $K$ 具有再生性，即满足

$$K(\cdot,x)\cdot f=f(x)$$

及

$$K(\cdot,x)\cdot K(\cdot,z) = K(x,z)$$

称为再生核。

通过上述过程可得

$$K(x,z)=\phi(x)\cdot \phi(z)$$

表明 $K(x,z)$ 为 $\mathcal{X}\times\mathcal{X}$ 上的核函数。

### 2.3.3 常用核函数

通常人们会从一些常用的核函数中选择（根据问题和数据的不同，选择不同的参数，实际上就是得到了不同的核函数），常用核函数有以下几种：


1.多项式核函数(Polynomial Kernel Function)

$$K(x,z)=(\langle x,z\rangle +R)^d$$

显然刚才我们上述举的例子是这里多项式核的一个特例( $R = 1，d = 2$ )。虽然比较麻烦，而且没有必要，不过这个核所对应的映射实际上是可以写出来的，该空间的维度是 $\binom{m+d}{d}$ ,其中 $m$ 是原始空间的维度。

2.高斯核函数(Gaussian Kernel Function)

$$K(x,z)=\exp\biggl(-\frac{\|x-z\|^2}{2\sigma^2}\biggr)$$

这个核会将原始空间映射为无穷维空间。不过，如果 $\sigma$ 选得很大的话，高次特征上的权重实际上衰减得非常快，所以实际上（数值上近似一下）相当于一个低维的子空间；反过来，如果 $\sigma$ 选得很小，则可以将任意的数据映射为线性可分——当然，这并不一定是好事，因为随之而来的可能是非常严重的过拟合问题。不过，总的来说，通过调控参数 $\sigma$ ，高斯核实际上具有相当高的灵活性，也是使用最广泛的核函数之一。下图所示的例子便是把低维线性不可分的数据通过高斯核函数映射到了高维空间：

![](https://darknessbeforedawn.github.io/test-book/images/SVM11.png)

3.线性核核函数(Linear Kernel Function)

$$K(x,z) = \langle x,z\rangle$$

这实际上就是原始空间中的内积。这个核存在的主要目的是使得“映射后空间中的问题”和“映射前空间中的问题”两者在形式上统一起来了。

### 2.3.4 非线性支持向量分类机

(1) 选取适当的核函数 $K(x,z)$ 和适当参数 $C$ ,构造并求解最优化问题

$$\begin{aligned}  
&\min_\alpha \  \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\  
&s.t.  \ \ \ \sum_{i=1}^N\alpha_iy_i=0 \\
& \ \ \ \ \ \ \ \ \ 0\leq\alpha_i\leq C  , i=1,2,\cdots,N
\end{aligned}$$

求最优解 $\alpha^* =(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$

(2) 选择 $\alpha^*$ 的一个正分量 $0<\alpha_i< C$ ,计算

$$b^*=y_j-\sum_{i,j=1}^N\alpha_i^*y_iK(x_i,x_j)$$

(3)构造决策函数

$$f(x) = \sum_{i=1}^N\alpha_i^*y_iK(x,x_i) + b^*$$

## 2.4 序列最小最优化算法

支持向量机的学习问题可以形式化为求解凸二次规划问题，这样的凸二次规划问题具有全局最优解，并且有许多最优化算法可以用于这一问题的求解。但是当数据量非常大使，这些算法往往非常低效。序列最小最优化(Sequential Minimal Optimization, SMO)算法就是一种比较高效的解决支持向量机的算法。

SMO算法要解如下凸二次规划的对偶问题：

$$\begin{aligned}  
&\min_\alpha \  \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\  
&s.t.  \ \ \ \sum_{i=1}^N\alpha_iy_i=0 \\
& \ \ \ \ \ \ \ \ \ 0\leq\alpha_i\leq C  , i=1,2,\cdots,N
\end{aligned}$$

其中变量是拉格朗日乘子，一个变量 $\alpha_i$ 对应于一个样本点 $(x_i,y_i)$ ;变量总数等于训练样本容量 $N$ .

SMO算法是一种启发式算法，其基本思路是:如果所有变量的解都满足此最优化问题的 $KKT$ 条件，那么 这个最优化问题就解决了。否则，选择两个变量固定其他变量，针对这两个变量构建一个二次规划问题。这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题，因为这会使得原始二次规划问题的目标函数值变的更新，并且这个子问题可以通过解析方法求解从而大大提高整个算法的计算速度，子问题的两个变量，一个是违反 $KKT$ 条件最严重的那一个，另一个由约束条件自动确定。如此，SMO算法将原问题不断分解为子问题并将子问题求解，进而达到求解原问题的目的。

子问题中只有一个是自由变量，假设 $\alpha_1,\alpha_2$ 为两个变量， $\alpha_3,\alpha_4,\cdots,\alpha_N$ 固定，那么有等式约束可得

$$\alpha_1=-y\sum_{i=2}^N\alpha_iy_i$$

如果 $\alpha_2$ 确定，那么 $\alpha_1$ 也随之确定。所以子问题中同时更新两个变量。

整个SMO算法包括两个部分：求解两个变量二次规划的解析方法和选择变量的启发式方法。


## 2.4.1 两个变量二次规划的求解方法

假设 $\alpha_1,\alpha_2$ 为两个变量， $\alpha_3,\alpha_4,\cdots,\alpha_N$ 固定，则SMO的最优化问题的子问题可以写成

$$\begin{aligned}  
&\min_{\alpha_1,\alpha_2}W(\alpha_1,\alpha_2)=\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2-(\alpha_1+\alpha_2)+y_1\alpha_1\sum_{i=3}^Ny_i\alpha_iK_{i1}+y_2\alpha_2\sum_{i=3}^Ny_i\alpha_iK_{i2}\\  
&\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ s.t.  \ \ \ \ \alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^N\alpha_iy_i=\varsigma \\
&\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 0\leq\alpha_i\leq C  , i=1,2,\cdots,N
\end{aligned}$$

其中， $K_{ij}=K(x_i,x_j),i,j=1,2,\cdots,N$ , $\varsigma$ 是常数，目标函数省略了不含 $\alpha_1,\alpha_2$ 的常数项。