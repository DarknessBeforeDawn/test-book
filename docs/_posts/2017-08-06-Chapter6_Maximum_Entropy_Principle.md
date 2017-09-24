---
title: 最大熵模型
layout: post
share: false
---

# 1.熵
信息是个很抽象的概念，接下来我们从[文件压缩](http://www.ruanyifeng.com/blog/2014/09/information-entropy.html)问题来说明信息熵，以便于理解。

#### 压缩原理
压缩的本质就是将重复出现的字符用更短的符号代替，就是找出文件内容的概率分布，将出现概率高德部分替代成更短的形式。因此，内容越是重复的文件就可以压缩的越小。比如，"ABABABABABABAB"可以压缩为"7AB"。而内容毫无重复，就很难进行压缩，例如无理数$$\pi$$就很难压缩。

#### 压缩极限

压缩可以分解成两个步骤。第一步是得到文件内容的概率分布；第二步是对文件进行编码，用较短的符号替代那些重复出现的部分。

比如扔硬币的结果，那么只要一个二进制位就够了，1表示正面，0表示表示负面。足球赛的结果，最少需要两个二进制位。扔骰子的结果，最少需要三个二进制位。

在均匀分布的情况下，假定一个字符（或字符串）在文件中出现的概率是$$p$$，那么该文件种最多可能出现$$\frac{1}{p}$$种字符（字符串）。则需要$$\log_2\frac{1}{p}$$个二进制位才能表示$$\frac{1}{p}$$种情况。也就是每种字符（或字符串）需要占用$$\log_2\frac{1}{p}$$个二进制位。那么全部压缩该文件总共需要$$\frac{1}{p}\log_2\frac{1}{p}$$个二进制位。

更一般情况，假定文件有n个部分组成，并且每个部分的内容在文件中的出现概率分别为$$p_1,p_2,\cdots p_n$$。可推出如下公式：

$$
\log_2\frac{1}{p_1}+\log_2\frac{1}{p_2}+\cdots+\log_2\frac{1}{p_n}
=\sum_{i=1}^n\log_2\frac{1}{p_i}
$$

上述公式即为压缩极限，表示压缩所需要的二进制位数。

#### 信息熵
在均匀分布的情况下$$(p_i=\frac{1}{n})$$，压缩每个字符（字符串）平均需要$$\sum\limits_{i=1}^n\frac{1}{n}\log_2\frac{1}{p_i}$$个二进制位。对于一般情况下$$p_i$$不等，压缩每个字符（字符串）平均需要$$\sum\limits_{i=1}^np_i\log_2\frac{1}{p_i}$$个二进制位。$$\sum\limits_{i=1}^np_i\log_2\frac{1}{p_i}$$即为信息熵公式。

假定有两个文件都包含1024个符号，在ASCII码的情况下，它们的长度是相等的，都是1KB。甲文件的内容50%是a，30%b，20%是c，则平均每个符号要占用1.49个二进制位。

$$0.5*\log_2\frac{1}{0.5}+0.3*log_2\frac{1}{0.3} + 0.2*log_2\frac{1}{0.2}=1.47$$

乙文件的内容10%是a，10%是b,$$\cdots$$,10%是j，则平均每个符号要占用3.32个二进制位。

$$0.1*\log_2\frac{1}{0.1}*10=3.32$$

可以看到文件内容越是分散（随机），所需要的二进制位就越长。所以，这个值可以用来衡量文件内容的随机性（又称不确定性）。这就叫做信息熵（information entropy）。

注:

（1）信息熵只反映内容的随机性，与内容本身无关。

（2）信息熵越大，表示占用的二进制位越长，因此就可以表达更多的符号(信息量并不一定大也许是一堆无序没意义的字符)。

（3）信息熵与热力学的熵，基本无关。

# 2.最大熵模型

## 2.1熵的定义

假设随机变量$$X$$的概率分布为$$P(X)$$，$$X$$取值为$$x_1,x_2,\cdots,x_n$$,$$p(x_i)=P(X=x_i)$$，则由上节我们可知息熵的计算公式如下：

$$\begin{equation}
H(X)=-\sum_xP(X)\log P(X)=-\sum_{i=1}^np(x_i)\log p(x_i) = E(\log(\frac{1}{p(x_i)}))
\end{equation}$$

用$$I(x_i)=-\log p(x_i)$$来表示事件$$x_i$$的信息量，则$$H(X)$$即为随机变量$$X$$的平均信息量(期望)。熵满足下列不等式：

$$0\leqslant H(X) \leqslant \log n$$

由Jensen不等式：$$\varphi(E(X)) \leqslant E(\varphi(X)),\varphi(x)$$为凸函数，若$$\varphi(x)$$为凹函数，则有$$\varphi(E(X)) \geqslant E(\varphi(X))$$，$$\log(x)$$为凹函数可得：

$$H(X)=E(\log(\frac{1}{p(x_i)})) \leqslant \log(E(\frac{1}{p(x_i)}))=\log(\sum_{i=1}^n p(x_i)\frac{1}{p(x_i)})=\log n $$

其中$$n$$是$$X$$的取值个数，当随机变量退化为定值时（概率为1），那么此时的熵值为0。另一个极限就是：当随机变量服从均匀分布的时候，此时的熵值最大$$\log n$$。

#### 联合熵

对于服从$$p(x,y)$$的两个随机变量$$X,Y$$，可以形成联合熵Joint Entropy，用$$H(X,Y)$$表示:

$$\begin{equation}
H(X,Y)=-\sum_{x\in X, y\in Y}p(x,y)\log p(x,y) = -E(\log(p(x,y))
\end{equation}$$

#### 条件熵

在随机变量$$X$$发生的前提下，随机变量$$Y$$发生所新带来的熵定义为$$Y$$的条件熵，用$$H(Y$$\|$$X)$$表示，用来衡量在已知随机变量$$X$$的条件下随机变量$$Y$$的不确定性。

若$$(X,Y)\sim p(x,y)$$，条件熵为：
$$
\begin{align*}
H(Y|X) &= \sum_{x \in X}p(x)H(Y|X=x)\\
&=-\sum_{x \in X}p(x)\sum_{y\in Y}p(y|x)\log p(y|x)\\
&=-\sum_{x \in X}\sum_{y\in Y}p(x,y)\log p(y|x)
\end{align*}
$$

联合熵等于其中一个随机变量的熵加上另一个随机变量的条件熵，即：$$H(X,Y)=H(Y$$\|$$X)+H(X)$$。证明如下：

$$
\begin{align*}
H(X,Y) &= -\sum_{x\in X}\sum_{y\in Y}p(x,y)\log p(x,y)\\
&=-\sum_{x\in X}\sum_{y\in Y}p(x,y)\log p(y|x)p(x)\\
&=-\sum_{x\in X}\sum_{y\in Y}p(x,y)\log p(x)-\sum_{x\in X}\sum_{y\in Y}p(x,y)\log p(y|x)\\
&=-\sum_{x\in X}p(x)\log p(x)-\sum_{x\in X}\sum_{y\in Y}p(x,y)\log p(y|x)\\
&=H(X)+H(Y|X)
\end{align*}
$$

#### 相对熵
又称互熵，交叉熵，鉴别信息，Kullback熵，Kullback-Leible散度等。$$p(x),q(x)$$是$$X$$中取值的两个概率分布，则$$p$$对$$q$$的相对熵是：

$$
\begin{equation}
D(p\|q)=\sum_xp(x)\log\frac{p(x)}{q(x)}=E_{p(x)}\log\frac{p(x)}{q(x)}
\end{equation}
$$

并约定$$0\log\frac{0}{0}=0$$，$$0\log\frac{p}{0}=\infty$$。因此，如果存在$$x\in X$$，使得$$p(x)>0,q(x)=0$$，则有$$D(p\|q)=\infty$$。

在一定程度上，相对熵可以度量两个随机变量的“距离”，且有$$D(p\|q)\neq D(q\|p)$$，可以通过证明知道相对熵总是非负的，而且，当且仅当$$p=q$$时为零。

#### 互信息

两个随机变量$$X,Y$$的互信息定义为$$X,Y$$的联合分布$$p(x,y)$$和各自独立分布乘积的$$p(x)p(y)$$相对熵，用$$I(X,Y)$$表示：

$$
\begin{align*}
I(X,Y) &= \sum_{x\in X}\sum_{y\in Y}p(x,y)\log \frac{p(x,y)}{p(x)p(y)}\\
&=D(p(x,y)\|p(x)p(y))\\
&=E_{p(x,y)}\frac{p(X,Y)}{p(X)p(Y)}
\end{align*}
$$

互信息是一个随机变量包含另一个随机变量信息量的度量。互信息也是在给定另一个随机变量知识的条件下，原随机变量不确定度的缩减量。通过查看表达式，即可知道互信息具有对称性，同样可以简单证明得互信息的非负性。

#### 互信息与熵的关系

$$
\begin{align*}
I(X,Y) &= \sum_{x\in X}\sum_{y\in Y}p(x,y)\log \frac{p(x,y)}{p(x)p(y)}\\
&=\sum_{x\in X}\sum_{y\in Y}p(x,y)\log \frac{p(x|y)}{p(x)}\\
&=-\sum_{x\in X}\sum_{y\in Y}p(x,y)\log p(x)-(-\sum_{x\in X}\sum_{y\in Y}p(x,y)\log p(x|y))\\
&=-\sum_{x\in X}p(x)\log p(x)-(-\sum_{x\in X}p(x,y)\log p(x|y))\\
&=H(X)-H(X|Y)
\end{align*}
$$

由此可见，互信息$$I(x,y)$$是在给定另一个随机变量$$Y$$知识的条件下，$$X$$不确定度的缩减量。

由于互信息的对称性，可得：

$$I(X,Y)=I(Y,X)=H(Y)-H(Y|X)$$

将$$H(X,Y)=H(X)+H(Y$$\|$$X)$$代入上式得：

$$I(X,Y)=H(x)+H(Y)-H(X,Y)$$

一些重要表达式:

$$I(X,Y)=H(Y)-H(Y|X)$$

$$I(X,Y)=H(X)-H(X|Y)$$

$$I(X,Y)=H(x)+H(Y)-H(X,Y)$$

$$I(X,Y)=I(Y,X)$$

韦恩图可以很好地表示以上各变量之间的关系，对着上面的公式，应该可以很好理解下面的图表达的含义。

![](https://darknessbeforedawn.github.io/test-book/images/entropy.png)
  
## 2.2最大熵模型
最大熵原理认为要选择的概率模型首先必须满足已有的事实，即约束条件。在没有更多信息的情况下，那些不确定的部分都是“等可能的”(无偏性)。“最大熵模型”的核心两点：1.承认已知事物（或知识）；2.对未知事物不做任何假设，没有任何偏见。
### 2.2.1无偏原则
例如，一篇文章中出现了“学习”这个词，那这个词可以作主语、谓语、宾语。换言之，已知“学习”可能是动词，也可能是名词，故“学习”可以被标为主语、谓语、宾语、定语等等。令$$x_1$$表示“学习”被标为名词， $$x_2$$表示“学习”被标为动词。令$$y_1$$表示“学习”被标为主语， $$y_2$$表示被标为谓语， $$y_3$$表示宾语， $$y_4$$表示定语。则有：

$$p(x_1)+p(x_2)=1,\sum_{i=1}^4p(y_i)=1$$

如果没有其他的知识，根据信息熵的理论，概率趋向于均匀。所以有： 

$$p(x_1)=p(x_2)=0.5,p(y_1)=p(y_2)=p(y_3)=p(y_4)=0.25$$

若我们已知“学习”被标为定语的可能性很小，只有0.05。引入这个新的知识：在满足了这个约束的情况下，其他的事件我们尽可能的让他们符合均匀分布（无偏原则）：

$$p(x_1)=p(x_2)=0.5,p(y_1)=p(y_2)=p(y_3)=\frac{0.95}{3}$$

再加入另一个约束条件：当“学习”被标作名词的时候，它被标作谓语的概率为0.95。即：

$$p(y_2|x_1)=0.95$$

此时我们仍然坚持无偏见原则，但随着已知的知识点越来越多，其实也就是约束越来越多，求解的过程会变得越来越复杂。我们可以通过使得熵尽可能的大来保持无偏性。将上述优化及约束转化为数学公式：

$$\max H(Y|X)=-\sum_{x=\{x_1,x_2\}\atop y=\{y_1,y_2,y_3,y_4\}}p(x,y)\log p(y|x)$$

且满足以下4个约束条件：

$$p(x_1)+p(x_2)=1$$

$$\sum_{i=1}^4p(y_i)=1$$

$$p(y_4)=0.05$$

$$p(y_2|x_1)=0.95$$

以上表达中，一般用$$H(Y$$\|$$X)$$，和用$$H(X,Y)$$效果一样，由于$$H(X,Y)=H(Y$$\|$$X)+H(X)$$,$$X$$为训练集集合，分布已知，即$$H(X)$$是一个定值，因此$$H(X,Y)$$最大等价于$$H(Y$$\|$$X)$$最大。

### 2.2.2最大熵模型表示

最大熵模型的一般表达式：

$$
\begin{equation}
\max_{p\in P} H(Y|X) = - \sum_{x,y}p(x,y)\log p(y|x)
\end{equation}
$$

 其中，$$P=\{p$$ \| $$p$$是$$X$$上满足条件的概率分布$$\}$$,此处对数为自然对数。

假设分类模型是一个条件概率分布$$P(Y$$\|$$X),X\in \mathcal{X} \subseteq \mathbf{R}^2$$表示输入，$$Y\in \mathcal{Y}$$表示输出。这个模型表示对于给定输入$$\mathcal{X}$$,以条件概率$$P(Y$$\|$$X)$$输出$$Y$$。给定训练集$$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n),\}$$，学习的目标是用最大熵原理选择最好的分类模型。

特征：$$(x,y),$$其中$$x$$为这个特征中的上下文信息，$$y$$为这个特征中需要确定的信息。样本：即$$(x_i,y_i)$$特征对的训练数据集。

对于一个特征$$(x_i,y_i)$$，定义特征函数：

$$
\begin{equation}
f(x,y)=\begin{cases}1,&x=x_i~~and~~y=y_i\\0,&else\end{cases}
\end{equation}
$$

则特征函数关于经验分布$$\tilde{p}(X,Y)$$在样本中的期望：

$$
\begin{equation}
E_{\tilde{p}}(f)=\sum_{x,y}\tilde{p}(x,y)f(x,y)
\end{equation}
$$

其中$$\tilde{p}(x,y)$$为$$(x,y)$$在样本中出现的概率。

特征函数关于模型$$p(Y$$\|$$X)$$与经验分布$$\tilde{p}(X)$$的期望值为：

$$
\begin{equation}
E_{p}(f)=\sum_{x,y}\tilde{p}(x)p(y|x)f(x,y)
\end{equation}
$$

其中$$\tilde{p}(x)$$为$$x$$在样本中出现的概率。

换言之，如果能够获取训练数据中的信息，那么上述这两个期望值相等，即：

$$
\begin{equation}
E_{p}(f)=E_{\tilde{p}}(f)
\end{equation}
$$

上面这个等式就是最大熵模型中的限制条件，那么最大熵模型的完整提法是： 

$$\begin{equation}
p^*=\arg\max_{p\in P}H(Y|X)= - \sum_{x,y}\tilde{p}(x)p(y|x)\log p(y|x)
\end{equation}
$$

$$P=\biggl\{p(y|x)~|~~\begin{array}\forall f_i: E_{p}(f_i)=E_{\tilde{p}}(f_i)\\\forall x: \sum_yp(y|x)=1\end{array}\biggr\}$$

### 2.2.3求解最大熵模型

针对原问题，首先引入拉格朗日乘子$$λ_0,λ_1,\cdots,λ_i$$定义拉格朗日函数，转换为对偶问题求其极大化：

$$
\begin{equation}
\Lambda(p,\overrightarrow{\lambda})=H(Y|X)+\sum_{i=1}^n\lambda_i(E_{E_{p}(f)-\tilde{p}}(f))+\lambda_0(\sum_yp(y|x)-1)
\end{equation}
$$

其中参数$$p$$表示$$p(y$$\|$$x)$$,$$\overrightarrow{\lambda}$$表示$$\lambda$$的向量形式，即：

$$L=\sum_{x,y}\tilde{p}(x)p(y|x)\log\frac{1}{p(y|x)}+\sum_{i=1}^n\lambda_i\sum_{x,y}f_i(x,y)\biggl(\tilde{p}(x)p(y|x)-\tilde{p}(x,y)\biggr)+\lambda_0\biggl(\sum_yp(y|x)-1\biggr)$$

按照Lagrange乘子法的思路，对参数$$p$$求偏导，可得：

$$
\begin{align*}
\frac{\partial L}{\partial p(y|x)}&=\sum_{x,y}\tilde{p}(x)(-\log p(y|x)-1)+\sum_{x,y}\tilde{p}(x)\sum_{i=1}^n\lambda_if_i(x,y)+\sum_y\lambda_0\\
&=\sum_{x,y}\tilde{p}(x)\biggl(-\log p(y|x)-1+\sum_{i=1}^n\lambda_if_i(x,y)+\lambda_0\biggr)\\
&\triangleq 0
\end{align*}$$

求得：

$$p^*(y|x)=\exp\biggl(\sum_{i=1}^n\lambda_if_i(x,y)+\lambda_0-1\biggr)=\frac{\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)}{\exp(1-\lambda_0)}$$

由于$$\sum_yp(y$$\|$$x)=1$$，得

$$
\begin{equation}
p^*(y|x)=\frac{1}{Z_{\lambda}(x)}\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)
\end{equation}
$$

$$
\begin{equation}
Z_{\lambda}(x)=\exp(1-\lambda_0)=\sum_y\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)
\end{equation}
$$

这里$$\exp(1-\lambda_0)$$起到了归一化的作用。

现将求得的最优解$$p^*(y$$\|$$x)$$带回之前建立的拉格朗日函数L,可得到关于$$\lambda$$的函数：


$$
\begin{align*}
L(\lambda)&=-\sum_{x,y}\tilde{p}(x)p(y|x)\log p(y|x)+\sum_{i=1}^n\lambda_i\sum_{x,y}f_i(x,y)\biggl[\tilde{p}(x)p(y|x)-\tilde{p}(x,y)\biggr]+\lambda_0\biggl[\sum_yp(y|x)-1\biggr]\\
&=-\sum_{x,y}\tilde{p}(x)p_\lambda(y|x)\log p_\lambda(y|x)+\sum_{i=1}^n\lambda_i\sum_{x,y}f_i(x,y)\biggl[\tilde{p}(x)p_\lambda(y|x)-\tilde{p}(x,y)\biggr]\\
&= -\sum_{x,y}\tilde{p}(x)p_\lambda(y|x)\log p_\lambda(y|x)+\sum_{i=1}^n\lambda_i\tilde{p}(x)p_\lambda(y|x)\sum_{x,y}f_i(x,y)-\sum_{i=1}^n\lambda_i\tilde{p}(x,y)\sum_{x,y}f_i(x,y)\\
&=-\sum_{x,y}\tilde{p}(x)p_\lambda(y|x)\log \biggl[\frac{1}{Z_{\lambda}(x)}\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)\biggr]+\sum_{i=1}^n\lambda_i\tilde{p}(x)p_\lambda(y|x)\sum_{x,y}f_i(x,y)-\sum_{i=1}^n\lambda_i\tilde{p}(x,y)\sum_{x,y}f_i(x,y)\\
&=\sum_{x,y}\tilde{p}(x)p_\lambda(y|x)\log Z_{\lambda}(x) -\sum_{i=1}^n\lambda_i\tilde{p}(x,y)\sum_{x,y}f_i(x,y)
\end{align*}$$


从$$Z_{\lambda}(x)=\sum\limits_y\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)$$可以看出，最大熵模型模型属于对数线性模型，因为其包含指数函数，所以几乎不可能有解析解。我们需要使用其他方式进行逼近。


### 2.2.4极大似然估计

极大似然估计的MLE一般形式表示为：

$$L_{\tilde{p}}=\prod_xp(x)^{\tilde{p}(x)}$$

其中$$p(x)$$是对模型进行估计的概率分布，$$\tilde{p}(x)$$是实验结果得到的概率分布。两边取对数，得到对数似然估计:

$$L_{\tilde{p}}=\log\prod_xp(x)^{\tilde{p}(x)}=\sum_x\tilde{p}(x)\log p(x)$$

则对于$$p(x,y)$$的对数似然估计为：

$$
\begin{align*}
L_{\tilde{p}}(p)&=\sum_{x,y}\tilde{p}(x,y)\log p(x,y)\\
&=\sum_{x,y}\tilde{p}(x,y)\log [p(y|x)\tilde{p}(x)]\\
&=\sum_{x,y}\tilde{p}(x,y)\log p(y|x) +\sum_{x,y}\tilde{p}(x,y)\log \tilde{p}(x)
\end{align*}$$

上述式子最后结果的第二项是常数项,所以最终结果等价于：

$$L_{\tilde{p}}(p)=\sum_{x,y}\tilde{p}(x,y)\log p(y|x)$$

将之前得到的最大熵的解带入MLE:

$$
\begin{align*}
L_{\tilde{p}}(p)&=\sum_{x,y}\tilde{p}(x,y)\log p(y|x)\\
&=\sum_{x,y}\tilde{p}(x,y)\biggl[\sum\limits_{i=1}^n\lambda_if_i(x,y)-\log Z_\lambda(x)\biggr]\\
&=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\lambda_if_i(x,y)-\sum_{x,y}\tilde{p}(x,y)\log Z_\lambda(x)\\
&=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\lambda_if_i(x,y)-\sum_x\tilde{p}(x)\log Z_\lambda(x)\\
&=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\lambda_if_i(x,y)-\sum_x\tilde{p}(x)\biggl(\sum_yp_\lambda(y|x)\biggr)\log Z_\lambda(x)\\
&=-\biggl(\sum_{x,y}\tilde{p}(x)p_\lambda(y|x)\log Z_\lambda(x)-\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)
\end{align*}$$

上述结果跟之前得到的对偶问题的极大化解

$$\sum_{x,y}\tilde{p}(x)p_\lambda(y|x)\log Z_{\lambda}(x) -\sum_{i=1}^n\lambda_i\tilde{p}(x,y)\sum_{x,y}f_i(x,y)$$

只差一个“-”号，所以只要把原对偶问题的极大化解也加个负号，等价转换为对偶问题的极小化解,则与极大似然估计的结果具有完全相同的目标函数。因此，最大熵模型的对偶问题的极小化等价于最大熵模型的极大似然估计。

熵是表示不确定性的度量，似然表示的是与知识的吻合程度，进一步，最大熵模型是对不确定度的无偏分配，最大似然估计则是对知识的无偏理解。 且根据MLE的正确性，可以断定：最大熵的解（无偏的对待不确定性）同时是最符合样本数据分布的解，进一步证明了最大熵模型的合理性。

## 2.3改进的迭代尺度法(IIS)

最大熵模型为：

$$
\begin{equation}
p^*(y|x)=\frac{1}{Z_{\lambda}(x)}\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)
\end{equation}
$$

$$
\begin{equation}
Z_{\lambda}(x)=\sum_y\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)
\end{equation}
$$

对数似然函数为：

$$L_{\tilde{p}}(p)=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\lambda_if_i(x,y)-\sum_x\tilde{p}(x)\log Z_\lambda(x)$$

通过极大似然函数求解最大熵模型的参数，即求上述对数似然函数参数$$\lambda$$的极大值。此时，通常通过迭代算法求解，比如改进的迭代尺度法IIS、梯度下降法、牛顿法或拟牛顿法。这里主要介绍下其中的改进的迭代尺度法IIS。

 改进的迭代尺度法IIS的核心思想是：假设最大熵模型当前的参数向量是$$\lambda$$，希望找到一个新的参数向量$$\lambda+\delta$$，使得当前模型的对数似然函数值L增加。重复这一过程，直至找到对数似然函数的最大值。

下面，计算参数$$\lambda$$变到$$\lambda+\delta$$的过程中，对数似然函数的增加量，用$$L(\lambda+\delta)-L(\lambda)$$表示，同时利用不等式：$$-\ln x \geqslant 1-x , x>0$$，可得到对数似然函数增加量的下界，如下：

$$
\begin{align*}
L(\lambda+\delta)-L(\lambda)&=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\delta_if_i(x,y)-\sum_x\tilde{p}(x)\log \frac{Z_{\lambda+\delta}(x)}{Z_\lambda(x)}\\
&\geqslant \sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\delta_if_i(x,y)+1-\sum_x\tilde{p}(x)\frac{Z_{\lambda+\delta}(x)}{Z_\lambda(x)}\\
&=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\delta_if_i(x,y)+1-\sum_x\tilde{p}(x)\frac{Z_{\lambda+\delta}(x)}{Z_\lambda(x)}\\
&=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\delta_if_i(x,y)+1-\sum_x\tilde{p}(x)\frac{\sum_y\exp\biggl(\sum\limits_{i=1}^n(\lambda_i+\delta_i)f_i(x,y)\biggr)}{\sum_y\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)}\\
&=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\delta_if_i(x,y)+1-\sum_x\tilde{p}(x)\sum_yp_\lambda (y|x)\exp\biggl(\sum\limits_{i=1}^n\delta_if_i(x,y)\biggr)\\
\end{align*}
$$

 将上述求得的下界结果记为$$A(\delta$$\|$$\lambda)$$,并引入$$f^\sharp(x,y)=\sum_if_i(x,y)$$,f是一个二值函数，故$$f^\sharp(x,y)$$表示的是所有特征$$(x,y)$$出现的次数，然后利用Jensen不等式：

$$
\begin{align*}
A(\delta|\lambda)&=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\delta_if_i(x,y)+1-\sum_x\tilde{p}(x)\sum_yp_\lambda (y|x)\exp\biggl(\sum\limits_{i=1}^n\delta_if_i(x,y)\biggr)\\
&=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\delta_if_i(x,y)+1-\sum_x\tilde{p}(x)\sum_yp_\lambda (y|x)\exp\biggl(f^\sharp(x,y)\sum\limits_{i=1}^n\delta_i\frac{f_i(x,y)}{f^\sharp(x,y)}\biggr)\\
&\geqslant \sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\delta_if_i(x,y)+1-\sum_x\tilde{p}(x)\sum_yp_\lambda (y|x)\sum\limits_{i=1}^n\frac{f_i(x,y)}{f^\sharp(x,y)}\exp\biggl(\delta_if^\sharp(x,y)\biggr)
\end{align*}
$$

把上述式子求得的$$A(\delta$$\|$$\lambda)$$的下界记为$$B(\delta$$\|$$\lambda)$$：

$$B(\delta|\lambda)=\sum_{x,y}\tilde{p}(x,y)\sum\limits_{i=1}^n\delta_if_i(x,y)+1-\sum_x\tilde{p}(x)\sum_yp_\lambda (y|x)\sum\limits_{i=1}^n\frac{f_i(x,y)}{f^\sharp(x,y)}\exp\biggl(\delta_if^\sharp(x,y)\biggr)$$

则$$L(\lambda+\delta)-L(\lambda)\geqslant B(\delta$$\|$$\lambda)$$,对$$B(\delta$$\|$$\lambda)$$求偏导：

$$
\begin{align*}
\frac{\partial B(\delta|\lambda)}{\partial \delta_i}&=\sum_{x,y}\tilde{p}(x,y)f_i(x,y)-\sum_x\tilde{p}(x)\sum_yp_\lambda (y|x)f_i(x,y)\exp\biggl(\delta_if^\sharp(x,y)\biggr)\\
&=E_{\tilde{p}}(f_i)-\sum_{x,y}\tilde{p}(x)p_\lambda (y|x)f_i(x,y)\exp\biggl(\delta_if^\sharp(x,y)\biggr)
\end{align*}
$$

令偏导为0,若$$f^\sharp(x,y)=M$$为常数：

$${\delta _i} = \frac{1}{M}\log \frac{E_{\tilde{p}}( {f_i})}{E_p(f_i)}$$

若$$f^\sharp(x,y)$$不是常数，那么无法直接求得令偏导为0时候$$\delta$$的值，令偏导表示为：

$$
g(\delta_i)=\sum_{x,y}\tilde{p}(x)p_\lambda (y|x)f_i(x,y)\exp\biggl(\delta_if^\sharp(x,y)\biggr)-E_{\tilde{p}}(f_i)
$$

转为$$g(\delta_i)=0$$的根，用牛顿法：

$$\delta_i^{k+1}=\delta_i^k-\frac{g(\delta_i)}{g'(\delta_i)}$$

计算$$g(\delta _i) = 0$$的根而不是求$$g(\delta _i)$$的极小值,所以牛顿法中是函数值除以一阶导，而不是一阶导除以二阶导。实践中，可采用拟牛顿法BFGS解决。

将上述求解过程中得到的参数$$\lambda$$，回代到下式中，即可得到最大熵模型的最优估计:

$$
\begin{equation}
p^*(y|x)=\frac{1}{Z_{\lambda}(x)}\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)
\end{equation}
$$

$$
\begin{equation}
Z_{\lambda}(x)=\sum_y\exp\biggl(\sum\limits_{i=1}^n\lambda_if_i(x,y)\biggr)
\end{equation}
$$



