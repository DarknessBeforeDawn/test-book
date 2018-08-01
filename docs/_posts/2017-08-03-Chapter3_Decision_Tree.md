---
title: 决策树
layout: post
share: false
---

决策树学习，假设给定训练数据集：


$$
D=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}
$$

其中 $x_i=(x_i^{(1)},x_i^{(2)},\cdots,x_i^{(n)})^T$ 为输入实例， $n$ 为特征个数，$$y\in\{1,2,\cdots,K\}$$为类标记， $N$ 为样本容量。学习目标是根据训练数据构建一个决策树模型，使它能够对实例正确分类。

熵（entropy）是表示随机变量不确定性的度量。设 $X$ 是一个取有限个值的离散随机变量，其概率分布为：

$$P(X=x_i)=p_i,  i=1,2,\cdots,n$$

则随机变量 $X$ 的熵定义为：

$$\begin{equation}
H(X)=-\sum_{i=1}^np_i\log p_i
\end{equation}$$

熵只依赖于 $X$ 的分布，而与 $X$ 的取值无关，因此可以将 $X$ 的熵记做 $H(P)$ ,熵越大随机变量的不确定性就越大，且 $0 \leqslant H(p) \leqslant \log n$ .

设随机变量 $(X,Y)$ ,其联合概率分布为：


$$\begin{equation}
P(X,Y)=P(X=x_i,Y=y_i)=p_{ij},  i=1,2,\cdots,n;j=1,2,\cdots,m
\end{equation}$$

条件熵 $H(Y$ \| $X)$ 表示在已知随机变量 $X$ 的条件下随机变量 $Y$ 的不确定性。 $X$ 给定条件下 $Y$ 的条件概率分布的熵对 $X$ 的数学期望：

$$\begin{equation}
H(Y|X)=\sum_{i=1}^np_iH(Y|X_i),p_i=P(X=x_i),i=1,2,\cdots,n
\end{equation}$$

#### 经验熵和经验条件熵

由数据估计(极大似然估计)得到的熵和条件熵。

如数据集 $D$ , 由 $K$ 个类别。经验熵是

$$\begin{equation}
H(D)=-\sum_{k=1}^K\frac{|C_k|}{|D|}\log\frac{|C_k|}{|D|}
\end{equation}$$

特征 $A$ 根据取值把数据集 $D$ 划分为 $n$ 个子集，则给定特征 $A$ 时数据集 $D$ 的经验条件熵是：

$$\begin{equation}
H(D|A)=-\sum_{i=1}^n\frac{|D_i|}{|D|}H(D_i)=-\sum_{i=1}^n\frac{|D_i|}{|D|}\sum_{k=1}^K\frac{|D_{ik}|}{|D_i|}\log\frac{|D_{ik}|}{|D_i|}
\end{equation}$$


# 1.信息增益

特征 $A$ 对训练数据集 $D$ 的信息增益 $g(D,A)$ ，定义为集合 $D$ 的经验熵 $H(D)$ 与特征 $A$ 给定条件下 $D$ 的经验熵 $H(D$ \| $A)$ 之差，即：

$$\begin{equation}
g(D,A)=H(D)-H(D|A)
\end{equation}$$

 $H(D)$ 表示对数据集 $D$ 进行分类的不确定性， $H(D$ \| $A)$ 表示按特征 $A$ 分类过后的所有熵的总和；二者的差表示分类前后不确定性减少的程度，因此信息增益大的特征有更强的分类能力。


设数据集为 $D$ ,\| $D$ \|表示样本个数，设有 $K$ 个类 $C_k$ ,\| $C_k$ \|为属于 $C_k$ 的样本个数， $\sum_{k=1}^K$ \| $C_k$ \| $=$ \| $D$ \|。设特征 $A$ 有 $n$ 个不同取值 $\{a_1,a_2,\cdots,a_n\}$ ，根据 $A$ 的取值将 $D$ 划分为 $n$ 个子集 $D_1,D_2,\cdots,D_n$ ,\| $D_i$ \|为 $D_i$ 的样本个数， $\sum_{i=1}^n$ \| $D_i$ \| $=$ \| $D$ \|,记子集 $D_i$ 中属于类 $C_k$ 的样本的集合为 $D_{ik}$ ,即 $D_{ik}=D_i\bigcap C_k,$ \| $D_{ik}$ \|为 $D_{ik}$ 的样本个数，则


$$\begin{equation}
H(D)=-\sum_{k=1}^K\frac{|C_k|}{|D|}\log\frac{|C_k|}{|D|}
\end{equation}$$

$$\begin{equation}
H(D|A)=-\sum_{i=1}^n\frac{|D_i|}{|D|}H(D_i)=-\sum_{i=1}^n\frac{|D_i|}{|D|}\sum_{k=1}^K\frac{|D_{ik}|}{|D_i|}\log\frac{|D_{ik}|}{|D_i|}
\end{equation}$$

#### 信息增益的问题

对于取值很多的特征，比如连续型数据(时间)。每一个取值几乎都可以确定一个样本。即这个特征就可以划分所有的样本数据。

- 信息增益不适合连续型、取值多的特征
- 使得所有分支下的样本集合都是纯的，极端情况每一个叶子节点都是一个样本
- 数据更纯，信息增益更大，选择它作为根节点，结果就是庞大且深度很浅的树


# 2.信息增益比
特征 $A$ 对训练数据集 $D$ 的信息增益比：

$$\begin{equation}
g_R(D,A)=\frac{g(D,A)}{H_A(D)}
\end{equation}$$

其中 $H_A(D)$ 为数据集 $D$ 关于特征 $A$ 的值的熵, $n$ 是特征 $A$ 取值的个数。

$$\begin{equation}
H_A(D)=-\sum_{i=1}^n\frac{|D_i|}{|D|}\log\frac{|D_i|}{|D|}
\end{equation}$$


解决信息增益的问题：特征 $A$ 分的类别越多， $D$ 关于 $A$ 的熵就越大，作为分母，所以信息增益 $g_R(D,A)$ 就越小。在信息增益的基础上增加了一个分母惩罚项。

信息增益比的问题：实际上偏好可取类别数目较少的特征。

# 3. $ID3$ 算法

决策树的生成， $ID3$ 算法以信息增益最大为标准选择特征。递归构建，不断选择最优特征对训练集进行划分。

递归终止条件：

- 当前节点的所有样本，属于同一类别 $C_k$ ，无需划分。该节点为叶子节点，类标记为 $C_k$ 
- 当前属性集为空，或所有样本在属性集上取值相同
- 当前节点的样本集合为空，没有样本

在集合 $D$ 中，选择信息增益最大的特征 $A_g$ :

- 增益小于阈值，则不继续向下分裂，到达叶子节点。该节点的标记为该节点所有样本中的 majority class $C_k$ 。 这也是预剪枝
- 增益大于阈值，按照特征 $A_g$ 的每一个取值 $A_g=a_i$ 把D划分为各个子集 $D_i$ ，去掉特征 $A_g$ 

继续对每个内部节点进行递归划分。


#### $C4.5$ 算法

$C4.5$ 是 $ID3$ 的改进， $C4.5$ 以信息增益率最大为标准选择特征。


# 4.决策树剪枝

决策树递归生成算法，易出现过拟合现象，这是由于在学习时过多的考虑如何提高对训练数据的正确分类，从而导致构建的决策树过于复杂，我们可以通过对决策树进行剪枝来简化决策树。

决策树剪枝通过极小化损失函数来实现。假设树 $T$ 的叶节点个数为\| $T$ \|, $t$ 是树\| $T$ \|的叶节点，该叶节点有 $N_t$ 个样本点，其中 $k$ 类的样本点有 $N_{tk}$ 个， $H_t(T)$ 位叶节点 $t$ 上的经验熵， $\alpha \geqslant 0$ 为参数，则决策树的损失函数可定义为：

$$\begin{equation}
C_\alpha(T)=\sum_{t=1}^{|T|}N_tH_t(T)+\alpha |T|
\end{equation}$$

其中经验熵为：

$$\begin{equation}
H_t(T)=-\sum_k\frac{N_{tk}}{N_t}\log\frac{N_{tk}}{N_t}
\end{equation}$$

令

$$\begin{equation}
C(T)=\sum_{t=1}^{|T|}N_tH_t(T)=-\sum_{t=1}^{|T|}\sum_{k=1}^KN_{tk}\log\frac{N_{tk}}{N_t}
\end{equation}$$

则 $C_\alpha(T) =C(T)+\alpha$ \| $T$ \| ，其中 $C(T)$ 表示模型对训练数据的预测误差，即模型与数据的拟合程度，\| $T$ \|表示模型的复杂程度，较大的 $\alpha$ 促使选择简单的模型，较小的 $\alpha$ 促使选择较复杂的模型。

剪枝就是当 $\alpha$ 确定时，选择损失函数最小的模型，假设一组叶节点回缩到其父节点之前与之后的整体树分别为 $T_B$ 和 $T_A$ ，其对应的损失函数数值分别为 $C_\alpha(T_B)$ 和 $C_\alpha(T_A)$ ,如果 $C_\alpha(T_B)\geqslant C_\alpha(T_A)$ ，则进行剪枝，即将父节点变为新的叶节点。

# 5.CART算法

## 5.1.回归生成树

首先，选择切分变量 $j$ 与切分点 $s$ ，将每个区域切分成两个子区域

$$\begin{equation}
R_1(j,s)=\{x|x^{(j)}\leqslant s\} , R_2(j,s)=\{x|x^{(j)}> s\}
\end{equation}$$

然后选择最优切分变量 $j$ 和切分点 $s$ ，具体地，求解：

$$\begin{equation}
min_{j,s}\biggl [min_{c_1}\sum_{x_i\in R_1(j,s)} (y_i-c_1)^2+min_{c_2}\sum_{x_i\in R_2(j,s)} (y_i-c_2)^2 \biggr]
\end{equation}$$

然后用选定的对 $(j,s)$ 划分区域并决定相应的输出值：

$$\begin{equation}
\hat{c}_m=\frac{1}{N_m}\sum_{x_i\in R_m(j,s)} y_i, x\in R_m,m=1,2
\end{equation}$$

递归对子区域做上述操作，直到满足停止条件。将输入空间划分为 $M$ 个区域 $R_1,R_2,\cdots,R_M$ ，生成决策树：

$$\begin{equation}
f(x) = \sum_{m=1}^M\hat{c}_mI(x\in R_m)
\end{equation}$$

## 5.2.分类树生成

### 尼基指数

分类问题中，假设有 $K$ 个类，样本点属于第 $k$ 类的概率为 $p_k$ ,则概率分布的尼基指数定义为:

$$\begin{equation}
Gini(p) = \sum_{k=1}^Kp_k(1-p_k)=1-\sum_{k=1}^Kp_k^2
\end{equation}$$

对于给定样本集合 $D$ ，其尼基指数为：

$$\begin{equation}
Gini(D) =1-\sum_{k=1}^K\biggl(\frac{|C_k|}{|D|}\biggr)^2
\end{equation}$$

这里 $C_k$ 是 $D$ 中属于第 $k$ 类的样本子集， $K$ 是类的个数。

如果样本集合 $D$ 根据特征 $A$ 是否取某一可能值 $a$ 被分割成 $D_1$ 和 $D_2$ 两部分，即 $D_1=\\{(x,y)\in D \| A(x)=a\\},D_2=D-D_1$ ，则在特征 $A$ 的条件下，集合 $D$ 的尼基指数定义为：

$$\begin{equation}
Gini(D,A) =\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)
\end{equation}$$

 $Gini(D)$ 表示集合 $D$ 的不确定性， $Gini(D,A)$ 表示经过 $A=a$ 分割后集合 $D$ 的不确定性。尼基指数越大，样本集合的不确定性越大同熵。

决策树生成：

1.计算现有特征对数据集的尼基指数，对每一个特征 $A$ ，对其可能取得每个值 $a$ ,根据是否 $A=a$ 将 $D$ 切割成 $D_1,D_2$ ,并计算 $A=a$ 时的尼基指数。

2.在所有可能的特征 $A$ 以及它们所有可能的切分点 $a$ 中，选择尼基指数最小的特征及其对应的切分点最为最优特征与最优切分点。一最优特征和最优切分点，从现有节点生成两个子节点，将训练数据集以特征分配到两个子节点中。

对子节点递归做1,2操作直至满足停止条件，生成CART决策树。

# 6.集成化(Ensemble)

### Bagging(bootstrap aggregating)
首先，从大小为 $N$ 的数据集 $D$ 中，分别独立进行多次随机抽取 $n$ 个数据，并使用每次抽取的数据训练弱分类器 $C_i$ 。

然后，对新的数据进行分类时，首先将新的数据放进每一个分类器 $C_i$ 中进行分类，得到每一个分类器对新数据的分类结果，并进行投票后获得优胜的结果。

### 随机森林(Random forest)
首先，从原始数据集从大小为 $N$ 的数据集 $D$ 中，有放回的抽取 $N$ 个数据，同一条数据可以重复被抽取，假设进行 $k$ 次，然后根据 $k$ 个数据集构建 $k$ 个决策树。

其次，设有 $n$ 个特征，在每一棵树的每个节点随机抽取 $m$ 个特征，并计算特征蕴含的信息量，并选择一个最具分类能力的特征进行节点分裂。每棵树最大限度的生长，不做任何剪裁。

最后，将生成的多棵树组成随机森林，用随机森林进行分类，分类结果桉树分类器投票多少而定。

### Boosting

假设训练集上一个有 $n$ 个样例，并对每个样例赋上一个权重 $W_i$ ,在训练起初每个点的权重相同；训练过程中提高在上次训练中分类错误的样例的权重，并降低分类正确样例的权重；在全部训练完成后得到 $M$ 个模型，对 $M$ 个模型的分类结果通过加权投票决定：

$$\begin{equation}
C(x)=sign\biggl[\sum_m^M\alpha_mC_m(x)\biggr]
\end{equation}$$

另外还有[AdaBoost](../Chapter4_AdaBoost/)和[GBDT](../Chapter5_Gradient_Boosting_Decision_Tree/)等。

[决策树相关示例代码](https://github.com/DarknessBeforeDawn/test-book/blob/master/code/decision_tree/decision_tree.ipynb)

[Kaggle房价预测示例代码](https://github.com/DarknessBeforeDawn/test-book/blob/master/code/house_price/house_price.ipynb)

