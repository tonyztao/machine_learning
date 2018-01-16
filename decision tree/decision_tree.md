
## 决策树
#### 决策树基本概念
决策树是一种既可以用于回归又可以用于分类的机器学习算法。它呈现出树形结构，在分类问题中，表示基于特征对实例进行分类的过程。决策树最大的优势就是比较直观易懂，可读性比较强，比较容易理解。总体上来说，决策树学习通常包括3个步骤：
- 特征选择
- 决策树的生成
- 决策树的剪枝

在本章节中会介绍三种典型的决策树算法：ID3和C4.5以及CART算法。这三种算法的主要区别在于属性选择，也就是最优特征选择上方法的差别，下面会详细进行说明。

#### 决策树方法的优缺点（翻译自sklearn）
* 优点  

（1）易于理解，可以产生可视化的分类规则，产生的模型具有可解释性  
（2）不需要过多的数据预处理；其他方法常常需要数据归一化、哑变量处理、缺失变量处理等  
（3）可以同时处理连续变量和离散变量；  
（4）可以处理多分类、多输出的情形；

* 缺点  

（1）容易产生过拟合问题。通过设置最大的叶节点个数或者树最大深度可帮助解决过拟合问题；  
（2）决策树的不稳定性，数据的微小变动会导致产生一个完全不同的决策树；  
（3）如果某些分类占优势，决策树将会创建一棵有偏差的树。因此，建议在训练之前，先抽样使样本均衡。

#### 数学辅助知识

在决策树的特征选择时，其目标是在选择了某个特征时，可以让各个子树的划分尽可能地“纯”，也就是尽可能的属于一类。这里面引入了信息论中熵的概念。熵是来表示不确定性的度量，根据真实分布，我们能够找到一个最优策略，以最小的代价消除系统的不确定性，而这个代价大小就是信息熵，记住，信息熵衡量了系统的不确定性，而我们要消除这个不确定性，所要付出的最小努力的大小就是信息熵。信息量与不确定性相对应：也就是提供的信息量越大，不确定性就越小，信息熵越小；相反如果提供的信息量越小，不确定就越大，信息熵也就越大。  
熵的数学公式如下：
 $$H=-\sum_{i=1}^np\left(x_{i}\right)\log_{2}p\left(x_{i}\right)$$

条件熵：
 $$H(Y|X)=\sum_{i=1}^np\left(X=x_{i}\right)H(Y |X=x_{i})$$

#### 决策树三种方法
* ID3算法

ID3算法以信息增益为准则来进行属性特征选择。
信息增益表示得知特征X的信息使得类Y的信息的不确定性减少程度。
特征A对训练数据集D的信息增益g(D,A)，定义为集合D的信息熵H(D)与特征A给定条件下的条件熵H(D|A)之差。

熵H(Y)与条件熵H(Y|X)之差称为互信息，即g(D,A)。
信息增益大表明信息增多，信息增多，则不确定性就越小。信息增益大的特征具有更强的分类能力。
  
信息增益公式：  
 $$Gain(D,A)=H(D)-H(D|A)$$  
 
对训练数据集，计算其每个特征的信息增益，并比较它们的大小，选择信息增益最大的特征进行分类。从根节点开始递归调用以上方法，构建决策树。  

优缺点：倾向于选择数量较多的变量，可能导致训练得到一个庞大且深度浅的树；另外输入变量必须是分类变量（连续变量必须离散化）；最后无法处理空值。

* C4.5算法

针对于以信息增益算法进行训练数据集划分时，存在偏向于选择取值较多的特征的问题。C4.5算法以信息增益率为准则来进行属性特征选择，来对以上问题进行校正。

信息增益率：特征A对训练数据集D的信息增益比定义为其信息增益与训练数据D关于特征A的值的熵$H_{A}(D)$之比

给出信息增益率公式：  
 $$Gain_{ratio}(D,A)=\frac{Gain(D,A)}{H_{A}(D)}=\frac{H(D)-H(D|A)}{H_{A}(D)}$$ 
 
需要注意的是，信息增益率对取值数目较少的属性有所偏好，因此C4.5算法并不是直接选择增益率最大的候选划分属性，而是使用了一个启发式：先从候选划分属性中找出信息增益率高于平均水平的属性，再从中选择增益率最高的。  

优缺点：对非离散数据也能处理；能够对不完整数据（有缺失数据）进行处理

* CART算法
  
CART树又名分类回归树，既能是分类树，又能是分类树。当CART是分类树时，采用GINI值作为节点分裂的依据；当CART是回归树时，采用样本的最小方差作为节点分裂的依据。CART是一棵二叉树。  

当分类树时：GINI系数反应了从数据集D中随机抽取两个样本，其类别标记不一致的概率。因此，GINI越小，则数据集D的纯度越高。故选择基尼系数最小的属性作为最优划分属性。  

GINI公式：
 $$Gini(D)=1-\sum_{i=1}^np_{k}^2$$   
 
#### 剪枝方法
决策树方法容易产生过拟合问题，故通过剪枝过程来简化决策树模型，从而提升泛化性能。决策树剪枝可分为预剪枝和后剪枝两类。

预剪枝是指在决策树生成过程中，对每个结点在划分前进行估计，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶结点；  
后剪枝则是先从训练集生成一颗完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能的提升，则将该子树替换为叶结点。


#### 代码实现以及注意事项



