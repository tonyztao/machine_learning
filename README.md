# machine_learning
机器学习，代码记录，主要关键点整理。

## Linear Regression
主要包含如下关键点: 
1. 数据准备: 训练集和测试集的划分，训练集数据的数量占比在2/3到4/5
2. 特征归一化: 可以使得梯度下降收敛的更快
3. 算法模型: BGDT（批量梯度下降法） and SGDT （随机梯度下降法），区别在于每次训练时候是否使用全部的训练集进行参数迭代计算；
4. 正则化 :防止过拟合
5. 拟合曲线：展现训练拟合过程，直观感受
6. 评估模型:MSE均方误差，或者计算预测值和真实值之间的相关性
7. 其他的线性回归方法：ridge回归（L2 penalty）和lasso回归（L1 penalty），引入其他线性回归算法，因为Linear regression一般只对low dimension适用，而且变量间还不存在multicolinearity。多重共线性时表明变量间存在相关性，$\mathbf{X}^\mathrm{T}\mathbf{X}$不可逆。这两种方法能够对具有多重共线性的模型进行变量剔除。

      -ridge回归（L2 penalty）
      
      公式如下：
      $$x=\frac{1}{2m}\left[\sum_{i=1}^m\left(h_{\theta}\left(x^{i}\right)-y^{i}\right)^{2}+\lambda\sum_{j=1}^n\theta_{j}^2\right]$$
      1. 岭回归可以解决特征数量比样本量多的问题
      2. 使用岭回归和缩减技术，首先需要对特征作标准化处理，使得每个特征具有相同的重要性，这样才能从得到的系数中反应各个参数的重要程度
      3. 岭回归作为一种缩减算法可以判断哪些特征重要或者不重要，有点类似于降维的效果
      4. 缩减算法可以看作是对一个模型增加偏差的同时减少方差，也就是减少过拟合的程度，更具有泛化的表现
      5. 岭回归中另外一个关键是参数$\lambda$的选取，一般是等间隔选取多个$\lambda$选择最优表现的值
      6. 为了定量的找到最佳参数值，需要通过交叉验证的方式得到最佳的$\lambda$
    
   
    -lasso回归（L1 penalty）
      
      公式如下：
      $$x=\frac{1}{2m}\left[\sum_{i=1}^m\left(h_{\theta}\left(x^{i}\right)-y^{i}\right)^{2}+\lambda\sum_{j=1}^n\left|\theta_{j}\right|\right]$$
      1. Lasso回归使得一些系数变小，甚至还是一些绝对值较小的系数直接变为0，因此特别适用于参数数目缩减与参数的选择，因而用来估计稀疏参数的线性模型
      2. 区别：Ridge回归在不抛弃任何一个特征的情况下，缩小了回归系数，使得模型相对而言比较的稳定，但和Lasso回归比，这会使得模型的特征留的特别多，模型解释性差。

*  [Linear Regression Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/Linear_Regression.py/)
* [Least Square Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/Least_Square_LR.py)
* [Ridge Regression Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/ridge_regresion.py)
## Logistic Regression
本节介绍逻辑回归算法， 这是最流行最广泛使用的一种学习算法，是一种分类算法。
主要包含如下关键点:
1. 算法的通用流程：
      - 寻找h函数（即预测函数）
      - 构造J函数（损失函数） 
      - 想办法使得J函数最小并求得回归参数（θ） 

2. 构造预测函数$h_{\theta}\left(X\right)$：  
逻辑回归的输入是一个线性组合，与线性回归一样，但输出变成了概率，通过引入Logit函数将结果输出在范围$\left(0,1\right)$，这样的话使得结果更容易被解释，通过概率来判断属于哪一个类别。
逻辑回归模型的假设是：$h_{\theta}=g\left(\theta^TX\right)$  其中:$X$代表特征向量,$g$代表逻辑函数（logistic function)是一个常用的逻辑函数为S形函数（Sigmoid function），公式为：$g\left(z\right)=\frac{1}{1+e^{-z}}$  
![Sigmoid ](https://github.com/tonyztao/machine_learning/blob/master/logistic_regression/sigmoid%E5%87%BD%E6%95%B0.png)  
我们可以看到，sigmoid的函数输出是介于（0，1）之间的，中间值是0.5，于是之前的公式 $h_{\theta}\left(X\right)$ 的含义就很好理解了，因为 $h_{\theta}\left(X\right)$ 输出是介于（0，1）之间，也就表明了数据属于某一类别的概率，例如 :  
$h_{\theta}\left(X\right)<0.5$则说明当前数据属于A类；     
$h_{\theta}\left(X\right)>0.5$ 则说明当前数据属于B类。  
 构造的预测函数为：$h_{\theta}\left(X\right)=g\left(\theta^TX\right)=\frac{1}{1+e^{-\theta^T}X}$

3. 构造损失函数$J\left(\theta\right)$： 
 J函数如下，它们是基于最大似然估计推导得到的，具体推导过程略。 
 ![损失函数 ](https://github.com/tonyztao/machine_learning/blob/master/logistic_regression/%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%85%AC%E5%BC%8F.png)  
 4. 求最小的参数$\min\limits_{\theta}J$：  
  在得到代价函数以后，就可以用梯度下降算法来求得能使代价函数最小的参数了。算法为：  
   ![梯度下降算法 ](https://github.com/tonyztao/machine_learning/blob/master/logistic_regression/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.png)  
    最后还有一点，我们之前在谈线性回归时讲到的特征缩放，我们看到了特征缩放是如何提高梯度下降的收敛速度的，这个特征缩放的方法，也适用于逻辑回归
  
  5. 实现代码：
* [Logistic Regression Code](https://github.com/tonyztao/machine_learning/blob/master/logistic_regression/Logistic_Regression.py)
		


## Scikit-learn Realization
*  linear regression
* [Linear Regression Sklearn Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/Siciket_learn_LR.py)


