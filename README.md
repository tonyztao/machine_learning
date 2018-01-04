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
      $$x=\frac{1}{2m}\left[\sum_{i=1}^m\left(h_{\theta}\left(x^{i}\right)-y^{i}\right)^{2}+\lambda\left|\theta_{j}\right|\right]$$


*  [Linear Regression Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/Linear_Regression.py/)
* [Least Square Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/Least_Square_LR.py)
* [Ridge Regression Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/ridge_regresion.py)
## Logistic Regression

## Scikit-learn Realization
*  linear regression
* [Linear Regression Sklearn Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/Siciket_learn_LR.py)


