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
7. 其他的线性回归方法：ridge回归（L2 penalty）和lasso回归（L1 penalty），引入其他线性回归算法，因为Linear regression一般只对low dimension适用，而且变量间还不存在multicolinearity.
    -ridge回归（L2 penalty）
    
    -lasso回归（L1 penalty）

$$x=\frac{-b\pm\sqrt{b^2\pm-4ac}}{2a}$$

*  [Linear Regression Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/Linear_Regression.py/)
*  [Least Square Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/Least_Square_LR.py)

## Logistic Regression

## Scikit-learn Realization
*  linear regression
* [Linear Regression Sklearn Code](https://github.com/tonyztao/machine_learning/blob/master/linear_regression/Linear%20Regression/Siciket_learn_LR.py)


