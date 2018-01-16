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
 代码实现关键点如下：  
      - 示例中的数据来源 * [adult classification dataset](http://archive.ics.uci.edu/ml/datasets/Adult)    
      - 数据预处理：包括连续数据归一化，分类数据one-hot编码。通过pandas的get_dummies实现，要强调一点，生成m个虚拟变量后，只要引入m-1个虚拟变量到数据集中，未引入的一个是作为基准对比的，同时在样本中把要处理的特征列删除掉。  
      - 数据可视化：通过seaborn考察各特征与结果Y（0,1）的关系；考察变量间的相关性；  
       ![Seaborn对年龄的可视化](https://github.com/tonyztao/machine_learning/blob/master/logistic_regression/age_distribution.png)    
       - 模型性能评估：通过Accuracy、Recall、Precision、F1 Value、ROC曲线、Auc曲线下面积对性能进行了评估；  
       ![ROC曲线](https://github.com/tonyztao/machine_learning/blob/master/logistic_regression/ROC%E6%9B%B2%E7%BA%BF.png)  
		
