from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split  
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def feature_scaling(X, axis=0):
    new = X - np.mean(X, axis=0)
    return new / np.std(new, axis=0)  

# reading data
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_Y = diabetes.target

# feature scaling 
diabetes_x = feature_scaling(diabetes_X)
diabetes_y = feature_scaling(diabetes_Y)

# data split into training data and testing data
X_train,X_test, y_train, y_test =  train_test_split(diabetes_x,diabetes_y,test_size=0.4, random_state=0) 

# model 
linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)

# print parameters 
#常数项
print (linreg.intercept_)
#特征的参数
print (linreg.coef_) 

yhat = linreg.intercept_ + linreg.coef_*X_train

LR_fig = plt.figure()
ax = LR_fig.add_subplot(111)
ax.plot(X_train, y_train, 'ro')
ax.plot(X_train, yhat, 'm', lw=1.0, label='fitting Curve')
ax.legend(loc='upper left')
plt.title('Linear Regression')
# plt.grid(True)
plt.show()


#模型拟合测试集
y_pred = linreg.predict(X_test)

# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        