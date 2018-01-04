from sklearn import cross_validation
from sklearn.cross_validation import train_test_split  
from sklearn import linear_model
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

if __name__ == "__main__":
    flag = 'Sklearn'
    xArr,yArr = loadDataSet(r'D:\git_repository\linear_regression\Linear Regression\ridge_regression_data.txt')
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas)

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha = a, fit_intercept=False)
        ridge.fit(xMat, yMat)
        coefs.append(ridge.coef_[0])


    # Display results

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis

    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()

        
