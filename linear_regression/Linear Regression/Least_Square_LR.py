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

    #最小二乘法，前提是得可逆
    xArr,yArr = loadDataSet(r'D:\git_repository\linear_regression\Linear Regression\least_square_data.txt')
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
    ws = xTx.I * (xMat.T*yMat)
    print(ws)

    #拟合曲线

    yhat = xMat*ws
    LR_fig = plt.figure()
    ax = LR_fig.add_subplot(111)
    ax.plot(xMat[:,1], yMat[:], 'ro')
    ax.plot(xMat[:,1], yhat, 'm', lw=1.0, label='fitting Curve')

    ax.legend(loc='upper left')
    plt.title('Linear Regression')

    plt.show()

