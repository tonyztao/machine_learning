import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split  
import matplotlib as mpl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation
import matplotlib.ticker as mtick
from time import sleep
import copy

# parameter
plt.rcParams['animation.ffmpeg_path'] = 'C://Program Files//ffmpeg//bin//ffmpeg.exe'
plt.rcParams['animation.convert_path'] = 'C:\Program Files\ImageMagick-7.0.7-Q16\convert.exe' 

class LinearRegression:

    def __init__(self, learning_rate = 0.01, iterations = 50, verbose = True, l2 = 0, 
                tolerance = 0, intercept = True,learning_algorithm = 'BGD'):
        """
        :param learning_rate: learning rate constant
        :param iterations: how many iterations
        :param tolerance: the error value in which to stop training
        :param intercept: whether to fit an intercept
        :param verbose: whether to spit out error rates while training
        :param l2: L2 regularization term in order to prevent overfitting 
        :learning_algorithm: BGD or SGD
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.intercept = intercept
        self.verbose = verbose
        self.theta = None
        self.learning_algorithm = learning_algorithm if learning_algorithm=='BGD' else 'SGD'
        self.l2 = l2


    def fit(self, X, y):
        """
        Gradient descent, loops over theta and updates to
        take steps in direction of steepest decrease of J.
        :return: value of theta that minimizes J(theta) and J_history
        """
        if self.intercept:
            intercept = np.ones((np.shape(X)[0],1))
            X = np.concatenate((intercept, X), 1)
            
        num_examples, num_features = np.shape(X)

        #record the historic theta value
        history_theta = []

        # initialize theta to 1
        self.theta = np.ones(num_features)
 
        if self.learning_algorithm =='BGD':

            for i in range(self.iterations):
                # make prediction
                predicted = np.dot(X, self.theta.T)
                # update theta with gradient descent
                self.theta = (self.theta * (1 - (self.learning_rate * self.l2 / num_examples))) - self.learning_rate / num_examples * np.dot((predicted - y).T, X)
                # sum of squares cost
                error = predicted - y
                cost = np.sum(error**2) / (2 * num_examples)

                history_theta.append(copy.copy(self.theta))

                if i % 10 == 0 and self.verbose == True:
                    print('iteration:', i)
                    print('theta:', self.theta)
                    print('cost:', cost)
                    
                if cost < self.tolerance:
                    return self.theta
                    break
                
        if self.learning_algorithm =='SGD':

            for i in range(self.iterations):
                for k in range(num_examples):
                    #产生随机索引
                    randIndex = int(random.uniform(0, num_examples)) 
                    predicted = np.dot(X[randIndex], self.theta.T)
                    #计算错误率
                    error = predicted -y[randIndex]
                    #更新参数
                    self.theta = (self.theta * (1 - (self.learning_rate * self.l2 / num_examples)))  - self.learning_rate/ num_examples  * (error * X[randIndex])


                    cost = np.sum(error**2) / (2 * num_examples)
                    history_theta.append(copy.copy(self.theta))
                
                    if i % 10 == 0 and self.verbose == True:
                        print('iteration:', i)
                        print('theta:', self.theta)
                        print('cost:', cost)
                        
                    if cost < self.tolerance:
                        return self.theta
                        break

        return self.theta,history_theta 

    def predict(self, X):
        """
        Make linear prediction based on cost and gradient descent
        :param X: new data to make predictions on
        :return: return prediction
        """
        if self.intercept:
            intercept = np.ones((np.shape(X)[0],1))
            X = np.concatenate((intercept, X), 1)
        
        num_examples, num_features = np.shape(X)
        prediction = []
        for sample in range(num_examples):
            yhat = 0
            for value in range(num_features):
                yhat += X[sample, value] * self.theta[value]
            prediction.append(yhat)
                
        return prediction

def feature_scaling(X, axis=0):
    new = X - np.mean(X, axis=0)
    return new / np.std(new, axis=0)    

# mean squared error，MSE
def MSE(target, predictions):
    squared_deviation = np.power(target - predictions, 2)
    return np.mean(squared_deviation)


def main():

# reading data and use one feature at first 
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_Y = diabetes.target

# feature scaling 
    diabetes_x = feature_scaling(diabetes_X)
    diabetes_y = feature_scaling(diabetes_Y)

# data split into training data and testing data
    X_train,X_test, y_train, y_test =  train_test_split(diabetes_x,diabetes_y,test_size=0.4, random_state=0) 
        
# initialize linear regression parameters
    iterations = 1060
    learning_rate = 0.001
    l2 = 0.001
    learning_algorithm = 'BGD'


    linearReg_BGD = LinearRegression(learning_rate = learning_rate, iterations = iterations, verbose = 1, l2 = l2, learning_algorithm = learning_algorithm)

    iterations = 4
    learning_algorithm = 'SGD'
    linearReg_SGD = LinearRegression(learning_rate = learning_rate, iterations = iterations, verbose = 1, l2 = l2, learning_algorithm = learning_algorithm)

# learning linear regression parameters
    Xparameter_bgd,history_theta_BGD = linearReg_BGD.fit(X_train, y_train)

    Xparameter_sgd,history_theta_SGD = linearReg_SGD.fit(X_train, y_train)


    # plot the fitting curve : static pic
    LR_fig = plt.figure()
    ax = LR_fig.add_subplot(111)
    ax.plot(X_train, y_train, 'ro')
    X_value = np.concatenate((np.ones((np.shape(X_train)[0],1)), X_train), 1)
    y_bgd = np.dot(X_value,Xparameter_bgd.T)
    y_sgd = np.dot(X_value,Xparameter_sgd.T)
    ax.plot(X_train, y_bgd, 'm', lw=1.0, label='fitting Curve by BGD')
    ax.plot(X_train, y_sgd, 'g', lw=1.0, label='fitting Curve by SGD')
    ax.legend(loc='upper left')
    plt.title('Linear Regression')
    # plt.grid(True)
    plt.show()
    LR_fig.savefig(r"D:\git_repository\linear_regression\Linear Regression\Pic\LR_2methods_FittingCurve.png")
    plt.close(LR_fig)


    # fitting Curve gif
    # bgd & sgd

    fC_gd_gif_fig = plt.figure()
    ax = fC_gd_gif_fig.add_subplot(111)
    line1, = ax.plot([], [], 'r', lw=2.0, label='Batch Gradient Descent')
    line2, = ax.plot([], [], 'g', lw=2.0, label='Stochastic Gradient Descent')
    ax.legend(loc='upper left')
    plt.title('Linear Regression solved by Gradient Descent GIF')
    plt.grid(True)



        # for drawing fitting curve GIF
    def drawLine1(theta):
        X_value = np.concatenate((np.ones((np.shape(X_train)[0],1)), X_train), 1)
        y_bgd = np.dot(X_value,theta)
        line1.set_data(X_train.reshape(np.shape(X_train)[0]),y_bgd)
        return line1,

    # for drawing fitting curve GIF
    def drawLine2(theta):
        X_value = np.concatenate((np.ones((np.shape(X_train)[0],1)), X_train), 1)
        y_sgd = np.dot(X_value,theta)
        line2.set_data(X_train.reshape(np.shape(X_train)[0]),y_sgd)
        return line2,

    # for drawing fitting curve GIF
    def init():
        m, n = np.shape(X_value)
        ax = fC_gd_gif_fig.add_subplot(111)
        # ax.scatter(X[:,1], y, s=30, c='b', alpha=0.5)
        ax.plot(X_train, y_train, 'ro')
        drawLine1(np.ones((n,1)))
        drawLine2(np.ones((n,1)))
        return

    # for drawing fitting curve GIF
    def animate(i):
        drawLine1(history_theta_BGD[i].reshape(2,1))
        drawLine2(history_theta_SGD[i].reshape(2,1))
        return


    anim = animation.FuncAnimation(fC_gd_gif_fig, animate, init_func=init,
                                    frames=len(history_theta_BGD),
                                    interval=1,
                                    repeat=False,
                                    blit=False
                                    )

    plt.show()
    anim.save(r"D:\git_repository\linear_regression\Linear Regression\Pic\LR_GD_FittingCurve.gif", writer='imagemagick', fps=50)
    plt.close(fC_gd_gif_fig)
   

# prediction
    yhat_BGD = linearReg_BGD.predict(X_test)

    yhat_SGD = linearReg_SGD.predict(X_test)

# evaluation the performance
    mean_squared_error_BGD = MSE(y_test,yhat_BGD)
    print("MSE of BGD Algorithm:",mean_squared_error_BGD)

    mean_squared_error_SGD = MSE(y_test,yhat_SGD)
    print("MSE of SGD Algorithm:",mean_squared_error_SGD)



if __name__ == "__main__":
    main()

