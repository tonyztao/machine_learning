#在scikit-learn中，提供了3中朴素贝叶斯分类算法：GaussianNB(高斯朴素贝叶斯)、MultinomialNB(多项式朴素贝叶斯)、BernoulliNB(伯努利朴素贝叶斯)
#! /usr/bin/env python
#coding=utf-8

# simple naive bayes classifier to classify iris dataset

# 代码功能：简易朴素贝叶斯分类器，直接取iris数据集，根据花的多种数据特征，判定是什么花

from sklearn import datasets
iris = datasets.load_iris()

#我们假定sepal length, sepal width, petal length, petal width 4个量独立且服从高斯分布，用贝叶斯分类器建模
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
right_num = (iris.target == y_pred).sum()
print("Total testing num :%d , naive bayes accuracy :%f" %(iris.data.shape[0], float(right_num)/iris.data.shape[0]))