import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics

import matplotlib.pylab as plt

train = pd.read_csv('train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
train['Disbursed'].value_counts() 

x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']

rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y)
print (rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:,1]
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

#我们首先对n_estimators进行网格搜索：
param_test1 = {'n_estimators':list(range(10,71,10))}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


#接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
param_test2 = {'max_depth':list(range(3,14,2)), 'min_samples_split':list(range(50,201,20))}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, 
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)