import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split  
from sklearn.cross_validation import  cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc  

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

def feature_scaling(X, axis=0):
    new = X - np.mean(X, axis=0)
    return new / np.std(new, axis=0)    


def main():

# reading data and data preprocessing
    adult_data_train_df = pd.read_csv(r'D:\git_repository\logistic_regression\adult_data_train_new.csv',names  =['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'])
    adult_data_test_df  = pd.read_csv(r'D:\git_repository\logistic_regression\adult_data_test_new.csv',names  =['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'])
    
# dealing with missing data    
    #将所有缺失值？替换为Null
    adult_data_train_df.replace('?',np.nan,inplace=True)

    adult_data_test_df.replace('?',np.nan,inplace=True)    
    #各列缺失missing data占比，或者通过df.info()粗略浏览一下
    total = adult_data_train_df.isnull().sum().sort_values(ascending=False)
    percent = (adult_data_train_df.isnull().sum()/adult_data_train_df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    #将每列none值替换为列值最多的值 occupation       workclass     native-country    

    adult_data_train_df['workclass'].fillna(adult_data_train_df['workclass'].value_counts().sort_values(ascending=False).index[0],inplace=True)
    adult_data_train_df['occupation'].fillna(adult_data_train_df['occupation'].value_counts().sort_values(ascending=False).index[0],inplace=True)
    adult_data_train_df['native-country'].fillna(adult_data_train_df['native-country'].value_counts().sort_values(ascending=False).index[0],inplace=True)


    adult_data_test_df['workclass'].fillna(adult_data_test_df['workclass'].value_counts().sort_values(ascending=False).index[0],inplace=True)
    adult_data_test_df['occupation'].fillna(adult_data_test_df['occupation'].value_counts().sort_values(ascending=False).index[0],inplace=True)
    adult_data_test_df['native-country'].fillna(adult_data_test_df['native-country'].value_counts().sort_values(ascending=False).index[0],inplace=True)


    #Y值改为0或者1 
    adult_data_train_df.loc[adult_data_train_df['income']=='<=50K','income']=0
    adult_data_train_df.loc[adult_data_train_df['income']=='>50K','income']=1

    adult_data_test_df.loc[adult_data_test_df['income']=='<=50K.','income']=0
    adult_data_test_df.loc[adult_data_test_df['income']=='>50K.','income']=1
    #强制转换数据类型
    adult_data_train_df['income'] = adult_data_train_df[['income']].astype(int)

    adult_data_test_df['income'] = adult_data_test_df[['income']].astype(int)
    
    #变量处理（特征离散化处理） 
    ################################################################################################################# 
    #1# 年龄 通过可视化选取最好的分界点 划分年龄区间
    plt.figure(1)
    facet = sns.FacetGrid(adult_data_train_df, hue="income",aspect=4)
    facet.map(sns.kdeplot,'age',shade= True)
    facet.set(xlim=(0, adult_data_train_df['age'].max()))
    facet.add_legend()
    
    plt.figure(1)
    average_age = adult_data_train_df[["age", "income"]].groupby(['age'],as_index=False).mean()
    sns.barplot(x='age', y='income', data=average_age)

    #年龄离散化
    adult_data_train_df.loc[adult_data_train_df['age']<=33,'age']= 0
    adult_data_train_df.loc[(adult_data_train_df['age']>33) & (adult_data_train_df['age']<62),'age']= 1
    adult_data_train_df.loc[adult_data_train_df['age']>=62,'age']= 2


    adult_data_test_df.loc[adult_data_test_df['age']<=33,'age']= 0
    adult_data_test_df.loc[(adult_data_test_df['age']>33) & (adult_data_test_df['age']<62),'age']= 1
    adult_data_test_df.loc[adult_data_test_df['age']>=62,'age']= 2


    #2# workclass  哑变量处理
    sns.factorplot('workclass','income', data=adult_data_train_df,size=8)

    workclass_dummies_train  = pd.get_dummies(adult_data_train_df['workclass'],prefix='workclass')
    workclass_dummies_train.drop(['workclass_Without-pay'], axis=1, inplace=True)
    adult_data_train_df = adult_data_train_df.join(workclass_dummies_train)
    adult_data_train_df.drop(['workclass'],axis =1 ,inplace =True)


    workclass_dummies_test  = pd.get_dummies(adult_data_test_df['workclass'],prefix='workclass')
    workclass_dummies_test.drop(['workclass_Without-pay'], axis=1, inplace=True)
    adult_data_test_df = adult_data_test_df.join(workclass_dummies_test)
    adult_data_test_df.drop(['workclass'],axis =1 ,inplace =True)

    
    #3# education  与  education_num 两列相同 留下一列即可 删除education
    adult_data_train_df.drop(['education'],axis =1 ,inplace =True)

    adult_data_test_df.drop(['education'],axis =1 ,inplace =True)


    #4# marital-status 婚姻关系 原理类似
    marital_dummies_train  = pd.get_dummies(adult_data_train_df['marital-status'],prefix='marital')
    marital_dummies_train.drop(['marital_Never-married'], axis=1, inplace=True)
    adult_data_train_df = adult_data_train_df.join(marital_dummies_train)
    adult_data_train_df.drop(['marital-status'],axis =1 ,inplace =True) 

    marital_dummies_test  = pd.get_dummies(adult_data_test_df['marital-status'],prefix='marital')
    marital_dummies_test.drop(['marital_Never-married'], axis=1, inplace=True)
    adult_data_test_df = adult_data_test_df.join(marital_dummies_test)
    adult_data_test_df.drop(['marital-status'],axis =1 ,inplace =True) 

    #5# occupation 工作
    occupation_dummies_train  = pd.get_dummies(adult_data_train_df['occupation'],prefix='occupa')
    occupation_dummies_train.drop(['occupa_Armed-Forces'], axis=1, inplace=True)
    adult_data_train_df = adult_data_train_df.join(occupation_dummies_train)
    adult_data_train_df.drop(['occupation'],axis =1 ,inplace =True)    

    occupation_dummies_test  = pd.get_dummies(adult_data_test_df['occupation'],prefix='occupa')
    occupation_dummies_test.drop(['occupa_Armed-Forces'], axis=1, inplace=True)
    adult_data_test_df = adult_data_test_df.join(occupation_dummies_test)
    adult_data_test_df.drop(['occupation'],axis =1 ,inplace =True)  

    #6# relation 工作
    relation_dummies_train  = pd.get_dummies(adult_data_train_df['relationship'],prefix='relation')
    relation_dummies_train.drop(['relation_Own-child'], axis=1, inplace=True)
    adult_data_train_df = adult_data_train_df.join(relation_dummies_train)
    adult_data_train_df.drop(['relationship'],axis =1 ,inplace =True)  

    relation_dummies_test  = pd.get_dummies(adult_data_test_df['relationship'],prefix='relation')
    relation_dummies_test.drop(['relation_Own-child'], axis=1, inplace=True)
    adult_data_test_df = adult_data_test_df.join(relation_dummies_test)
    adult_data_test_df.drop(['relationship'],axis =1 ,inplace =True)  

    #7# race 工作
    race_dummies_train  = pd.get_dummies(adult_data_train_df['race'],prefix='race')
    race_dummies_train.drop(['race_Other'], axis=1, inplace=True)
    adult_data_train_df = adult_data_train_df.join(race_dummies_train)
    adult_data_train_df.drop(['race'],axis =1 ,inplace =True)  

    race_dummies_test  = pd.get_dummies(adult_data_test_df['race'],prefix='race')
    race_dummies_test.drop(['race_Other'], axis=1, inplace=True)
    adult_data_test_df = adult_data_test_df.join(race_dummies_test)
    adult_data_test_df.drop(['race'],axis =1 ,inplace =True)  

    #8# sex 性别
    sex_dummies_train  = pd.get_dummies(adult_data_train_df['sex'],prefix='sex')
    sex_dummies_train.drop(['sex_Female'], axis=1, inplace=True)
    adult_data_train_df = adult_data_train_df.join(sex_dummies_train)
    adult_data_train_df.drop(['sex'],axis =1 ,inplace =True)  

    sex_dummies_test  = pd.get_dummies(adult_data_test_df['sex'],prefix='sex')
    sex_dummies_test.drop(['sex_Female'], axis=1, inplace=True)
    adult_data_test_df = adult_data_test_df.join(sex_dummies_test)
    adult_data_test_df.drop(['sex'],axis =1 ,inplace =True)  
    #8# country
    country_dummies_train  = pd.get_dummies(adult_data_train_df['native-country'],prefix='country')
    country_dummies_train.drop(['country_Outlying-US(Guam-USVI-etc)'], axis=1, inplace=True)
    adult_data_train_df = adult_data_train_df.join(country_dummies_train)
    adult_data_train_df.drop(['native-country'],axis =1 ,inplace =True)  

    country_dummies_test  = pd.get_dummies(adult_data_test_df['native-country'],prefix='country')
    country_dummies_test.drop(['country_Outlying-US(Guam-USVI-etc)'], axis=1, inplace=True)
    adult_data_test_df = adult_data_test_df.join(country_dummies_test)
    adult_data_test_df.drop(['native-country'],axis =1 ,inplace =True)  

    adult_data_test_df['country_Holand-Netherlands']=0

    #8# 其他连续变量归一化
    adult_data_train_df['fnlwgt'] = feature_scaling(adult_data_train_df['fnlwgt'])
    adult_data_train_df['capital-gain'] = feature_scaling(adult_data_train_df['capital-gain'])
    adult_data_train_df['capital-loss'] = feature_scaling(adult_data_train_df['capital-loss'])
    adult_data_train_df['hours-per-week'] = feature_scaling(adult_data_train_df['hours-per-week'])

    adult_data_test_df['fnlwgt'] = feature_scaling(adult_data_test_df['fnlwgt'])    
    adult_data_test_df['capital-gain'] = feature_scaling(adult_data_test_df['capital-gain'])
    adult_data_test_df['capital-loss'] = feature_scaling(adult_data_test_df['capital-loss'])
    adult_data_test_df['hours-per-week'] = feature_scaling(adult_data_test_df['hours-per-week'])
    
    #9# 特征和结果分离   
    
    X_train = adult_data_train_df.drop("income",axis=1)
    Y_train = adult_data_train_df["income"]


    X_test = adult_data_test_df.drop("income",axis=1)
    Y_test = adult_data_test_df["income"]     

# initialize linear regression parameters
    iterations = 100
    learning_rate = 0.001
    l2 = 0.001
    learning_algorithm = 'BGD'


    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)
    print('parameter sklearn:',classifier.coef_)

# evaluation the performance
#各指标的含义：
# 正确率accuracy是样本中正样本预测为正，负样本预测为负的总和比上总的样本数
# 精确率precision是针对我们预测结果而言的，它表示的是预测为正的样本中有多少是真正的正样本
# 召回率recall是针对我们原来的样本而言的，它表示的是样本中的正例有多少被预测正确了

    scores = cross_val_score(classifier, X_train, Y_train, cv=5)  
    print('accuracy:',np.mean(scores), scores)  
    precisions = cross_val_score(classifier, X_train, Y_train, cv=5, scoring='precision')  
    print('precison:', np.mean(precisions), precisions)  
    recalls = cross_val_score(classifier, X_train, Y_train, cv=5, scoring='recall')  
    print('recall:', np.mean(recalls), recalls)  

    #综合评价指标  
    #F1 值，是精确率和召回率的调和均值
    f1s = cross_val_score(classifier, X_train, Y_train, cv=5, scoring='f1')  
    print('f1:', np.mean(f1s), f1s)  
 
    
    #ROC AUC  
    #ROC曲线（Receiver Operating Characteristic，ROC curve）可以用来可视化分类器的效果。和准确率不同，ROC曲线对分类比例不平衡的数据集不敏感，ROC曲线显示的是对超过限定阈值的所有预测结果的分类器效果。ROC曲线画的是分类器的召回率与误警率（fall-out）的曲线。误警率也称假阳性率，是所有阴性样本中分类器识别为阳性的样本所占比例：  
    #F=FP/(TN+FP) AUC是ROC曲线下方的面积，它把ROC曲线变成一个值，表示分类器随机预测的效果. from sklearn.metrics import roc_curve, auc  

    predictions = classifier.predict_proba(X_test)  
    false_positive_rate, recall, thresholds = roc_curve(Y_test, predictions[:, 1])  
    roc_auc = auc(false_positive_rate, recall)  
    print('roc_auc: ',roc_auc)
    plt.title('Receiver Operating Characteristic')  
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)  
    plt.legend(loc='lower right')  
    plt.plot([0, 1], [0, 1], 'r--')  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.0])  
    plt.ylabel('Recall')  
    plt.xlabel('Fall-out')  
    plt.show()    



if __name__ == "__main__":
    main()

