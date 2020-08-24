# coding: utf-8
"""This program aims to achieve classify 2 statues of epilepsy with 4 classifiers to figure out the seizure stage.
"""

# Authors: RM Peng <rmpeng19@gmail.com>
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from xgboost import XGBClassifier,plot_importance
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score,f1_score,classification_report
from utils import _classification as evaluate
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt

# load feature matrix
file2= 'dataset/bonn_feature/normal_feature_new.npz'
file3= 'dataset/bonn_feature/inter_feature_new.npz'
file4= 'dataset/bonn_feature/ictal_feature_new.npz'

test1=np.load(file2, allow_pickle=True)
test2 = np.load(file3, allow_pickle=True)
test3 = np.load(file4, allow_pickle=True)

time_fea = np.vstack((test1['time'],test2['time'],test3['time']))
freq_fea = np.vstack((test1['freq'],test2['freq'],test3['freq']))
tf_fea = np.vstack((test1['tf'],test2['tf'],test3['tf']))
en_fea = np.vstack((test1['entropy'],test2['entropy'],test3['entropy']))
all_fea = np.hstack((time_fea,freq_fea,tf_fea,en_fea))
# select the features you want to use for epilepsy detection
X = all_fea # or time_fea, freq_fea, tf_fea, en_fea
y=np.zeros((500,1))

for i in range(500):
    if i <400:
        y[i,:] = 0
    else:
        y[i,:] = 1

y = y.squeeze()

#'''if you choose the classifier as KNN, the best K should be found out first. This part gives a way to figure
# out the best k value.
# '''
# k_error = []
# k_range = range(1,31)
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X, y, cv=6, scoring='accuracy')
#     k_error.append(1 - scores.mean())
# plt.plot(k_range, k_error)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Error')
# plt.show()
#the results shows that k=9/11 make the least error
k_neighbor = 11

#params for RF
Num_of_Learners = 150
N_JOBS = 20
max_leafs = None

#10 folds cross-validation experiment
kf = KFold(n_splits=10,shuffle=True,random_state=2)

#initilize the evaluation for the performance of classify
acc = 0
sen = 0
spec = 0
f1_avg =0

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # scale
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

#svm
#     model1 = SVC(kernel='linear')
#     model1.fit(X=X_train, y=y_train)
#     y_pred = model1.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     acc+=accuracy
#     Sensitivity = recall_score(y_test, y_pred, pos_label=0)
#     sen+=Sensitivity
#     f1 = f1_score(y_test, y_pred, pos_label=0)
#     f1_avg +=f1
#     print('acc:', accuracy)
#     print('sen:', Sensitivity)
#     print('f1', f1)
# print('acc_avg',acc/10)
# print('sen_avg',sen/10)
# print('F1_avg',f1_avg/10)

#xgb
    model = XGBClassifier(learning_rate=0.1, n_estimators=1000,  # 树的个数--1000棵树建立xgboost
                                          max_depth=6,  # 树的深度
                                          min_child_weight=1,  # 叶子节点最小权重
                                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                                          subsample=0.8,  # 随机选择80%样本建立决策树
                                          colsample_btree=0.8,  # 随机选择80%特征建立决策树
                                          objective='binary:logistic',  # 指定损失函数
                                          scale_pos_weight=1,  # 解决样本个数不平衡的问题
                                          random_state=27  # 随机数
                                          )
    model.fit(X_train, y_train,eval_set=[(X_test, y_test)],
                  eval_metric="logloss",
                  early_stopping_rounds=10,
                  verbose = False)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    acc += accuracy
    Sensitivity = recall_score(y_test, y_pred, pos_label=1)
    sen += Sensitivity
    specificity = evaluate.specificity_score(y_test, y_pred)
    spec += specificity
    f1 = f1_score(y_test, y_pred, pos_label=0)
    f1_avg += f1
    print('acc:', accuracy)
    print('sen:', Sensitivity)
    print('spec', specificity)
print('acc_avg', acc / 10)
print('sen_avg', sen / 10)
print('spec_avg', spec / 10)

#random forest
#     model = GridSearchCV(RandomForestClassifier(n_estimators=Num_of_Learners, max_leaf_nodes=max_leafs),
#         cv=5,
#         param_grid={"min_samples_leaf": range(5, 26)},
#         iid=True,
#         n_jobs=N_JOBS)
#
#     model.fit(X_train, y_train)
#     model_best_ = model.best_estimator_
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     acc += accuracy
#     Sensitivity = recall_score(y_test, y_pred, pos_label=0)
#     sen += Sensitivity
#     f1 = f1_score(y_test, y_pred, pos_label=0)
#     f1_avg += f1
#     print('acc:',accuracy)
#     print('sen:',Sensitivity)
#     print('f1',f1)
# print('acc_avg',acc/10)
# print('sen_avg',sen/10)
# print('F1_avg',f1_avg/10)

#KNN
#
#     model1 = KNeighborsClassifier(n_neighbors=k_neighbor, weights='uniform')
#     model1.fit(X=X_train, y=y_train)
#     y_pred = model1.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     acc+=accuracy
#     Sensitivity = recall_score(y_test, y_pred, pos_label=0)
#     sen+=Sensitivity
#     f1 = f1_score(y_test, y_pred, pos_label=0)
#     f1_avg +=f1
#     print('acc:', accuracy)
#     print('sen:', Sensitivity)
#     print('f1', f1)
# print('acc_avg',acc/10)
# print('sen_avg',sen/10)
# print('F1_avg',f1_avg/10)