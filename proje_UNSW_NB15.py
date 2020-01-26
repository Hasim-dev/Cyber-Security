# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:39:34 2019

@author: Hasim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

training_data=pd.read_csv("UNSW_NB15_training-set.csv")
testing_data=pd.read_csv("UNSW_NB15_testing-set.csv")

data=pd.concat([training_data,testing_data])
data.info()

data.drop(["id"],axis=1,inplace = True)
data.drop(["state","service"],axis=1,inplace=Tru
#%%
udp=data[data.proto == "udp" ]
tcp=data[data.proto == "tcp" ]
data=pd.concat([tcp,udp])

proto=data.iloc[:,1:2]

data.proto = [1 if each == "tcp" else 0 for each in data.proto]
data.drop(["state","service"],axis=1,inplace=True

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data.attack_cat=le.fit_transform(data.attack_cat)

sns.countplot(x="attack_cat",data=data);
corr_values=data.corr()["attack_cat"].sort_values()
print(corr_values)
plt.show()

#%%Correlation with output variable
cor_target = abs(corr_values)
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.02]
features=pd.DataFrame()
for i in relevant_features.index:
    features=pd.concat([features,data[i]],axis=1)
features.drop(["label"],axis=1,inplace=True)

#%%Using Pearson Correlation
import seaborn as sns
plt.figure(figsize=(10,10))
cor = features.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#%%
data_selected=pd.DataFrame()
data_selected=pd.concat([data_selected,features],axis=1)

#%%assign Class_att column as y attribute
y = data_selected.attack_cat.values

#drop Class_att column, remain only numerical columns
new_data = data_selected.drop(["attack_cat"],axis=1)

#Normalize values to fit between 0 and 1.
x = (new_data-np.min(new_data))/(np.max(new_data)-np.min(new_data)).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =0)

# %%Logistic regression classification
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
accuracy_lr = lr_model.score(x_test,y_test)
print("\nLogistic Regression accuracy is :",accuracy_lr)

# %%KNN Classification
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(x_train,y_train)
predicted_y = knn_model.predict(x_test)
accuracy_knn = knn_model.score(x_test,y_test)
print("\nKNN accuracy according to K=3 is :",accuracy_knn)

# %%SVM Classification
from sklearn.svm import SVC
svc_model = SVC(random_state = 1)
svc_model.fit(x_train,y_train)
accuracy_svc = svc_model.score(x_test,y_test)
print("\nSVM accuracy is :",accuracy_svc)

# %%Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(x_train,y_train)
accuracy_nb = nb_model.score(x_test,y_test)
print("\nNaive Bayes accuracy is :",accuracy_nb)
# %%Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
sonuc=dt_model.predict(x_test)
accuracy_dt = dt_model.score(x_test,y_test)
print("\nDecision Tree accuracy is :",accuracy_dt)

# %%Random Forest Classification - 5
from sklearn.ensemble import RandomForestClassifier
rf_model_initial = RandomForestClassifier(n_estimators = 5, random_state = 1)
rf_model_initial.fit(x_train,y_train)
accuracy_rf = rf_model_initial.score(x_test,y_test)
print("\nRandom Forest accuracy for 5 trees is :",rf_model_initial.score(x_test,y_test))
#%%
# %%Random Forest Classification - 8
rf_model_eight = RandomForestClassifier(n_estimators = 8, random_state = 1)
rf_model_eight.fit(x_train,y_train)
accuracy_rf_eight = rf_model_eight.score(x_test,y_test)
print("\nRandom Forest accuracy for 8 trees is :",rf_model_eleven.score(x_test,y_test))
# %%Random Forest Classification - 11
rf_model_eleven = RandomForestClassifier(n_estimators = 11, random_state = 1)
rf_model_eleven.fit(x_train,y_train)
accuracy_rf_eleven = rf_model_eleven.score(x_test,y_test)
print("\nRandom Forest accuracy for 11 trees is :",rf_model_eleven.score(x_test,y_test))

# %%Random Forest Classification - 15
rf_model_fifteen = RandomForestClassifier(n_estimators = 15, random_state = 1)
rf_model_fifteen.fit(x_train,y_train)
accuracy_rf_fifteen = rf_model_fifteen.score(x_test,y_test)
print("\nRandom Forest accuracy for 5 trees is :",rf_model_fifteen.score(x_test,y_test))

# %%Confusion Matrix libraries
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score

#%%for Logistic Regression
cm_lr = confusion_matrix(y_test,lr_model.predict(x_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_lr, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Logistic Regression")
plt.show()

#%%for KNN Classification (n_neighbors=3)
cm_knn=confusion_matrix(y_test,knn_model.predict(x_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_knn, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of KNN Classification")
plt.show()

#%%for SVM
cm_svc=confusion_matrix(y_test,svc_model.predict(x_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_svc, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of SVM")
plt.show()

#%%for Naive Bayes Classification
cm_nb=confusion_matrix(y_test,nb_model.predict(x_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_nb, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Naive Bayes Classification")
plt.show()

#%%for Decision Tree Classification
cm_dt=confusion_matrix(y_test,dt_model.predict(x_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_dt, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Decision Tree Classification")
plt.show()

#%%for Random Forest Classification - 5
cm_rf=confusion_matrix(y_test,rf_model_initial.predict(x_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_rf, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Random Forest Classification - N=5")
plt.show()

#%%for Random Forest Classification - 8
cm_rf=confusion_matrix(y_test,rf_model_eight.predict(x_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_rf, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Random Forest Classification - N=8")
plt.show()

#%%for Random Forest Classification - 11
cm_rf=confusion_matrix(y_test,rf_model_eleven.predict(x_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_rf, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Random Forest Classification - N=11")
plt.show()

#%%for Random Forest Classification - 15
cm_rf=confusion_matrix(y_test,rf_model_fifteen.predict(x_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_rf, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Random Forest Classification - N=15")
plt.show()

# %%the function that prints all scores
def print_scores(headline, y_true, y_pred):
    print(headline)
    acc_score = accuracy_score(y_true, y_pred)
    print("accuracy: ",acc_score)
    pre_score = precision_score(y_true, y_pred, average='micro')
    print("precision: ",pre_score)
    rec_score = recall_score(y_true, y_pred, average='micro')
    print("recall: ",rec_score)
    f_score = f1_score(y_true, y_pred, average='weighted')
    print("f1_score: ",f_score)

#%%
print_scores("Logistic Regression;",y_test, lr_model.predict(x_test))
print_scores("SVM;",y_test, svc_model.predict(x_test))
print_scores("KNN;",y_test, knn_model.predict(x_test))
print_scores("Naive Bayes;",y_test, nb_model.predict(x_test))
print_scores("Decision Tree;",y_test, dt_model.predict(x_test))
print_scores("Random Forest-5;",y_test, rf_model_initial.predict(x_test))
print_scores("Random Forest-8;",y_test, rf_model_eight.predict(x_test))
print_scores("Random Forest-11;",y_test, rf_model_eleven.predict(x_test))
print_scores("Random Forest;-15",y_test, rf_model_fifteen.predict(x_test
