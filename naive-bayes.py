# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:24:12 2020

@author: Erwing_fc
"""

from sklearn import datasets

dataset=datasets.load_breast_cancer()
print(dataset)
print()
# llaves

print('Información del dataset')
print(dataset.keys())
print()

# Caracteristicas

print('Caracteristicas del dataset')
print(dataset.DESCR)
print()

#datos de las columnas

x=dataset.data
y=dataset.target

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB

algoritmo=GaussianNB()
algoritmo.fit(x_train,y_train)
y_pred=algoritmo.predict(x_test)

#verifico la matriz de confusion
from sklearn.metrics import confusion_matrix

Matriz=confusion_matrix(y_test,y_pred)
print('Matriz de confusión')
print(Matriz)

#Precision del modelo
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
print('Precisión del modelo')
print(precision)

#la exactitud del modelo
from sklearn.metrics import accuracy_score
exactitud=accuracy_score(y_test,y_pred)
print('Exactitud del modelo')
print(exactitud)

#sensibilidad del modelo
from sklearn.metrics import recall_score
sensibilidad=recall_score(y_test,y_pred)
print('Sensibilidad del modelo')
print(sensibilidad)

#puntaje f1 del modelo
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
print('Calculo de f1 del modelo')
print(f1)

#CAlculo de la curva ROC - AUC del modelo
from sklearn.metrics import roc_auc_score
roc_auc=roc_auc_score(y_test,y_pred)
print('curva ROC - AUC del modelo')
print(roc_auc)
