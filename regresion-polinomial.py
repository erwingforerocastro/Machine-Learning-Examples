# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:37:55 2020

@author: Erwing_fc
"""

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

boston = datasets.load_boston()
print(boston)
print()


print('informacion de el dataset')
print(boston.keys())
print()

print('caracteristicas del dataset')
print(boston.DESCR)
print()

print('cantidad de datos del dataset')
print(boston.data.shape)
print()

print('nombres de datos del dataset')
print(boston.feature_names)
print()

x_p=boston.data[:,np.newaxis,5]
y_p=boston.target

plt.scatter(x_p,y_p)
plt.show()

from sklearn.model_selection import train_test_split

#se separan los datos de entrenamiento y prediccion de un grupo, no del total
x_train_p,x_test_p,y_train_p,y_test_p=train_test_split(x_p,y_p,test_size=0.2)

from sklearn.preprocessing import PolynomialFeatures

poli_reg= PolynomialFeatures(degree=2)

x_train_poli=poli_reg.fit_transform(x_train_p)
x_test_poli=poli_reg.fit_transform(x_test_p)

#instancia del modelo de regresion lineal de sklearn
pr=linear_model.LinearRegression()

#se entrena el algoritmo
pr.fit(x_train_poli,y_train_p)

#se predicen los valores
y_pred_pr=pr.predict(x_test_poli)

plt.scatter(x_test_p,y_test_p)
plt.plot(x_test_p,y_pred_pr,color='red',lineWidth=3)
plt.show()

print()
print('Datos del modelo de regresion polinomial' )
print()
print('Valor de la pendiente o coeficiente "a" ')
print(pr.coef_)
print('Valor de la interseccion o  coeficiente "b" ' )
print(pr.intercept_)

print('Precision del modelo')
print(pr.score(x_train_poli,y_train_p))