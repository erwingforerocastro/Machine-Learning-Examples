# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:16:46 2020

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

x_svr=boston.data[:,np.newaxis,5]
y_svr=boston.target

plt.scatter(x_svr,y_svr)
plt.show()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_svr,y_svr,test_size=0.2)

from sklearn.svm import SVR
#definiendo los parametros manualmente
svr=SVR(kernel='linear',C=0.1,epsilon=0.2)

#entrenar el algoritmo
svr.fit(x_train,y_train) 

#realizar la predicción
y_pred=svr.predict(x_test)

plt.scatter(x_svr,y_svr)
plt.plot(x_test,y_pred,color='red',lineWidth=3)
plt.show()

print()
print('Datos del modelo de regresion simple' )
print()
print('Valor de la pendiente o coeficiente "a" ')
print(svr.coef_)
print('Valor de la interseccion o  coeficiente "b" ' )
print(svr.intercept_)
print('ecuacion')
print('y= ',svr.coef_,'X ',svr.intercept_)
print()

print('Precision del modelo')
print(svr.score(x_train,y_train))
