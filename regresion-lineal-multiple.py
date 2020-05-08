# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:42:02 2020

@author: Erwing_fc
"""

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

boston=datasets.load_boston()

x_multiple = boston.data[:,5:8]
#print(x_multiple)
y_multiple=boston.target
#print(y_multiple)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_multiple,y_multiple,test_size=0.2)

lr_multiple=linear_model.LinearRegression()

lr_multiple.fit(x_train,y_train)

Y_pred_mult=lr_multiple.predict(x_test)


print()
print('Datos del modelo de regresion simple' )
print()
print('Valor de la pendiente o coeficiente "a" ')
print(lr_multiple.coef_)
print('Valor de la interseccion o  coeficiente "b" ' )
print(lr_multiple.intercept_)


print('Precision del modelo')
print(lr_multiple.score(x_train,y_train ))