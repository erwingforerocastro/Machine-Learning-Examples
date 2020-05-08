# -*-Autor Erwing Forero
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

X = boston.data[:,np.newaxis,5]

Y =  boston.target

plt.scatter(X,Y)
plt.xlabel('Numero de habitaciones')
plt.ylabel('Valor medio')
plt.show()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

lr=linear_model.LinearRegression()
lr.fit(x_train,y_train)
Y_pred=lr.predict(x_test)

plt.scatter(x_test,y_test)
plt.plot(x_test,Y_pred,color="red",lineWidth=3)
plt.title("regresion lineal simple")
plt.xlabel('Numero de habitaciones')
plt.ylabel('valor medio')
plt.show()

print()
print('Datos del modelo de regresion simple' )
print()
print('Valor de la pendiente o coeficiente "a" ')
print(lr.coef_)
print('Valor de la interseccion o  coeficiente "b" ' )
print(lr.intercept_)
print('ecuacion')
print('y= ',lr.coef_,'X ',lr.intercept_)
print()

print('Precision del modelo')
print(lr.score())


