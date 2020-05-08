# -*-Autor Erwing Forero erwingforerocastro@gmail.com
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
 
x_adr=boston.data[:,np.newaxis,5]

y_adr=boston.target

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_adr,y_adr,test_size=0.2)

from sklearn.tree import DecisionTreeRegressor

adr=DecisionTreeRegressor(max_depth=5) #profundidad del arbol 

adr.fit(x_train,y_train)

y_pred=adr.predict(x_test)

x_grid=np.arange(min(x_test),max(x_test),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x_test,y_test)
plt.plot(x_grid,adr.predict(x_grid),color='red',lineWidth=3)
plt.show()

print('Precisión del modelo')
print(adr.score(x_train,y_train))
