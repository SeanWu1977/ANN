import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import AdalineSGD as ppn

import plot_decision_regions as pdr

df=pd.read_csv('iris.data',header=None)
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-versicolor' ,-1,1)
X = df.iloc[0:100,[0,2]].values # get first 100 rows and col0 and col2
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()
ppn3 = ppn.AdalineSGD(0.01 , 15, True, 1)
ppn3.fit(X_std,y)

pdr.plot_decision_regions(X_std, y, classifier=ppn3)
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title('Adaline - Stochastic Gradient Descent')
plt.legend(loc="upper left")
plt.show()

plt.plot(range(1,len(ppn3.cost_)+1),ppn3.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()





