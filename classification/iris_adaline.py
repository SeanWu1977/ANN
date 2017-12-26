import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Adaline as ppn

import plot_decision_regions as pdr

df=pd.read_csv('iris.data',header=None)
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-versicolor' ,-1,1)
X = df.iloc[0:100,[0,2]].values # get first 100 rows and col0 and col2


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

ppn1 = ppn.Adaline(0.01 , 10)
ppn1.fit(X,y)
ax[0].plot(range(1,len(ppn1.cost_)+1),np.log10(ppn1.cost_),marker='*')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate {}'.format(ppn1.eta))


ppn2 = ppn.Adaline(0.0001 , 10)
ppn2.fit(X,y)
ax[1].plot(range(1,len(ppn2.cost_)+1),ppn2.cost_,marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate {}'.format(ppn2.eta))
plt.show()


# standardization
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()
ppn3 = ppn.Adaline(0.01 , 15)
ppn3.fit(X_std,y)
pdr.plot_decision_regions(X_std, y, classifier=ppn3)
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.title('Adaline - Gradient Descent')
plt.legend(loc="upper left")
plt.show()

plt.plot(range(1,len(ppn3.cost_)+1),ppn3.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()



