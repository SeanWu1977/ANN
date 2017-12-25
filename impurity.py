from sklearn import datasets
import numpy as np
import plot_decision_regions as pdr
import matplotlib.pyplot as plt


# 決策樹中不是的impurity(不純度) 測量公式
def gini(p):
    return (p)*(1-(p)) + (1 - p)*(1-(1 - p))

def entropy(p):
    # p = 0 要排除
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def error(p):
    return 1 - np.max([p,(1-p)])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]

err = [error(i) for i in x]

plt.figure(num = 'ed', figsize = (10, 10)) # plt.figure('dfg')
ax = plt.subplot(111)
for i, lab, ls, c, in zip(
    [ent, sc_ent, gini(x), err],
    ['Entropy', 'Entropy(scaled)', 'Gini Impurity', 'Misclassification Error'],
    ['-', '-', '--', '-.'],
    ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.15),
           ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1,color='k', linestyle='--')
ax.axhline(y=1.5, linewidth=1,color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('impurit index')
plt.show()


# 由圖可看到，節點屬於同一類時，impurity為0
# 父節點impurity大，子節點impurity小時，分類效果最好
# information gain (IG) 資訊增益 父不純度- 所有(各別子不純度*各別子個數/所有子個數)和
