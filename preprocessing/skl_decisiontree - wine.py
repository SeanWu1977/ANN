import numpy as np,pandas as pd
import matplotlib.pyplot as plt


# 將資料分為訓練集與測試集函數
from sklearn.model_selection import train_test_split


# 決策樹

from sklearn.ensemble import RandomForestClassifier as rfc



df_wine = pd.read_csv('wine.data', header=None)
df_wine.columns = [
'class level',
'Alcohol',
'Malic acid',
'Ash',
'Alcalinity of ash',
'Magnesium',
'Total phenols',
'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity',
'Hue',
'OD280/OD315 of diluted wines',
'Proline'
]

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
# 將資料分為訓練集(70%)與測試集(30%)
# random_state 0 表示不執行seed(), 即每次執行順序會不同
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

feat_labels = df_wine.columns[1:]

# 模型訓練{'entropy' or 'gini'(default)}

# n_estimators = 隨機樹數量  ，  n_jobs cpu使用量
forest = rfc( n_jobs=-1, random_state=1, n_estimators=10000)
forest.fit(X_train, y_train)

# importances 為每個特徵的重要性，所有特徵重要性加總為1
importances = forest.feature_importances_
# np.argsort(importances) 排序由小到大，在用[::-1]反轉
indices = np.argsort(importances)[::-1]
# importances總合為1
# 直接取 > 0.15 的重要特微，之後就用x_selected 取代X_train_new
# x_selected = forest.transform(X_train_new, threshold=0.15)

for f in range(X_train.shape[1]):
	print("%2d) %-*s %f" % (f+1, 30,feat_labels[f],importances[indices[f]]))
	
plt.title('Feature Importances')
# 畫長條圖
plt.bar(range(X_train.shape[1]),importances[indices],color='lightblue',align='center')
plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()

plt.show()
