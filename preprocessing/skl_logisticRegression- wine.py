from sklearn import datasets
import numpy as np
import pandas as pd
import plot_decision_regions as pdr
import matplotlib.pyplot as plt


'''
利用正規化將個各權重縮小，甚至到0 (稀疏)，
如此達到降維

透過LogisticRegression 的 penalty 設為 L1 方法,
另外調整正規化參數(C為正規化參數之倒數)來調整權重稀疏程度。

LogisticRegression
會算出每一類的權重值，
因此每一筆資料會有對每一類的預測機率
取最大權重為期預測結果

'''



# 將資料分為訓練集與測試集函數
from sklearn.model_selection import train_test_split

# 將資料正規化函數
from sklearn.preprocessing import StandardScaler

# 感知器函數
from sklearn.linear_model import LogisticRegression
# 用以下函式效果同上，差別在有提供partial_fit來處理大量資料，去除記憶體不足問題。線上學習概念。
# from sklearn.linear_model import SGDClassifier
# ppn = SGDClassifier(loss='log')




# 效能指標模組
# accuracy_score 算正確率
from sklearn.metrics import accuracy_score

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

# 建立正規化物件
sc = StandardScaler()

# 利用現有資料計算正規化所需的平均值與標準差
# 不執行此指令則之後的動作都無法操作
sc.fit(X_train) 

# 查平均值與標準差，回傳陣列[平均值，標準差]
# sc.mean_

# 利用sc.fit算出來的平均值與標準差來正規化訓練集與測試集
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 模型訓練
lr = LogisticRegression(penalty='l1',C=0.1, random_state=0)  # 正規化L1/L2, 預設 penalty: ‘l2’
lr.fit(X_train_std, y_train)

# 預測
print("訓練集準確率 : {:.2%}".format(lr.score(X_train_std, y_train)))
print("測試集準確率 : {:.2%}".format(lr.score(X_test_std, y_test)))

print(lr.intercept_)

print(lr.coef_)



                                             
##y_pred = lr.predict(X_test_std)
##print("預測準確率 : {:.2%}".format(accuracy_score(y_test, y_pred)))


# 正規化參數對權重稀疏的影響
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']
weights, params = [], []


for c in np.arange(-4, 6,dtype=float):
    lr = LogisticRegression(C=10**c, penalty='l1',random_state=0)
    lr.fit(X_train_std, y_train)
    ### 選擇看對那一個分類(y)的特徵權重
    ### lr.coef_ 列出所有分類的權重
    weights.append(lr.coef_[0])
    params.append(10**c)
    y_pred = lr.predict(X_test_std)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]),colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1],color=color)
plt.axhline(0,color="black", linestyle='--',linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper right')
ax.legend(loc='upper center', bbox_to_anchor=(1.38,1.03), ncol=1, fancybox=True)
plt.show()


##
### 預測第n個樣本的機率
pd=lr.predict_proba(X_test_std)

print("預測第n個樣本分類機率")
i = 0
for s in pd:
    print("class 1:{:.2%}, class 2: {:.2%}, class 3: {:.2%}, 實際: {}".format(s[0], s[1], s[2],y_test[i]))
    i += 1

### 預測第n個樣本的分類
print(lr.predict(X_test_std[0].reshape(1,13)))
