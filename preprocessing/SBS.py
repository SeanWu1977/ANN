from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

class SBS():
    '''
        Sequential Backward Selection
        循序特徵選擇(sequential feature selection) : 用於不支援正規化的演算法
             循序向後選擇(sequential back selection, SBS) : 每輪移除一個影響準確率
                                                            (準則函數, criterion function)最少的特徵，
                                                            不斷重覆此步驟直到設定特徵數。 
    '''
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        #shape 會顯示(row, col), 取col值
        dim = X_train.shape[1] 

        # 建立 0 ~ col 的陣列，並轉為tuple(內容不可改變)
        self.indices_ = tuple(range(dim))

        # 記錄每一輪最佳的feature set [(0,1,...,col),(,,,),...]
        self.subsets_ = [self.indices_]

        # 傳入訓練資料/測試資料/特徵欄位索引 計算準確率(criterion function)
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)


        # 記錄每一輪準確率
        self.scores_ = [score]

        # 每輪計算所有組合的準確率，取最大為該輪最佳feature
        while dim > self.k_features:

            # 存放本輪每個組合的準確率            
            scores = []
            # 存放本輪所有組合
            subsets = []

            # 利用combinations來進行 N 取 m 的組合(1,2,3)  => (1,2),(1,3),(2,3)
            for p in combinations(self.indices_, r=dim-1):
                score= self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)

            # 更新indices_ 給下一輪combinatons使用
            self.indices_ = subsets[best]
            # 儲存本輪最佳feature indices list & 準確率
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best])

            dim -= 1

        # 定義最後準確率
        self.k_score_ = self.scores_[-1]
        return self

    
    
    def transform(self, X):
        returnX[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

def main():
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=2, metric='minkowski')
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    # 取5個特徵，看圖表準確率100%有5~10個特徵，取最小100%的特徵(5):特徵縮放
    print(sbs.subsets_)
    k5 = list(sbs.subsets_[8])
    print(df_wine.columns[1:][k5])
    knn.fit(X_train_std, y_train)
    print('13 train features: {}'.format(knn.score(X_train_std,y_train)))
    knn.fit(X_test_std, y_test)
    print('13 test features: {}'.format(knn.score(X_test_std,y_test)))
    
    knn.fit(X_train_std[:, k5], y_train)
    print('5 train features: {}'.format(knn.score(X_train_std[:, k5],y_train)))
    knn.fit(X_test_std[:,k5], y_test)
    print('5 test features: {}'.format(knn.score(X_test_std[:, k5],y_test)))
    
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.show()


    
if __name__=='__main__':
    '''
        循序特徵選擇演算法
    '''
    main() 
