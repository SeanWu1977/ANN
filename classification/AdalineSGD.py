import numpy as np
class AdalineSGD(object):
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)
            
    def fit(self,X,y):
        ''' 訓練資料學習用
        '''
        self._initialized_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                # 隨機重新排列輸入陣列
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self,X,y):
        ''' online learning
            給新進資料學習用
        '''
        if not self.w_initialized:
            self._initialized_weights(X.shape[1])
            # 新資料有預期輸出值是否大於1筆
        if y.reval().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, target)            
        return self
            
    def _shuffle(self, X, y):
        # 產生數字為0~len(y)的隨機順序
        r=np.random.permutation(len(y))
        # 以r為array index 回傳以此index為順序的序列
        return X[r], y[r]

    def _initialized_weights(self, m):
        self.w_ = np.zeros(1+m)
        self.w_initialized = False
            

    def _update_weights(self, xi, target):
        # 計算每一筆記錄的加權值和
        output = self.net_input(xi)
        # 計算每一筆輸出與預期輸出差
        error = (target - output)
        # 每一特徵值的 X*deltaW 相加 * eta 
        self.w_[1:] += self.eta * xi.dot(error)
        # 調整門檻值
        self.w_[0] += self.eta * error
        # 計算成本, 分析用
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0 ,1, -1)

