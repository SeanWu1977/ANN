import numpy as np
class Adaline(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self,X,y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            # 計算每一筆記錄的加權值和
            output = self.net_input(X)

            # 計算每一筆輸出與預期輸出差
            errors = (y - output)

            # 每一特徵值的所有值 X*deltaW 相加*eta 
            self.w_[1:] += self.eta * X.T.dot(errors)

            # 調整門檻值
            self.w_[0] += self.eta * errors.sum()

            #計算成本
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0 ,1, -1)

