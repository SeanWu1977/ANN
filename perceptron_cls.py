import numpy as np
class Perception(object):
    def __init__(self, eta=0.01, n_iter=10, threshold=1.0 ):
        self.eta = eta
        self.n_iter = n_iter
        self.th = threshold
    def fit(self,X,y):
        self.w_ = np.zeros(X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X,self.w_) + self.th

    def perdict(self, X):
        return np.where(self.net_input(X) >=0.0 ,1, -1)
