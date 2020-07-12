import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score


class KNNClassifier:
    def __init__(self, k):
        assert k >= 1, "K must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert self.k <= X_train.shape[0], "K must be valid"
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to y_train"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not  None and self._y_train is not None, \
            "must fit before predict !"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train !"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        assert self._X_train.shape[1] == x.shape[0], "The feature number of x must be equal to X_train"
        distance = []
        # 计算点x与训练集中其它点的距离
        for x_train in self._X_train:
            d = sqrt(np.sum((x_train - x) ** 2))
            distance.append(d)
        # 将计算的距离按照下标排序
        nearest = np.argsort(distance)
        # 选出前k个距离最近点的预测值
        top_k = [self._y_train[i] for i in nearest[: self.k]]
        votes = Counter(top_k)  # 返回一个tuple,tuple中是一个列表
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据X_test和y_test值计算模型准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k


