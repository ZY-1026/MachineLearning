import numpy as np
from metrics import r2_score


class LinearRegression:
    """
    线性回归模型
    """

    def __init__(self):
        self.coef_ = None  # 系数
        self.interception_ = None  # 截距
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """
        利用公式法求解
        :param X_train: 训练集
        :param y_train: 训练集
        :return: self
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """
        批量梯度下降
        :param X_train: 训练集
        :param y_train: 训练集
        :param eta: 学习率
        :param n_iters: 迭代次数
        :return:self
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 计算损失函数
        def J(theta, X_b, y,):
            try:
                return np.sum(y - X_b.dot(theta) ** 2) / len(X_b)
            except:
                return float("inf")

        # 计算梯度
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            return (X_b.T.dot(X_b.dot(theta) - y) * 2) / len(X_b)

        def dJ_debug(theta, X_b, y, epsilon=0.01):
            result = np.empty(len(theta))

            for i in range(len(theta)):
                theta_1 = theta.copy()
                theta_1[i] += epsilon
                theta_2 = theta.copy()
                theta_2[i] += epsilon
                result[i] = (J(theta_1, X_b, y) - J(theta_2, X_b, y)) / (2 * epsilon)
                return result

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            i_iter = 0
            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - gradient * eta
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break
                i_iter += 1
            return theta
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """
        利用随机梯度下降法训练数据
        学习率计算公式 eta = t0 / (t1 + t)，t代表迭代次数，学习率随着迭代次数增加而减小
        :param X_train: 训练集
        :param y_train: 训练集
        :param n_iters: 迭代次数
        :param t0: 计算学习率的参数t0，
        :param t1: 计算学习率的参数t1
        :return: self
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1, "n_iters must be validate"

        # 任取一个样本进行计算
        def dJ_sgd(theta, X_b_i, y_i):
            return (X_b_i.T.dot(X_b_i.dot(theta) - y_i)) * 2

        def sgd(X_b, y, initial_theta, n_iters):

            def learning_theta(t):
                """
                计算学习率
                :param t: 迭代次数
                :return: t0 / (t + t1)
                """
                return t0 / (t + t1)

            theta = initial_theta
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(len(X_b))
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(len(X_b)):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_theta(cur_iter * len(X_b) + i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def predict(self, X_predict):
        assert self.coef_ is not None and self.interception_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        return r2_score(y_test, self.predict(X_test))

    def __repr__(self):
        return "LinearRegression()"

