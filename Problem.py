import numpy as np
import math
import base
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone


class Problem:
    def __init__(self, minimized):
        self.minimized = minimized

    def fitness(self, sol):
        return 10*sol.shape[0]+np.sum(sol**2-10*np.cos(2*math.pi*sol))
        # return np.sum(sol**2)

    def worst_fitness(self):
        w_f = float('inf') if self.minimized else float('-inf')
        return w_f

    def is_better(self, first, second):
        if self.minimized:
            return first < second
        else:
            return first > second


class F1(Problem):

    def __init__(self):
        Problem.__init__(self, minimized=True)

    def fitness(self, sol):
        return np.sum(sol**2)


class F2(Problem):

    def __init__(self):
        Problem.__init__(self, minimized=True)

    def fitness(self, sol):
        abs_sol = np.abs(sol)
        return np.sum(abs_sol) + np.prod(abs_sol)


class F3(Problem):

    def __init__(self):
        Problem.__init__(self, minimized=True)

    def fitness(self, sol):
        square = sol**2
        sum_up_to = [np.sum(square[:i+1]) for i in range(len(square))]
        return np.sum(sum_up_to)


class FeatureSelection(Problem):

    def __init__(self, X_train, y_train):
        Problem.__init__(self, minimized=True)
        self.y_train = np.copy(y_train)
        self.no_instances = X_train.shape[0]
        self.X_train_coef = np.append(X_train, np.ones((self.no_instances, 1)), axis=1)

    def fitness(self, sol):
        output = np.dot(sol, self.X_train_coef.T)
        diff = output-self.y_train
        return np.sum(diff**2)+0.1*np.sum(np.abs(sol))
