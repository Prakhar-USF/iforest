import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import random

# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.height_limit = int(np.ceil(np.log2(sample_size)))
        self.trees = []
        self.n_trees = n_trees

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        for i in range(self.n_trees):
            idx = np.random.randint(X.shape[0], size=self.sample_size)
            X_sample = X[idx, :]
            self.trees.append(IsolationTree(self.height_limit, 0).fit(X_sample, improved=improved))
        #             print("done fit")
        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        path_lengths = np.zeros([len(X), 1])

        if isinstance(X, pd.DataFrame):
            X = X.values

        for i, x in enumerate(X):
            temp = 0
            for T in self.trees:
                temp += compute_path_length(x, T, e=0)
            path_lengths[i] = temp / self.n_trees

        return path_lengths

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        n = X.shape[0]
        euc = 0.5772156649
        if n == 2:
            c = 1
        elif n < 2:
            c = 0
        else:
            c = (2 * (np.log(n - 1) + euc)) - (2 * (n - 1) / n)
        return 2.0 ** (-1.0 * self.path_length(X) / c)

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        scores_thr = scores > threshold
        return scores_thr.astype(int)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:

        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


def compute_path_length(x,T,e):
#     print ("height_limit", T.height_limit)
#     print ("height_current", T.height_current)
#     print ("split_att", T.split_att)
#     print ("split_val", T.split_val)
#     print ("right", T.right)
#     print ("left", T.left)
#     print ("size", T.size)
#     print ("exnodes", T.exnodes)
#     print ("n_nodes", T.n_nodes)
#     import ipdb
#     ipdb.set_trace()
    if T.exnodes == 1:
        n = T.size
        euc = 0.5772156649
        if n == 2:
            return e+1
        elif n < 2:
            return e
        else:
            return e+(2*(np.log(n-1)+euc))-(2*(n-1)/n)
    else:
        a = T.split_att
        if x[a] < T.split_val:
            return compute_path_length(x,T.left, e+1)
        else:
            return compute_path_length(x,T.right, e+1)


class IsolationTree:
    def __init__(self, height_limit, height_current):
        self.height_limit = height_limit
        self.height_current = height_current
        self.split_att = None
        self.split_val = None
        self.right = None
        self.left = None
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if X.shape[0] <= 1 or self.height_current >= self.height_limit:
            self.exnodes = 1
            self.size = X.shape[0]
            #             print ("done_exnode")
            return self

        Q = X.shape[1]
        q = np.random.randint(0, Q - 1)
        attrib = X[:, q]

        if attrib.min() == attrib.max():
            self.exnodes = 1
            self.size = X.shape[0]
            #             print ("done_equal")
            return self

        if improved:
            # p_min = attrib.min()
            # p_max = attrib.max()
            # p = np.random.uniform(p_min, p_max)
            # x_l = X[attrib < p, :]
            # ratio = x_l.shape[0] / X.shape[0]
            #
            # while not (ratio > 0.8 or ratio < 0.2 or X.shape[0] < 10):
            #     #                 print("tried", p, p_min, p_max, ratio, x_l.shape[0], X.shape[0])
            #     p = np.random.uniform(p_min, p_max)
            #     x_l = X[attrib < p, :]
            #     ratio = x_l.shape[0] / X.shape[0]
            # #                 print("tried", p, p_min, p_max, ratio, x_l.shape[0], X.shape[0])
            # #             print ("done_equal")
            # x_r = X[attrib >= p, :]

            p = random.betavariate(0.5,0.5)*(attrib.max() - attrib.min()) + attrib.min()
            x_l = X[attrib < p, :]
            x_r = X[attrib >= p, :]

            self.size = X.shape[0]
            self.split_att = q
            self.split_val = p
            #             print ("done_node_add")

            self.left = IsolationTree(self.height_limit, self.height_current + 1).fit(x_l, improved=improved)
            self.right = IsolationTree(self.height_limit, self.height_current + 1).fit(x_r, improved=improved)
            self.n_nodes = self.left.n_nodes + self.right.n_nodes

        else:

            p = np.random.uniform(attrib.min(), attrib.max())
            x_l = X[attrib < p, :]
            x_r = X[attrib >= p, :]

            #             print(x_l,x_r)
            #             print("p,q",p,q)
            self.size = X.shape[0]
            self.split_att = q
            self.split_val = p
            #             print ("done")

            self.left = IsolationTree(self.height_limit, self.height_current + 1).fit(x_l, improved=improved)
            self.right = IsolationTree(self.height_limit, self.height_current + 1).fit(x_r, improved=improved)
            self.n_nodes = self.left.n_nodes + self.right.n_nodes

        return self


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """

    threshold = 1

    while threshold > 0:
        scores_thr = scores > threshold
        y_pred = scores_thr.astype(int)
        TN, FP, FN, TP = confusion_matrix(y, y_pred).flat
        TPR = TP / (TP + FN)
        if TPR >= desired_TPR:
            return threshold, FP/(FP+TN)

        threshold = threshold - 0.001

    return threshold, FPR