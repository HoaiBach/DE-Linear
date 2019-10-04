import JADE
import Problem
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold
from sklearn import svm, preprocessing
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
import base

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    max_iterations = 100
    to_print = 'Maximum number of iterations: %d \n' % max_iterations
    pop_size = 200
    to_print += 'Population size: %d \n' % pop_size
    run = int(sys.argv[2])

    seed = 1617*run
    np.random.seed(seed)

    #load data
    mat = scipy.io.loadmat('/home/nguyenhoai2/Grid/data/FSMathlab/'+dataset+'.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]

    # testing
    # X = -1 + np.random.rand(10, 5)*2
    # W = np.array([-2, 1, 3, 1, -4])
    # y = np.ravel(np.dot(X, np.reshape(W, (5, 1))))
    # y[y > 0] = 1
    # y[y < 0] = 0

    # ensure that y label start from 0, not 1
    num_class, count = np.unique(y, return_counts=True)
    n_classes = np.unique(y).shape[0]
    min_class = np.min(count)
    if np.max(y) >= len(num_class):
        y = y-1
    y = np.int8(y)
    # convert to -1 and 1
    # assert(n_classes == 2)
    # y[np.where(y == 0)] = -1

    # ensure that the division is the same for all algorithms, in all runs
    n_splits = min(min_class, 5)
    skf = StratifiedKFold(n_splits=n_splits, random_state=1617)

    fold_count = 1
    svm_accs_full = []
    knn_accs_full = []
    svm_accs_sel = []
    knn_accs_sel = []
    nf = []

    for train_index, test_index in skf.split(X, y):
        to_print += '=========Fold ' + str(fold_count) + '=========\n'
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        no_train_ins, no_test_ins = X_train.shape[0], X_test.shape[0]

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

        # normalize so all values in the range 0 and 1
        # it is very important to ensure that the feture values are greater than 0
        # for local search
        X_min = np.min(X_train, axis=0)
        X_max = np.max(X_train, axis=0)
        non_dup = np.where(X_min != X_max)[0]
        X_min = X_min[non_dup]
        X_max = X_max[non_dup]
        X_train = X_train[:, non_dup]
        X_test = X_test[:, non_dup]
        X_train = (X_train-X_min)/(X_max-X_min)
        X_test = (X_test-X_min)/(X_max-X_min)
        no_features = X_train.shape[1]

        prob = Problem.FeatureSelection(X_train, y_train)
        de = JADE.JADE(problem=prob, popsize=pop_size, dims=no_features+1, maxiters=max_iterations,
                       minpos=-1.0, maxpos=1.0, c=0.1, p=0.05)

        best_sol, best_fit, evo_process = de.evolve()
        to_print += evo_process

        sol_print = 'Solution: '
        for entry in best_sol:
            sol_print += str(entry)+', '
        sol_print += '\n'
        to_print += sol_print
        output = np.dot(best_sol, prob.X_train_coef.T)

        f_weight = best_sol[0:no_features-1]
        f_weight_mean = np.mean(np.abs(f_weight))
        f_selected = np.where(np.abs(f_weight) > f_weight_mean)[0]
        X_train_sel = X_train[:, f_selected]
        X_test_sel = X_test[:, f_selected]

        to_print += 'Selected features: '
        for f_idx in f_selected:
            to_print += str(f_idx)+', '
        to_print += '\n'

        knn = KNN(metric='euclidean')
        knn.fit(X_train, y_train)
        knn_full_acc = balanced_accuracy_score(y_test, knn.predict(X_test))
        knn.fit(X_train_sel, y_train)
        knn_sel_acc = balanced_accuracy_score(y_test, knn.predict(X_test_sel))
        to_print += 'Full KNN: %f \n' % knn_full_acc
        to_print += 'Sel KNN: %f \n' % knn_sel_acc

        clf = svm.LinearSVC(random_state=seed, C=1.0)
        clf.fit(X_train, y_train)
        svm_full_acc = balanced_accuracy_score(y_test, clf.predict(X_test))
        clf.fit(X_train_sel, y_train)
        svm_sel_acc = balanced_accuracy_score(y_test, clf.predict(X_test_sel))
        to_print += 'Full SVM: %f \n' % svm_full_acc
        to_print += 'Sel SVM: %f \n' % svm_sel_acc

        knn_accs_full.append(knn_full_acc)
        knn_accs_sel.append(knn_sel_acc)
        svm_accs_full.append(svm_full_acc)
        svm_accs_sel.append(svm_sel_acc)
        nf.append(len(f_selected)+0.0)

    to_print += '----------------Final----------------\n'
    to_print += 'Full KNN: %f \n' % np.mean(knn_accs_full)
    to_print += 'Select KNN: %f \n' % np.mean(knn_accs_sel)
    to_print += 'Full SVM: %f \n' % np.mean(svm_accs_full)
    to_print += 'Select SVM: %f \n' % np.mean(svm_accs_sel)
    to_print += 'Ave nf: %f \n' % np.mean(nf)
    f = open(str(run)+'.txt', 'w')
    f.write(to_print)
    f.close()
