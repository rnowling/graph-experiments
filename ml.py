import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def create_features_1(er_sp_distr, pp_sp_distr):
    n_features = max(max(map(len, er_sp_distr)),
                     max(map(len, pp_sp_distr)))
    n_samples = len(er_sp_distr) + len(pp_sp_distr)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    r = 0
    for row in er_sp_distr:
        X[r, :] = row
        r += 1

    for row in pp_sp_distr:
        X[r, :] = row
        y[r] = 1.0
        r += 1

    return X, y

def create_features_2(er_sp_distr, pp_sp_distr):
    n_features = max(max(map(len, er_sp_distr)),
                     max(map(len, pp_sp_distr)))
    n_samples = len(er_sp_distr) * len(pp_sp_distr) + \
                len(er_sp_distr) * len(er_sp_distr) + \
                len(pp_sp_distr) * len(pp_sp_distr) + \
                len(pp_sp_distr) * len(er_sp_distr)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    r = 0
    for er_sample in er_sp_distr:
        for pp_sample in pp_sp_distr:
            diff = er_sample - pp_sample
            X[r, :] = diff
            y[r] = 0.0
            r += 1

    for er_sample_1 in er_sp_distr:
        for er_sample_2 in er_sp_distr:
            diff = er_sample_1 - er_sample_2
            X[r, :] = diff
            y[r] = 1.0
            r += 1

    for pp_sample_1 in pp_sp_distr:
        for pp_sample_2 in pp_sp_distr:
            diff = pp_sample_1 - pp_sample_2
            X[r, :] = diff
            y[r] = 1.0
            r += 1

    for pp_sample in pp_sp_distr:
        for er_sample in er_sp_distr:
            diff = pp_sample - er_sample
            X[r, :] = diff
            y[r] = 0.0
            r += 1

    return X, y

def create_features_3(er_sp_distr, pp_sp_distr):
    n_features = max(max(map(len, er_sp_distr)),
                     max(map(len, pp_sp_distr)))
    n_samples = len(er_sp_distr) * len(pp_sp_distr) + \
                len(er_sp_distr) * len(er_sp_distr) + \
                len(pp_sp_distr) * len(pp_sp_distr) + \
                len(pp_sp_distr) * len(er_sp_distr)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    r = 0
    for er_sample in er_sp_distr:
        for pp_sample in pp_sp_distr:
            diff = er_sample - pp_sample
            X[r, :] = np.abs(diff)
            y[r] = 0.0
            r += 1

    for er_sample_1 in er_sp_distr:
        for er_sample_2 in er_sp_distr:
            diff = er_sample_1 - er_sample_2
            X[r, :] = np.abs(diff)
            y[r] = 1.0
            r += 1

    for pp_sample_1 in pp_sp_distr:
        for pp_sample_2 in pp_sp_distr:
            diff = pp_sample_1 - pp_sample_2
            X[r, :] = np.abs(diff)
            y[r] = 1.0
            r += 1

    for pp_sample in pp_sp_distr:
        for er_sample in er_sp_distr:
            diff = pp_sample - er_sample
            X[r, :] = np.abs(diff)
            y[r] = 0.0
            r += 1

    return X, y

def create_features_4(er_sp_distr, pp_sp_distr):
    n_features = 1
    n_samples = len(er_sp_distr) * len(pp_sp_distr) + \
                len(er_sp_distr) * len(er_sp_distr) + \
                len(pp_sp_distr) * len(pp_sp_distr) + \
                len(pp_sp_distr) * len(er_sp_distr)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    r = 0
    for er_sample in er_sp_distr:
        for pp_sample in pp_sp_distr:
            diff = er_sample - pp_sample
            X[r, 0] = np.dot(diff, diff)
            y[r] = 0.0
            r += 1

    for er_sample_1 in er_sp_distr:
        for er_sample_2 in er_sp_distr:
            diff = er_sample_1 - er_sample_2
            X[r, 0] = np.dot(diff, diff)
            y[r] = 1.0
            r += 1

    for pp_sample_1 in pp_sp_distr:
        for pp_sample_2 in pp_sp_distr:
            diff = pp_sample_1 - pp_sample_2
            X[r, 0] = np.dot(diff, diff)
            y[r] = 1.0
            r += 1

    for pp_sample in pp_sp_distr:
        for er_sample in er_sp_distr:
            diff = pp_sample - er_sample
            X[r, 0] = np.dot(diff, diff)
            y[r] = 0.0
            r += 1

    return X, y

def evaluate_classifier(clf, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    roc_aucs = []
    accuracies = []
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_X = X[train_idx, :]
        train_y = y[train_idx]
        test_X = X[test_idx, :]
        test_y = y[test_idx]
        
        clf.fit(train_X, train_y)
        pred_labels = clf.predict(test_X)
        pred_probs = clf.predict_proba(test_X)

        roc_auc = roc_auc_score(test_y, pred_probs[:, 1])
        acc = accuracy_score(test_y, pred_labels)
        fpr, tpr, thresholds = roc_curve(test_y, pred_probs[:, 1])
        roc_aucs.append(roc_auc)
        accuracies.append(acc)

    print "Average ROC AUC", np.mean(roc_aucs)
    print "Std ROC AUC", np.std(roc_aucs)
    print
    print "Average accuracy", np.mean(accuracies)
    print "Std accuracy", np.std(accuracies)
    print

    return fpr, tpr
        
        
