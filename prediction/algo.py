# encoding: UTF-8

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


def get_target_distribution(target):
    return pd.DataFrame(target).groupby('outcome').size()


def get_regressor(regressor, params={}):
    if regressor == "random_forest":
        args = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 0,
            "oob_score": True,
            "n_jobs": 4
        }
        args.update(params)
        return RandomForestRegressor(**args)
    if regressor == "gradient_boosting":
        args = {
            "n_estimators": 100,
            "learning_rate": 0.2,
            "max_depth": 1,
            "random_state": 0,
            "loss": 'ls',
            "subsample": 0.5
        }
        args.update(params)
        return GradientBoostingRegressor(**args)


def get_predictions(regressor, features, target):
    return cross_val_predict(regressor, features, target, n_jobs=-1, cv=50)


def get_metrics(target, predictions):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    precision_score = metrics.average_precision_score(target, predictions)
    return (
        false_positive_rate,
        true_positive_rate,
        thresholds,
        precision_score,
        roc_auc
    )


def get_confusion_matrix(target, predictions, threshold, normalize=True):
    cm = metrics.confusion_matrix(target, predictions >= threshold)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return pd.DataFrame(cm)


def get_feature_importance(regressor, features, target):
    regressor.fit(features, target)
    return pd.DataFrame(zip(
        features.columns,
        regressor.feature_importances_)).sort_values(1, ascending=False)


def plot_roc_curve(target, predicted):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(target, predicted)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
                                label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
