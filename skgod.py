import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF, AdaBoostClassifier as AB, \
        GradientBoostingClassifier as GB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from pprint import pprint

class OneHotEncoder:
    def __init__(self, df, col_name, max_cats=5):
        self.col_name = col_name
        if not self.col_name in df.columns:
            return df
        # record dominant categories
        all_cats = df[col_name].value_counts().index.values
        n = len(all_cats)
        if n <= max_cats:
            self.cats = all_cats[:n]
        else:
            self.cats = all_cats[:max_cats]

        self.new_col_names = []
        for cat in self.cats:
            self.new_col_names.append(col_name + "=" + str(cat))

    def __call__(self, df):
        # execute encoding for each category in self.cats
        #ipdb.set_trace()

        if not self.col_name in df.columns:
            return df

        for i,cat in enumerate(self.cats):
            new_col = self.new_col_names[i]

            #pandas gets upset when a column is all nan
            if df[self.col_name].isnull().all():
                df[new_col] = 0
            else:
                df[new_col] = (df[self.col_name] == cat).astype(int)

        # drop old column
        df = df.drop(self.col_name,axis=1)
        return df


def load_n_clean():
    df = pd.read_csv('data/churn.csv')
    cols = df.columns.values
    cols = [col.lower().replace(' ','_').replace("'","").replace("?","") for col in cols]
    df.columns = cols
    df.churn = np.where(df.churn=="True.", True, False)
    df.intl_plan = np.where(df.intl_plan=='yes', True, False)
    df.vmail_plan = np.where(df.vmail_plan=='yes', True, False)
    df.drop(['state','area_code','phone'], inplace=True, axis=1)

    y = df.pop('churn').values
    X = df.values.astype(np.float)

    return X,y

def LR_classif():
    # LogisticRegression
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001,
    #     C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None,
    #     solver=’liblinear’, max_iter=100, multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1)
    hypers = {
        'penalty': 'l2',  # type of regularization
        'C': .1 ,           # inverted lambda
        'class_weight': 'balanced'
    }
    return LR(**hypers)

def DT_classif():
    # Decision Tree
    # http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    # sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None,
    #     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
    #     random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
    #     class_weight=None, presort=False)
    hypers = {
        'max_depth': 5,
        'class_weight': 'balanced',
    }
    return DT(**hypers)

def RF_classif():
    # RandomForestClassifier
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None,
    #     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’,
    #     max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
    #     oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    hypers = {
        'n_estimators': 100,
        'max_depth': 5 ,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    return RF(**hypers)

def AB_classif():
    # AdaBoostClassifier
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    # sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50,
    #     learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)
    hypers = {
        'base_estimator': DT_classif(),
        'n_estimators': 100,
        'learning_rate' : 0.1
    }
    return AB(**hypers)

def GB_classif():
    # GradientBoostingClassifier
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
    # sklearn.ensemble.GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1,
    #     n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2,
    #     min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
    #     min_impurity_split=None, init=None, random_state=None, max_features=None,
    #     verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’)
    hypers = {
        'n_estimators': 400,
        'learning_rate' : 0.05,
        # 'subsample': 0.4
        'max_depth' : 4
    }
    return GB(**hypers)

def SVC_classif():
    # Support Vector Machine classifier
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True,
    #     probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    #     max_iter=-1, decision_function_shape=’ovr’, random_state=None)
    hypers = {
        'C':1,
        'class_weight':'balanced',
        'gamma' : 'auto'
    }
    return SVC(**hypers)


def cross_val(X,y):
    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    # models = [LR_classif, DT_classif, RF_classif, AB_classif, GB_classif, SVC_classif]
    # models = [RF_classif, GB_classif, SVC_classif]
    # models = [SVC_classif]
    models = [GB_classif]
    for model in models:
        print(model().__class__)
        classifier = make_pipeline(StandardScaler(), model())
        y_pred = cross_val_predict(classifier, X, y, cv=5)
        pprint( classification_report(y, y_pred) )


if __name__ == '__main__':
    X,y = load_n_clean()
    cross_val(X,y)
