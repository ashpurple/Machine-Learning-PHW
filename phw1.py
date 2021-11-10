import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings(action='ignore')

## Preprocessing
def preprocessing(df):
    # check outlier
    df.replace('?', np.NAN, inplace=True)
    df.iloc[:, 6] = pd.to_numeric(df.iloc[:, 6])

    # check NAN
    df_drop_NAN = df.dropna(axis=0)

    X = df_drop_NAN.iloc[:, 1:-2].copy() # feature
    y = df_drop_NAN.iloc[:, -1].copy() # target

    y.replace(2, 0, inplace=True)
    y.replace(4, 1, inplace=True)

    return X, y

# function for set hyper parameters and run find_best
def setCombination(X, y):

    # 1. Scaler List : Standard, MinMax, Robust
    standard = StandardScaler()
    minMax = MinMaxScaler()
    robust = RobustScaler()
    scalers = {"standard scaler": standard, "minMax scaler": minMax, "robust scaler": robust}

    # 2. Model List: Decision tree(entropy), Decision tree(Gini), Logistic regression, SVM
    decisionTree_entropy = tree.DecisionTreeClassifier(criterion="entropy")
    decisionTree_gini = tree.DecisionTreeClassifier(criterion="gini")
    logistic = LogisticRegression()
    svm_model = svm.SVC()
    models = {"decisionTree_entropy": decisionTree_entropy, 
            "decisionTree_gini": decisionTree_gini,
            "logistic": logistic, 
            "svm": svm_model}

    # 3. Parameters Setting
    params_dict = {"decisionTree_entropy": {"max_depth": [x for x in range(3, 9, 1)],
                                            "min_samples_split": [x for x in range(2, 10, 1)],
                                            "min_samples_leaf": [x for x in range(3, 10, 1)]},
                   "decisionTree_gini": {"max_depth": [x for x in range(3, 9, 1)],
                                         "min_samples_split": [x for x in range(2, 10, 1)],
                                         "min_samples_leaf": [x for x in range(3, 10, 1)]},
                   "logistic": {"C": [0.001, 0.01, 0.1, 1, 10],
                   'penalty': ['l1', 'l2', 'elasticnet', 'none']},
                   "svm": {"C": [ 0.1, 1, 10], 
                        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                        "gamma": [0.01, 0.1, 1]}
                   }

    # K-fold's n_splits List: 25%, 20%, 10%
    k_fold_list = [4, 5, 10]
    
    # K-fold's cross validation List: 5, 10
    cv_list = [5, 10]

    return X, y, scalers, models, k_fold_list, params_dict, cv_list

# function for store combination that has the best accuracy 
def getBestCombination(X, y, scalers, models, k_folds_list, params_dict):
    best_accuracy = {}

    # find the best parameter by using grid search
    for scaler_key, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        print("\n----------------------------------")
        print(f'<    scaler: {scaler_key}    >')
        for model_key, model in models.items():
            print(f'\n<   model: {model_key}   >')
            for k_fold_num in k_folds_list:
                print(f'\n<  k-fold: {k_fold_num}  >')
                for train_idx, test_idx in KFold(n_splits=k_fold_num).split(X):

                    # train test split
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # grid search
                    grid = GridSearchCV(model, param_grid=params_dict[model_key])
                    grid.fit(X_train, y_train)
                    print(f'parameters: {grid.best_params_}')
                    best_model = grid.best_estimator_
                    predict = best_model.predict(X_test)
                    accuracy = accuracy_score(y_test, predict)

                    # save the 10 highest accuracy and parameters each models
                    list_size = 10
                    list_size -= 1
                    flag = False

                    target_dict = {'accuracy': accuracy, 
                        'scaler': scaler_key,
                        'model': model_key, 
                        'k_fold_num': k_fold_num,
                        'param': grid.best_params_}

                    # save accuracy
                    if model_key not in best_accuracy.keys():
                        best_accuracy[model_key] = []
                    if len(best_accuracy[model_key]) <= list_size:
                        best_accuracy[model_key].append(target_dict)

                    # insert accuracy
                    elif best_accuracy[model_key][-1]['accuracy'] < accuracy:
                        for i in range(1, list_size):
                            if best_accuracy[model_key][list_size - 1 - i]['accuracy'] > accuracy:
                                best_accuracy[model_key].insert(list_size - i, target_dict)
                                best_accuracy[model_key].pop()
                                flag = True
                                break
                        if flag is False:
                            best_accuracy[model_key].insert(0, target_dict)
                            best_accuracy[model_key].pop()

                    print(f'accuracy: {accuracy}', end='')

    return best_accuracy

def crossValidation(X, y, scalers, models, result_dict, cv_list):
    best_score = {}
    for i in models.keys():
        best_score[i] = {'best_score': 0}

    for model_name, result_list in result_dict.items():
        for result in result_list:
            for cv in cv_list:
                scaler_name = result['scaler']
                X_scaled = scalers[scaler_name].fit_transform(X)
                models[model_name].set_params(**result['param'])
                scores = cross_val_score(models[model_name], X_scaled, y, cv=cv)
                result['cv'] = cv
                score_mean = scores.mean()

                if best_score[model_name]['best_score'] < score_mean:
                    best_score[model_name]['best_score'] = score_mean
                    best_score[model_name]['params'] = result
    print("-----------------------")
    print(f'<    best score    >')

    for m in models.keys():
        print(m)
        print(best_score[m])

# function for display result_dict
def displayBestCombination(result_dict):
    print("-----------------------")
    print("<Best Combination>")
    for model_name, result_list in result_dict.items():
        print(model_name)
        for result in result_list:
            print(result)
        print()

if __name__ == "__main__":
    # read data
    df = pd.read_csv('breast-cancer-wisconsin.data', header=None) 

    # preprocessing
    X, y = preprocessing(df)

    # set scalers, models, params, k values
    X, y, scalers, models, k_fold_list, params_dict, cv_list = setCombination(X, y)

    # get best combination dictionary
    result_dict = getBestCombination(X, y, scalers, models, k_fold_list, params_dict)
    displayBestCombination(result_dict)

    # cross validation
    crossValidation(X, y, scalers, models, result_dict, cv_list)
