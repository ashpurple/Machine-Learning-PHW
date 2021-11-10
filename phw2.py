import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import LabelEncoder


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import estimate_bandwidth
from pyclustering.cluster.clarans import clarans;

from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

import warnings

warnings.filterwarnings(action='ignore')


# Preprocessing
def findMissingValue(df):
    # check missing value
    # only 'totla_bedrooms' has missing values, fill median
    df.total_bedrooms.fillna(df.total_bedrooms.median(), inplace=True)

    return df


# function for set hyper parameters and run find_best
def setCombination():
    # Scaler List
    standard = StandardScaler()
    minMax = MinMaxScaler()
    robust = RobustScaler()
    maxAbs = MaxAbsScaler()
    scalers = {"standard scaler": standard, "minMax scaler": minMax, "robust scaler": robust, "maxAbs scaler": maxAbs}

    # Encoder List
    label = LabelEncoder()
    encoders = {"label encoder": label}

    # Model List
    kmeans = KMeans()
    gmm = GaussianMixture()

    dbscan = DBSCAN()
    meanshift = MeanShift()

    models = {"kmeans": kmeans,
              #"clarans": [],
              "gmm": gmm,
              "dbscan": dbscan,
              #"meanshift": meanshift

    }

    # Parameters
    params_dict = {"kmeans": {"n_clusters": [x for x in range(3, 5)]},
                   "gmm": {"n_components": [x for x in range(3, 5)]},
                   "dbscan": {"eps": [0.1, 0.5]},
                   "meanshift": {"bandwidth": [1, 2]}
                   }

    return scalers, encoders, models, params_dict


def knee_method(X):
    nearest_neighbors = NearestNeighbors(n_neighbors=11)
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    distances = np.sort(distances[:, 10], axis=0)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.savefig("Distance_curve.png", dpi=300)
    plt.title("Distance curve")
    plt.show()
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    fig = plt.figure(figsize=(5, 5))
    knee.plot_knee()
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.show()
    print(distances[knee.knee])


def purity_scorer(target, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(target, y_pred)
    score = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    score = silhouette_score(X, labels, metric='euclidean')
    return score


def display_silhouette_plot(X, labels):
    sil_score = metrics.silhouette_score(X, labels, metric='euclidean')
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("For n_clusters =", n_clusters_, "The average silhouette score is :", sil_score)
    # compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, labels)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters_ + 1) * 10])
    y_lower = 10
    for i in range(n_clusters_):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters_)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=sil_score, color="red", linestyle='--')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for clustering on sample data with n_clusters = %d" % n_clusters_),
                 fontsize=14, fontweight='bold')
    plt.show()
    return sil_score


def featureCombination(df, index):
    if index == 0:
        feature = df[['population', 'ocean_proximity']]
    elif index == 1:
        feature = df[['population', 'households', 'median_income']]
    elif index == 1:
        feature = df[['housing_median_age', 'total_rooms', 'total_bedrooms']]
    elif index == 2:
        feature = df[['population', 'households']]
    elif index == 3:
        feature = df[['population', 'median_income']]
    elif index == 4:
        feature = df[['households', 'median_income']]
    elif index == 5:
        feature = df[['total_rooms', 'total_bedrooms']]
    elif index == 6:
        feature = df[['median_income', 'ocean_proximity']]
    elif index == 7:
        feature = df[['median_income', 'total_rooms']]
    return feature



# function for store combination that has the best accuracy
def findBestCombination(df, scalers, encoders, models, params_dict):
    best_combination = {}
    best_score = 0
    # Sample Data
    for index in range(8):
        X = featureCombination(df, index)
        feature = X.columns.tolist()
        print(f'\n[feature: {feature}]')

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.to_list()     # numerical value
        cat_cols = X.select_dtypes(include=['object']).columns.to_list()               # categorical value
        # find the best parameter by using grid search
        for scaler_key, scaler in scalers.items():
            scaled_X = scaler.fit_transform(X[num_cols])
            print(f'\n[scaler: {scaler_key}]')
            #Label encoding if there is categorical feature
            if 'ocean_proximity' in X.columns :
                label=LabelEncoder()            
                encoded_X=label.fit_transform(X[cat_cols])               
                encoded_X=encoded_X.reshape(-1,1)              
                # add encoded_x to scaled_x
                scaled_X=np.concatenate((scaled_X, encoded_X), axis=1)
            print(f'[encoder: label encoder]\n')
            #print(scaled_X)
            knee_method(scaled_X)
           
            for model_key, model in models.items():
                print(f'\n[model: {model_key}]')

                cv = [(slice(None), slice(None))]
                if (model_key == 'clarans'):
                    clarans_obj = clarans(scaled_X, 6, 3, 5)
                    clarans_obj.process()
                    clst = clarans_obj.get_clusters()
                    label = []
                    for i in range(len(clst)):
                        for j in clst[i]:
                            label.insert(j, i)
                    score = silhouette_score(scaled_X, label, metric='euclidean')
                    if (best_score < score):
                        best_score = score
                        best_X = scaled_X
                        best_label = label

                    target_dict = {'silhouette': score,
                                        'scaler': scaler_key,
                                        'model': model_key,
                                        'param': {'n_cluster':6,'local minima':3},
                                        'feature': feature
                                        }

                else:  # Grid Search
                    if (model_key == 'meanshift'):
                        grid = GridSearchCV(estimator=model,
                                            param_grid=estimate_bandwidth(scaled_X),
                                            scoring=silhouette_scorer,
                                            cv=cv)
                    else:
                        grid = GridSearchCV(estimator=model,
                                            param_grid=params_dict[model_key],
                                            scoring=silhouette_scorer,
                                            cv=cv)
                        grid.fit(scaled_X)
                    print(f'best_parameters: {grid.best_params_}')
                    score = grid.best_score_
                    if (best_score < score):
                        best_score = score
                        best_X = scaled_X
                        best_label = grid.best_estimator_
                        

                        target_dict = {'silhouette': score,
                                        'scaler': scaler_key,
                                        'model': model_key,
                                        'param': grid.best_params_, 
                                        'feature': feature
                                        }
                
                list_size = 10
                list_size -= 1
                flag = False

                # save accuracy
                if model_key not in best_combination.keys():
                    best_combination[model_key] = []
                if len(best_combination[model_key]) <= list_size:
                    best_combination[model_key].append(target_dict)

                # insert accuracy
                elif best_combination[model_key][-1]['silhouette'] < score:
                    for i in range(1, list_size):
                        if best_combination[model_key][list_size - 1 - i]['silhouette'] > score:
                            best_combination[model_key].insert(list_size - i, target_dict)
                            best_combination[model_key].pop()
                            flag = True
                            break
                    if flag is False:
                        best_combination[model_key].insert(0, target_dict)
                        best_combination[model_key].pop()

                    print(f'silhouette score: {score}', end='')
                print()

    return best_combination, best_X, best_label

# read data
df = pd.read_csv("housing.csv")

# preprocessing
df = findMissingValue(df)

# set scalers, models, params, k values
scalers, encoders, models, params_dict = setCombination()

# get best combination dictionary
best_result, best_X, best_label = findBestCombination(df, scalers, encoders, models, params_dict)
print("\n\n-----------result-----------")
print("[Best Results]")
best_score = 0
for model_name, result_list in best_result.items():
    print(model_name)
    for result in result_list:
        print(result)
        if (best_score < result['silhouette']):
            best_score = result['silhouette']
            best_combi = result
    print()

print("[Best Combination]")
print(best_combi)
display_silhouette_plot(best_X, best_label.fit_predict(best_X))
