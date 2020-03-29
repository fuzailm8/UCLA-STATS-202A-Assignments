# -*- coding: utf-8 -*-
"""

 Stat 202A 2019 Fall - Homework 07
 Author: Fuzail Mujahid Khan
 Date : 11/20/19

 INSTRUCTIONS: Please fill in the corresponding function. Do not change function names, 
 function inputs or outputs. Do not write anything outside the function.
 
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, GridSearchCV
from matplotlib import pyplot as plt
import xgboost as xgb

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

#############################################################################################################
# TODO (1) Perform 5-fold validation for cancer data.
# You may consider use KFold function in sklearn.model_selection
# Print the mean and std of the 5-fold validation accuracy
#############################################################################################################

kfold = KFold(n_splits=5, shuffle=True, random_state=123)
accuracy = []

for train_index, test_index in kfold.split(X):
    X_train, Y_train = X[train_index], y[train_index]
    X_test, Y_test = X[test_index], y[test_index]
    
    xgbm = xgb.XGBClassifier(objective="binary:logistic", random_state=123)
    xgbm.fit(X_train, Y_train)
    
    y_pred = xgbm.predict(X_test)
    accu = y_pred == Y_test
    accuracy.append(np.mean(accu))
    
print('The mean accuracy after 5-fold CV is ',np.mean(accuracy))
print('The standard deviation of the accuracy after 5-fold CV is ',np.std(accuracy))

#############################################################################################################
# TODO (2) Perform Grid Search for parameter max_depth in [3,5,7] and min_child_weight in [0.1, 1, 5]
# For each combination use 5-Fold validation
# You may consider use GridSearchCV function in sklearn.modelselection
# Print the grid search mean test score for each paramter combination (use cv_results_)
#############################################################################################################

params = {
    "max_depth": [3,5,7],
    "min_child_weight": [0.1,1,5]
}

xgbm = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

search = GridSearchCV(xgbm,
                        scoring='accuracy',
                        param_grid = params,
                        cv=5,
                        verbose=1
                      )

search.fit(X,y)

print('These are the 9 parameter combinations between max_depth and min_child_weight : \n',search.cv_results_['params'])
print('These are the corresponding mean test scores for each parameter combination after 5 fold CV : \n',search.cv_results_['mean_test_score'])

# The below part is for plotting graphs as attached in the report

params = {
    "max_depth": [3,5,7,9,11],
    "min_child_weight": [0.1,1,5]
}

xgbm = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

search = GridSearchCV(xgbm,
                        scoring='accuracy',
                        param_grid = params,
                        cv=5,
                        verbose=1
                      )

search.fit(X,y)

print(search.cv_results_)
mean_test_scores = search.cv_results_['mean_test_score']
accu_min_child_0_1 = [mean_test_scores[i] for i in [0,3,6,9,12]]
accu_min_child_1 = [mean_test_scores[i] for i in [1,4,7,10,13]]
accu_min_child_5 = [mean_test_scores[i] for i in [2,5,8,11,14]]

test_err_min_child_0_1 = [1-i for i in accu_min_child_0_1]
test_err_min_child_1 = [1-i for i in accu_min_child_1]
test_err_min_child_5 = [1-i for i in accu_min_child_5]

max_depth_range = [3,5,7,9,11]

plt.plot(max_depth_range, test_err_min_child_0_1, '-o')
plt.title('Test error for min_child_weight = 0.1')
plt.show()

plt.plot(max_depth_range, test_err_min_child_1, '-o')
plt.title('Test error for min_child_weight = 1')
plt.show()

plt.plot(max_depth_range, test_err_min_child_5, '-o')
plt.title('Test error for min_child_weight = 5')
plt.show()



params = {
    "n_estimators" : [i for i in range(100,500,5)]
}

xgbm = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

search = GridSearchCV(xgbm,
                        scoring='accuracy',
                        param_grid = params,
                        cv=5,
                        verbose=1
                      )

search.fit(X,y)

accu = search.cv_results_['mean_test_score']
test_error_n_estimator = [1-i for i in accu]
n_estimators_range = [i for i in range(100,500,5)]

plt.plot(n_estimators_range, test_error_n_estimator, '-o')

 
#############################################################################################################
# TODO (3) Plot the feature importance of the best model
# You may fit a new xgboost model using all the data and then plot the importance using xgb.plot_importance()
#############################################################################################################

#Tuning hyper parameters again for given ranges for max_depth and min_child_weight

params = {
    "max_depth": [3,5,7],
    "min_child_weight": [0.1,1,5]
}

xgbm = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

search = GridSearchCV(xgbm,
                        scoring='accuracy',
                        param_grid = params,
                        cv=5,
                        verbose=1
                      )

search.fit(X,y)


best_max_depth = search.best_params_['max_depth']
best_min_child_weight = search.best_params_['min_child_weight']
print('The optimal parameters are max_depth:',best_max_depth,' min_child_weight:',best_min_child_weight)

#Creating model with best parameters
xgbm = xgb.XGBClassifier(objective="binary:logistic", random_state=42, max_depth = best_max_depth, min_child_weight = best_min_child_weight)
xgbm.fit(X, y)

from xgboost import plot_importance

ax = plot_importance(xgbm)
ax.figure.set_size_inches(10,8)
plt.show()
