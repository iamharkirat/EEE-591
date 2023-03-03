import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def perceptron_model(X, y):
    # scale the features
    X_scaled, scaler = scale_features(X)

    # split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

    # create a Perceptron model
    model = Perceptron(random_state=42)

    # define the hyperparameters to tune
    params = {'alpha': np.logspace(-5, -1, num=100),
              'max_iter': [1000, 2000, 3000],
              'tol': np.logspace(-5, -1, num=100)}

    # perform randomized search to find the best hyperparameters
    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='accuracy', cv=5, n_iter=50)
    random_search.fit(X_train, y_train)

    # get the best hyperparameters
    best_params = random_search.best_params_

    # train the model with the best hyperparameters
    model = Perceptron(random_state=42, **best_params)
    model.fit(X_train, y_train)

    # make predictions on the testing data
    y_pred = model.predict(X_test)

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:', accuracy)

def logistic_regression(X, y):
    # scale the features
    X_scaled, scaler = scale_features(X)

    # split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # create a logistic regression model
    model = LogisticRegression()

    # define the hyperparameter grid to search over
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'class_weight': [None, 'balanced']
    }

    # create a GridSearchCV object to find the best hyperparameters
    grid_search = RandomizedSearchCV(model, param_grid, cv=4, scoring='accuracy', n_jobs=-1)

    # fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # evaluate the best model on the testing data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print('accuracy:', accuracy)   

def svm(X,y):
    # scale the features
    X_scaled, scaler = scale_features(X)

    # split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # define the hyperparameters to test
    param_dist = {'C': np.logspace(-3, 3, 7),
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': ['scale', 'auto']}

    # create a SVM model
    svm_model = SVC(random_state=42)

    # perform randomized search to find the best hyperparameters
    random_search = RandomizedSearchCV(svm_model, param_distributions=param_dist, n_iter=50, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # make predictions on the testing data using the best model
    best_svm_model = random_search.best_estimator_
    y_pred = best_svm_model.predict(X_test)

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:', accuracy)

def decision_tree(X, y):
    # scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # define the decision tree model to test
    tree_model = DecisionTreeClassifier(random_state=42)

    # define the hyperparameters to test
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }

    # perform grid search to find the best hyperparameters
    grid_search = RandomizedSearchCV(tree_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # get the best hyperparameters
    best_params = grid_search.best_params_

    # train the model with the best hyperparameters
    tree_model = DecisionTreeClassifier(random_state=42, **best_params)
    tree_model.fit(X_train, y_train)

    # make predictions on the testing data
    y_pred = tree_model.predict(X_test)

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:', accuracy)

def random_forest(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # define the Random Forest model to test
    rf_model = RandomForestClassifier(random_state=42)

    # define the hyperparameters to test
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }

    # perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(rf_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # test the model with the best hyperparameters on the testing set
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:', accuracy)

def knn(X, y):
    # scale the features
    X_scaled, scaler = scale_features(X)

    # split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # define the KNN model to test
    knn_model = KNeighborsClassifier()

    # define the hyperparameters to test
    param_grid = {'n_neighbors': np.arange(1,50), 
                  'weights': ['uniform', 'distance'], 
                  'metric': ['euclidean', 'manhattan']}

    # perform grid search to find the best hyperparameters
    grid_search = RandomizedSearchCV(knn_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # print the best parameters and accuracy score
    print("accuracy: ", grid_search.best_score_)


def main():
    heart_df = pd.read_csv('heart1.csv')
    
    # Split the data into features (X) and target (y)
    X = heart_df.drop('a1p2', axis=1)
    y = heart_df['a1p2']
    
    print("PERCEPTRON MODEL ==================================================")
    perceptron_model(X,y)
    print("\nLOGISTIC REGRESSION ===============================================")
    logistic_regression(X,y)
    print("\nSVM ===============================================================")
    svm(X,y)
    print("\nDECISION TREE =====================================================")
    decision_tree(X,y)
    print("\nRANDOM FOREST =====================================================")
    random_forest(X,y)
    print("\nKNN ===============================================================")
    knn(X,y)


main()