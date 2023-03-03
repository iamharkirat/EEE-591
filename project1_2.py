import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def perceptron_model(X, y):
    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

    model = Perceptron(tol=0.001, max_iter=1000, alpha=0.0001, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def logistic_regression(X, y):
    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    model = LogisticRegression(solver='liblinear', penalty='l2', class_weight=None, C=0.01)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy 

def svm(X, y):
    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    svm_model = SVC(kernel='sigmoid', gamma='scale', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy 

def decision_tree(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    tree_model = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=2, max_depth=9, random_state=42)
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf_model = RandomForestClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=2, n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def knn(X, y):
    X_scaled, _ = scale_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    knn_model = KNeighborsClassifier(weights='uniform', n_neighbors=19, metric='euclidean')
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    # Load the heart dataset
    heart_df = pd.read_csv('heart1.csv')
    
    # Split the data into features (X) and target (y)
    X = heart_df.drop('a1p2', axis=1)
    y = heart_df['a1p2']
    
    # Initialize an empty dataframe to store the results
    results_df = pd.DataFrame(columns=['Model', 'Accuracy'])
    
    # Perceptron Model
    perceptron_acc = perceptron_model(X, y)
    results_df = results_df.append({'Model': 'Perceptron', 'Accuracy': perceptron_acc}, ignore_index=True)
    
    # Logistic Regression
    logistic_acc = logistic_regression(X, y)
    results_df = results_df.append({'Model': 'Logistic Regression', 'Accuracy': logistic_acc}, ignore_index=True)
    
    # SVM
    svm_acc = svm(X, y)
    results_df = results_df.append({'Model': 'SVM', 'Accuracy': svm_acc}, ignore_index=True)
    
    # Decision Tree
    decision_tree_acc = decision_tree(X, y)
    results_df = results_df.append({'Model': 'Decision Tree', 'Accuracy': decision_tree_acc}, ignore_index=True)
    
    # Random Forest
    random_forest_acc = random_forest(X, y)
    results_df = results_df.append({'Model': 'Random Forest', 'Accuracy': random_forest_acc}, ignore_index=True)
    
    # KNN
    knn_acc = knn(X, y)
    results_df = results_df.append({'Model': 'KNN', 'Accuracy': knn_acc}, ignore_index=True)
    
    # Print the results dataframe
    print("\nRESULTS ===========================================================")
    print(results_df)

main()