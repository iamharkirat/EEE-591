import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from warnings import filterwarnings
from sklearn.metrics import confusion_matrix
import seaborn as sns

filterwarnings('ignore')

# Load the data from a csv file
df = pd.read_csv('sonar_all_data_2.csv', header=None)

# Get the number of columns in the dataframe
n = len(df.columns)

# Select the features and labels
X = df.iloc[:,0:n-1].values
y = df.iloc[:,n-1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

y_pred_list = []     # initialize list to store predictions
acc = []             # initialize list to store accuracies

for i in range(1, n):
    pca = PCA(n_components=i)
    print('Number of Components:', i)

    # Apply PCA to the training and test sets
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # Define the MLPClassifier model
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', batch_size=20, max_iter=500,
                          solver='lbfgs', learning_rate='adaptive', alpha=0.00001, random_state=0)

    # Fit the model to the training data
    model.fit(X_train_pca, y_train)

    # Predict the classes of the test set
    y_pred = model.predict(X_test_pca)

    # Append predictions to the list
    y_pred_list.append(y_pred)

    # Compute and print the accuracy score
    acc_score = accuracy_score(y_test, y_pred)
    print('Accuracy: %.2f' % acc_score)

    # Append accuracy to the list
    acc.append(acc_score)

    print("\==================================================================================\n")

# Find the index of the maximum accuracy
max_acc_index = np.argmax(acc)

# Plot the accuracy scores
array = np.arange(1, n, 1)
plt.plot(array, acc)
plt.xlabel("Number of PCA components")
plt.ylabel("Accuracy")
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_list[max_acc_index], labels=['R', 'M'])
confusion_matrix = pd.DataFrame(conf_matrix, index=['true:R', 'true:M'], columns=['pred:R', 'pred:M'])
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
