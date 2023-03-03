import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
DATASET_FILENAME = "heart1.csv"
TARGET_VARIABLE = "a1p2"
TOP_COUNT = 5

# Strings
PRINT_TOP_FEATURE_CORRELATIONS = "Top feature correlations:\n"
PRINT_TOP_TARGET_CORRELATIONS = "Top target correlations:\n"
PRINT_TOP_FEATURE_COVARIANCES = "Top feature covariances:\n"
PRINT_TOP_TARGET_COVARIANCES = "Top target covariances:\n"

def calculate_correlation(dataframe):
    corr_matrix = dataframe.corr().abs()
    corr_matrix *= np.tri(*corr_matrix.values.shape, k=-1).T
    corr_unstacked = corr_matrix.unstack()
    corr_unstacked.sort_values(inplace=True, ascending=False)
    
    # Use f-strings instead of string concatenation
    print(f"{PRINT_TOP_FEATURE_CORRELATIONS}{corr_unstacked.head(TOP_COUNT).to_string()}\n")
    
    target_correlations = corr_unstacked.get(key=TARGET_VARIABLE)
    print(f"{PRINT_TOP_TARGET_CORRELATIONS}{target_correlations.head(TOP_COUNT).to_string()}\n")
    return corr_matrix.iloc[:13, :13]

def calculate_covariance(dataframe):
    cov_matrix = dataframe.cov().abs()
    cov_matrix *= np.tri(*cov_matrix.values.shape, k=-1).T
    cov_unstacked = cov_matrix.unstack()
    cov_unstacked.sort_values(inplace=True, ascending=False)
    
    # Use f-strings instead of string concatenation
    print(f"{PRINT_TOP_FEATURE_COVARIANCES}{cov_unstacked.head(TOP_COUNT).to_string()}\n")
    
    target_covariances = cov_unstacked.get(key=TARGET_VARIABLE)
    print(f"{PRINT_TOP_TARGET_COVARIANCES}{target_covariances.head(TOP_COUNT).to_string()}\n")
    return cov_matrix.iloc[:13, :13]

def create_pairplot(dataframe):
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(dataframe, height=1.5, plot_kws={"s": 1})
    plt.show()

def main():
    heart_data = pd.read_csv(DATASET_FILENAME)
    
    # Combine the two functions into one
    correlation = calculate_correlation(heart_data)
    print("Correlation Matrix:\n")
    print(correlation.to_string())
    print("")
    
    covariance = calculate_covariance(heart_data)
    print("Covariance Matrix:\n")
    print(covariance.to_string())
    print("")
    
    create_pairplot(heart_data)

if __name__ == "__main__":
    main()
