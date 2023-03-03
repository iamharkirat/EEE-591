import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_filename = "heart1.csv"
target_variable = "a1p2"

print_top_5_feature_correlations = "Top 5 feature correlations:\n"
print_top_5_target_correlations = "Top 5 target correlations:\n"
print_top_5_feature_covariances = "Top 5 feature covariances:\n"
print_top_5_target_covariances = "Top 5 target covariances:\n"

count = 5

def calculate_correlation(dataframe):
    corr_matrix = dataframe.corr().abs()
    corr_matrix *= np.tri(*corr_matrix.values.shape, k=-1).T
    corr_unstacked = corr_matrix.unstack()
    corr_unstacked.sort_values(inplace=True, ascending=False)
    print(print_top_5_feature_correlations, corr_unstacked.head(count).to_string())
    print("")
    target_correlations = corr_unstacked.get(key=target_variable)
    print(print_top_5_target_correlations, target_correlations.head(count).to_string())
    print("")
    return corr_matrix.iloc[:13,:13]

def calculate_covariance(dataframe):
    cov_matrix = dataframe.cov().abs()
    cov_matrix *= np.tri(*cov_matrix.values.shape, k=-1).T
    cov_unstacked = cov_matrix.unstack()
    cov_unstacked.sort_values(inplace=True, ascending=False)
    print(print_top_5_feature_covariances, cov_unstacked.head(count).to_string())
    print("")
    target_covariances = cov_unstacked.get(key=target_variable)
    print(print_top_5_target_covariances, target_covariances.head(count).to_string())
    print("")
    return cov_matrix.iloc[:13,:13]

def create_pairplot(dataframe):
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(dataframe, height=1.5, plot_kws={"s": 1})
    plt.show()

def main():
    heart_data = pd.read_csv(dataset_filename)
    correlation = calculate_correlation(heart_data)
    print("Correlation Matrix:\n")
    print(correlation.to_string())
    print("")
    covariance = calculate_covariance(heart_data)
    print("Covariance Matrix:\n")
    print(covariance.to_string())
    print("")
    create_pairplot(heart_data)

main()
