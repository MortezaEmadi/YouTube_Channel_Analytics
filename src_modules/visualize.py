# src/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(data, features, output_path=None):
    correlation_matrix = data[features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_distribution(data, column, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_count(data, column, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(data[column])
    plt.title(f'Count Plot of {column}')
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_box(data, column, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data[column])
    plt.title(f'Box Plot of {column}')
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_scatter(data, x_column, y_column, output_path=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[x_column], y=data[y_column])
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')
    if output_path:
        plt.savefig(output_path)
    plt.show()
