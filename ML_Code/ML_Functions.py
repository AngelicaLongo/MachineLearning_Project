# Importing essential libraries for data manipulation, analysis, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement
np.random.seed(42)

#Necessary pre-processing steps
class Preprocessing:
    def __init__(self, path,test_size, val_size):
        self.df = pd.read_csv(path) # Load the dataset
        self.test_size = test_size
        self.val_size = val_size
        self.standardized_data = None

    def remove_missing_values(self):
        # Remove missing values from the dataset
        self.missing_values = self.df.isnull().sum()
        self.df.dropna(inplace=True)

    def remove_outliers(self):
        # Remove outliers using the IQR method
        outliers = self.detect_outliers()
        self.df = self.df.loc[~self.df.index.isin(outliers.index)]

    def detect_outliers(self):
        # Detect outliers using the IQR method
        outliers = pd.DataFrame(index = self.df.index)
        for col in self.df.select_dtypes (include=['float64', 'int64']).columns:
                q1, q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                # Calculate the first (25th percentile) and third quartile (75th percentile)
                iqr = q3 - q1  # Calculate the Interquartile Range (IQR)
                lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers[col] = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
        return self.df[outliers.sum(axis=1) > 0]

    def correlations(self, correlation_threshold=0.8):
        # Analyze correlations and remove highly correlated columns
        correlation_matrix = self.df.corr()
        self.plot_heatmap(correlation_matrix)

        correlated_columns = self.high_correlations(correlation_matrix, correlation_threshold)
        self.df.drop(columns=correlated_columns, inplace=True)

    def high_correlations(self, correlation_matrix, threshold):
        # Find highly correlated columns
        high_corr = set()
        for col1 in correlation_matrix.columns:
            for col2 in correlation_matrix.columns:
                if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold:
                    high_corr.add(col2)
        return list(high_corr)

    def plot_heatmap(self, correlation_matrix):
        # Create a correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.savefig('heatmap.png')
        plt.close()

    def standardize_data(self):
        # Standardize data to have average = 0, sd = 1
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        X_standardized = (X - X.mean()) / X.std()
        self.standardized_data = pd.concat([X_standardized, y], axis=1)

    def summarize(self):
        # Provides descriptive statistics of variables and save them
        summary = self.df.describe()
        latex_table = summary.to_latex()
        with open('descriptive_statistics.tex', 'w') as file:
            file.write(latex_table)

    def plot_boxplots(self):
        # Create and save boxplots for all numerical variables
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(self.df.columns[:-1], 1):
            plt.subplot(2, 5, i)
            sns.boxplot(y=self.df[col])
            plt.title(col)
        plt.tight_layout()
        plt.savefig('boxplots.png')
        plt.close()

    def split_data(self):
        if self.standardized_data is None:
            raise ValueError("Data must be standardized before splitting.")

        # Split the standardized dataset into training, testing, and validation sets
        X = self.standardized_data.iloc[:, :-1].values
        y = self.standardized_data.iloc[:, -1].values

        # Mix data and split in training and test set
        indices = np.random.permutation(len(X))
        split_train = int(len(X) * (1 - self.test_size))
        train_indices, test_indices = indices[:split_train], indices[split_train:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        # Further divide the training set in training and validation set
        validation_size = int(len(X_train) * self.val_size)
        val_indices, train_final_indices = np.split(train_indices, [validation_size])

        X_validation, X_train_final = X[val_indices], X[train_final_indices]
        y_validation, y_train_final = y[val_indices], y[train_final_indices]

        self.X_validation, self.y_validation = X_validation, y_validation
        self.X_train, self.y_train = X_train_final, y_train_final



class Perceptron:
    def __init__(self, epoch_candidates=None, random_seed = 42):
        # Initialize the classifier with a list of candidate epochs for cross-validation
        self.epoch_candidates = epoch_candidates
        self.optimal_epochs = None
        self.highest_accuracy = 0
        self.weights = None
        np.random.seed(random_seed)

    def perceptron_train(self, features, labels, max_iterations):
        # Train the perceptron using features and labels for a fixed n of epochs
        num_samples, num_features = features.shape
        self.weights = np.zeros(num_features)  # Initialize weights to zero

        for epoch in range(max_iterations):
            changes_made = False  # Track whether any updates are performed in this epoch
            for idx in range(num_samples):
                if labels[idx] * np.dot(self.weights, features[idx]) <= 0:  # Update weights if prediction is incorrect
                    self.weights += labels[idx] * features[idx]
                    changes_made = True
            if not changes_made:  # Stop if no changes were made in an epoch
                print(f"Training completed in {epoch + 1} epochs.")
                break
        else:
            print(f"Reached the maximum allowed epochs: {max_iterations}")

        return self.weights

    def perceptron_predict(self, features):
        # Make predictions using the trained weights
        return np.sign(np.dot(features, self.weights))
        # Return the sign of the dot product between features and weights

    def perceptron_cross_validation(self, features, labels, k_splits=5):
        # Perform k-fold cross-validation to determine the best n of epochs
        num_samples = len(labels)
        fold_size = num_samples // k_splits

        for candidate_epoch in self.epoch_candidates:
            print(f"Evaluating for {candidate_epoch} epochs...")
            fold_accuracies = []

            for fold_idx in range(k_splits):
                # Split data into validation and training sets
                val_start = fold_idx * fold_size
                val_end = val_start + fold_size

                val_features, val_labels = features[val_start:val_end], labels[val_start:val_end]
                train_features = np.concatenate((features[:val_start], features[val_end:]), axis=0)
                train_labels = np.concatenate((labels[:val_start], labels[val_end:]), axis=0)

                # Train and evaluate
                self.perceptron_train(train_features, train_labels, candidate_epoch)
                predictions = self.perceptron_predict(val_features)
                accuracy = np.mean(predictions == val_labels)
                fold_accuracies.append(accuracy)

            avg_accuracy = np.mean(fold_accuracies)
            print(f"Average accuracy for {candidate_epoch} epochs: {avg_accuracy:.4f}")

            # Update the best epoch if performance improves
            if avg_accuracy > self.highest_accuracy:
                self.highest_accuracy = avg_accuracy
                self.optimal_epochs = candidate_epoch

        print(f"Optimal epochs: {self.optimal_epochs} with accuracy: {self.highest_accuracy:.4f}")
        return self.optimal_epochs

    def perceptron_test_accuracy(self, train_features, train_labels, test_features, test_labels):
        # Train the model with the best number of epochs and evaluate it on the test set
        if self.optimal_epochs is None:
            raise ValueError("Optimal epochs not determined. Run cross-validation first.")

        self.perceptron_train(train_features, train_labels, self.optimal_epochs)
        predictions = self.perceptron_predict(test_features)
        misclassification_rate = np.mean(predictions != test_labels)
        print(f"Test Misclassification Rate: {misclassification_rate:.4f}")
        return 1 - misclassification_rate



class PegasosSVM:
    def __init__(self, num_iterations_list, regularization_params, learning_rate_schedules, num_folds=5):
        self.num_iterations_list = num_iterations_list  # List of T values to test
        self.regularization_params = regularization_params  # List of lambda values to test
        self.learning_rate_schedules = learning_rate_schedules  # List of learning rate functions
        self.num_folds = num_folds  # Number of folds for cross-validation

        # Store optimal values
        self.optimal_accuracy = 0
        self.optimal_learning_rate = None
        self.optimal_lambda = None
        self.optimal_iterations = None
        self.weights = None

    def pegasos_train(self, num_iterations, lambda_param, features, labels, learning_rate_fn):
        # Trains the Pegasos SVM model
        num_samples, num_features = features.shape
        weights = np.zeros(num_features)  # Initialize weights to zero

        for iteration in range(1, num_iterations + 1):
            random_index = np.random.randint(num_samples)
            sample, label = features[random_index], labels[random_index]

            # Update rule based on hinge loss
            if label * np.dot(weights, sample) < 1:
                gradient = lambda_param * weights - label * sample
            else:
                gradient = lambda_param * weights

            # Adjust weights using the learning rate schedule
            weights -= learning_rate_fn(iteration) * gradient

        return weights

    def pegasos_predict(self, features):
        # Predicts labels for the given dataset using the trained model
        if self.weights is None:
            raise ValueError("The model has not been trained yet. Train the model before making predictions.")
        return np.sign(np.dot(features, self.weights))

    def pegasos_cross_validation(self, features, labels):
        # Executes k-fold cross-validation to tune hyperparameters
        num_samples = len(labels)
        fold_size = num_samples // self.num_folds

        for num_iterations in self.num_iterations_list:
            for lambda_param in self.regularization_params:
                for learning_rate_fn in self.learning_rate_schedules:
                    fold_accuracies = []

                    for fold in range(self.num_folds):
                        # Split data into validation and training subsets
                        start, end = fold * fold_size, (fold + 1) * fold_size
                        val_features, val_labels = features[start:end], labels[start:end]
                        train_features = np.concatenate((features[:start], features[end:]), axis=0)
                        train_labels = np.concatenate((labels[:start], labels[end:]), axis=0)

                        # Train the model on training folds and evaluate on the validation fold
                        weights = self.pegasos_train(num_iterations, lambda_param, train_features, train_labels, learning_rate_fn)
                        predictions = np.sign(np.dot(val_features, weights))
                        accuracy = np.mean(predictions == val_labels)
                        fold_accuracies.append(accuracy)

                    avg_accuracy = np.mean(fold_accuracies)

                    # Output performance and update optimal parameters if necessary
                    print(f"Iterations={num_iterations}, Lambda={lambda_param}, LR={learning_rate_fn.__name__}, Accuracy={avg_accuracy:.4f}")
                    if avg_accuracy > self.optimal_accuracy:
                        self.optimal_accuracy = avg_accuracy
                        self.optimal_learning_rate = learning_rate_fn
                        self.optimal_lambda = lambda_param
                        self.optimal_iterations = num_iterations

        if self.optimal_iterations is None:
            raise ValueError(
                "No optimal parameters found. Ensure that the dataset and hyperparameters are correctly defined.")

        print(f"Best Iterations: {self.optimal_iterations}, Best Lambda: {self.optimal_lambda}, "
              f"Best Learning Rate: {self.optimal_learning_rate.__name__}, Best Accuracy: {self.optimal_accuracy:.4f}")

    def pegasos_test_accuracy(self, train_features, train_labels, test_features, test_labels):
        # Evaluates the trained model on a test set using the best hyperparameters
        if self.optimal_iterations is None or self.optimal_lambda is None or self.optimal_learning_rate is None:
            raise ValueError("Hyperparameters are not optimized yet. Perform cross-validation first.")

        # Train the model with optimal parameters
        self.weights = self.pegasos_train(self.optimal_iterations, self.optimal_lambda, train_features, train_labels, self.optimal_learning_rate)

        # Evaluate on the test set
        test_predictions = self.pegasos_predict(test_features)
        error_rate = np.mean(test_predictions != test_labels)
        print(f"Test Set Error Rate: {error_rate:.4f}")
        return 1 - error_rate