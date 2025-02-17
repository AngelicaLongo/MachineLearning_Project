# Importing essential libraries for data manipulation, analysis, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement
np.random.seed(42)

#Necessary pre-processing steps
class PreProcessing:
    def __init__(self, path):
        """Dataset initialization."""
        self.df = pd.read_csv(path)

    def remove_missing_values(self):
        """Check and remove missing values from the dataset."""
        count_missing = self.df.isnull().sum().sum()
        if count_missing == 0:
            print("✅ There are no missing values in the dataset.")
        else:
            self.df.dropna(inplace=True)
            print(f"⚠️ Removed {count_missing} missing values from the dataset.")

    def plot_distributions(self):
        """Visualization of variables distribution."""
        print("Visualize distributions of variables:")
        plt.figure(figsize=(20, 15))
        for i, column in enumerate(self.df.columns, 1):
            plt.subplot(4, 3, i)
            sns.histplot(self.df[column], kde=True, color="cornflowerblue")
            plt.title(f'Distribution of {column}')
            plt.grid(True, linestyle='--', color='gray', alpha=0.5)
        plt.tight_layout()
        plt.savefig('Distributions.png')
        plt.show()

    def split_train_val_test(self, test_size=0.2, val_size=0.15, random_state=42):
        """Divide dataset into training, validation, and test set."""
        np.random.seed(random_state)
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        indices = np.random.permutation(len(X))  # Shuffle degli indici
        test_split = int(len(X) * (1 - test_size))
        val_split = int(test_split * (1 - val_size))

        train_indices = indices[:val_split]
        val_indices = indices[val_split:test_split]
        test_indices = indices[test_split:]

        self.X_train, self.y_train = X.iloc[train_indices], y.iloc[train_indices]
        self.X_val, self.y_val = X.iloc[val_indices], y.iloc[val_indices]
        self.X_test, self.y_test = X.iloc[test_indices], y.iloc[test_indices]

        print("\nDataset split into Train, Validation and Test sets.\n")

    def plot_boxplot(self):
        """Visualize boxplots of variables."""
        print("\nBoxplots of variables:")
        plt.figure(figsize=(18, 8))
        plt.boxplot(self.X_train, vert=False, labels=self.X_train.columns)
        plt.title("Boxplot delle variabili X1 - X10 (Training set)")
        plt.savefig('Boxplots_before_outliers.png')
        plt.show()

    def remove_outliers(self):
        """Remove outliers from training set using IQR method."""
        Q1 = self.X_train.quantile(0.25)
        Q3 = self.X_train.quantile(0.75)
        IQR = Q3 - Q1

        mask = ~((self.X_train < (Q1 - 1.5 * IQR)) | (self.X_train > (Q3 + 1.5 * IQR))).any(axis=1)
        outliers_removed = mask.shape[0] - mask.sum()

        self.X_train, self.y_train = self.X_train[mask], self.y_train[mask]

        print(f"\nRemoved outliers: {outliers_removed}\n")

    def check_correlation(self, threshold=0.8):
        """Analyze correlations and remove high correlated variables."""
        print("\nCheck correlations:")
        sns.pairplot(self.X_train)
        plt.suptitle('Scatterplot', y=1.02)
        plt.savefig('Scatterplot.png')
        plt.show()

        correlation_matrix = self.X_train.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.savefig('Heatmap_before.png')
        plt.show()

        # Identify high correlated variables
        high_corr_pairs = [(col1, col2) for col1 in correlation_matrix.columns
                           for col2 in correlation_matrix.columns
                           if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold]

        self.vars_to_remove = set()
        for col1, col2 in high_corr_pairs:
            if col1 not in self.vars_to_remove:
                self.vars_to_remove.add(col2)

        print(f"\nHigh correlated variables, then removed: {self.vars_to_remove}\n")

        self.X_train = self.X_train.drop(columns=list(self.vars_to_remove))
        self.X_test = self.X_test.drop(columns=list(self.vars_to_remove))

        # New heatmap after correlated variables removal
        correlation_matrix_after = self.X_train.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix_after, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix after removal')
        plt.savefig('Heatmap_after.png')
        plt.show()

    def show_statistics(self, message):
        """Show descriptive statistics of the training set."""
        print(f"\n📊 {message}")
        display(self.X_train.describe())

    def standardize(self):
        """Standardize data only on the training set."""
        self.show_statistics("Statistics of the training set:")

        self.train_mean = self.X_train.mean()
        self.train_std = self.X_train.std()

        self.X_train = (self.X_train - self.train_mean) / self.train_std
        self.X_test = (self.X_test - self.train_mean) / self.train_std

        self.show_statistics("Statistics of the standardized training set:")

    def check_class_balance(self):
        """Verify labels balance in the training set."""
        unique, counts = np.unique(self.y_train, return_counts=True)
        print("\nDistribution of labels (Training Set):")
        for u, c in zip(unique, counts):
            print(f"Classe {u}: {c} ({c / len(self.y_train) * 100:.2f}%)")

    def run(self):
        """Run all steps of preprocessing"""
        print("\n▶️ Preprocessing running...\n")
        self.plot_distributions()
        self.split_train_val_test()
        self.plot_boxplot()
        self.remove_outliers()
        self.check_correlation()
        self.standardize()
        self.check_class_balance()
        print("\n✅ Preprocessing completed!\n")


class Perceptron:
    def __init__(self, epoch_candidates=None, random_seed = 42):
        # Initialize the classifier with a list of candidate epochs for cross-validation
        self.epoch_candidates = epoch_candidates if epoch_candidates is None else [100, 1000, 2000, 5000]
        self.optimal_epochs = None
        self.highest_accuracy = 0
        self.weights = None
        np.random.seed(random_seed)

    def perceptron_train(self, X_train, y_train, max_epochs):
        # Train the perceptron model using the given training set.
        num_samples, num_features = X_train.shape
        self.weights = np.zeros(num_features)  # Initialize weights to zero

        for epoch in range(max_epochs):
            changes_made = False  # Track whether any updates are performed in this epoch
            for idx in range(num_samples):
                if y_train[idx] * np.dot(self.weights, X_train[idx]) <= 0:  # Update weights if prediction is incorrect
                    self.weights += y_train[idx] * X_train[idx]
                    changes_made = True
            if not changes_made:  # Stop if no changes were made in an epoch
                print(f"Training converged in {epoch + 1} epochs.")
                break
        else:
            print(f"Reached the maximum allowed epochs: {max_epochs}")

        return self.weights

    def perceptron_predict(self, X):
        # Make predictions using the trained weights
        return np.sign(np.dot(X, self.weights))
        # Return the sign of the dot product between features and weights

    def perceptron_cross_validation(self, X_train, y_train, k_splits=5):
        # Perform k-fold cross-validation to determine the best n of epochs
        num_samples = len(y_train)
        fold_size = num_samples // k_splits

        for candidate_epoch in self.epoch_candidates:
            print(f"Evaluating for {candidate_epoch} epochs...")
            fold_accuracies = []

            for fold_idx in range(k_splits):
                # Split data into validation and training sets
                val_start = fold_idx * fold_size
                val_end = val_start + fold_size

                X_val, y_val = X_train[val_start:val_end], y_train[val_start:val_end]
                X_train_fold = np.concatenate((X_train[:val_start], X_train[val_end:]), axis=0)
                y_train_fold = np.concatenate((y_train[:val_start], y_train[val_end:]), axis=0)

                # Train and evaluate
                self.perceptron_train(X_train_fold, y_train_fold, candidate_epoch)
                predictions = self.perceptron_predict(X_val)
                accuracy = np.mean(predictions == y_val)
                fold_accuracies.append(accuracy)

            avg_accuracy = np.mean(fold_accuracies)
            print(f"Average accuracy for {candidate_epoch} epochs: {avg_accuracy:.4f}")

            # Update the best epoch if performance improves
            if avg_accuracy > self.highest_accuracy:
                self.highest_accuracy = avg_accuracy
                self.optimal_epochs = candidate_epoch

        print(f"\nOptimal epochs: {self.optimal_epochs} with accuracy: {self.highest_accuracy:.4f}")
        return self.optimal_epochs

    def perceptron_evaluate(self, X_train, y_train, X_test, y_test):
        # Train the model with the best number of epochs and evaluate it on the test set
        if self.optimal_epochs is None:
            raise ValueError("Optimal epochs not determined. Run cross-validation first.")

        self.perceptron_train(X_train, y_train, self.optimal_epochs)
        predictions = self.perceptron_predict(X_test)
        misclassification_rate = np.mean(predictions != y_test)
        print(f"Test Misclassification Rate: {misclassification_rate:.4f}")
        return 1 - misclassification_rate #Return accuracy



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



class Logistic:
    def __init__(self, T_options, lambda_options, learning_rates, num_folds=5):
        self.T_options = T_options
        self.lambda_options = lambda_options
        self.learning_rates = learning_rates
        self.num_folds = num_folds
        self.optimal_params = {'T': None, 'lambda': None, 'eta': None, 'accuracy': 0}
        self.weights = None

    def logistic_activation(self, z):
        return 1 / (1 + np.exp(-z))

    def logistic_train(self, X, y, reg_strength, iterations, lr_schedule):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for step in range(1, iterations + 1):
            index = np.random.randint(num_samples)
            sample_x, sample_y = X[index], y[index]

            prediction = self.logistic_activation(np.dot(sample_x, self.weights))
            target = (sample_y + 1) / 2

            gradient = (prediction - target) * sample_x + reg_strength * self.weights
            self.weights -= lr_schedule(step) * gradient

        return self.weights

    def logistic_classify(self, X):
        probabilities = self.logistic_activation(np.dot(X, self.weights))
        return np.where(probabilities >= 0.5, 1, -1)

    def logistic_evaluate_hyperparameters(self, X, y):
        num_samples = len(y)
        fold_size = num_samples // self.num_folds

        for T in self.T_options:
            for reg in self.lambda_options:
                for lr in self.learning_rates:
                    fold_scores = []

                    for fold in range(self.num_folds):
                        start, end = fold * fold_size, (fold + 1) * fold_size
                        X_valid, y_valid = X[start:end], y[start:end]
                        X_train = np.concatenate((X[:start], X[end:]), axis=0)
                        y_train = np.concatenate((y[:start], y[end:]), axis=0)

                        self.weights = self.logistic_train(X_train, y_train, reg, T, lr)
                        preds = self.logistic_classify(X_valid)
                        fold_scores.append(np.mean(preds == y_valid))

                    avg_score = np.mean(fold_scores)
                    print(f"T={T}, lambda={reg}, eta={lr.__name__}: mean accuracy={avg_score:.4f}")

                    if avg_score > self.optimal_params['accuracy']:
                        self.optimal_params.update({'T': T, 'lambda': reg, 'eta': lr, 'accuracy': avg_score})

        best = self.optimal_params
        print(
            f"Best: T={best['T']}, lambda={best['lambda']}, eta={best['eta'].__name__}, accuracy={best['accuracy']:.4f}")
        return best['T'], best['lambda'], best['eta']

    def logistic_test_performance(self, X_train, y_train, X_test, y_test):
        best_T, best_lambda, best_eta = self.optimal_params['T'], self.optimal_params['lambda'], self.optimal_params['eta']
        self.weights = self.logistic_train(X_train, y_train, best_lambda, best_T, best_eta)
        predictions = self.logistic_classify(X_test)
        error_rate = np.mean(y_test != predictions)
        print(f"Test misclassification rate: {error_rate:.4f}")
        return 1 - error_rate



class feature_expansion:
    def __init__(self):
        pass

    def expand_polynomial_features(self, X, max_degree=2):
        num_samples, num_features = X.shape

        if max_degree != 2:
            raise NotImplementedError("Currently, only second-degree expansion is supported.")

        expanded_features = [X]

        # Generate squared terms
        squared_terms = [X[:, i:i + 1] ** 2 for i in range(num_features)]
        expanded_features.extend(squared_terms)

        # Generate interaction terms
        interaction_terms = [X[:, i:i + 1] * X[:, j:j + 1] for i in range(num_features) for j in
                             range(i + 1, num_features)]
        expanded_features.extend(interaction_terms)

        return np.hstack(expanded_features)


class WeightVisualizer:
    def __init__(self):
        pass

    def generate_latex_table(self, perceptron_weights, pegasos_weights, logistic_weights, dataframe):
        feature_names = dataframe.columns[:-1].tolist()
        weight_data = {
            'Feature': feature_names,
            'Perceptron': perceptron_weights,
            'Pegasos': pegasos_weights,
            'Logistic': logistic_weights
        }
        weight_df = pd.DataFrame(weight_data).set_index('Feature')
        latex_output = weight_df.to_latex(index=True)

        with open('weights_original.tex', 'w') as file:
            file.write(latex_output)

    def plot_weight_comparison(self, perceptron_weights, pegasos_weights, logistic_weights):
        plt.figure(figsize=(12, 8))
        plt.plot(self.features, perceptron_weights, label='Perceptron', color='b', marker='o')
        plt.plot(self.features, pegasos_weights, label='Pegasos', color='r', marker='s')
        plt.plot(self.features, logistic_weights, label='Logistic', color='g', marker='^')
        plt.xlabel('Features')
        plt.ylabel('Weights')
        plt.title('Comparison of Feature Weights Across Models')
        plt.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('weights.png')

    def generate_polynomial_latex_table(self, poly_perceptron, poly_pegasos, poly_logistic):
        base_variables = [f"x{i + 1}" for i in range(len(self.features))]
        polynomial_terms = [f"{var}^2" for var in base_variables]
        interaction_terms = [f"{base_variables[i]}*{base_variables[j]}" for i, j in
                             combinations_with_replacement(range(len(base_variables)), 2) if i != j]

        feature_names = base_variables + polynomial_terms + interaction_terms
        poly_weight_df = pd.DataFrame({
            'Feature': feature_names,
            'Perceptron': poly_perceptron,
            'Pegasos': poly_pegasos,
            'Logistic': poly_logistic
        })

        chunked_dataframes = [poly_weight_df[i:i + 22].reset_index(drop=True) for i in
                              range(0, len(poly_weight_df), 22)]
        final_df = pd.concat(chunked_dataframes, axis=1)
        latex_output = final_df.to_latex(index=True)

        with open('weights_poly.tex', 'w') as file:
            file.write(latex_output)


class KernelizedPerceptron:
    def __init__(self):
        self.support_vectors = []  # Support vectors
        self.labels_sv = []  # Labels of support vectors

    def kp_gaussian_kernel(self, vector1, vector2, sigma):
        # Compute the Radial Basis Function (RBF) kernel between two vectors
        return np.exp(-np.linalg.norm(vector1 - vector2) ** 2 / (2 * sigma ** 2))

    def kp_polynomial_kernel(self, vector1, vector2, degree=3, constant=1):
        # Compute the polynomial kernel between two vectors
        return (np.dot(vector1, vector2) + constant) ** degree

    def kp_train(self, X_train, y_train, kernel_func, max_epochs=1):
        # Train the kernelized perceptron using the provided kernel function
        is_converged = False
        for epoch in range(max_epochs):
            incorrect_count = 0
            for idx in range(len(X_train)):
                current_vector = X_train[idx]
                current_label = y_train[idx]
                kernel_sum = sum(self.labels_sv[i] * kernel_func(self.support_vectors[i], current_vector)
                                 for i in range(len(self.support_vectors)))
                predicted_label = np.sign(kernel_sum)

                if predicted_label != current_label:
                    self.support_vectors.append(current_vector)
                    self.labels_sv.append(current_label)
                    incorrect_count += 1

            if incorrect_count == 0:
                print(f"Converged after {epoch + 1} epochs.")
                is_converged = True
                break

        if not is_converged:
            print("Failed to converge within the specified epochs.")
        return self.support_vectors, self.labels_sv

    def kp_predict(self, X_test, kernel_func):
        # Predict labels for the test set using the trained model
        predictions = []
        for vector in X_test:
            kernel_sum = sum(self.labels_sv[i] * kernel_func(self.support_vectors[i], vector)
                             for i in range(len(self.support_vectors)))
            predictions.append(np.sign(kernel_sum))
        return np.array(predictions)

    def kp_optimize_hyperparameters(self, X_train_ext, y_train_ext, X_val, y_val, kernel_type, param_grid, epoch_options):
        # Optimize hyperparameters by evaluating combinations of parameters
        best_accuracy = -np.inf
        best_params = {}

        if kernel_type == 'rbf':
            for sigma in param_grid['sigmas']:
                for epochs in epoch_options:
                    print(f"Testing sigma={sigma} and epochs={epochs}")
                    self.support_vectors, self.labels_sv = self.kp_train(X_train_ext, y_train_ext,
                                                                      lambda v1, v2: self.kp_gaussian_kernel(v1, v2,
                                                                                                             sigma),
                                                                      epochs)
                    predictions = self.kp_predict(X_val, lambda v1, v2: self.kp_gaussian_kernel(v1, v2, sigma))
                    accuracy = np.mean(predictions == y_val)
                    print(f'Accuracy: {accuracy}')
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {'sigma': sigma, 'epochs': epochs}
            best_kernel = 'rbf'

        elif kernel_type == 'polynomial':
            for degree in param_grid['degrees']:
                for constant in param_grid['constants']:
                    for epochs in epoch_options:
                        print(f"Testing degree={degree}, constant={constant}, and epochs={epochs}")
                        self.support_vectors, self.labels_sv = self.kp_train(X_train_ext, y_train_ext,
                                                                          lambda v1, v2: self.kp_polynomial_kernel(
                                                                              v1, v2, degree, constant), epochs)
                        predictions = self.kp_predict(X_val, lambda v1, v2: self.kp_polynomial_kernel(v1, v2, degree,
                                                                                                        constant))
                        accuracy = np.mean(predictions == y_val)
                        print(f'Accuracy: {accuracy}')
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {'degree': degree, 'constant': constant, 'epochs': epochs}
            best_kernel = 'polynomial'

        print(f"Best kernel: {best_kernel}")
        print(f"Best hyperparameters: {best_params}")
        print(f"Best validation accuracy: {best_accuracy}")
        return best_params

    def kp_evaluate_accuracy(self, X_train, y_train, X_test, y_test, kernel_type):
        # Evaluate the model's accuracy using the best hyperparameters
        if kernel_type == 'rbf':
            if 'sigma' not in self.best_params or 'epochs' not in self.best_params:
                raise ValueError("RBF kernel parameters are not set. Run optimize_hyperparameters first.")
            kernel_func = lambda v1, v2: self.kp_gaussian_kernel(v1, v2, self.best_params['sigma'])
            self.support_vectors, self.labels_sv = self.kp_train(X_train, y_train, kernel_func, self.best_params['epochs'])

        elif kernel_type == 'polynomial':
            if 'degree' not in self.best_params or 'constant' not in self.best_params or 'epochs' not in self.best_params:
                raise ValueError("Polynomial kernel parameters are not set. Run optimize_hyperparameters first.")
            kernel_func = lambda v1, v2: self.kp_polynomial_kernel(v1, v2, self.best_params['degree'],
                                                                        self.best_params['constant'])
            self.support_vectors, self.labels_sv = self.kp_train(X_train, y_train, kernel_func, self.best_params['epochs'])

        else:
            raise ValueError("Invalid kernel type. Choose 'rbf' or 'polynomial'.")

        predictions = self.kp_predict(X_test, kernel_func)
        misclassified_count = np.sum(predictions != y_test)
        accuracy = 1 - misclassified_count / len(y_test)
        print(f"Misclassification Rate: {misclassified_count / len(y_test):.4f}")
        return accuracy


class KernelizedPegasos:
    def __init__(self):
        self.dual_coeffs = None  # Dual coefficients for support vectors

    def kpe_gaussian_kernel(self, vec1, vec2, sigma):
        # Compute the Radial Basis Function (RBF) kernel between two vectors
        return np.exp(-np.linalg.norm(vec1 - vec2) ** 2 / (2 * sigma ** 2))

    def kpe_polynomial_kernel(self, vec1, vec2, degree=3, constant=1):
        # Compute the polynomial kernel between two vectors
        return (np.dot(vec1, vec2) + constant) ** degree

    def kpe_fit(self, support_vectors, lambda_, max_iter, kernel_fn, X_train, y_train):
        # Train the Kernelized Pegasos model using the specified kernel function
        self.dual_coeffs = np.zeros(len(support_vectors), dtype=float)
        for iteration in range(1, max_iter + 1):
            index = np.random.randint(0, len(support_vectors))
            kernel_sum = 0
            for j in range(len(support_vectors)):
                kernel_sum += self.dual_coeffs[j] * y_train[j] * kernel_fn(X_train[index], X_train[j])
            if y_train[index] * kernel_sum < lambda_:
                self.dual_coeffs[index] += 1
        return self.dual_coeffs

    def kpe_predict(self, X_new, X_train, y_train, kernel_fn):
        # Predict using the Kernelized Pegasos model
        kernel_sum = 0
        for j in range(len(X_train)):
            kernel_sum += self.dual_coeffs[j] * y_train[j] * kernel_fn(X_new, X_train[j])
        prediction = np.sign(kernel_sum)
        return prediction

    def kpe_tune_parameters(self, X_train_extended, y_train_extended, X_val, y_val, kernel_type, param_grid, iter_values,
                        lambda_values):
        # Tune hyperparameters for the Kernelized Pegasos model
        best_accuracy = -np.inf
        best_params = {}

        if kernel_type == 'rbf':
            for sigma in param_grid['sigmas']:
                for lambda_ in lambda_values:
                    for max_iter in iter_values:
                        print(f"Testing sigma={sigma}, lambda={lambda_}, max_iter={max_iter}")
                        kernel_fn = lambda x1, x2: self.kpe_gaussian_kernel(x1, x2, sigma)
                        support_vectors = X_train_extended
                        self.dual_coeffs = self.fit(support_vectors, lambda_, max_iter, kernel_fn, X_train_extended,
                                                    y_train_extended)
                        y_pred = np.array(
                            [self.predict(x, X_train_extended, y_train_extended, kernel_fn) for x in X_val])
                        accuracy = np.mean(y_pred == y_val)
                        print(f'Accuracy: {accuracy}')
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {'sigma': sigma, 'lambda': lambda_, 'max_iter': max_iter}
            best_kernel = 'rbf'

        elif kernel_type == 'polynomial':
            for degree in param_grid['degrees']:
                for constant in param_grid['constants']:
                    for lambda_ in lambda_values:
                        for max_iter in iter_values:
                            print(
                                f"Testing degree={degree}, constant={constant}, lambda={lambda_}, max_iter={max_iter}")
                            kernel_fn = lambda x1, x2: self.kpe_polynomial_kernel(x1, x2, degree, constant)
                            support_vectors = X_train_extended
                            self.dual_coeffs = self.fit(support_vectors, lambda_, max_iter, kernel_fn, X_train_extended,
                                                        y_train_extended)
                            y_pred = np.array(
                                [self.predict(x, X_train_extended, y_train_extended, kernel_fn) for x in X_val])
                            accuracy = np.mean(y_pred == y_val)
                            print(f'Accuracy: {accuracy}')
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = {'degree': degree, 'constant': constant, 'lambda': lambda_,
                                               'max_iter': max_iter}
            best_kernel = 'polynomial'

        print(f"Best kernel: {best_kernel}")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_accuracy}")

    def kpe_evaluate_accuracy(self, X_train, y_train, X_test, y_test, kernel_type):
        # Compute accuracy on test data using the best hyperparameters
        if kernel_type == 'rbf':
            if 'sigma' not in self.best_params or 'lambda' not in self.best_params or 'max_iter' not in self.best_params:
                raise ValueError("RBF kernel parameters not set. Run tune_parameters first.")
            kernel_fn = lambda x1, x2: self.compute_rbf_kernel(x1, x2, self.best_params['sigma'])
            support_vectors = X_train
            self.dual_coeffs = self.fit(support_vectors, self.best_params['lambda'], self.best_params['max_iter'],
                                        kernel_fn, X_train, y_train)

        elif kernel_type == 'polynomial':
            if 'degree' not in self.best_params or 'constant' not in self.best_params or 'lambda' not in self.best_params or 'max_iter' not in self.best_params:
                raise ValueError("Polynomial kernel parameters not set. Run tune_parameters first.")
            kernel_fn = lambda x1, x2: self.compute_polynomial_kernel(x1, x2, self.best_params['degree'],
                                                                      self.best_params['constant'])
            support_vectors = X_train
            self.dual_coeffs = self.fit(support_vectors, self.best_params['lambda'], self.best_params['max_iter'],
                                        kernel_fn, X_train, y_train)

        else:
            raise ValueError("Invalid kernel type. Choose 'rbf' or 'polynomial'.")

        y_pred = np.array([self.predict(x, X_train, y_train, kernel_fn) for x in X_test])
        misclassified = np.sum(y_pred != y_test)
        accuracy = 1 - misclassified / len(y_test)
        print(f"Misclassification Rate: {misclassified / len(y_test):.4f}")
        return accuracy