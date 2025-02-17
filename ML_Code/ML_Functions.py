# Importing essential libraries for data manipulation, analysis, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
np.random.seed(42)

#Necessary pre-processing steps
class PreProcessing:
    def __init__(self, path):
        """Dataset initialization."""
        self.df = pd.read_csv(path)

    def remove_missing_values(self):
        """Check and remove missing values from the dataset."""
        count_missing = self.df.isnull().sum().sum()
        self.df.dropna(inplace=True)
        print(f"Removed {count_missing} missing values from the dataset.")

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

    def split_train_test(self, test_size=0.2):
        """Divide dataset into training and test set."""
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        indices = np.random.permutation(len(X))  # Shuffle degli indici
        test_split = int(len(X) * (1 - test_size))

        train_indices = indices[:test_split]
        test_indices = indices[test_split:]

        self.X_train, self.y_train = X.iloc[train_indices], y.iloc[train_indices]
        self.X_test, self.y_test = X.iloc[test_indices], y.iloc[test_indices]

        print("\nDataset split into Train and Test sets.\n")

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
        self.split_train_test()
        self.plot_boxplot()
        self.remove_outliers()
        self.check_correlation()
        self.standardize()
        self.check_class_balance()
        print("\n✅ Preprocessing completed!\n")

#Algorithms implementation
class Perceptron:
    def __init__(self, epoch_candidates):
        # Initialize the classifier with a list of candidate epochs for cross-validation
        self.epoch_candidates = epoch_candidates
        self.optimal_epochs = None
        self.highest_accuracy = 0
        self.weights = None

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
     #   return 1

class PegasosSVM:
    def __init__(self, num_iterations_list, regularization_params, learning_rate_schedules, num_folds=5):

        self.num_iterations_list = num_iterations_list
        self.regularization_params = regularization_params
        self.learning_rate_schedules = learning_rate_schedules
        self.num_folds = num_folds

        # Optimal parameters
        self.optimal_iterations = None
        self.optimal_lambda = None
        self.optimal_learning_rate = None
        self.optimal_accuracy = 0
        self.weights = None

    def pegasos_train(self, X_train, y_train, num_iterations, lambda_param, learning_rate_fn):
        """Train of the model Pegasos SVM."""
        num_samples, num_features = X_train.shape
        weights = np.zeros(num_features)  #Weights initialization

        for iteration in range(1, num_iterations + 1):
            random_idx = np.random.randint(num_samples)
            x_i, y_i = X_train[random_idx], y_train[random_idx]
            eta_t = learning_rate_fn(iteration)

            # Weights update
            if y_i * np.dot(weights, x_i) < 1:
                weights = (1 - eta_t * lambda_param) * weights + eta_t * y_i * x_i
            else:
                weights = (1 - eta_t * lambda_param) * weights

        return weights

    def pegasos_predict(self, X):
        """Predicting labels with the trained model."""
        if self.weights is None:
            raise ValueError("The model hasn't been trained. Train the model before making predictions.")
        return np.sign(np.dot(X, self.weights))

    def pegasos_cross_validation(self, X_train, y_train):
        """K-fold cross-validation k-fold to optimize parameters."""
        num_samples = len(y_train)
        fold_size = num_samples // self.num_folds

        for num_iterations in self.num_iterations_list:
            for lambda_param in self.regularization_params:
                for learning_rate_fn in self.learning_rate_schedules:
                    fold_accuracies = []

                    for fold in range(self.num_folds):
                        start, end = fold * fold_size, (fold + 1) * fold_size
                        X_val, y_val = X_train[start:end], y_train[start:end]
                        X_train_fold = np.concatenate((X_train[:start], X_train[end:]), axis=0)
                        y_train_fold = np.concatenate((y_train[:start], y_train[end:]), axis=0)

                        weights = self.pegasos_train(X_train_fold, y_train_fold, num_iterations, lambda_param,
                                                     learning_rate_fn)
                        predictions = np.sign(np.dot(X_val, weights))
                        accuracy = np.mean(predictions == y_val)
                        fold_accuracies.append(accuracy)

                    avg_accuracy = np.mean(fold_accuracies)
                    print(
                        f"Iterations = {num_iterations}, Lambda = {lambda_param}, Average accuracy = {avg_accuracy:.4f}")

                    if avg_accuracy > self.optimal_accuracy:
                        self.optimal_accuracy = avg_accuracy
                        self.optimal_iterations = num_iterations
                        self.optimal_lambda = lambda_param
                        self.optimal_learning_rate = learning_rate_fn

        if self.optimal_iterations is None:
            raise ValueError(
                "Optimal parameters haven't been found. Check the dataset and the hyperparameters values.")

        print(
            f"\nBest parameters founded:\nIterations: {self.optimal_iterations}\nLambda: {self.optimal_lambda}\nAverage accuracy: {self.optimal_accuracy:.4f}\n")

    def pegasos_evaluate(self, X_train, y_train, X_test, y_test):
        """Train the model with the best parameters and test it on the test set."""
        if self.optimal_iterations is None or self.optimal_lambda is None or self.optimal_learning_rate is None:
            raise ValueError("Hyperparameters haven't been optimized. Do cross-validation before the evaluation.")

        self.weights = self.pegasos_train(X_train, y_train, self.optimal_iterations, self.optimal_lambda,
                                          self.optimal_learning_rate)
        predictions = self.pegasos_predict(X_test)
        misclassification_rate = np.mean(predictions != y_test)
        print(f"Test Misclassification Rate: {misclassification_rate:.4f}")
        #return 1

class Logistic:
    def __init__(self, num_iterations_list, regularization_params, learning_rate_schedules, num_folds=5):
        self.num_iterations_list = num_iterations_list
        self.regularization_params = regularization_params
        self.learning_rate_schedules = learning_rate_schedules
        self.num_folds = num_folds
        self.optimal_iterations = None
        self.optimal_lambda = None
        self.optimal_learning_rate = None
        self.optimal_accuracy = 0
        self.weights = None

    def logistic_activation(self, z):
        """Logistic (Sigmoid) activation function."""
        return 1 / (1 + np.exp(-z))

    def logistic_train(self, X_train, y_train, num_iterations, lambda_param, learning_rate_fn):
        """Train the logistic regression model."""
        num_samples, num_features = X_train.shape
        weights = np.zeros(num_features)  # Initialize weights to zero

        for iteration in range(1, num_iterations + 1):
            random_idx = np.random.randint(num_samples)
            x_i, y_i = X_train[random_idx], y_train[random_idx]

            prediction = self.logistic_activation(np.dot(x_i, weights))
            target = (y_i + 1) / 2  # Convert labels to 0 or 1 (from -1 and 1)

            gradient = (prediction - target) * x_i + lambda_param * weights  # Regularization added to gradient
            eta_t = learning_rate_fn(iteration)  # Calculate learning rate based on function
            weights -= eta_t * gradient  # Update weights

        return weights

    def logistic_classify(self, X):
        """Classify using the logistic regression model."""
        probabilities = self.logistic_activation(np.dot(X, self.weights))
        return np.where(probabilities >= 0.5, 1, -1)

    def logistic_cross_validation(self, X_train, y_train):
        """Perform k-fold cross-validation to optimize hyperparameters."""
        num_samples = len(y_train)
        fold_size = num_samples // self.num_folds

        for num_iterations in self.num_iterations_list:
            for lambda_param in self.regularization_params:
                for learning_rate_fn in self.learning_rate_schedules:
                    fold_accuracies = []

                    for fold in range(self.num_folds):
                        start, end = fold * fold_size, (fold + 1) * fold_size
                        X_val, y_val = X_train[start:end], y_train[start:end]
                        X_train_fold = np.concatenate((X_train[:start], X_train[end:]), axis=0)
                        y_train_fold = np.concatenate((y_train[:start], y_train[end:]), axis=0)

                        self.weights = self.logistic_train(X_train_fold, y_train_fold, num_iterations, lambda_param,
                                                           learning_rate_fn)
                        preds = self.logistic_classify(X_val)
                        accuracy = np.mean(preds == y_val)
                        fold_accuracies.append(accuracy)

                    avg_accuracy = np.mean(fold_accuracies)
                    print(f"Iterations = {num_iterations}, Lambda = {lambda_param}, Average accuracy = {avg_accuracy:.4f}")

                    if avg_accuracy > self.optimal_accuracy:
                        self.optimal_accuracy = avg_accuracy
                        self.optimal_iterations = num_iterations
                        self.optimal_lambda = lambda_param
                        self.optimal_learning_rate = learning_rate_fn

        if self.optimal_iterations is None:
            raise ValueError("Optimal parameters haven't been found. Check the dataset and hyperparameter values.")

        print(
            f"\nBest parameters found:\nIterations: {self.optimal_iterations}\nLambda: {self.optimal_lambda}\nAverage accuracy: {self.optimal_accuracy:.4f}\n")

    def logistic_evaluate(self, X_train, y_train, X_test, y_test):
        """Train with optimal hyperparameters and evaluate the model."""
        if self.optimal_iterations is None or self.optimal_lambda is None or self.optimal_learning_rate is None:
            raise ValueError("Hyperparameters haven't been optimized. Run cross-validation first.")

        # Train the model with optimal parameters
        self.weights = self.logistic_train(X_train, y_train, self.optimal_iterations, self.optimal_lambda,
                                           self.optimal_learning_rate)
        predictions = self.logistic_classify(X_test)
        misclassification_rate = np.mean(predictions != y_test)
        print(f"Test Misclassification Rate: {misclassification_rate:.4f}")
       # return 1

#Feature expansion and weights comparison
class FeatureExpansion:
    def __init__(self):
        pass

    def expand_polynomial_features(self, X, max_degree=2):
        """
        Expands the features of X to include polynomial terms up to max_degree.

        Args:
        - X (numpy.ndarray): Input feature matrix with shape (num_samples, num_features)
        - max_degree (int): Maximum degree of polynomial expansion (default is 2 for second-degree expansion)

        Returns:
        - numpy.ndarray: Expanded feature matrix
        """
        num_samples, num_features = X.shape
        expanded_features = [X]  # Start with the original features

        # Add higher-degree features (squared terms, etc.)
        for degree in range(2, max_degree + 1):
            for i in range(num_features):
                expanded_features.append(X[:, i:i + 1] ** degree)  # Add powers of the features

        # Generate interaction terms for all pairs (i < j)
        interaction_terms = [X[:, i:i + 1] * X[:, j:j + 1] for i in range(num_features) for j in range(i + 1, num_features)]
        expanded_features.extend(interaction_terms)

        return np.hstack(expanded_features)  # Combine all the expanded features into one matrix

    def _generate_feature_names(self, original_features):
        """
        Generate names for polynomial expanded features.

        Args:
        - original_features (list): List of original feature names

        Returns:
        - list: Expanded feature names
        """
        poly_features = original_features.copy()

        # Generate squared and interaction terms
        for i, f1 in enumerate(original_features):
            for f2 in original_features[i:]:
                poly_features.append(f"{f1}*{f2}")

        return poly_features

    def describe_expansion(self, X, X_poly):
        """
        Prints information about the feature expansion.

        Args:
        - X (numpy.ndarray): Original feature matrix
        - X_poly (numpy.ndarray): Expanded feature matrix
        """
        original_features = [f"x{i+1}" for i in range(X.shape[1])]
        expanded_features = self._generate_feature_names(original_features)

        print(f"Shape of original features: {X.shape}\n")
        print(f"Shape of features after polynomial expansion: {X_poly.shape}\n")
        print("New list of features:", expanded_features)

class WeightVisualizer:
    def __init__(self):
        self.features = None  # Initialize the variable

    def generate_latex_table(self, perceptron_weights, pegasos_weights, logistic_weights, dataframe):
        # If the dataframe is an array, it is converted into a dataframe
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The given dataframe is not a Pandas dataframe!")

        self.features = dataframe.columns.tolist()  # Features names

        # Check if weights and features have the same lenghts
        num_features = len(self.features)
        if len(perceptron_weights) != num_features or len(pegasos_weights) != num_features or len(logistic_weights) != num_features:
            raise ValueError(f"Weights must have {num_features} elements, but have respectively"
                             f"{len(perceptron_weights)}, {len(pegasos_weights)}, {len(logistic_weights)}")

        # Create weights dataframe
        weight_data = {
            'Feature': self.features,
            'Perceptron': perceptron_weights.tolist(),
            'Pegasos': pegasos_weights.tolist(),
            'Logistic': logistic_weights.tolist()
        }
        self.weight_df = pd.DataFrame(weight_data).set_index('Feature')

        # Convert in latex and save on file
        latex_output = self.weight_df.to_latex(index=True)
        with open('weights_original.tex', 'w') as file:
            file.write(latex_output)

        return self.weight_df

    def plot_weight_comparison(self, perceptron_weights, pegasos_weights, logistic_weights):
        """Plot weights comparison among different models"""
        if not self.features:
            raise ValueError("Features have not been initialized. Perform generate_latex_table before this method.")

        plt.figure(figsize=(10, 8))
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
        plt.show()

    def generate_polynomial_latex_table(self, poly_perceptron_weights, poly_pegasos_weights, poly_logistic_weights):
        """Create a latex table for polynomial weights"""
        if not self.features:
            raise ValueError("Features have not been initialized. Perform generate_latex_table before this method.")

        base_variables = [f"x{i + 1}" for i in range(len(self.features))]
        polynomial_terms = [f"{var}^2" for var in base_variables]
        interaction_terms = [f"{base_variables[i]}*{base_variables[j]}" for i, j in combinations(range(len(base_variables)), 2)]

        feature_names = base_variables + polynomial_terms + interaction_terms

        self.poly_weight_df = pd.DataFrame({
            'Feature': feature_names,
            'Perceptron': poly_perceptron_weights.tolist(),
            'Pegasos': poly_pegasos_weights.tolist(),
            'Logistic': poly_logistic_weights.tolist()
        })

        # Divide in blocchi di 22 righe per gestire tabelle molto lunghe
        chunked_dataframes = [self.poly_weight_df.iloc[i:i + 22].reset_index(drop=True) for i in range(0, len(self.poly_weight_df), 22)]
        final_df = pd.concat(chunked_dataframes, axis=1)
        latex_output = final_df.to_latex(index=False)

        with open('weights_poly.tex', 'w') as file:
            file.write(latex_output)

        return self.poly_weight_df

#Kernelized algorithms
class KernelizedPerceptron:
    def __init__(self):
        self.support_vectors = []  # Support vectors
        self.labels_sv = []  # Labels of support vectors

    def kp_gaussian_kernel(self, vector1, vector2, sigma):
        return np.exp(-np.linalg.norm(vector1 - vector2) ** 2 / (2 * sigma ** 2))

    def kp_polynomial_kernel(self, vector1, vector2, degree=3, constant=1):
        return (np.dot(vector1, vector2) + constant) ** degree

    def kp_train(self, X_train, y_train, kernel_func, max_epochs=1):
        self.support_vectors = []  # Reset dei support vectors
        self.labels_sv = []
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
        predictions = []
        for vector in X_test:
            kernel_sum = sum(self.labels_sv[i] * kernel_func(self.support_vectors[i], vector)
                             for i in range(len(self.support_vectors)))
            predictions.append(np.sign(kernel_sum))
        return np.array(predictions)

    def kp_cross_validation(self, X, y, kernel_type, param_grid, epoch_options, k=5):
        n = len(X)
        fold_size = n // k
        best_accuracy = -np.inf
        best_params = {}

        if kernel_type == 'rbf':
            for sigma in param_grid.get('sigmas', [None]):
                for epochs in epoch_options:
                    print(f"Testing kernel type = {kernel_type}, sigma = {sigma}, epochs = {epochs}")

                    scores = []
                    for i in range(k):
                        # Split data into train and validation sets
                        val_start = i * fold_size
                        val_end = (i + 1) * fold_size if i < k - 1 else n
                        X_val, y_val = X[val_start:val_end], y[val_start:val_end]
                        X_train = np.vstack((X[:val_start], X[val_end:]))
                        y_train = np.hstack((y[:val_start], y[val_end:]))

                        # Use the RBF kernel
                        kernel_func = lambda v1, v2: self.kp_gaussian_kernel(v1, v2, sigma)

                        # Train the model with the selected parameters
                        self.kp_train(X_train, y_train, kernel_func, epochs)
                        # Predict using the trained model
                        predictions = self.kp_predict(X_val, kernel_func)
                        accuracy = np.mean(predictions == y_val)
                        scores.append(accuracy)

                    # Calculate the average accuracy for this set of parameters
                    avg_score = np.mean(scores)
                    print(f"Accuracy with max_epochs = {epochs}, sigma = {sigma} is {avg_score:.4f}")

                    # Update best parameters if needed
                    if avg_score > best_accuracy:
                        best_accuracy = avg_score
                        best_params = {'sigma': sigma, 'epochs': epochs}

        elif kernel_type == 'polynomial':
            for degree in param_grid.get('degrees', [None]):
                for constant in param_grid.get('constants', [None]):
                    for epochs in epoch_options:
                        print(f"Testing kernel type = {kernel_type}, degree = {degree}, constant = {constant}, epochs = {epochs}")

                        scores = []
                        for i in range(k):
                            # Split data into train and validation sets
                            val_start = i * fold_size
                            val_end = (i + 1) * fold_size if i < k - 1 else n
                            X_val, y_val = X[val_start:val_end], y[val_start:val_end]
                            X_train = np.vstack((X[:val_start], X[val_end:]))
                            y_train = np.hstack((y[:val_start], y[val_end:]))

                            # Use the Polynomial kernel
                            kernel_func = lambda v1, v2: self.kp_polynomial_kernel(v1, v2, degree, constant)

                            # Train the model with the selected parameters
                            self.kp_train(X_train, y_train, kernel_func, epochs)
                            # Predict using the trained model
                            predictions = self.kp_predict(X_val, kernel_func)
                            accuracy = np.mean(predictions == y_val)
                            scores.append(accuracy)

                        # Calculate the average accuracy for this set of parameters
                        avg_score = np.mean(scores)
                        print(f"Accuracy with max_epochs = {epochs}, degree = {degree}, constant = {constant} is {avg_score:.4f}")

                        # Update best parameters if needed
                        if avg_score > best_accuracy:
                            best_accuracy = avg_score
                            best_params = {'degree': degree, 'constant': constant, 'epochs': epochs}

        print(f"Best hyperparameters: {best_params}")
        print(f"Best cross-validation accuracy: {best_accuracy}")
        return best_params

    def kp_evaluate_test(self, X_test, y_test, kernel_func):
        predictions = self.kp_predict(X_test, kernel_func)
        accuracy = np.mean(predictions == y_test)
        print(f"Test set accuracy: {accuracy:.4f}")
        #return accuracy

class KernelizedPegasos:
    def __init__(self):
        self.support_vectors = []  # Support vectors
        self.labels_sv = []  # Labels of support vectors
        self.dual_coeffs = []  # Dual coefficients for support vectors

    def kpe_gaussian_kernel(self, vector1, vector2, sigma):
        return np.exp(-np.linalg.norm(vector1 - vector2) ** 2 / (2 * sigma ** 2))

    def kpe_polynomial_kernel(self, vector1, vector2, degree=3, constant=1):
        return (np.dot(vector1, vector2) + constant) ** degree

    def kpe_train(self, X_train, y_train, kernel_func, lambda_, max_iter):
        self.support_vectors = X_train
        self.labels_sv = y_train
        self.dual_coeffs = np.zeros(len(X_train))

        for iteration in range(1, max_iter + 1):
            index = np.random.randint(0, len(X_train))
            kernel_sum = sum(
                self.dual_coeffs[j] * self.labels_sv[j] * kernel_func(X_train[index], X_train[j])
                for j in range(len(X_train))
            )
            if y_train[index] * kernel_sum < 1:
                self.dual_coeffs[index] += 1 / (lambda_ * iteration)

        return self.support_vectors, self.labels_sv, self.dual_coeffs

    def kpe_predict(self, X_test, kernel_func):
        predictions = []
        for vector in X_test:
            kernel_sum = sum(
                self.dual_coeffs[i] * self.labels_sv[i] * kernel_func(self.support_vectors[i], vector)
                for i in range(len(self.support_vectors))
            )
            predictions.append(np.sign(kernel_sum))
        return np.array(predictions)

    def kpe_cross_validation(self, X, y, kernel_type, param_grid, lambda_values, iter_values, k=5):
        n = len(X)
        fold_size = n // k
        best_accuracy = -np.inf
        best_params = {}

        if kernel_type == 'rbf':
            for sigma in param_grid.get('sigmas', [None]):
                for lambda_ in lambda_values:
                    for max_iter in iter_values:
                        print(f"Testing RBF kernel: sigma={sigma}, lambda={lambda_}, max_iter={max_iter}")
                        scores = []
                        for i in range(k):
                            val_start = i * fold_size
                            val_end = (i + 1) * fold_size if i < k - 1 else n
                            X_val, y_val = X[val_start:val_end], y[val_start:val_end]
                            X_train = np.vstack((X[:val_start], X[val_end:]))
                            y_train = np.hstack((y[:val_start], y[val_end:]))

                            kernel_func = lambda v1, v2: self.kpe_gaussian_kernel(v1, v2, sigma)
                            self.kpe_train(X_train, y_train, kernel_func, lambda_, max_iter)
                            predictions = self.kpe_predict(X_val, kernel_func)
                            accuracy = np.mean(predictions == y_val)
                            scores.append(accuracy)
                        avg_score = np.mean(scores)
                        print(f"Accuracy: {avg_score:.4f}")
                        if avg_score > best_accuracy:
                            best_accuracy = avg_score
                            best_params = {'sigma': sigma, 'lambda': lambda_, 'max_iter': max_iter}

        elif kernel_type == 'polynomial':
            for degree in param_grid.get('degrees', [None]):
                for constant in param_grid.get('constants', [None]):
                    for lambda_ in lambda_values:
                        for max_iter in iter_values:
                            print(
                                f"Testing Polynomial kernel: degree={degree}, constant={constant}, lambda={lambda_}, max_iter={max_iter}")
                            scores = []
                            for i in range(k):
                                val_start = i * fold_size
                                val_end = (i + 1) * fold_size if i < k - 1 else n
                                X_val, y_val = X[val_start:val_end], y[val_start:val_end]
                                X_train = np.vstack((X[:val_start], X[val_end:]))
                                y_train = np.hstack((y[:val_start], y[val_end:]))

                                kernel_func = lambda v1, v2: self.kpe_polynomial_kernel(v1, v2, degree, constant)
                                self.kpe_train(X_train, y_train, kernel_func, lambda_, max_iter)
                                predictions = self.kpe_predict(X_val, kernel_func)
                                accuracy = np.mean(predictions == y_val)
                                scores.append(accuracy)
                            avg_score = np.mean(scores)
                            print(f"Accuracy: {avg_score:.4f}")
                            if avg_score > best_accuracy:
                                best_accuracy = avg_score
                                best_params = {'degree': degree, 'constant': constant, 'lambda': lambda_,
                                               'max_iter': max_iter}

        print(f"Best hyperparameters: {best_params}")
        print(f"Best cross-validation accuracy: {best_accuracy:.4f}")
        return best_params

    def kpe_evaluate_test(self, X_test, y_test, kernel_func):
        predictions = self.kpe_predict(X_test, kernel_func)
        accuracy = np.mean(predictions == y_test)
        print(f"Test set accuracy: {accuracy:.4f}")
        # return accuracy