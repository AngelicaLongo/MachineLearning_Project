import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
np.random.seed(42)

#### IQR method to remove outliers

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]




##### PERCEPTRON
class Perceptron:
    # Initialize with maximum number of epochs and a vector of weights
    def __init__(self, max_epochs=1000):
        self.max_epochs = max_epochs
        self.w = None

    def reset(self):
        self.w = None

    def get_params(self, deep=True):
        # Return parameters as a dictionary
        return {"max_epochs": self.max_epochs}

    def set_params(self, **params):
        # Set model parameters
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        # Updates weights vector until it reaches convergence or maximum number of epochs
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        # Print the parameters used
        print(f"Training with parameters: max_epochs={self.max_epochs}")

        for epoch in range(self.max_epochs):
            errors = 0
            for i in range(len(X)):
                if y[i] * (np.dot(X[i], self.w)) <= 0:
                    self.w += y[i] * X[i]
                    errors += 1
            if errors == 0:
                print(f"Convergence reached at epoch number {epoch + 1}")
                break
        if errors > 0:
            print(f"Max epochs ({self.max_epochs}) reached. NO convergence.")

        # Calculate and print the final accuracy on the training set
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        print(f"Accuracy on training set: {accuracy:.4f}")

        return self

    def predict(self, X):
        # Returns an array of predictions
        return np.sign(np.dot(X, self.w))

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

#### PEGASOS SVM
class PegasosSVM(BaseEstimator):
    #Initialise parameters (lamdbda and maximum number of iterations) and a weights vector
    def __init__(self, lambda_param=0.01, max_iter=1000):
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.w = None

    def reset(self):
        self.w = None

    def get_params(self, deep=True):
        # Return parameters as a dictionary
        return {"lambda_param": self.lambda_param, "max_iter": self.max_iter}

    def set_params(self, **params):
        # Set parameters of the model
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        cumulative_w = np.zeros(n_features)

        print(f'Training with parameters: max_iter: {self.max_iter}, lambda: {self.lambda_param}')

        for t in range(1, self.max_iter + 1):
            i = np.random.randint(0, n_samples)
            learning_rate = 1 / (self.lambda_param * t)

            if y[i] * (np.dot(X[i], self.w)) < 1:
                self.w = (1 - learning_rate * self.lambda_param) * self.w + learning_rate * y[i] * X[i]
            else:
                self.w = (1 - learning_rate * self.lambda_param) * self.w

            cumulative_w += self.w #var made for mean

        # Final weights mean
        self.w = cumulative_w / self.max_iter

        # Calculate and print the final accuracy on the training set
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        print(f"Accuracy on training set: {accuracy:.4f}")

        return self

    def predict(self, X):
        # Returns an array of predictions
        return np.sign(np.dot(X, self.w))

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)



#### LOGISTIC CLASSIFICATION

class PegasosLogisticClassification(BaseEstimator):
    def __init__(self, lambda_param=0.01, max_iter=1000):
        # Initialise parameters (lambda and maximum number of iterations) and a weights vector
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.w = None

    def reset(self):
        """ Reset the model weights to None for fresh training. """
        self.w = None

    def get_params(self, deep=True):
        # Return parameters as a dictionary
        return {"lambda_param": self.lambda_param, "max_iter": self.max_iter}

    def set_params(self, **params):
        # Imposta i parametri del modello
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def sigmoid(self, z):
        # Maps linear combination (z) to probability (0, 1)
        z = np.clip(z, -700, 700)  # Limiting Z to avoid overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # Weight initialization
        cumulative_w = np.zeros(n_features)  # Mean purposes

        print(f'Training with parameters: max_iter: {self.max_iter}, lambda: {self.lambda_param}')

        for t in range(1, self.max_iter + 1):
            # Random selection of an index
            i = np.random.randint(0, n_samples)
            learning_rate = 1 / (self.lambda_param * t)

            # Computing probability with sigmoid function
            margin = y[i] * (np.dot(X[i], self.w))
            prob = self.sigmoid(margin)

            # Updating weights
            self.w = (1 - learning_rate * self.lambda_param) * self.w + learning_rate * y[i] * X[i] * (1 - prob)

            # For mean purposes
            cumulative_w += self.w

        # Average of final weights
        self.w = cumulative_w / self.max_iter

        return self

    def predict_proba(self, X):
        linear_output = np.dot(X, self.w)
        return self.sigmoid(linear_output)

    def predict(self, X):
        # Returns an array of predictions
        prob = self.predict_proba(X)
        return np.where(prob >= 0.5, 1, -1)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)




#### POLYNOMIAL FEATURE EXPANSION

def polynomial_feature_expansion(X):
    n_samples, n_features = X.shape
    new_features = []

    new_features.append(X)  # add original features

    # Add quadratic features
    for i in range(n_features):
        for j in range(i, n_features):
            new_features.append(X[:, i] * X[:, j])

    # Combine all new features in 1 array
    X_poly = np.column_stack(new_features)

    return X_poly




#### KERNELISED PERCEPTRON (GAUSSIAN AND POLYNOMIAL)

# define gaussian and polynomial kernel functions

def gaussian_kernel(X, Y, gamma=0.1):
    # Computing Gaussian Kernel
    return np.exp(-np.linalg.norm(X - Y) ** 2 / (2 * gamma))

def polynomial_kernel(X, Y, degree=3):
    # Computing Polynomial Kernel
    return (1 + np.dot(X, Y.T)) ** degree

# Main algorithm

class KernelPerceptron:
    def __init__(self, max_epochs=1000, kernel=polynomial_kernel, **kernel_params):
        self.max_epochs = max_epochs
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.support_vectors = []
        self.support_labels = []

    def reset(self):
        self.support_vectors = []
        self.support_labels = []

    def get_params(self, deep=True):
        # Return parameters as a dictionary
        return {"max_epochs": self.max_epochs, "kernel": self.kernel, **self.kernel_params}

    def set_params(self, **params):
        # Imposta i parametri del modello
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        n_samples = len(y)

        for epoch in range(self.max_epochs):
            errors = 0
            for t in range(n_samples):
                # Compute prediction only using support vectors
                kernel_eval = sum(self.support_labels[i] * self.kernel(self.support_vectors[i], X[t], **self.kernel_params)
                                      for i in range(len(self.support_vectors)))
                y_pred = np.sign(kernel_eval)

                # If prediction is wrong, add the example to support vectors
                if y_pred != y[t]:
                    self.support_vectors.append(X[t])
                    self.support_labels.append(y[t])
                    errors +=1
            if errors == 0:
                print(f"Convergence reached at epoch {epoch + 1}")
                break
        if errors > 0:
            print(f"Did not converge after {self.max_epochs} epochs.")

    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = 0
            for j in range(len(self.support_vectors)):
                prediction += self.support_labels[j] * self.kernel(self.support_vectors[j], x)
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)




#### KERNELISED PEGASOS WITH GAUSSIAN AND POLYNOMIAL KERNELS SVM

class KernelPegasosSVM:
    def __init__(self, lambda_param=0.01, max_iter=1000, kernel=gaussian_kernel, **kernel_params):
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None

    def reset(self):
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None

    def get_params(self, deep=True):
        # Restituisce i parametri del modello
        return {"lambda_param": self.lambda_param, "max_iter": self.max_iter, "kernel": self.kernel,
                **self.kernel_params}

    def set_params(self, **params):
        # Imposta i parametri del modello
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        n_samples = len(X)
        self.alpha = np.zeros(n_samples)
        self.support_vectors = X
        self.support_labels = y



        for t in range(1, self.max_iter + 1):
            i_t = np.random.randint(0, n_samples)
            sum_term = sum(self.alpha[j] * y[j] * self.kernel(X[i_t], X[j], **self.kernel_params) for j in range(n_samples))

            if y[i_t] * (1 / (self.lambda_param * t)) * sum_term < 1:
                self.alpha[i_t] += 1

        return self

    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = np.sign(
                sum(self.alpha[j] * self.support_labels[j] * self.kernel(self.support_vectors[j], x)
                    for j in range(len(self.support_vectors)))
            )
            y_pred.append(prediction)
        return np.array(y_pred)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy



#### COMPARE ORIGINAL AND EXPANDED WEIGHTS

def compare_weights(model, feature_names, feature_names_poly, X_train, X_train_poly, y_train):
    # Training model on original features
    model.fit(X_train, y_train)
    original_weights = model.w

    # Training model on polynomial features
    model.fit(X_train_poly, y_train)
    poly_weights = model.w

    # Create a dataframe to visualise original weights
    original_weights_df = pd.DataFrame({'Feature': feature_names, 'Weight': original_weights})
    original_weights_df['Abs_Weight'] = original_weights_df['Weight'].abs()
    original_weights_df = original_weights_df.sort_values(by='Abs_Weight', ascending=False)

    print("Original features weights:")
    print(original_weights_df)

    # Create a dataframe to visualise polynomial weighTs
    poly_weights_df = pd.DataFrame({'Feature': feature_names_poly, 'Weight': poly_weights})
    poly_weights_df['Abs_Weight'] = poly_weights_df['Weight'].abs()
    poly_weights_df = poly_weights_df.sort_values(by='Abs_Weight', ascending=False)

    print("Features weights after polynomial expansion:")
    print(poly_weights_df)




#### CROSS-VALIDATION ON SVM AND LOG_REG (done automatically with original and expanded features)

def KFold(X, y, k):
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    return folds


def cross_val_score(model, X, y, folds):
    scores = []
    for fold in folds:
        X_train = np.delete(X, fold, axis=0)
        y_train = np.delete(y, fold, axis=0)
        X_val = X[fold]
        y_val = y[fold]

        model.reset()
        model.fit(X_train, y_train)
        accuracy = model.score(X_val, y_val)
        scores.append(accuracy)

    return np.array(scores)


def cross_val_model(model, param_grid, X_train, y_train, X_train_poly, k=5):
    folds = KFold(X_train, y_train, k)

    best_score = -np.inf
    best_params = None
    best_feature_type = None

    for lambda_param in param_grid['lambda_param']:
        for max_iter in param_grid['max_iter']:
            model.set_params(lambda_param=lambda_param, max_iter=max_iter)

            # Cross-validation on original features
            scores = cross_val_score(model, X_train, y_train, folds)
            mean_score = np.mean(scores)
            print('Accuracy with lambda (', lambda_param, ') and max_iter (', max_iter, ') is ', mean_score)

            # Update best parameters
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'lambda_param': lambda_param, 'max_iter': max_iter}
                best_feature_type = 'original features'

            # Cross-validation on polynomial features
            folds_poly = KFold(X_train_poly, y_train, k)
            scores_poly = cross_val_score(model, X_train_poly, y_train, folds_poly)
            mean_score_poly = np.mean(scores_poly)
            print('Accuracy with max_iter (', max_iter, ') and lambda (',lambda_param,') is', mean_score_poly)

            if mean_score_poly > best_score:
                best_score = mean_score_poly
                best_params = {'lambda_param': lambda_param, 'max_iter': max_iter}
                best_feature_type = 'polynomial features'

    print(f'Best Parameters: {best_params}')
    print(f'Best Cross-Validation Accuracy: {best_score:.4f}')
    print(f'Model with {best_feature_type} is better.')

    return best_params




#### CROSS-VALIDATION ON PERCEPTRON (distinct function because only one parameter)

def cross_val_model_perceptron(model, max_iter_grid, X_train, y_train, X_train_poly, k=5):
    folds = KFold(X_train, y_train, k)

    best_score = -np.inf
    best_params = None
    best_feature_type = None

    for max_iter in max_iter_grid:
        model.set_params(max_epochs=max_iter)

        # Cross-validation on original features
        scores = cross_val_score(model, X_train, y_train, folds)
        mean_score = np.mean(scores)
        print('Accuracy with max_iter (', max_iter, ') is ', mean_score)

        # Updating best parameters
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'max_iter': max_iter}
            best_feature_type = 'original features'

        # Cross-validation on polynomial features
        folds_poly = KFold(X_train_poly, y_train, k)
        scores_poly = cross_val_score(model, X_train_poly, y_train, folds_poly)
        mean_score_poly = np.mean(scores_poly)
        print('Accuracy with max_iter (', max_iter ,') is', mean_score_poly)

        if mean_score_poly > best_score:
            best_score = mean_score_poly
            best_params = {'max_iter': max_iter}
            best_feature_type = 'polynomial features'

    print(f'Best Parameters for Perceptron: {best_params}')
    print(f'Best Cross-Validation Accuracy: {best_score:.4f}')
    print(f'Model with {best_feature_type} is better.')

    return best_params




#### CROSS-VALIDATION ON KERNELISED PERCEPTRON (GAUSSIAN OR POLYNOMIAL)
def cross_val_model_kernel(model, param_grid, X_train, y_train, kernel_type='gaussian', k=5):
    folds = KFold(X_train, y_train, k)

    best_score = -np.inf
    best_params = None

    for max_epochs in param_grid['max_epochs']:
        if kernel_type == 'gaussian':
            for gamma in param_grid['gamma']:
                model.reset()
                model.set_params(max_epochs=max_epochs, gamma=gamma)
                scores = cross_val_score(model, X_train, y_train, folds)
                mean_score = np.mean(scores)
                print('Accuracy with max_epochs (', max_epochs, ') and gamma (', gamma, ') is', mean_score)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'gamma': gamma, 'max_epochs': max_epochs}

        elif kernel_type == 'polynomial':
            for degree in param_grid['degree']:
                model.reset()
                model.set_params(max_epochs=max_epochs, degree=degree)
                scores = cross_val_score(model, X_train, y_train, folds)
                mean_score = np.mean(scores)
                print('Accuracy with max_epochs (', max_epochs, ') and degree (', degree, ') is', mean_score)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'degree': degree, 'max_epochs': max_epochs}

    print(f'Best Parameters for Perceptron ({kernel_type} kernel): {best_params}')
    print(f'Best Cross-Validation Accuracy for Kernelised Perceptron: {best_score:.4f}')

    return best_params



#### CROSS-VALIDATION ON KERNELISED PEGASOS SVM (GAUSSIAN OR POLYNOMIAL)
def cross_val_model_kernelSVM(model, param_grid, X_train, y_train, kernel_type='gaussian', k=5):
    folds = KFold(X_train, y_train, k)

    best_score = -np.inf
    best_params = None

    if kernel_type == 'gaussian':
        for gamma in param_grid['gamma']:
            for lambda_param in param_grid['lambda_param']:
                model.reset()
                model.set_params(lambda_param=lambda_param, gamma=gamma)
                scores = cross_val_score(model, X_train, y_train, folds)
                mean_score = np.mean(scores)
                print(f'Accuracy with lambda ({lambda_param}) and gamma ({gamma}) is {mean_score}')
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'gamma': gamma, 'lambda_param': lambda_param}

    elif kernel_type == 'polynomial':
        for degree in param_grid['degree']:
            for lambda_param in param_grid['lambda_param']:
                model.reset()
                model.set_params(degree=degree, lambda_param=lambda_param)
                scores = cross_val_score(model, X_train, y_train, folds)
                mean_score = np.mean(scores)
                print(f'Accuracy with lambda ({lambda_param}) and degree ({degree}) is {mean_score}')
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'degree': degree, 'lambda_param': lambda_param}

    print(f'Best Parameters for Pegasos SVM ({kernel_type} kernel): {best_params}')
    print(f'Best Cross-Validation Accuracy for Pegasos SVM: {best_score:.4f}')

    return best_params



#### APPLYING BEST MODEL TO TEST SET AND COMPUTING ACCURACY

def evaluate_model_on_test(model, X_train_poly, y_train, X_test_poly, y_test):
    # Training model with the best parameters on whole training set
    model.fit(X_train_poly, y_train)

    # Predict labels of test set
    y_pred = model.predict(X_test_poly)

    # Compute accuracy
    accuracy = np.mean(y_test == y_pred)
    loss = np.mean(y_test != y_pred)
    print(f'Accuracy on test set: {accuracy:.4f}')
    print(f'Loss on test set: {loss:.4f}')