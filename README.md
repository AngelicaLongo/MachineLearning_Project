# MachineLearning_Project
 
Kernelized Linear Classification

The goal is to learn how to classify the Y labels based on the numerical features X1, ..., X10 according to the 0-1 loss. 
The dataset is explored and the appropriate preprocessing steps performed. 

In the project, the following machine learning algorithms are implemented and analyzied:

- The Perceptron
- Support Vector Machines (SVMs) using the Pegasos algorithm
- Regularized logistic classification (i.e., the Pegasos objective function with logistic loss instead of hinge loss)
- Perceptron with polynomial feature expansion of degree 2
- Pegasos with polynomial feature expansion of degree 2
- Logistic with polynomial feature expansion of degree 2
- The kernelized Perceptron with the Gaussian and the polynomial kernels
- The kernelized Pegasos with the Gaussian and the polynomial kernels for SVM

Hyperparameters tuning is performed using k-folds cross-validation and the best configuration is used for training the whole training set and compute the test accuracy. 
Linear weights corresponding to the various numerical features of base and expanded models are compared.
