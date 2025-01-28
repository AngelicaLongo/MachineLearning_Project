# MachineLearning_Project
 
Kernelized Linear Classification

The goal is to learn how to classify the Y labels based on the numerical features X1, ..., X10 according to the 0-1 loss, which is the metric you should adopt when evaluating the trained models. Explore the dataset and perform the appropriate preprocessing steps. Please be mindful of data leakage between the training and test sets. Implement from scratch (without using libraries such as Scikit-learn) the following machine learning algorithms:

- The Perceptron
- Support Vector Machines (SVMs) using the Pegasos algorithm
- Regularized logistic classification (i.e., the Pegasos objective function with logistic loss instead of hinge loss)

Test the performance of these models. Next, attempt to improve the performance of the previous models by using polynomial feature expansion of degree 2. Include and compare the linear weights corresponding to the various numerical features you found after the training phase.

Then, try using kernel methods. Specifically, implement from scratch (again, without using libraries such as Scikit-learn):

- The kernelized Perceptron with the Gaussian and the polynomial kernels
- The kernelized Pegasos with the Gaussian and the polynomial kernels for SVM (refer to the kernelized Pegasos paper with its pseudo-code here in Figure 3. Note that there is a typo in the pseudo-code. Identify and correct it.)

Evaluate the performance of these models as well.

Remember that relevant hyperparameter tuning is a crucial part of the project and must be performed using a sound procedure.

Ensure that the code you provide is polished, working, and, importantly, well-documented. Write a report discussing your findings, with particular attention to the adopted methodology, and provide a thorough discussion of the models’ performance and their theoretical interpretation. Include comments on the presence of overfitting or underfitting and discuss the computational costs.
