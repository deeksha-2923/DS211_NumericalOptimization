import numpy as np
import pandas as pd

# use pandas to load real_estate_dataset.csv
df = pd.read_csv("real_estate_dataset.csv")

# get the number of samples and features
num_samples, num_features = df.shape
print("Number of samples, features: ", num_samples, num_features)

# get the names of the columns
column_names = df.columns

# save the column names to a file for accessing later as text file
np.savetxt('column_names.txt', column_names, fmt='%s')

# Use Square_Feet, Garage_Size, Location_Score, Distance_to_Center as features and price as target
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']].values
y = df['Price'].values
print("X shape: ", X.shape)
print(f"Data type of X: {X.shape}")
print("y shape: ", y.shape)

n_samples, n_features = X.shape
# Build a linear model to predict price from the four features in X
# make an array of coefs of the size of the number of features in X + 1 (for the bias term) and initialize it to 1.

coefs = np.ones(n_features + 1)

# predict the price of each sample in X using the coefs
predictions_by_defn = X @ coefs[1:] + coefs[0]
# if all coefficients are 0, the predicted price will be the bias for all samples. no information about sample implying predict the bias. so bias is the expected value of the target.

# append a column of 1s to X to account for the bias term
X = np.hstack((np.ones((n_samples, 1)), X))

# predict the price of each sample in X using the coefs
predictions = X @ coefs

# check if all entries in predictions_by_defn and predictions are the same
is_same = np.allclose(predictions_by_defn, predictions)
print("Are the predictions the same? ", is_same)

# calculate the error using predictions and y
errors = y - predictions

# print the size of errors and its L2 norm
print("Size of errors: ", errors.shape)
print("L2 norm of errors: ", np.linalg.norm(errors))

# how do we know if the L2 norm of errors is small or large? 

# calculate the relative error
rel_errors = errors / y

print("L2 norm of relative errors: ", np.linalg.norm(rel_errors)) # should ideally be less than 1

# calculate the mean of square of errors using a loop
loss_loop = 0
for i in range(n_samples):
    loss_loop += errors[i] ** 2
loss_loop = loss_loop/n_samples
print("Mean squared error: ", loss_loop)

# calculate the mean of square of errors using matrix operations
loss_matrix = np.transpose(errors) @ errors /n_samples

# compare the two methods of calculating mean squared error
print("Are the two methods of calculating mean squared error the same? ", np.isclose(loss_loop, loss_matrix))


# What is my optimization problem?
# I want to find the coefs that minimize the L2 norm of errors or mean squared error (MSE)
# this problem is called the least squares problem

# Objective function is clear: f(coefs)  = 1/n_samples * \sum_{i=1}^{n_samples} (y_i - X_i^T coefs)^2

# What is a solution?
# A solution is a set of coefs that minimize the objective function

# How do I find a solution?
#  By searching for the coefs at which the gradient of the objective function is zero
#  set the gradient of the objective function to zero and solve for coefs

# write the loss matrix in terms of the data and coefs
loss_matrix = (y - X @ coefs).T @ (y - X @ coefs) / n_samples

# write the gradient of the loss matrix with respect to coefs
grad_loss_matrix = -2/n_samples * X.T @ (y - X @ coefs)

# Loss function is a scalar(12 mn) with respect to coefs (4)+1 ie a vector so gradient is a vector of size 5
# we set grad_loss_matrix to zero and solve for coefs
# X.T @ y = X.T@X@coefs
# X.T @ X @ coefs = X.T @ y This is the normal equation
# coefs = (X.T @ X)^-1 @ X.T @ y
coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# save coefs to a file for accessing later
np.savetxt('coefs.csv', coefs, delimiter=',')

# calculate the predictions using the new coefs
predictions_model = X @ coefs

# calculate the errors using the new coefs
errors_model = y - predictions_model

# print the L2 norm of the errors using the new coefs
print("L2 norm of errors using the new coefs: ", np.linalg.norm(errors_model))

# print the L2 norm of the relative errors using the new coefs
rel_errors_model = errors_model / y
print("L2 norm of relative errors using the new coefs: ", np.linalg.norm(rel_errors_model))

# use all the features in the dataset to build a linear model to predict price
X_all = df.drop(columns=['Price']).values
y_all = df['Price'].values

# get the number of samples and features
n_samples_all, n_features_all = X_all.shape
print("Number of samples, features: ", n_samples_all, n_features_all)

# solve the linear model using the normal equation
X_all = np.hstack((np.ones((n_samples_all, 1)), X_all))
coefs_all = np.linalg.inv(X_all.T @ X_all) @ X_all.T @ y_all

# save coefs to a file for accessing later
np.savetxt('coefs_all.csv', coefs_all, delimiter=',')

# the most expensive step of this process is the inversion of X.T @ X
# calculate the rank of X.T @ X
rank_XTX = np.linalg.matrix_rank(X_all.T @ X_all)
print("Rank of X.T @ X: ", rank_XTX)
# Rank is 12 = no. of features + 1 so it is full rank i.e. invertible
# What if X.T @ X is not invertible?
# solve the normal equation using matrix decomposition techniques
# QR factorization
Q, R = np.linalg.qr(X_all)
print("Q shape: ", Q.shape)
print("R shape: ", R.shape)

# Write R to a file named R.csv
np.savetxt('R.csv', R, delimiter=',')

# R*coeffs= b
# can easily find coeffs using back substitution
# R is upper triangular so back substitution is easy

# Q is orthogonal so Q.T @ Q = I
sol = Q.T @ Q # to check if Q.T@Q is indeed identity. It is. 10^-17,18 is just the machine precision
np.savetxt('sol.csv', sol, delimiter=',')

# X = QR
# X.T @ X = R.T @ Q.T @ Q @ R = R.T @ R
# X.T @ y = R.T @ Q.T @ y
# R @ coefs = Q.T @ y
# coefs = R^-1 @ Q.T @ y
b = Q.T @ y_all
print("Shape of b: ", b.shape)
print("Shape of R: ", R.shape)

# coefs_qr = np.linalg.inv(R) @ b # instead of inverting R, use back substitution

# loop to solve R*coefs = b using back substitution
coefs_qr_loop = np.zeros(n_features_all + 1)
for i in range(n_features_all, -1, -1):
    coefs_qr_loop[i] = (b[i] - np.dot(R[i, i + 1:], coefs_qr_loop[i + 1:])) / R[i, i] 

# save coefs_qr_loop to a file for accessing later
np.savetxt('coefs_qr_loop.csv', coefs_qr_loop, delimiter=',')


# find the errors using the coefs_svd
predictions_qr = X_all @ coefs_qr_loop
errors_qr = y_all - predictions_qr
# print the L2 norm of the errors using the SVD coefs
print("L2 norm of errors using the QR coefs: ", np.linalg.norm(errors_qr))

# print the L2 norm of the relative errors using the new coefs
rel_errors_model_qr = errors_qr / y_all
print("L2 norm of relative errors using the SVD coefs: ", np.linalg.norm(rel_errors_model_qr))

# solve the normal equation using SVD
# X = USV^T

# eigen decomposition of a square matrix
# A = V D V^T
# A^-1 = V D^-1 V^T
# X*coefs = y
# A = XT @ X
# Doing an SVD of X is equivalent to doing an eigen dexomposition of XT @ X

# Normal equation: X.T @ X @ coefs = X.T @ y
# Xdagger = (X.T @ X)^-1 @ X.T 
# Xdagger is the pseudo inverse of X

U, S, Vt = np.linalg.svd(X_all, full_matrices=False)

# Find the inverse of X in the least squares sense
S_inv = np.diag(1 / S)

# Compute the coefficients using the SVD decomposition
coefs_svd = Vt.T @ S_inv @ U.T @ y
 
# save coefs_svd to a file for accessing later
np.savetxt('coefs_svd.csv', coefs_svd, delimiter=',')


# find the errors using the coefs_svd
predictions_svd = X_all @ coefs_svd
errors_svd = y_all - predictions_svd
# print the L2 norm of the errors using the SVD coefs
print("L2 norm of errors using the SVD coefs: ", np.linalg.norm(errors_svd))

# print the L2 norm of the relative errors using the new coefs
rel_errors_model_svd = errors_svd / y_all
print("L2 norm of relative errors using the SVD coefs: ", np.linalg.norm(rel_errors_model_svd))