from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.023
iterations = 100
#y1=0
#y2=0
# plot predictions for every iteration?
do_plot = True

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)

#########################################
# Write your code here
x1=[1650,3]
x2=[3000,4]
# Create two new samples: (1650, 3) and (3000, 4)
x1=(x1-mean_vec)/std_vec
x2=(x2-mean_vec)/std_vec
# Calculate the hypothesis for each sample, using the trained parameters theta_final
#h2=calculate_hypothesis(x2,theta_final,0)

#h1=calculate_hypothesis(x1,theta_final,0)
h1= theta_final[0] + theta_final[1] * x1[0,0] + theta_final[2]* x1[0,1]
h2= theta_final[0] + theta_final[1] * x2[0,0] + theta_final[2]* x2[0,1]


# Make sure to apply the same preprocessing that was applied to the training data
# Print the predicted prices for the two samples
#print(h1)
print(theta_final)
print('price of the house is:')
print(h1)
print('price of the house is:')
print(h2)
########################################/
