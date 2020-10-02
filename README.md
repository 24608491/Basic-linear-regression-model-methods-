#Linear regression model methods

Three methods for training a linear regresion model are investigated. Namely Stochastic gradient descent, mini-batch gradient descent and a multi-layered perceptron. All of this is done using R-Studio.

Stochastic gradient descent samples a data point from the training data to compute the gradient and uses it to adjust the parameters of the regression model.
Mini-Batch gradient descent samples several data points from the training data at once to compute the gradient and uses it to adjust the parameters of the regression model.
A multi-layered perceptron requires an architecture with the right activation function that is capable of modeling a linear regression model from the input data 

csv files containing the training and testing x and y values were used.Mean sqaured error function was used as the loss function for all the three methods during the training. In the case of the multi-layered perceptron,it's architecture had one hidden layer with five nodes. The 5 nodes were using a ReLu function as the activation function and the output layer used a linear activation function.
