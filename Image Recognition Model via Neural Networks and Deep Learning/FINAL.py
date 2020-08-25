# Import data
import mnist_load
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

import NeuralNetworkAlgorithm as nt

# Train basic network to test out setup
# Using cross entropy cost function
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 10, 10, 0.5, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Best: 94.73%

# Adjust weight initialization
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost)
net.default_weight_initializer() # Not necessary since this is default method
net.SGD(training_data, 10, 10, 0.5, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Best: 95.92%
# Better starting point

# Adjust cost function from cross entropy to quadratic
net = nt.Network([784, 30, 10], cost=nt.QuadraticCost)
net.default_weight_initializer()
net.SGD(training_data, 10, 10, 0.5, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Best: 95.51%
# Cross entropy is slightly better

# Add regularization
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost)
net.default_weight_initializer()
net.SGD(training_data, 10, 10, 0.5, 
        lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Best: 95.84%
# Comparable to model with similar hyper-parameters without regularization

# Increase number of epochs to 30
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost)
net.default_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, 
        lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 30 epochs
# Best result so far, but training stagnated pretty quickly

# Decrease number of epochs back to 10 to speed up training,
# but increase hidden layer to 100
net = nt.Network([784, 100, 10], cost=nt.CrossEntropyCost)
net.default_weight_initializer()
net.SGD(training_data, epochs=10, mini_batch_size=10, 
        eta=0.5, lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Best: 97.59%
# Significant improvement

# After ReLU was added, hyper-parameters had to be adjusted to produce 
# decent results. This is especially true for learning rate

# Typical run of 10 epochs with 39 hidden neurons
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 10, 5, 0.05, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Best: 95.17%

# Testing if ReLU can beat Sigmoid with 100 hidden neurons
net = nt.Network([784, 100, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 10, 5, 0.05, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Best: 97.82% 
# Only slightly better than sigmoid network

# Forgot to test regularization with ReLU
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 10, 5, 0.05, 
        lmbda = 5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Best: 95.82%
# Noticeably better than without regularization

# Different lambda
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 10, 5, 0.05, 
        lmbda = 50,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Best: 95.77%
# Different learning, but similar outcome

# Code for displaying a single MNIST digit
img_index = 54
from matplotlib import pyplot as plt
import numpy as np
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt
gen_image(test_data[img_index][0]).show()

# Testing out calculating and accessing predictions and actual values
# Prediction
net.feedforward(test_data[img_index][0])
net.predict(test_data[img_index][0])
# Actual
np.argmax(training_data[img_index][1])
test_data[img_index][1]

# Loop through test data and gather all mis-classifications
err = []
for pixels, digit in test_data:
    p = net.feedforward(pixels)
    if (digit != np.argmax(p)):
        err.append([pixels,digit,p])

# Sample wrong prediction
gen_image(err[1][0]).show() # Image
np.argmax(err[1][2])        # Predicted digit
np.round(err[1][2], 4)      # NN output array
err[1][1]                   # Actual digit

# Softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Softmax input
softmax(net.feedforward(test_data[img_index][0]))
softmax(err[1][2])
softmax(err[1][2]*10)

# ADDING VARIABLE LEARNING RATE TO THE NETWORK CODE

import datetime
print(datetime.datetime.now())
net = nt.Network([784, 100, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 60, 5, 0.05, 
        lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
        early_stopping_n=4,
        variable_learning_coef=0.5)
print(datetime.datetime.now())


# Just for quick reference evaluate network with 2 hidden layers
# (30 and 30 neurons) and same hyper-parameters. 
print(datetime.datetime.now())
net = nt.Network([784, 30, 30, 10], 
                 cost=nt.CrossEntropyCost, 
                 neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 60, 5, 0.05, 
        lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
        early_stopping_n=4,
        variable_learning_coef=0.5)
print(datetime.datetime.now())
# Stopped training after 32 epochs. 
# Best: 96.57%