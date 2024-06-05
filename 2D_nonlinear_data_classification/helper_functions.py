import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def load_flower_dataset(num_samples=500, petals=4, petal_length=4, noise=0.2, angle=30)->Tuple[np.ndarray,np.ndarray]:
    '''
    Create synthetic flower 2D dataset with two classes(0/1)
    
    Parameters
    ----------
    num_samples : int (default=500)
        number of overall samples to be created
    petals : int (default=4)
        represents half the number of petals of flower
    petal_length : int (default=4)
    noise : float (default=0.2)
    angle : int (default=30)
        angle in degrees to couter-clockwise rotate the flower
    
    Returns
    -------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data; dtype: float64
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1; dtype: uint8
    '''
    np.random.seed(1)
    m = num_samples # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*np.pi,(j+1)*np.pi,N) + np.random.randn(N)*noise # theta, classes mixing imperfection
        r = petal_length*np.sin(petals*t) + np.random.randn(N)*noise # radius, petal shape imperfection
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    # rotating data points
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    X = np.dot(X, rotation_matrix)
    # previous shape is (num_samples, dim), we need to transpose it to (dim, num_samples
    X = X.T
    # previous shape is (num_samples,), we need to reshape it to (1, num_samples)
    Y = Y.T

    return X, Y


def plot_scatter(X:np.ndarray, Y:np.ndarray):
    '''
    Show the scatter plot of flower dataset
    
    Parameters
    ----------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data; dtype: float64
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1; dtype: uint8
    '''
    scatter=plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()
    

def plot_decision_boundary(w:np.ndarray, b:np.float64, X:np.ndarray, Y:np.ndarray):
    """
    Plot the decision boundary for logistic regression
    
    Parameters
    ----------
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron; dtype: float64
    b : np.float64
        bias used by neuron
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data; dtype: float64
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1; dtype: uint8
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    # Z = predict(w,b,np.c_[xx.ravel(), yy.ravel()].T)
    z = np.matmul(w.T,np.c_[xx.ravel(), yy.ravel()].T)+b
    a = 1/(1+np.exp(-z))        # shape(1,m)
    Z = (a >0.5).astype(int)    # shape(1,m)
    
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plot_scatter(X,Y)


def plot_1hidden_layer_nn_decision_boundary(W1, b1, W2, b2, X, Y):
    """
    Plot the decision boundary for logistic regression
    
    Parameters
    ----------
    W1 : numpy.ndarray [shape: (#units_in_layer1, #units_in_layer0)]
        matrix containing weights of neurons in first layer; dtype: float64
    b1 : numpy.ndarray [shape: (#units_in_layer1, 1)]
        array containing biases of neurons in first layer; dtype: float64
    W2 : numpy.ndarray [shape: (#units_in_layer2, #units_in_layer1)]
        matrix containing weights of neurons in second layer; dtype: float64
    b2 : numpy.ndarray [shape: (#units_in_layer2, 1)]
        array containing biases of neurons in second layer; dtype: float64
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data; dtype: float64
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1; dtype: uint8
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    # Z = predict(w,b,np.c_[xx.ravel(), yy.ravel()].T)
    Z1 = np.matmul(W1,np.c_[xx.ravel(), yy.ravel()].T) + b1         # shape(n_h,m)
    A1 = np.tanh(Z1)                  # shape(n_h,m)
    Z2 = np.matmul(W2,A1) + b2        # shape(n_y,m)
    A2 = 1/(1 + np.exp(-Z2))          # shape(n_y,m)
    Y_pred = (A2 >0.5).astype(int)    # shape(n_y,m)
    
    Y_pred = Y_pred.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Y_pred, cmap=plt.cm.Spectral)
    plot_scatter(X,Y)