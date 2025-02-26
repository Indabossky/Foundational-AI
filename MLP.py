import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple
import math

def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    assert len(train_x) == len(train_y), "Number of samples in X and y do not match."
    batch_x = np.array_split(train_x, math.ceil(len(train_x) / batch_size), axis=0)
    batch_y = np.array_split(train_y, math.ceil(len(train_y) / batch_size), axis=0)
    return batch_x, batch_y


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigma_x = self.forward(x)
        return sigma_x * (1 - sigma_x)

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.square(np.tanh(x))

class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerically stable softmax
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the full Jacobian matrix for each sample in x.
        Given input x of shape (N, K), compute s = softmax(x) of shape (N, K).
        Returns a tensor of shape (N, K, K) where for each sample:
            J[i,j] = s_i * (delta_ij - s_j)
        """
        s = self.forward(x)  # shape (N, K)
        N, K = s.shape
        jacobian = np.zeros((N, K, K))
        # Set diagonal: for each sample, jacobian[n, i, i] = s[n, i]
        jacobian[np.arange(N)[:, None], np.arange(K), np.arange(K)] = s
        # Outer product for each sample
        outer = np.einsum('ni,nj->nij', s, s)
        return jacobian - outer
    
class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
class Mish(ActivationFunction):
    @staticmethod
    def softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(x))
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        sp = Mish.softplus(x)
        return x * np.tanh(sp)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sp = Mish.softplus(x)
        tanh_sp = np.tanh(sp)
        sig = Mish.sigmoid(x)
        return tanh_sp + x * sig * (1 - np.square(tanh_sp))
    

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * np.mean(np.square(y_pred - y_true))
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_true.shape[0]
        

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps) #to avoid log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_true.shape[0]

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float = 0.0):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        #Glorot uniform initialization
        sigma = np.sqrt(6 / (self.fan_in + self.fan_out))
        self.W = np.random.uniform(-sigma, sigma, (self.fan_in, self.fan_out))
        self.b = np.zeros((1, self.fan_out))

        self.input = None      # Input to the layer
        self.Z = None          # Linear combination
        self.activations = None  # Activation output
        self.delta = None      # Delta for backpropagation

    
    def forward(self, h: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        self.input = h
        self.Z = h @ self.W + self.b
        activation = self.activation_function.forward(self.Z)

        if training and self.dropout_rate > 0:
            keep_prob = 1 - self.dropout_rate
            mask = np.random.binomial(1, keep_prob, size=activation.shape) / keep_prob
            activation = activation * mask
        
        self.activations = activation
        return activation

    def backward(self, delta: np.ndarray, is_output_layer: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        For the output layer using Softmax with CrossEntropy, the gradient computed
        in train is already simplified (i.e., (y_pred - y_true)/N) and no further
        derivative multiplication is necessary.
        """
        if is_output_layer and isinstance(self.activation_function, Softmax):
            pre_delta = delta  # Use the precomputed loss gradient directly.
        else:
            # Standard backpropagation for non-output layers.
            pre_delta = delta * self.activation_function.derivative(self.Z)
        
        dL_dW = self.input.T @ pre_delta
        dL_db = np.sum(pre_delta, axis=0, keepdims=True)
        self.delta = pre_delta @ self.W.T
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, loss_grad: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []
        delta = loss_grad

        for i, layer in enumerate(reversed(self.layers)):
            # For the output layer, pass the flag is_output_layer=True.
            if i == 0:
                dl_dw, dl_db = layer.backward(delta, is_output_layer=True)
            else:
                dl_dw, dl_db = layer.backward(delta, is_output_layer=False)
        
            dl_dw_all.insert(0, dl_dw)
            dl_db_all.insert(0, dl_db)
            delta = layer.delta
        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, 
              learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, use_rmsprop: bool = False, decay_rate: float = 0.9, 
              epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        x_batches, y_batches = batch_generator(train_x, train_y, batch_size)
        training_losses = []
        validation_losses = []

        if use_rmsprop:
            weight_cache = [np.zeros_like(layer.W) for layer in self.layers]
            bias_cache = [np.zeros_like(layer.b) for layer in self.layers]

        for epoch in range(epochs):
            total_loss = 0
            for bx, by in zip(x_batches, y_batches):
                y_pred = self.forward(bx, training=True)
                loss = loss_func.loss(by, y_pred)
                total_loss += loss
                loss_grad = loss_func.derivative(by, y_pred)
                weight_grad, bias_grad = self.backward(loss_grad)

                if use_rmsprop:
                    # For RMSProp, update the cache and then adjust the weights and biases
                    for i, layer in enumerate(self.layers):
                        weight_cache[i] = decay_rate * weight_cache[i] + (1 - decay_rate) * np.square(weight_grad[i])
                        bias_cache[i] = decay_rate * bias_cache[i] + (1 - decay_rate) * np.square(bias_grad[i])
                        layer.W -= learning_rate * weight_grad[i] / (np.sqrt(weight_cache[i]) + epsilon)
                        layer.b -= learning_rate * bias_grad[i] / (np.sqrt(bias_cache[i]) + epsilon)
                else:
                    #Implement Vanilla Gradient Descent
                    for i, layer in enumerate(self.layers):
                        layer.W -= learning_rate * weight_grad[i]
                        layer.b -= learning_rate * bias_grad[i]
            val_loss = loss_func.loss(val_y, self.forward(val_x))
            train_loss = total_loss / len(x_batches)

            training_losses.append(train_loss)
            validation_losses.append(val_loss)

            print(f"Epoch {epoch+1}  :::  Train Loss={train_loss}  :::  Val Loss={val_loss}")
        return np.array(training_losses), np.array(validation_losses)