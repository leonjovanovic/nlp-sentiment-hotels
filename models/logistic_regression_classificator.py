import numpy as np
from preprocess import prepare_dataset
from utils import ModelType

class LogisticRegression:
    def __init__(self, type: ModelType) -> None:
        self.W = None
        self.type = type
        self.k = 3 if self.type == ModelType.BOTH else 1


    def split_data_input_output(self, data):
        return data[:, :-1], data[:, -1] # mini_batch_size x num_of_features, mini_batch_size x 1


    def calculate_hypotesis(self, X):
        # Multiplying parameters with input features which need to be transponed for vector multiplication to work
        Z = np.dot(self.W, X.transpose()) # 1 x mini_batch_size
        if self.type != ModelType.BOTH:
            # Applying Sigmoid function to get hypotesis h(x) = 1 / (1 + e^-(w0*1 + w1*x1 + w2*x2 + ... + wn*xn)) = 1 / (1 + e^-W*X^T) = 1 / (1 + e^-Z) = H
            H = 1 / (1 + np.exp(-Z)) # 1 x mini_batch_size
        else:
            # Applying Softmax function to get hypotesis h(x) = 1 / sum(e^Wk*X^T) * [e^W1*X^T, e^W2*X^T, ...] = 1 / sum(Z) * Z = H
            H = 1 / np.sum(Z, axis=0) * Z # num_of_classes x mini_batch_size
        return H


    def calculate_cost_function(self, H, Y):
        if self.type != ModelType.BOTH:
            L = -Y * np.log(H) - (1 - Y) * np.log(1 - H) # mini_batch_size x 1
            J = -np.mean(L) # 1 x 1
        else:
            L = -np.log(H) # num_of_classes x mini_batch_size
            one_hot_Y = np.eye(self.k)[np.array(Y).reshape(-1)] # mini_batch_size x num_of_classes
            J = np.mean(np.sum(one_hot_Y * L.transpose(), axis=1)) # 1 x 1
        return J


    def compute(self, data):
        X, Y = self.split_data_input_output(data)
        # Adding ones as first element in every row for future multiplication with w0 (bias term)
        ones_bias_term = np.ones([mini_batch_size, 1])
        X = np.hstack(ones_bias_term, X) # mini_batch_size x 1(one) + num_of_features
        H = self.calculate_hypotesis(X)
        J = self.calculate_cost_function(H, Y)
        return X, Y, H, J


    def update_parameters(self, X, Y, H):
        if self.type != ModelType.BOTH:
            dJ = np.dot((H - Y).transpose(), X)  # 1 x num_of_features + 1
        else:
            one_hot_Y = np.eye(self.k)[np.array(Y).reshape(-1)] # mini_batch_size x num_of_classes
            dJ = np.dot((one_hot_Y - H.transpose()).transpose(), X)  # num_of_classes x num_of_features + 1
        self.W -= (learning_rate / X.shape[0]) * dJ


    def train_mini_batch(self, data, mini_batch_index):
        mini_batch_data = data[mini_batch_index:mini_batch_index + mini_batch_size] # mini_batch_size x num_of_features + 1 (output)
        X, Y, H, J = self.compute(mini_batch_data)
        self.update_parameters(X, Y, H)
        return J


    def train(self, train_input_data, train_output_data, validation_input_data = None, validation_output_data = None):
        self.W = np.zeros([self.k, len(train_input_data.columns) + 1]) # Weights for Logistic regression 1 x num_of_features + 1 for bias term
        train_dataset = prepare_dataset(train_input_data, train_output_data, self.type).to_numpy()
        validation_dataset = prepare_dataset(validation_input_data, validation_output_data, self.type).to_numpy() if validation_input_data else None

        for iteration in range(num_of_iterations):        
            for mini_batch_index in range(0, train_dataset.shape[0], mini_batch_size):
                J_train = self.train_mini_batch(train_dataset, mini_batch_index)
                if  validation_dataset: 
                    _, _, H_validation, J_validation = self.compute(validation_dataset)

