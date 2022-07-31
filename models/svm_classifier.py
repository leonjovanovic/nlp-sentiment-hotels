

from matplotlib.pyplot import cla
import numpy as np

from preprocess import prepare_dataset
from utils import ModelType, read_hyperparameters


class SupportVectorMachine:
    # BINARY SO FAR
    def __init__(self, type: ModelType, hyperparameters=None) -> None:
        self.W = None
        self.type = type
        self.k = 3 if self.type == ModelType.BOTH else 1
        if not hyperparameters:
            hyperparameters = read_hyperparameters('logistic_regression', self.type)
        self.num_of_iterations = hyperparameters['num_of_iterations']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.cost_function_type = hyperparameters['cost_function_type']
        self.C = hyperparameters['C']

    def split_data_input_output(self, data):
        return data[:, :-1], data[:, -1] # mini_batch_size x num_of_features, mini_batch_size x 1

    def calculate_hypothesis(self, X):
        # Multiplying parameters with input features which need to be transposed for vector multiplication to work
        return np.dot(self.W, X.transpose()) # num_of_classes (1 in binary case) x mini_batch_size

    def calculate_cost_function(self, H, Y):
        L = np.maximum(0, 1 - Y * H) # num_of_classes (1 in binary case) x mini_batch_size
        if self.cost_function_type == 'L2':
            L = L**2
        J = self.C * np.sum(L) / Y.shape[0] + np.sum(self.W**2) / 2
        return J

    def compute(self, data):
        X, Y = self.split_data_input_output(data)
        # Adding ones as first element in every row for future multiplication with w0 (bias term)
        ones_bias_term = np.ones([X.shape[0], 1])
        X = np.hstack((ones_bias_term, X)) # mini_batch_size x 1(one) + num_of_features
        H = self.calculate_hypothesis(X)
        J = self.calculate_cost_function(H, Y) 
        return X, Y, H, J

    def update_parameters(self, X, Y, H):
        # https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2#1001
        # https://www.freecodecamp.org/news/support-vector-machines/
        # https://rti.etf.bg.ac.rs/rti/ms1opj/pdf/Metoda%20potpornih%20vektora.pdf
        dJ = 1 #<-------------------------------------------------
        self.W -= self.learning_rate * dJ

    def train_mini_batch(self, data, mini_batch_index):
        mini_batch_data = data[mini_batch_index:mini_batch_index + self.mini_batch_size] # mini_batch_size x num_of_features + 1 (output)
        X, Y, H, J = self.compute(mini_batch_data)
        self.update_parameters(X, Y, H) #<-------------------------------------------------
        return J

    def train(self, train_input_data, train_output_data, validation_input_data, validation_output_data):
        self.W = np.zeros([1, len(train_input_data.columns) + 1]) # 1 x number_of_features + 1        
        train_dataset = prepare_dataset(train_input_data, train_output_data, self.type).to_numpy()

        for iteration in range(self.num_of_iterations):        
            for mini_batch_index in range(0, train_dataset.shape[0], self.mini_batch_size):
                J_train = self.train_mini_batch(train_dataset, mini_batch_index) #<-------------------------------------------------
                #J_validation, accuracy_val = self.test(validation_input_data, validation_output_data)
            #print(f'        It{iteration}: J_train = {J_train}, J_val = {J_validation}, Acc_val = {accuracy_val}')




class SupportVectorMachineCombined:
    def __init__(self) -> None:
        pass