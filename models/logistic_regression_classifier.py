import numpy as np
import pandas as pd
from preprocess import prepare_dataset
from utils import ModelType, read_hyperparameters
from sklearn.metrics import f1_score


class LogisticRegression:
    def __init__(self, type: int, hyperparameters=None) -> None:
        if type > ModelType.BOTH:
            raise Exception('Bad value for ModelType enum')
        self.W = None
        self.type = type
        self.k = 3 if self.type == ModelType.BOTH else 1
        if not hyperparameters:
            hyperparameters = read_hyperparameters('logistic_regression', self.type)
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.num_of_iterations = hyperparameters['num_of_iterations']
        self.learning_rate = hyperparameters['learning_rate']
        self.regularization_type = hyperparameters['regularization_type']
        self.reg_lambda = hyperparameters['lambda']            

    def split_data_input_output(self, data):
        return data[:, :-1], data[:, -1] # mini_batch_size x num_of_features, mini_batch_size x 1

    def calculate_hypothesis(self, X):
        # Multiplying parameters with input features which need to be transposed for vector multiplication to work
        Z = np.dot(self.W, X.transpose()) # num_of_classes (1 in binary case) x mini_batch_size
        if self.type != ModelType.BOTH:
            # Applying Sigmoid function to get hypotesis h(x) = 1 / (1 + e^-(w0*1 + w1*x1 + w2*x2 + ... + wn*xn)) = 1 / (1 + e^-W*X^T) = 1 / (1 + e^-Z) = H
            H = 1 / (1 + np.exp(-Z)) # 1 x mini_batch_size
        else:
            # Applying Softmax function to get hypotesis h(x) = 1 / sum(e^Wk*X^T) * [e^W1*X^T, e^W2*X^T, ...] = 1 / sum(Z) * Z = H
            H = (1 / np.sum(np.exp(Z), axis=0)) * np.exp(Z) # num_of_classes x mini_batch_size
            # https://www.youtube.com/watch?v=pZbkVup7fYE
            #c = np.max(Z, axis=0) # 1 x mini_batch_size
            #H = np.exp(Z - c) / np.sum(np.exp(Z - c), axis=0) # num_of_classes x mini_batch_size
        return H

    def regularization(self, function:str) -> float:
        # Regularization
        if self.regularization_type == 'L1':
            if function == 'cost_function':
                return self.reg_lambda * np.sum(abs(self.W)) / 2
            elif function == 'derivative':
                return self.reg_lambda * np.sign(self.W) / 2
        elif self.regularization_type == 'L2':
            if function == 'cost_function':
                return self.reg_lambda * np.sum(self.W**2) / 2
            elif function == 'derivative':
                return self.reg_lambda * self.W
        return 0    

    def calculate_cost_function(self, H, Y):
        if self.type != ModelType.BOTH:
            L = Y * np.log(H) + (1 - Y) * np.log(1 - H) # mini_batch_size x 1
            J = -np.mean(L)# 1 x 1
        else:
            L = -np.log(H) # num_of_classes x mini_batch_size
            one_hot_Y = np.eye(self.k)[np.array(Y).reshape(-1).astype(int)] # mini_batch_size x num_of_classes
            J = np.mean(np.sum(one_hot_Y * L.transpose(), axis=1))# 1 x 1
        return J + self.regularization('cost_function')

    def compute(self, data):
        X, Y = self.split_data_input_output(data)
        # Adding ones as first element in every row for future multiplication with w0 (bias term)
        ones_bias_term = np.ones([X.shape[0], 1])
        X = np.hstack((ones_bias_term, X)) # mini_batch_size x 1(one) + num_of_features
        H = self.calculate_hypothesis(X)
        J = self.calculate_cost_function(H, Y)
        return X, Y, H, J

    def update_parameters(self, X, Y, H):
        if self.type != ModelType.BOTH:
            dJ = np.dot((H - Y), X)  # 1 x num_of_features + 1
        else:
            one_hot_Y = np.eye(self.k)[np.array(Y).reshape(-1).astype(int)] # mini_batch_size x num_of_classes
            dJ = -np.dot((one_hot_Y - H.transpose()).transpose(), X) / X.shape[0]  # num_of_classes x num_of_features + 1
        self.W -= (self.learning_rate / X.shape[0]) * (dJ  + self.regularization('derivative'))

    def train_mini_batch(self, data, mini_batch_index):
        mini_batch_data = data[mini_batch_index:mini_batch_index + self.mini_batch_size] # mini_batch_size x num_of_features + 1 (output)
        X, Y, H, J = self.compute(mini_batch_data)
        self.update_parameters(X, Y, H)
        return J

    def train(self, train_input_data, train_output_data, validation_input_data, validation_output_data):
        if self.type == ModelType.BOTH:
            np.random.seed(0)
            self.W = np.random.uniform(0.0, 1.0, (self.k, len(train_input_data.columns) + 1)) # Weights for Logistic regression 1 x num_of_features + 1 for bias term
        else:
            self.W = np.zeros([self.k, len(train_input_data.columns) + 1])
        train_dataset = prepare_dataset(train_input_data, train_output_data, self.type).to_numpy()

        for iteration in range(self.num_of_iterations):        
            for mini_batch_index in range(0, train_dataset.shape[0], self.mini_batch_size):
                J_train = self.train_mini_batch(train_dataset, mini_batch_index)
                J_validation, accuracy_val = self.test(validation_input_data, validation_output_data)
            #print(f'        It{iteration}: J_train = {J_train}, J_val = {J_validation}, Acc_val = {accuracy_val}')

    def test(self, input_data, output_data):
        data = prepare_dataset(input_data, output_data, self.type).to_numpy()
        _, Y, H, J = self.compute(data)
        if self.k > 1:
            H = H.argmax(axis=0) # Number of class which has highest probability (1 x batch_size)
        else:
            H[H>=0.5] = 1
            H[H<0.5] = 0
        return J, f1_score(Y, H, average='macro')
    
    def predict(self, X):
        # Adding ones as first element in every row for future multiplication with w0 (bias term)
        ones_bias_term = np.ones([X.shape[0], 1])
        X = np.hstack((ones_bias_term, X)) # mini_batch_size x 1(one) + num_of_features
        H = self.calculate_hypothesis(X)
        if self.k > 1:
            H = H.argmax(axis=0) # Number of class which has highest probability (1 x batch_size)
        else:
            H[H>=0.5] = 1
            H[H<0.5] = 0
        return H


class LogisticRegressionCombined:
    def __init__(self):
        self.regression_category = LogisticRegression(ModelType.CATEGORY)
        self.regression_sentiment = LogisticRegression(ModelType.SENTIMENT)

    def train(self, train_input_data, train_output_data, validation_input_data, validation_output_data):
        self.regression_category.train(train_input_data, train_output_data, validation_input_data, validation_output_data)
        self.regression_sentiment.train(train_input_data, train_output_data, validation_input_data, validation_output_data)
    
    def test(self, input_data, output_data):
        prediction = input_data.apply(lambda x: self.compute(pd.DataFrame(x).transpose()), axis=1)
        df_test = pd.concat([output_data, prediction], axis=1)
        return len(df_test[df_test[df_test.columns[-1]] != df_test[df_test.columns[-2]]])/len(df_test), \
            f1_score(df_test[df_test.columns[-2]], df_test[df_test.columns[-1]], average='macro')

    def compute(self, review):
        if self.regression_category.predict(review):
            return self.regression_sentiment.predict(review) + 1
        else:
            return 0

