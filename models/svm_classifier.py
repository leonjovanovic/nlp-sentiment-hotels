

import numpy as np
import pandas as pd

from preprocess import prepare_dataset
from utils import ModelType, read_hyperparameters
from sklearn.feature_extraction.text import CountVectorizer


class SupportVectorMachine:
    # BINARY SO FAR
    def __init__(self, type: ModelType, hyperparameters=None) -> None:
        self.W = None
        self.features_mapping = None
        self.type = type
        if not hyperparameters:
            hyperparameters = read_hyperparameters('svm', self.type)
        self.num_of_iterations = hyperparameters['num_of_iterations']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.learning_rate = hyperparameters['learning_rate']
        self.regularization_type = hyperparameters['regularization_type']
        self.cost_function_type = hyperparameters['cost_function_type']
        self.C = hyperparameters['C']

    def split_data_input_output(self, data):
        return data[:, :-1], data[:, -1] # mini_batch_size x num_of_features, mini_batch_size x 1

    def calculate_hypothesis(self, X):
        # Multiplying parameters with input features which need to be transposed for vector multiplication to work
        return np.dot(self.W, X.transpose()) # num_of_classes (1 in binary case) x mini_batch_size

    def calculate_cost_function(self, H, Y):
        distance = 1 - Y * H
        L = np.maximum(0, distance) # num_of_classes (1 in binary case) x mini_batch_size
        if self.cost_function_type == 'L2':
            L = L**2
        if self.regularization_type == 'L1':
            J = self.C * np.sum(L) / Y.shape[0] + np.sum(self.W)
        elif self.regularization_type == 'L2':
            J = self.C * np.sum(L) / Y.shape[0] + np.sum(self.W**2) / 2
        return distance, J

    def compute(self, data):
        X, Y = self.split_data_input_output(data)
        # Output is 0 and 1 and SVM classes are -1 and 1
        Y = Y * 2 - 1
        # Adding ones as first element in every row for future multiplication with w0 (bias term)
        ones_bias_term = np.ones([X.shape[0], 1])
        X = np.hstack((ones_bias_term, X)) # mini_batch_size x 1(one) + num_of_features
        H = self.calculate_hypothesis(X)
        distance, J = self.calculate_cost_function(H, Y) 
        return X, Y, H, distance, J

    def update_parameters(self, X, Y, D):
        M = np.array(D > 0, dtype=int).transpose() # mini_batch_size x 1
        if self.regularization_type == 'L1':
            dJ = -self.C * np.dot(Y.transpose(), X * M) + 1
        elif self.regularization_type == 'L2':
            dJ = -self.C * np.dot(Y.transpose(), X * M) + self.W
        self.W -= self.learning_rate * dJ

    def train_mini_batch(self, data, mini_batch_index):
        mini_batch_data = data[mini_batch_index:mini_batch_index + self.mini_batch_size] # mini_batch_size x num_of_features + 1 (output)
        X, Y, H, D, J = self.compute(mini_batch_data)
        self.update_parameters(X, Y, D) 
        return J

    def train(self, train_input_data, train_output_data, validation_input_data, validation_output_data):
        self.W = np.zeros([1, len(train_input_data.columns) + 1]) # 1 x number_of_features + 1        
        train_dataset = prepare_dataset(train_input_data, train_output_data, self.type).to_numpy()

        self.features_mapping = {}
        for ind, feature in enumerate(list(train_input_data.columns)):
            self.features_mapping[feature] = ind

        for iteration in range(self.num_of_iterations):        
            for mini_batch_index in range(0, train_dataset.shape[0], self.mini_batch_size):
                J_train = self.train_mini_batch(train_dataset, mini_batch_index)
                J_validation, accuracy_val = self.test(validation_input_data, validation_output_data)
            print(f'        It{iteration}: J_train = {J_train}, J_val = {J_validation}, Acc_val = {accuracy_val}')

    def test(self, input_data, output_data):
        vectorizer = CountVectorizer(vocabulary=self.features_mapping)
        input_data = pd.DataFrame(vectorizer.transform(input_data.hotel_review).toarray(), columns=vectorizer.get_feature_names_out())
        data = prepare_dataset(input_data, output_data, self.type).to_numpy()
        _, Y, H, distance, J = self.compute(data)
        H[H>=0] = 1
        H[H<0] = -1
        accuracy = (np.count_nonzero(H == Y)) / Y.shape[0]
        return J, accuracy
    
    def predict(self, review):
        vectorizer = CountVectorizer(vocabulary=self.features_mapping)
        X = pd.DataFrame(vectorizer.transform(review).toarray(), columns=vectorizer.get_feature_names_out())
        # Adding ones as first element in every row for future multiplication with w0 (bias term)
        ones_bias_term = np.ones([X.shape[0], 1])
        X = np.hstack((ones_bias_term, X)) # mini_batch_size x 1(one) + num_of_features
        H = self.calculate_hypothesis(X)
        # H[H>=0] = 1
        # H[H<0] = 0
        return H


class SupportVectorMachineCombined:
    def __init__(self):
        # should hyperparameters be passed to the constructor or should it use the best hyperparams found for each model?
        self.svm_category = SupportVectorMachine(ModelType.CATEGORY)
        self.svm_sentiment = SupportVectorMachine(ModelType.SENTIMENT)

    def train(self, train_input_data, train_output_data, validation_input_data, validation_output_data):
        self.svm_category.train(train_input_data, train_output_data, validation_input_data, validation_output_data)
        self.svm_sentiment.train(train_input_data, train_output_data, validation_input_data, validation_output_data)
    
    def test(self, input_data, output_data):
        input_data = pd.DataFrame(input_data)
        prediction = input_data.apply(lambda x: self.compute(x), axis=1)
        df_test = pd.concat([input_data, output_data, prediction.rename('prediction')], axis=1)
        return len(df_test[df_test['prediction'] != df_test[df_test.columns[-2]]])/len(df_test), \
               len(df_test[df_test['prediction'] == df_test[df_test.columns[-2]]])/len(df_test)*100.0

    def compute(self, review):
        if self.regression_category.predict(review) >= 0:
            return int(self.regression_sentiment.predict(review) >= 0) + 1
        else:
            return 0


class SupportVectorMachineOneVsRest:
    def __init__(self) -> None:
        self.svm_zero = SupportVectorMachine(ModelType.SVM_ZERO) # Preraditi podatke tako 0 odgovara -1, a 1 i 2 odgovaraju 1
        self.svm_one = SupportVectorMachine(ModelType.SVM_ONE) # Preraditi podatke tako 1 odgovara -1, a 0 i 2 odgovaraju 1
        self.svm_two = SupportVectorMachine(ModelType.SVM_TWO) # Preraditi podatke tako 2 odgovara -1, a 0 i 1 odgovaraju 1
        # Onaj model koji vrati najmanju razdaljinu od 0 prema -1 je pobednik

    def train(self, train_input_data, train_output_data, validation_input_data, validation_output_data):
        self.svm_zero.train(train_input_data, train_output_data, validation_input_data, validation_output_data)
        self.svm_one.train(train_input_data, train_output_data, validation_input_data, validation_output_data)
        self.svm_two.train(train_input_data, train_output_data, validation_input_data, validation_output_data)
    
    def test(self, input_data, output_data):
        prediction = self.svm_zero.predict(input_data.hotel_review) # 1 x batch_size
        prediction = np.vstack((prediction, self.svm_one.predict(input_data.hotel_review))) # 2 x batch_size
        prediction = np.vstack((prediction, self.svm_two.predict(input_data.hotel_review))) # 3 x batch_size
        prediction = np.argmin(prediction, axis=0)
        df_test = pd.concat([input_data, output_data, pd.Series(prediction)], axis=1)
        return len(df_test[df_test[df_test.columns[-1]] != df_test[df_test.columns[-2]]])/len(df_test), \
               len(df_test[df_test[df_test.columns[-1]] == df_test[df_test.columns[-2]]])/len(df_test)*100.0
