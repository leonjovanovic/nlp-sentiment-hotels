from enum import auto
from models.bayes_classifier import NaiveBayes, NaiveBayesCombined
from utils import split_dataset_into_two
from preprocess import preprocess
import numpy as np
from tqdm.auto import tqdm

k_fold = 10


def cross_validation(input_data, output_data, model, preprocess_params=None):
    J_values = []
    Accuracy_values = []
    for i in tqdm(range(k_fold)):
        train_input, test_or_val_input = split_dataset_into_two(input_data, i, k_fold)
        train_output, test_or_val_output = split_dataset_into_two(output_data, i, k_fold)
        
        is_bayes = isinstance(model, NaiveBayes) or isinstance(model, NaiveBayesCombined)
        train_input, test_or_val_input = preprocess(train_input.hotel_review, test_or_val_input.hotel_review, is_bayes, preprocess_params)

        model.train(train_input, train_output, test_or_val_input, test_or_val_output)
        J, accuracy = model.test(test_or_val_input, test_or_val_output)
        J_values.append(J)
        Accuracy_values.append(accuracy)
        # print(f'\t\tIteration {i}: Model accuracy is {accuracy}, cost function is {J}')
    return np.mean(J_values), np.mean(Accuracy_values)


def nested_cross_validation(input_data, output_data, model, preprocess_params=None):
    J_values = []
    Accuracy_values = []
    for i in range(k_fold):
        train_input, test_input = split_dataset_into_two(input_data, i, k_fold)
        train_output, test_output = split_dataset_into_two(output_data, i, k_fold)
        J, accuracy = cross_validation(train_input, train_output, model, preprocess_params)
        J_values.append(J)
        Accuracy_values.append(accuracy)
        # print(f'\tOuter iteration {i} finished: Average model accuracy is {accuracy}, average cost function is {J}')
    return np.mean(J_values), np.mean(Accuracy_values)
