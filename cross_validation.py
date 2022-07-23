from utils import split_dataset_into_two, bag_of_words, tokenize
import numpy as np

k_fold = 10

def cross_validation(input_data, output_data, model):
    J_values = []
    Accuracy_values = []
    for i in range(k_fold):
        train_input, test_or_val_input = split_dataset_into_two(input_data, i, k_fold)
        train_output, test_or_val_output = split_dataset_into_two(output_data, i, k_fold)
        train_input = bag_of_words(train_input.hotel_review)
        test_or_val_input = test_or_val_input.apply(lambda x: tokenize([x.hotel_review]), axis=1)
        model.train(train_input, train_output)
        J, accuracy = model.test(test_or_val_input, test_or_val_output)
        J_values.append(J)
        Accuracy_values.append(accuracy)
        print(f'Iteration {i}: Model accuracy is {accuracy}, cost function is {J}')
    return np.mean(J_values), np.mean(Accuracy_values)
