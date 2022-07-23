from utils import split_dataset_into_two, bag_of_words,tokenize
import numpy as np

k_fold = 10

def cross_validation(input_data, output_data, ratio, model):
    J_values = []
    for i in range(k_fold):
        train_input, test_or_val_input = split_dataset_into_two(input_data, ratio)
        train_output, test_or_val_output = split_dataset_into_two(output_data, ratio)
        train_input = bag_of_words(train_input.hotel_review)
        test_or_val_input = test_or_val_input.apply(lambda x: tokenize([x.hotel_review]), axis=1)
        model.train(train_input, train_output)
        J_values.append(model.test(test_or_val_input, test_or_val_output))
        print(f'{i}: {J_values[i]}')
    return np.mean(J_values)
