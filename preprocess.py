import pandas as pd
from utils import ModelType, bag_of_words, split_into_words, read_preprocess_parameters, tf_idf


def checkBayesParamsNotValid(is_bayes: bool, params: dict) -> bool:
    if not is_bayes:
        return False
    if params['preprocess_type'] != 'bag_of_words' or params['binary'] == True:
        return True
    return False


def preprocess(train_input: pd.Series, test_or_val_input: pd.Series, is_bayes: bool, preprocess_params=None):
    if preprocess_params is None:
        params = read_preprocess_parameters()
    else:
        params = preprocess_params
    
    if checkBayesParamsNotValid(is_bayes, params):
        raise Exception("Wrong preprocessing type for bayes classifier!")

    if params['preprocess_type'] == 'bag_of_words':
        train_input, vectorizer = bag_of_words(train_input, None, params)
        if is_bayes:
            test_or_val_input = split_into_words(test_or_val_input, params)
        else:
            test_or_val_input, _ = bag_of_words(test_or_val_input, vectorizer, params)
    elif params['preprocess_type'] == 'tf_idf' or params['preprocess_type'] == 'tf':
        train_input, vectorizer = tf_idf(train_input, None, params)
        test_or_val_input, _ = tf_idf(test_or_val_input, vectorizer, params)
    else: 
        raise Exception("Wrong preprocess type!")
    return train_input, test_or_val_input


def prepare_dataset(input_data, output_data, model_type):
    if model_type == ModelType.CATEGORY or model_type == ModelType.SVM_ZERO:
        return pd.concat([input_data, output_data.apply(lambda x: 1 if x[0] == 2 else x[0], axis=1).rename(output_data.columns[0])], axis=1)
    elif model_type == ModelType.SENTIMENT:
        data = pd.concat([input_data, output_data], axis=1)
        data = data[data[data.columns[-1]] != 0]
        data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: x - 1)
        data.reset_index(drop=True, inplace=True)
        return data
    elif model_type == ModelType.BOTH:
        return pd.concat([input_data, output_data], axis=1)
    elif model_type == ModelType.SVM_ONE:
        return pd.concat([input_data, output_data.apply(lambda x: 0 if x[0] == 1 else 1, axis=1).rename(output_data.columns[0])], axis=1)
    elif model_type == ModelType.SVM_TWO:
        return pd.concat([input_data, output_data.apply(lambda x: 0 if x[0] == 2 else 1, axis=1).rename(output_data.columns[0])], axis=1)