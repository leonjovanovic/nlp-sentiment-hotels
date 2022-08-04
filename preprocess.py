import pandas as pd
from utils import ModelType, PreprocessType, import_annotated_json, split_dataset, bag_of_words, split_into_words


def preprocess(train_input: pd.Series, test_or_val_input: pd.Series, preprocess_type: int, is_bayes: bool):
    if preprocess_type == PreprocessType.BAG_OF_WORDS:
        train_input, vocabulary = bag_of_words(train_input)
        if is_bayes:
            test_or_val_input = pd.DataFrame(test_or_val_input.apply(lambda x: split_into_words(x)))
        else:
            test_or_val_input, _ = bag_of_words(test_or_val_input, vocabulary)
    elif preprocess_type == PreprocessType.BAG_OF_WORDS_BINARY:
        a = 1
    elif preprocess_type == PreprocessType.TF:
        a = 1
    elif preprocess_type == PreprocessType.IDF:
        a = 1
    elif preprocess_type == PreprocessType.TF_IDF:
        a = 1
    elif preprocess_type == PreprocessType.BIGRAM:
        a = 1
    elif preprocess_type == PreprocessType.TRIGRAM:
        a = 1
    else: 
        raise Exception("Wrong preprocess type!")
    return train_input, test_or_val_input


def prepare_dataset(input_data, output_data, model_type):
    if model_type == ModelType.CATEGORY or model_type == ModelType.SVM_ZERO:
        return pd.concat([input_data, output_data.apply(lambda x: 1 if x[0] == 2 else x[0], axis=1)], axis=1)
    elif model_type == ModelType.SENTIMENT:
        data = pd.concat([input_data, output_data], axis=1)
        data = data[data[data.columns[-1]] != 0]
        data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: x - 1)
        data.reset_index(drop=True, inplace=True)
        return data
    elif model_type == ModelType.BOTH:
        return pd.concat([input_data, output_data], axis=1)
    elif model_type == ModelType.SVM_ONE:
        return pd.concat([input_data, output_data.apply(lambda x: 0 if x[0] == 1 else 1, axis=1)], axis=1)
    elif model_type == ModelType.SVM_TWO:
        return pd.concat([input_data, output_data.apply(lambda x: 0 if x[0] == 2 else 1, axis=1)], axis=1)


def initial_preprocess(series:pd.Series, lowercasing=True, frequency_filtering=False, stop_words_filtering=False):
    if lowercasing:
        series = lowercase(series)
    if frequency_filtering:
        series = filter_frequent(series)
    if stop_words_filtering:
        series = filter_stop_words(series)
    return series


def lowercase(series):
    series.apply(lambda review : review.lower(), axis=1)
    return series


def filter_frequent(series):
    # TODO
    pass


def filter_stop_words(series):
    # TODO
    pass