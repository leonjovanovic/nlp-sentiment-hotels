import pandas as pd
from utils import ModelType, import_annotated_json, split_dataset, bag_of_words, tokenize


def preprocess(train_split_ratio=0.8, validation_split=True):
    df = import_annotated_json()
    df_train, df_val, df_test = split_dataset(df, train_split_ratio, validation_split)
    train_input = bag_of_words(df_train.hotel_review)
    val_input = None
    test_input = df_test.apply(lambda x: tokenize([x.hotel_review]), axis=1)
    return df_train, df_val, df_test, train_input, val_input, test_input


def prepare_dataset(input_data, output_data, type):
    if type == ModelType.CATEGORY:
        return pd.concat([input_data, output_data.apply(lambda x: 1 if x[0] == 2 else x[0], axis=1)], axis=1)
    elif type == ModelType.SENTIMENT:
        data = pd.concat([input_data, output_data], axis=1)
        data = data[data[data.columns[-1]] != 0]
        data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: x - 1)
        data.reset_index(drop=True, inplace=True)
        return data
    return pd.concat([input_data, output_data], axis=1)