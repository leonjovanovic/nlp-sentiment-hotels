import utils
import pandas as pd

def preprocess(train_split_ratio=0.8, validation_split=True):
    df = utils.import_annotated_json()
    df_train, df_val, df_test = utils.split_dataset(df, train_split_ratio, validation_split)
    train_input = utils.bag_of_words(df_train.hotel_review)
    val_input = None
    test_input = df_test.apply(lambda x: utils.tokenize([x.hotel_review]), axis=1)
    return df_train, df_val, df_test, train_input, val_input, test_input